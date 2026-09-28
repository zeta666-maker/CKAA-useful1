"""Adaptive Quantization for Stable Knowledge Acquisition (AQCL).

This module is intentionally self-contained and uses only PyTorch. It provides
asymmetric uniform fake quantization, Fisher-trace sensitivity estimation,
adaptive weight/activation bit allocation, and similarity-aware orthogonal
gradient modulation.
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from typing import Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _as_int(value) -> int:
    if isinstance(value, Tensor):
        return int(value.item())
    return int(value)


class UniformQuantizer(nn.Module):
    """Asymmetric uniform quantizer with a straight-through estimator."""

    def __init__(self, bits: int = 8, per_channel: bool = False):
        super().__init__()
        self.register_buffer("bits", torch.tensor(int(bits), dtype=torch.int64))
        self.per_channel = bool(per_channel)

    @property
    def num_bits(self) -> int:
        return _as_int(self.bits)

    def set_bits(self, bits: int) -> None:
        self.bits.fill_(int(bits))

    def forward(self, x: Tensor) -> Tensor:
        bits = self.num_bits
        if bits >= 32 or bits <= 0:
            return x
        if self.per_channel and x.ndim >= 2:
            reduce_dims = tuple(range(1, x.ndim))
            x_min = x.detach().amin(dim=reduce_dims, keepdim=True)
            x_max = x.detach().amax(dim=reduce_dims, keepdim=True)
        else:
            x_min = x.detach().amin()
            x_max = x.detach().amax()
        level = (x_max - x_min) / float((1 << bits) - 1)
        level = level.clamp_min(torch.finfo(x.dtype).eps)
        q = torch.round((x - x_min) / level) * level + x_min
        return x + (q - x).detach()


class QATLinear(nn.Linear):
    def __init__(self, source, out_features=None, bias=True, device=None, dtype=None, low_bits: int = 8):
        if isinstance(source, nn.Linear):
            super().__init__(
                source.in_features,
                source.out_features,
                bias=source.bias is not None,
                device=source.weight.device,
                dtype=source.weight.dtype,
            )
            self.load_state_dict(source.state_dict())
        else:
            super().__init__(source, out_features, bias=bias, device=device, dtype=dtype)
        self.weight_quant = UniformQuantizer(low_bits, per_channel=True)
        self.activation_quant = UniformQuantizer(low_bits)

    def forward(self, x: Tensor) -> Tensor:
        xq = self.activation_quant(x)
        wq = self.weight_quant(self.weight)
        return F.linear(xq, wq, self.bias)


class QATConv1d(nn.Conv1d):
    def __init__(self, source: nn.Conv1d, low_bits: int = 8):
        super().__init__(
            source.in_channels,
            source.out_channels,
            source.kernel_size,
            stride=source.stride,
            padding=source.padding,
            dilation=source.dilation,
            groups=source.groups,
            bias=source.bias is not None,
            padding_mode=source.padding_mode,
            device=source.weight.device,
            dtype=source.weight.dtype,
        )
        self.load_state_dict(source.state_dict())
        self.weight_quant = UniformQuantizer(low_bits, per_channel=True)
        self.activation_quant = UniformQuantizer(low_bits)

    def forward(self, x: Tensor) -> Tensor:
        xq = self.activation_quant(x)
        wq = self.weight_quant(self.weight)
        return self._conv_forward(xq, wq, self.bias)


def convert_to_qat(
    model: nn.Module,
    low_bits: int = 8,
) -> OrderedDict[str, nn.Module]:
    """Replace Linear/Conv1d layers while preserving parameter names."""
    targets: OrderedDict[str, nn.Module] = OrderedDict()
    weight_replacements: dict[int, Tensor] = {}

    def visit(parent: nn.Module, prefix: str = "") -> None:
        for name, child in list(parent.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, (QATLinear, QATConv1d)):
                targets[full_name] = child
            elif type(child) is nn.Linear:
                replacement = QATLinear(child, low_bits=low_bits)
                weight_replacements[id(child.weight)] = replacement.weight
                parent._modules[name] = replacement
                targets[full_name] = replacement
            elif type(child) is nn.Conv1d:
                replacement = QATConv1d(child, low_bits=low_bits)
                weight_replacements[id(child.weight)] = replacement.weight
                parent._modules[name] = replacement
                targets[full_name] = replacement
            else:
                visit(child, full_name)

    visit(model)
    # CKAA's IntermReader keeps a hard reference to attention.qkv.weight.
    # Point it at the replacement parameter after quantization wrapping.
    for module in model.modules():
        other_args = getattr(module, "other_args", None)
        if not isinstance(other_args, dict):
            continue
        for key, value in list(other_args.items()):
            if isinstance(value, Tensor) and id(value) in weight_replacements:
                other_args[key] = weight_replacements[id(value)]
    return targets


class AQCLContext:
    """Training-time state and deployment helpers for AQCL."""

    def __init__(self, model: nn.Module, args) -> None:
        self.args = args
        self.mode = getattr(args, "aqcl_mode", "rpq_saou")
        self.low_bits = int(getattr(args, "aqcl_low_bits", 8))
        self.high_bits = int(getattr(args, "aqcl_high_bits", max(2 * self.low_bits, 2)))
        self.fixed_bits = int(getattr(args, "aqcl_bits", self.low_bits))
        self.lambda_threshold = float(getattr(args, "aqcl_lambda", 10.0))
        self.alpha = float(getattr(args, "aqcl_alpha", 5.0))
        self.theta = float(getattr(args, "aqcl_theta", 0.5))
        self.fisher_batches = int(getattr(args, "aqcl_fisher_batches", 8))
        self.targets = convert_to_qat(model, low_bits=self.low_bits)
        self.weight_sensitivity: dict[str, Tensor] = {}
        self.activation_sensitivity: dict[str, Tensor] = {}
        self.covariance: dict[str, Tensor] = {}
        self.covariance_count: dict[str, int] = {}
        self.subspaces: dict[str, dict[str, Tensor]] = {}
        self.last_mac_profile: dict[str, float] = {}
        self.active_weight_bits: dict[str, int] = {}
        self.active_activation_bits: dict[str, int] = {}
        self.quantization_enabled = True
        self.before_task(0)

    @property
    def enabled(self) -> bool:
        return True

    def _linear_targets(self) -> Iterable[tuple[str, QATLinear]]:
        for name, module in self.targets.items():
            if isinstance(module, QATLinear):
                yield name, module

    def set_bits(self, weight_bits: dict[str, int], activation_bits: dict[str, int]) -> None:
        self.active_weight_bits = dict(weight_bits)
        self.active_activation_bits = dict(activation_bits)
        if not self.quantization_enabled:
            return
        for name, module in self.targets.items():
            module.weight_quant.set_bits(weight_bits.get(name, self.low_bits))
            module.activation_quant.set_bits(activation_bits.get(name, self.low_bits))

    def set_quantization_enabled(self, enabled: bool) -> None:
        self.quantization_enabled = bool(enabled)
        if enabled:
            for name, module in self.targets.items():
                module.weight_quant.set_bits(
                    self.active_weight_bits.get(name, self.low_bits)
                )
                module.activation_quant.set_bits(
                    self.active_activation_bits.get(name, self.low_bits)
                )
        else:
            for module in self.targets.values():
                module.weight_quant.set_bits(32)
                module.activation_quant.set_bits(32)

    def register_target(self, name: str, module: nn.Module) -> None:
        if isinstance(module, (QATLinear, QATConv1d)):
            self.targets[name] = module
            if self.quantization_enabled:
                module.weight_quant.set_bits(
                    self.active_weight_bits.get(name, self.low_bits)
                )
                module.activation_quant.set_bits(
                    self.active_activation_bits.get(name, self.low_bits)
                )
            else:
                module.weight_quant.set_bits(32)
                module.activation_quant.set_bits(32)

    def before_task(self, taskid: int) -> None:
        if self.mode == "fixed":
            bits = {name: self.fixed_bits for name in self.targets}
            self.set_bits(bits, bits)
        elif not self.weight_sensitivity:
            # Sensitivity is estimated only after the first task. Start from
            # the safe high precision so the first task is not pre-compressed.
            bits = {name: self.high_bits for name in self.targets}
            self.set_bits(bits, bits)
        elif self.mode not in ("rpq", "rpq_saou"):
            raise ValueError(f"Unsupported AQCL mode: {self.mode}")

    def _allocate_bits(self) -> None:
        if self.mode == "fixed":
            return
        if not self.weight_sensitivity:
            return
        weight_values = torch.stack(list(self.weight_sensitivity.values()))
        activation_values = torch.stack(list(self.activation_sensitivity.values()))
        weight_threshold = weight_values.mean()
        activation_threshold = activation_values.mean()
        weight_bits = {
            name: (
                self.high_bits
                if value >= weight_threshold
                else self.low_bits
            )
            for name, value in self.weight_sensitivity.items()
        }
        activation_bits = {
            name: (
                self.high_bits
                if value >= activation_threshold
                else self.low_bits
            )
            for name, value in self.activation_sensitivity.items()
        }
        self.set_bits(weight_bits, activation_bits)

    def _fisher_and_covariance(
        self,
        model: nn.Module,
        dataloader,
        taskid: int,
        criterion: nn.Module,
    ) -> None:
        args = self.args
        device = next(model.parameters()).device
        was_training = model.training
        model.train()

        previous_requires_grad = {
            name: module.weight.requires_grad
            for name, module in self.targets.items()
        }
        for name, module in self.targets.items():
            module.weight.requires_grad_(True)

        weight_sums = {name: torch.zeros_like(module.weight, device=device) for name, module in self.targets.items()}
        weight_counts = {name: 0 for name in self.targets}
        activation_sums = {name: 0.0 for name in self.targets}
        activation_counts = {name: 0 for name in self.targets}
        covariance_local = {}
        covariance_counts = {}
        if self.mode == "rpq_saou":
            covariance_local = {
                name: torch.zeros(
                    module.in_features,
                    module.in_features,
                    device="cpu",
                    dtype=torch.float32,
                )
                for name, module in self._linear_targets()
            }
            covariance_counts = {name: 0 for name, _ in self._linear_targets()}

        def activation_hook(name: str):
            def hook(_module, grad_input, _grad_output):
                if grad_input and grad_input[0] is not None:
                    activation_sums[name] += float(grad_input[0].detach().float().pow(2).sum().item())
                    activation_counts[name] += int(grad_input[0].numel())
            return hook

        def covariance_hook(name: str):
            def hook(_module, inputs):
                if not inputs:
                    return
                x = inputs[0].detach().float()
                if isinstance(_module, nn.Linear):
                    x = x.reshape(-1, x.shape[-1])
                elif x.ndim == 3:
                    x = x.transpose(1, 2).reshape(-1, x.shape[1])
                else:
                    x = x.reshape(-1, x.shape[-1])
                covariance_local[name] += (x.t() @ x).cpu()
                covariance_counts[name] += int(x.shape[0])
            return hook

        handles = []
        for name, module in self._linear_targets():
            handles.append(module.register_full_backward_hook(activation_hook(name)))
            if self.mode == "rpq_saou":
                handles.append(module.register_forward_pre_hook(covariance_hook(name)))

        try:
            for batch_index, (images, target, _) in enumerate(dataloader):
                if batch_index >= self.fisher_batches:
                    break
                images = images.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                model.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=bool(getattr(args, "use_amp", False)),
                ):
                    _, logits, _ = model(
                        images,
                        mode=args.train_tool + "_train",
                        taskid=taskid,
                        max_taskid=taskid,
                    )
                    loss = criterion(logits, target)
                loss.backward()
                for name, module in self.targets.items():
                    if module.weight.grad is not None:
                        weight_sums[name] += module.weight.grad.detach().pow(2)
                        weight_counts[name] += int(module.weight.numel())
        finally:
            for handle in handles:
                handle.remove()
            for name, module in self.targets.items():
                module.weight.requires_grad_(previous_requires_grad[name])
            model.zero_grad(set_to_none=True)
            model.train(was_training)

        for name, module in self.targets.items():
            count = max(weight_counts[name], 1)
            sensitivity = (weight_sums[name].sum() / count).detach().cpu()
            if name in self.weight_sensitivity:
                self.weight_sensitivity[name] = 0.5 * self.weight_sensitivity[name] + 0.5 * sensitivity
            else:
                self.weight_sensitivity[name] = sensitivity
            act_sensitivity = activation_sums[name] / max(activation_counts[name], 1)
            act_tensor = torch.tensor(act_sensitivity, dtype=torch.float32)
            if name in self.activation_sensitivity:
                self.activation_sensitivity[name] = (
                    0.5 * self.activation_sensitivity[name] + 0.5 * act_tensor
                )
            else:
                self.activation_sensitivity[name] = act_tensor

        if self.mode != "rpq_saou":
            self._allocate_bits()
            return

        for name, module in self._linear_targets():
            count = covariance_counts[name]
            if count == 0:
                continue
            current = covariance_local[name] / float(count)
            if name in self.covariance:
                old_count = self.covariance_count[name]
                self.covariance[name] = (
                    (old_count * self.covariance[name] + count * current)
                    / float(old_count + count)
                )
                self.covariance_count[name] = old_count + count
            else:
                self.covariance[name] = current
                self.covariance_count[name] = count
            covariance = self.covariance[name]
            covariance = (covariance + covariance.t()) * 0.5
            covariance = torch.nan_to_num(covariance, nan=0.0, posinf=0.0, neginf=0.0)
            scale = float(torch.diagonal(covariance).abs().mean().clamp_min(1e-8))
            jitter = scale * 1e-6
            for _ in range(5):
                try:
                    singular_values, eigenvectors = torch.linalg.eigh(
                        covariance + jitter * torch.eye(
                            covariance.shape[0],
                            dtype=covariance.dtype,
                        )
                    )
                    break
                except RuntimeError:
                    jitter *= 10.0
            else:
                raise RuntimeError(f"Unable to decompose covariance for {name}")
            order = torch.argsort(singular_values, descending=True)
            singular_values = singular_values[order].clamp_min(0.0)
            u = eigenvectors[:, order]
            floor = singular_values[-1].clamp_min(1e-12)
            threshold = self.lambda_threshold * floor
            null_mask = singular_values <= threshold
            transition_mask = (
                (singular_values >= 0.5 * threshold)
                & (singular_values <= 2.0 * threshold)
            )
            if not null_mask.any():
                null_mask[-1] = True
            if not transition_mask.any():
                transition_mask = null_mask
            self.subspaces[name] = {
                "Uo": u[:, null_mask].detach().cpu(),
                "Up": u[:, transition_mask].detach().cpu(),
            }
        self._allocate_bits()

    def after_task(
        self,
        model: nn.Module,
        dataloader,
        taskid: int,
        criterion: nn.Module,
    ) -> None:
        self._fisher_and_covariance(model, dataloader, taskid, criterion)

    @torch.no_grad()
    def modulate_gradients(self, model: nn.Module) -> None:
        if self.mode != "rpq_saou":
            return
        for name, module in self._linear_targets():
            gradient = module.weight.grad
            subspace = self.subspaces.get(name)
            if gradient is None or subspace is None:
                continue
            uo = subspace["Uo"].to(gradient.device, gradient.dtype)
            up = subspace["Up"].to(gradient.device, gradient.dtype)
            if uo.numel() == 0:
                continue
            go = (gradient @ uo) @ uo.t()
            gp = gradient - go
            gp_norm = gp.norm()
            g_norm = gradient.norm().clamp_min(1e-12)
            if gp.numel() == 0 or gp_norm <= self.theta * g_norm:
                module.weight.grad = go
                continue
            if up.numel() == 0:
                module.weight.grad = go
                continue
            coefficients = gp @ up
            cosine = coefficients / gp_norm.clamp_min(1e-12)
            mask = torch.exp(-self.alpha * cosine.pow(2))
            module.weight.grad = go + (coefficients * mask) @ up.t()

    def deployment_state(self) -> dict[str, object]:
        weight_bits = {
            name: module.weight_quant.num_bits
            for name, module in self.targets.items()
        }
        activation_bits = {
            name: module.activation_quant.num_bits
            for name, module in self.targets.items()
        }
        return {
            "mode": self.mode,
            "weight_bits": weight_bits,
            "activation_bits": activation_bits,
            "training_only": {
                "fisher": self.weight_sensitivity,
                "covariance": self.covariance,
                "subspaces": self.subspaces,
            },
        }

    def model_size_mb(self, model: nn.Module) -> float:
        total_bits = 0
        for name, parameter in model.named_parameters():
            bits = 32
            if name.endswith(".weight"):
                module_name = name[: -len(".weight")]
                if module_name in self.targets:
                    bits = self.targets[module_name].weight_quant.num_bits
            total_bits += parameter.numel() * bits
        return total_bits / 8.0 / (1024.0 ** 2)

    def profile_macs(
        self,
        model: nn.Module,
        sample: Tensor,
        mode: str = "shared",
    ) -> dict[str, float]:
        counts: dict[str, float] = {}

        def make_hook(name: str):
            def hook(module, inputs, _output):
                x = inputs[0]
                if isinstance(module, nn.Linear):
                    counts[name] = counts.get(name, 0.0) + float(x.numel() // x.shape[-1]) * float(
                        module.in_features * module.out_features
                    )
                elif isinstance(module, nn.Conv1d):
                    output_length = _output.shape[-1]
                    counts[name] = counts.get(name, 0.0) + float(
                        _output.shape[0]
                        * output_length
                        * module.in_channels
                        * module.out_channels
                        * module.kernel_size[0]
                        / module.groups
                    )
            return hook

        handles = []
        for name, module in self.targets.items():
            handles.append(module.register_forward_hook(make_hook(name)))
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                model(sample, mode=mode)
        finally:
            for handle in handles:
                handle.remove()
            model.train(was_training)
        self.last_mac_profile = counts
        return counts

    def inference_report(self, model: nn.Module, sample: Tensor) -> dict[str, float]:
        counts = self.profile_macs(model, sample, mode="shared")
        gmac = sum(counts.values()) / 1e9
        gflops = 2.0 * gmac
        gbops = 0.0
        for name, macs in counts.items():
            module = self.targets[name]
            gbops += macs * module.weight_quant.num_bits * module.activation_quant.num_bits
        gbops /= 1e9
        return {
            "model_size_mb": self.model_size_mb(model),
            "inference_gmac": gmac,
            "inference_gflops": gflops,
            "inference_gbops": gbops,
            "inference_extra_memory_mb": 0.0,
            "weight_bits": sorted(
                {module.weight_quant.num_bits for module in self.targets.values()}
            ),
            "activation_bits": sorted(
                {module.activation_quant.num_bits for module in self.targets.values()}
            ),
        }
