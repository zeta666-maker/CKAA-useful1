# AQCL Implementation Notes

AQCL is implemented in `utils/aqcl.py` with the separate entry point
`train_aqcl.py`. The CKAA entry point `train_eval.py` contains no AQCL code
and is unaffected by the quantization implementation.

## Method Mapping

| AQCL paper component | Implementation |
| --- | --- |
| Eq. 1 asymmetric uniform quantization | `UniformQuantizer` with a straight-through estimator |
| Layer-wise weight quantization | `QATLinear`, `QATConv1d` |
| Eq. 11 normalized Hessian trace | Diagonal empirical Fisher in `AQCLContext._fisher_and_covariance` |
| Eq. 13 high/low bit allocation | `AQCLContext._allocate_bits` |
| Eq. 21-22 covariance accumulation | `AQCLContext._fisher_and_covariance` covariance hooks |
| Eq. 20 SVD null/transition subspaces | Symmetric eigendecomposition in `AQCLContext._fisher_and_covariance` |
| Eq. 23-24 SAOU gradient modulation | `AQCLContext.modulate_gradients` |
| Inference-only quantized model | `AQCLContext.deployment_state`; Fisher, covariance and subspaces remain in the training context and are not saved in the model checkpoint |
| Model size, GFLOPs, GBOPS, extra memory | `AQCLContext.inference_report` |

## Commands

Run the CKAA FP32 baseline:

```powershell
python .\train_eval.py -d fault_csv -t 3 -m vit_base_patch16_224.augreg_in21k -b 32 -e 1 -jt 0 -je 0 -et 1 --eval_batch_size 64 --temperature 28.0 --lr 0.005 --lr_scale 0.2 --lr_scale_patterns patch_embed --null_eta1 0.999 --null_eta2 0.999 -tc 2.0 --eval-tool adapter --eval-trained-task-router true --eval-local-task-head true --eval-task-weight false --freeze-shared-prompts-after-first true --prototype-confidence-weight 0.0 --seed 2024 --save-model false --logs-dir logs -sf aqcl_compare_fp32
```

Run AQCL with RPQ and SAOU:

```powershell
python .\train_aqcl.py -d fault_csv -t 3 -m vit_base_patch16_224.augreg_in21k -b 32 -e 1 -jt 0 -je 0 -et 1 --eval_batch_size 64 --temperature 28.0 --lr 0.005 --lr_scale 0.2 --lr_scale_patterns patch_embed --null_eta1 0.999 --null_eta2 0.999 -tc 2.0 --eval-tool adapter --eval-trained-task-router true --eval-local-task-head true --eval-task-weight false --freeze-shared-prompts-after-first true --prototype-confidence-weight 0.0 --seed 2024 --save-model false --logs-dir logs -sf aqcl_compare_rpq_saou_4_8 --aqcl-mode rpq_saou --aqcl-low-bits 4 --aqcl-high-bits 8 --aqcl-fisher-batches 1
```

Run all comparison cases:

```powershell
.\tools\run_aqcl_comparison.ps1
```

## Short-Run Results

The following is a 1-epoch smoke comparison on `fault_csv`, 3 incremental
tasks, seed 2024. It validates the implementation direction but is not the
paper's 80-epoch benchmark.

| Case | Last-acc | Avg-acc | Forgetting | Model size MB | GFLOPs | GBOPS | Inference extra memory MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FP32 | 75.67% | 85.28% | 16.88% | 344.92* | 33.7004 | 17254.6* | 0.0000 |
| Fixed 8-bit | 64.63% | 67.32% | 34.07% | 86.23 | 33.7004 | 1078.41 | 0.0000 |
| Fixed 4-bit | 26.23% | 38.74% | 42.62% | 44.02 | 33.7004 | 269.60 | 0.0000 |
| Fixed 2-bit | 11.11% | 20.37% | 0.00% | 22.91 | 33.7004 | 67.40 | 0.0000 |
| RPQ only (4/8-bit) | 24.12% | 33.50% | 37.97% | 58.67 | 33.7004 | 520.99 | 0.0000 |
| RPQ + SAOU (4/8-bit) | 54.08% | 48.58% | 23.42% | 51.06 | 33.7004 | 472.58 | 0.0000 |

`*` FP32 size and GBOPS are derived from the fixed-bit scaling because the
FP32 run does not instantiate quantizers.

The expected qualitative trend holds: lower fixed bit widths reduce accuracy,
RPQ improves over fixed 4-bit, and RPQ+SAOU improves over RPQ-only under the
same short budget. SAOU improves Last-acc from 24.12% to 54.08% and Avg-acc
from 33.50% to 48.58%. The
absolute values are not comparable to the paper's CIFAR-100/TinyImageNet
results because the fault dataset, backbone, task count and epoch budget
differ.

## Risks

- The fault dataset is not a paper benchmark. Its distribution, task count,
  and sample count differ from CIFAR-100 and TinyImageNet.
- A 1-epoch comparison is only a smoke test. The paper trains for 80 epochs
  with learning-rate decay.
- The implementation quantizes the tabular ViT and CKAA PEFT modules, but
  the paper's efficiency numbers are for ResNet-20. GFLOPs are reported for
  the tabular ViT architecture, so absolute GFLOPs differ from the paper.
- SAOU covariance/eigendecomposition is retained only in the training
  context. Model checkpoints contain the quantized weights and bit widths,
  not Fisher, covariance, SVD, or gradient-modulation state.
