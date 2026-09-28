"""Separate AQCL entry point.

This file adds quantization-aware training on top of the unmodified CKAA
entry point in train_eval.py. CKAA experiments should continue to use
train_eval.py; only AQCL experiments should use this file.
"""

import argparse
import os
import os.path as osp
import sys
from copy import deepcopy

import torch
from torch import nn
from torch.utils.data import DataLoader

import train_eval as ckaa
import timm
from utils import misc
from utils.aqcl import AQCLContext
from utils.dataset_builder import define_dataset
from utils.logging import Logger
from utils.mod_adam import ModAdam


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--aqcl-enable", type=misc.str2bool, default=True)
    parser.add_argument("--aqcl-mode", choices=("fixed", "rpq", "rpq_saou"), default="rpq_saou")
    parser.add_argument("--aqcl-bits", type=int, default=8)
    parser.add_argument("--aqcl-low-bits", type=int, default=8)
    parser.add_argument("--aqcl-high-bits", type=int, default=16)
    parser.add_argument("--aqcl-lambda", type=float, default=10.0)
    parser.add_argument("--aqcl-alpha", type=float, default=5.0)
    parser.add_argument("--aqcl-theta", type=float, default=0.5)
    parser.add_argument("--aqcl-fisher-batches", type=int, default=8)
    parser.add_argument("--aqcl-warmup-epochs", type=int, default=1)
    parser.add_argument("--aqcl-report", type=misc.str2bool, default=True)
    aqcl_args, remaining = parser.parse_known_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + remaining
    ckaa_args = ckaa.get_args()
    for key, value in vars(aqcl_args).items():
        setattr(ckaa_args, key, value)
    # AQCL experiments follow the tabular CKAA baseline: the shared prompt is
    # fixed after the first task, and routing uses the selected task head.
    ckaa_args.freeze_shared_prompts_after_first = True
    ckaa_args.eval_local_task_head_task_only = True
    ckaa.args = ckaa_args
    return ckaa_args


def make_optimizer(model, context):
    if context is None:
        return ModAdam

    class AQCLModAdam(ModAdam):
        _aqcl_context = context
        _aqcl_model = model

        def step(self, closure=None):
            AQCLModAdam._aqcl_context.modulate_gradients(AQCLModAdam._aqcl_model)
            return super().step(closure)

    return AQCLModAdam


def build_model(args, GVM, current_task_classes):
    prompt_args = misc.get_specific_args_dict(args, "prompt_")
    other_args = misc.get_specific_args_dict(args, "logit_")
    old_current = getattr(ckaa, "current_task_classes", None)
    ckaa.current_task_classes = current_task_classes
    head_args = ckaa.get_head_dim_arg_dict(GVM, args)
    if old_current is not None:
        ckaa.current_task_classes = old_current

    model = timm.create_model(
        args.model,
        pretrained=True,
        pretrained_strict=False,
        **head_args,
        prompt_args_dict=prompt_args,
        other_args_dict=other_args,
    )
    if args.dataset in ("tabular", "fault_csv"):
        model = ckaa.convert_vit_to_tabular(
            model,
            signal_length=args.tabular_window_length,
            patch_size=args.tabular_patch_size,
            in_chans=args.tabular_in_chans,
        )
    context = AQCLContext(model, args) if args.aqcl_enable else None
    model = nn.DataParallel(model, device_ids=args.device_ids)
    return model, context


def main():
    args = parse_args()
    ckaa.seed_etc_options(args.seed)
    logs_path = osp.join(args.logs_dir, args.logs_suffix + ".txt")
    os.makedirs(args.logs_dir, exist_ok=True)
    sys.stdout = Logger(logs_path)

    GVM = ckaa.GlobalVarsManager()
    GVM.init_from_args(args)
    GVM.cache_dict["exp_start_time"] = ckaa.ttime()
    print("***** Start AQCL training *****")

    context = None
    model = None
    original_train_one_epoch = ckaa.train_one_epoch

    def train_one_epoch_with_warmup(
        GVM_,
        taskid_,
        curr_epoch,
        dataloader_,
        model_,
        criterion_,
        optimizer_,
        *args_,
        **kwargs_,
    ):
        if context is not None:
            warmup_epochs = max(int(args.aqcl_warmup_epochs), 0)
            context.set_quantization_enabled(curr_epoch > warmup_epochs)
        return original_train_one_epoch(
            GVM_,
            taskid_,
            curr_epoch,
            dataloader_,
            model_,
            criterion_,
            optimizer_,
            *args_,
            **kwargs_,
        )

    ckaa.train_one_epoch = train_one_epoch_with_warmup

    for taskid, current_task_classes in GVM.cl_mngr:
        print(f"{'#' * 30} Task: [{taskid + 1}/{GVM.cl_mngr.num_tasks}] {'#' * 30}")
        print(f"Current classes ({len(current_task_classes)}): {current_task_classes}")

        if not args.consecutive_training or taskid == 0:
            ckaa.current_task_classes = current_task_classes
            model, context = build_model(args, GVM, current_task_classes)
            GVM.cache_dict["pretrained_cfg"] = deepcopy(model.module.pretrained_cfg)
            GVM.cache_dict["aqcl"] = context
            ckaa.ModAdam = make_optimizer(model, context)

        if args.consecutive_training and taskid > 0:
            ckaa.ModAdam = make_optimizer(model, context)

        not_pretrained = ckaa.find_not_pretrained_params(
            model.module,
            pretrained_cfg=model.module.pretrained_cfg,
        )
        GVM.cache_dict["not_pretrained_params"] = not_pretrained
        GVM.update_label_maps(taskid, current_task_classes)
        GVM.cache_dict["training_string"] = args.training_string
        misc.check_param_training(not_pretrained, args.training_string)

        original_train_one_task = ckaa.train_one_task

        def train_one_task_with_aqcl(GVM_, taskid_, task_classes_, model_, **kwargs):
            if context is not None:
                context.before_task(taskid_)
            result = original_train_one_task(
                GVM_,
                taskid_,
                task_classes_,
                model_,
                **kwargs,
            )
            if context is not None:
                dataset = define_dataset(
                    GVM_,
                    task_classes_,
                    training=True,
                    transform_type=args.transform_type,
                    target_map_to_local=args.seperate_head,
                    expand_times=1,
                )
                dataloader = DataLoader(
                    dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=0,
                    pin_memory=True,
                )
                criterion = nn.CrossEntropyLoss().to(model_.module.device)
                context.set_quantization_enabled(True)
                context.after_task(
                    model_.module if isinstance(model_, nn.DataParallel) else model_,
                    dataloader,
                    taskid_,
                    criterion,
                )
                if args.aqcl_report:
                    sample = dataset[0][0].unsqueeze(0).to(model_.module.device)
                    report = context.inference_report(model_.module, sample)
                    print(
                        "AQCL deployment: "
                        f"size={report['model_size_mb']:.4f}MB, "
                        f"GFLOPs={report['inference_gflops']:.4f}, "
                        f"GBOPS={report['inference_gbops']:.4f}, "
                        f"extra_memory={report['inference_extra_memory_mb']:.4f}MB, "
                        f"weight_bits={report['weight_bits']}, "
                        f"activation_bits={report['activation_bits']}"
                    )
            return result

        if args.evaluation:
            if taskid == GVM.cl_mngr.num_tasks - 1:
                ckaa.evaluate_tasks_sofar(
                    GVM,
                    taskid,
                    model,
                    pretrained_model_path=osp.join(
                        "specific-shared",
                        "logs",
                        args.dataset,
                        args.evaluation_model_name + ".pkl",
                    ),
                )
                ckaa.task_ending_info(GVM)
        else:
            ckaa.train_one_task = train_one_task_with_aqcl
            model = train_one_task_with_aqcl(
                GVM,
                taskid,
                current_task_classes,
                model,
            )
            ckaa.train_one_task = original_train_one_task
            ckaa.evaluate_tasks_sofar(GVM, taskid, model)
            ckaa.task_ending_info(GVM)

        if args.save_model:
            save_path = osp.join(
                "specific-shared",
                "logs",
                args.dataset,
                args.save_model_name + ".pkl",
            )
            os.makedirs(osp.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    main()
