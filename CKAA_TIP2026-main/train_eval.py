import sys
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ['TIMM_FUSED_ATTN'] = '0' if 'TIMM_FUSED_ATTN' not in os.environ else os.environ['TIMM_FUSED_ATTN']
import os.path as osp
from time import time as ttime
import argparse
import random
from collections import OrderedDict
import tqdm
from typing import Any, Literal
from copy import deepcopy
import warnings
import scipy.ndimage
warnings.filterwarnings('ignore')
from collections import Counter
import copy
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.cuda.amp.grad_scaler import GradScaler
import torch.utils.hooks
from torch.utils.data import DataLoader, TensorDataset
import torch.linalg
import torchvision
import timm
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2


from utils.mod_adam import ModAdam
import utils.vit_builder
from utils.vit_builder import VisionTransformer, ContrastiveLoss, convert_vit_to_tabular
from utils.dataset_builder import ImagePathDatasetClassManager, ImagePathDataset, Mixup, define_dataset
from utils.continual_manager import ClassIncrementalManager
from utils import misc
from utils.logging import Logger

import clip
from clip.clip import CLIP
from einops import rearrange, reduce, repeat
import time
import math
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn as nn

torch.set_float32_matmul_precision("high")


class GlobalVarsManager:
    args: argparse.Namespace
    path_data_dict: dict[str, ImagePathDataset]
    cl_mngr: ClassIncrementalManager
    acc_mat_dict: OrderedDict[str, np.ndarray]
    cache_dict: dict
    param_dict: dict[Literal['base_params', 'task_params_'], OrderedDict[str, Tensor]]
    label_map_g2l: dict[int, tuple[int, int, int]]

    def init_from_args(self, args):
        self.args = args
        _dataset_class_manager = ImagePathDatasetClassManager(**{args.dataset: args.data_root})
        self.path_data_dict = {'train': _dataset_class_manager[args.dataset](train=True),
                               'eval': _dataset_class_manager[args.dataset](train=False)}
        ## self.path_data_dict['train'].class_imgpath_dict: len=100;
        ## self.path_data_dict['train'].class_imgpath_dict[i]: image path
        ## self.path_data_dict['train'].class_int_str_map: len=100, class name;

        ## create split dataset
        self.cl_mngr = ClassIncrementalManager(self.path_data_dict['eval'].class_list, args.num_tasks, args.seed, shuffle=args.shuffle_classes)

        ## create metric: AccClassIncMat:10*10; AccClassIncList:10
        self.acc_mat_dict = OrderedDict(AccClassIncMat=np.zeros([_nt := self.cl_mngr.num_tasks, _nt]), AccClassIncList=np.zeros([_nt]))
        self.cache_dict = {}
        self.param_dict = {}
        self.label_map_g2l = {}

        self.stastics_dict = []

    def update_label_maps(self, taskid: int, task_classes: list[int]) -> tuple[dict[int, int], dict[str, int]]:
        _g2l_map = misc.make_label_maps(taskid, task_classes, self.cl_mngr.num_classes_per_task) ## map: {label: {taskid; idx in task; total idx}}
        if not all([_k not in self.label_map_g2l.keys() for _k in _g2l_map.keys()]):
            print("The global_to_local label map has been fully loaded, which is not expected.")
        self.label_map_g2l.update(_g2l_map) ## map --> self.label_map_g2l (Dict)
        return _g2l_map


def get_args():

    parser = argparse.ArgumentParser(description='Class-incremental Learning')
    parser.add_argument('-d', '--dataset', type=str, default='imagenet_a',
                    choices=('cifar100', 'imagenet_r', 'imagenet_a', 'sdomainet', 'cub', 'stanford_cars', 'tabular'), help='use lowercase')
    ## logs_out parameters
    parser.add_argument('--logs-dir', type=str, default='logs/')
    parser.add_argument('-sf', '--logs-suffix', type=str, default = '10s_imagenet_a')
    parser.add_argument('--logger-out-step', type=int, default=50)
    parser.add_argument('--seed', type=int, default=2024)

    ## training settings
    parser.add_argument('--save-model', type=bool, default=False)
    parser.add_argument('--save-model-name', type=str, default='model_10s')
    parser.add_argument('--evaluation', type=bool, default=False, help='evaluating through trained model')
    parser.add_argument('--evaluation-model-name', type=str, default='model_20s')

    parser.add_argument('-t', '--num_tasks', type=int, default=10, choices=(1, 2, 3, 5, 10, 20, 25, 50, 100)) 

    ## shuffle
    parser.add_argument('--shuffle_classes', type=misc.str2bool, default=True)
    parser.add_argument('-m', '--model', type=str, default='vit_base_patch16_224.augreg_in21k', 
                        help='vit_base_patch16_224.augreg_in21k, vit_base_patch16_clip_quickgelu_224.openai, vit_base_patch16_224.dino'
                        )
    parser.add_argument('-pt', '--pretrained-type', type=str, default='Sup-21k', choices=('Sup-21k', 'DINO', 'iBOT'))
    
    parser.add_argument('--head_dim_type', type=str, choices=('task_classes', 'pretrained', 'text_dim'), default='task_classes')
    parser.add_argument('--logit_type', type=str, choices=('head_out', 'sim_imgtext'), default='head_out')
    
    parser.add_argument('--logit_scale', type=float, default=4.605170249938965, help='0 | 4.605170249938965')
    parser.add_argument('--logit_scale_trainable', type=misc.str2bool, default=False)
    parser.add_argument('--prompt_len', type=int, default=4, help='0 means not using prompt') ## 4
    parser.add_argument('--prompt_init', type=str, choices=('uniform', 'zero'), default='uniform')
    parser.add_argument('--prompt_start_block', type=int, default=0)
    parser.add_argument('--prompt_end_block', type=int, default=11)
    parser.add_argument('--seperate_head', type=misc.str2bool, default=True)

    ## null-space
    parser.add_argument('--use_null_space', type=bool, default=True)

    parser.add_argument('--null_patterns', type=str, nargs='+', default=('sh_prompt'))
    parser.add_argument('--null_thres_mode', type=str, choices=('adaptive', 'times'), default='adaptive')
    parser.add_argument('--null_thres_value1', type=float, default=0.)
    parser.add_argument('--null_thres_value2', type=float, default=0.)
    parser.add_argument('--null_eta1', type=float, default=0.96) 
    parser.add_argument('--null_eta2', type=float, default=0.96) 
    parser.add_argument('--null_interm_accum', type=str, choices=('sum', 'mean'), default='sum')
    parser.add_argument('--ln_loss_lam', type=float, default=1.)
    parser.add_argument('--refine_head', type=misc.str2bool, default=False)
    parser.add_argument('--transform_type', type=str, choices=('timm', 'autoaug', 'prototype', 'clip'), default='autoaug')
    parser.add_argument('--prob_cutmixup', type=float, default=0)

    ## number of epochs
    parser.add_argument('-e', '--epochs', type=int, default=10) ## epochs for training each task
    parser.add_argument('-jt', '--workers', type=int, default=16)
    parser.add_argument('-je', '--eval_workers', type=int, default=2)

    ## expand_times for dataset
    parser.add_argument('-et', '--expand_times', type=int, default=10) ## expand_times for training in one epoch
    parser.add_argument('--temperature', type=float, default=28.) ## 28.
    parser.add_argument('--use_amp', type=misc.str2bool, default=True)
    parser.add_argument('--sample_type', type=str, choices=('path', 'image'), default='image')
    parser.add_argument('--consecutive_training', type=misc.str2bool, default=True, help="")
    parser.add_argument('--timeout', type=int, default=3000)
    parser.add_argument('--persistent_workers', type=misc.str2bool, default=False)
    
    parser.add_argument('-eb', '--eval_batch_size', type=int, default=100)
    parser.add_argument('--lr', '--learning_rate', type=float, default=0.01)
    parser.add_argument('--lr_scale', type=float, default=1.)
    parser.add_argument('--lr_scale_patterns', type=str, nargs='+')
    parser.add_argument('--optimizer', type=str, default='mod_adam')
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--lr_sch', type=str, default='multistep', choices=('cosine', 'step', 'multistep'))
    parser.add_argument('--warmup_epochs', type=int, default=0)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('-dm', '--decay_milestones', type=int, nargs='+', default=[5,8]) ## [5, 8]
    parser.add_argument('--decay_epochs', type=int, default=1000)
    parser.add_argument('--decay_rate', type=float, default=0.1)
    parser.add_argument('--show_bar', type=bool, default=True)
    parser.add_argument('--print_model', type=bool, default=True)

    ## additional hyper-parameters
    parser.add_argument('--prompt-len-sh', type=float, default=4)
    parser.add_argument('--prompt-len-sp', type=float, default=4)

    ## loss functions
    parser.add_argument('--sh-loss', type=bool, default=True)
    parser.add_argument('--sp-loss', type=bool, default=True)
    parser.add_argument('--ca-loss', type=bool, default=True)
    parser.add_argument('--fa-loss', type=bool, default=True)
    parser.add_argument('--classifier-aggregation-type', type=str, default='mean', choices=('weighted', 'mean', None))
    parser.add_argument('-tf', '--contrastive-temperature', type=float, default=0.05)
    parser.add_argument('-kg', '--simulation-k', type=int, default=20, help='knn-nearest neighbors for constructing graph')
    parser.add_argument('-tg', '--simulation-temperature', type=float, default=0.2, help='temperature for similarity metric')

    ## evaluation
    parser.add_argument('--task-detector', type=str, default='logit', choices=('logit',))
    parser.add_argument('--eval-task-weight', type=misc.str2bool, default=True)
    parser.add_argument('-tc', '--eval-tau', type=float, default=3.)
    parser.add_argument('--nearest-class-selection', type=bool, default=True)
    parser.add_argument('-kc', '--prototype-k', type=int, default=20)
    parser.add_argument('--prototype-temperature', type=float, default=0.1)
    parser.add_argument('--prototype-confidence-weight', type=float, default=0.0)

    parser.add_argument('--prototype-ratio', type=int, default=0.1)
    parser.add_argument('--topk-adapter-selection', type=bool, default=False, help='select topk adapter for evaluation')
    parser.add_argument('--adapter-k', type=int, default=100)
    parser.add_argument('--eval-adapter-scale', type=float, default=1.0)
    parser.add_argument(
        '--eval-adapter-ensemble',
        type=misc.str2bool,
        default=False,
        help='Select the most confident task adapter without external task IDs.',
    )
    parser.add_argument(
        '--eval-prototype-router',
        type=misc.str2bool,
        default=False,
        help='Route samples with task-specific adapter prototypes.',
    )
    parser.add_argument(
        '--eval-cross-adapter-prototypes',
        type=misc.str2bool,
        default=False,
        help='Classify against class prototypes computed in every seen adapter.',
    )
    parser.add_argument(
        '--eval-trained-task-router',
        type=misc.str2bool,
        default=False,
        help='Train a lightweight task router from adapter-logit patterns.',
    )
    parser.add_argument(
        '--eval-local-task-head',
        type=misc.str2bool,
        default=False,
        help='Use only the selected task head for final classification.',
    )
    parser.add_argument(
        '--freeze-shared-prompts-after-first',
        type=misc.str2bool,
        default=False,
        help='Freeze the shared prompt after the first task.',
    )
    parser.add_argument('--max-logit-score', type=bool, default=False)

    parser.add_argument('--previous-head', type=bool, default=True)
    parser.add_argument('--prototype-classifier', type=bool, default=True)
    parser.add_argument('--copy-weight', type=bool, default=False, help='copy adapter weights from the previous task')

    parser.add_argument('--training_string', type=str, nargs='+', 
                    default=('prompt','head', 'adapter', 'prototypes'))
    parser.add_argument('--train-tool', type=str, default='adapter', 
                    choices=('specific', 'shared', 'prompt', 'adapter'))
    parser.add_argument('--eval-tool', type=str, default='adapter', 
                    choices=('specific', 'shared', 'prompt', 'upper_bound', 'adapter'))

    parser.add_argument('-de', '--device-ids', type=int, default=[0])
    parser.add_argument('-dd', '--distributed-data', type=int, default=True)
    parser.add_argument('-b', '--batch_size', type=int, default=110)
    parser.add_argument('--tabular-window-length', type=int, default=1568)
    parser.add_argument('--tabular-patch-size', type=int, default=32)
    parser.add_argument('--tabular-in-chans', type=int, default=4)


    args = parser.parse_args()
    print(args)

    args.prompt_num_tasks = args.num_tasks
    if args.dataset == 'tabular' and 'patch_embed' not in args.training_string:
        args.training_string = tuple(args.training_string) + ('patch_embed',)
    if args.evaluation == True or 'debug' in args.logs_suffix:
        args.save_model = False
        
    if args.dataset == 'cifar100':
        args.data_root = 'A_CLData/cifar100-split'
    elif args.dataset == 'imagenet_r':
        args.data_root = 'A_CLData/imagenet-r'
    elif args.dataset == 'imagenet_a':
        args.data_root = 'A_CLData/imagenet-a'
    elif args.dataset == 'sdomainet':
        args.data_root = 'A_CLData/domainnet'
    elif args.dataset == 'cub':
        args.data_root = 'A_CLData/cub'
    elif args.dataset == 'stanford_cars':
        args.data_root = 'A_CLData/stanford_cars'
    elif args.dataset == 'tabular':
        args.data_root = 'A_CLData/tabular_ckaa'
    else:
        raise ValueError(args.dataset)

    print('Experiment description: weighted adapter for testing ==> \
        we use linearly increasing k for k-nearest prototype search')

    if args.optimizer not in ('mod_adam',):
        raise NotImplementedError(args.optimizer)

    if not args.use_null_space:
        if args.ln_loss_lam != 0:
            print("args.ln_loss_lam is set to 0 for not using null space.")
        args.ln_loss_lam = 0

    return args


def seed_etc_options(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.set_printoptions(precision=4, linewidth=256)
    torch.set_printoptions(linewidth=256)
    try:
        torchvision.set_image_backend('accimage')
    except (ImportError, RuntimeError):
        torchvision.set_image_backend('PIL')


def set_model_mode(GVM: GlobalVarsManager, taskid: int, model: VisionTransformer, training: bool, to_gpu: bool = True, 
                training_string: tuple[str] = ('prompt',)) -> VisionTransformer:
    
    for n, p in model.named_parameters():
        if training and any([_s in n for _s in training_string]):
            if getattr(GVM.args, 'dataset', None) == 'tabular' and taskid > 0 and 'patch_embed' in n:
                p.requires_grad_(False)
            elif (
                getattr(GVM.args, 'freeze_shared_prompts_after_first', False)
                and taskid > 0
                and 'sh_prompt' in n
            ):
                p.requires_grad_(False)
            elif 'sp_prompt' in n or 'keys' in n or 'task_adapter' in n or 'task_head' in n:
                if str(taskid) in n:
                    p.requires_grad_(True)
                else:
                    p.requires_grad_(False)
            else:
                p.requires_grad_(True)
        else:
            p.requires_grad_(False)
    params_requires_grad = [n for n, p in model.named_parameters() if p.requires_grad]

    print_num_params = False
    if print_num_params:
        all_params = []
        learnable_params = []
        for n, p in model.named_parameters():
            if training and any([_s in n for _s in training_string]):
                if 'sh_prompt' in n:
                    learnable_params.append(p)
            all_params.append(p)

        learnable_total = sum([param.nelement() for param in learnable_params])
        all_total = sum([param.nelement() for param in all_params])
        print('Number of learnable parameter: % .4fM' % (learnable_total / 1e6))
        print('Number of all parameter: % .4fM' % (all_total / 1e6))
        print(learnable_total / all_total)
                

    model.eval()
    for n, m in model.named_modules():
        if training and any([n.endswith(_s) and not isinstance(m, nn.Identity) for _s in training_string]):
            m.train()
        else:
            m.eval()
    modules_training = [n for n, m in model.named_modules() if m.training]

    if to_gpu:
        model = model.cuda()

    return model


def set_learning_rates(GVM: GlobalVarsManager, model: VisionTransformer, base_lr: float, lr_scale: float, lr_scale_patterns: str) -> list[dict[str: Tensor | float]]:
    param_lr_groups = [{'params': [], 'lr': base_lr},
                       {'params': [], 'lr': base_lr * lr_scale}]
    lr_param_dict = {_p['lr']: [] for _p in param_lr_groups}

    for n, p in model.named_parameters():
        if p.requires_grad:
            _group_idx = 1 if any(_s in n for _s in lr_scale_patterns) else 0
            param_lr_groups[_group_idx]['params'].append(p)
            lr_param_dict[param_lr_groups[_group_idx]['lr']].append(n)

    return param_lr_groups


def pairwise_distance(x, y):
   
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist_mat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(x, y.t(), beta=1, alpha=-2)
        
    return dist_mat


def knn_graph(feat1, feat2, k=20, temp=0.05):

    k = min(k, feat2.shape[0])
    graph = torch.softmax(torch.mm(F.normalize(feat1, dim=1), F.normalize(feat2, dim=1).t()) / temp, dim=1)
    mask = graph > graph.sort(dim=1, descending=True)[0][:, k-1].unsqueeze(1)
    graph = graph * torch.as_tensor(mask, dtype=torch.float)
    graph = graph.clamp(min=0.0)
    graph = graph / graph.sum(1).unsqueeze(1)

    return graph


def train_one_epoch(GVM: GlobalVarsManager, taskid: int, curr_epoch: int, dataloader: DataLoader, model: VisionTransformer, 
                criterion: nn.CrossEntropyLoss, optimizer: torch.optim.Optimizer, current_in_previous_list: list=None, 
                frozen_feature_list=None) -> str:
    
    args = GVM.args
    temperature: float = args.temperature
    use_amp: bool = args.use_amp
    assert temperature > 0.
    if not args.use_null_space:
        assert args.ln_loss_lam == 0
    else:
        assert args.ln_loss_lam == 1
    _use_cutmixup = args.prob_cutmixup > 0

    if _use_cutmixup:
        cutmixup_fn = Mixup(mixup_alpha=1., cutmix_alpha=1., prob=args.prob_cutmixup, switch_prob=0.5, mode='batch', num_classes=len(GVM.cl_mngr.current_task_classes))

    criterion_cl :nn.Module = ContrastiveLoss(temperature=args.contrastive_temperature)

    amp_scalar = GradScaler(enabled=use_amp)
    scalar_meter = misc.ScalarMeter(loss="samp_avg:.4f", batch_time="step_sum:.3f", acc_top1="samp_avg:>6.2%")
    _btimer = ttime()

    tt = ttime()
    # for i_batch, (images, target) in tqdm.tqdm(enumerate(dataloader, 1), total=len(dataloader), dynamic_ncols=True, disable=not GVM.args.show_bar):
    for i_batch, (images, target, index) in enumerate(dataloader, 1):

        images: Tensor = images.cuda(non_blocking=True) ## 256, 3, 224, 224
        target: Tensor = target.cuda(non_blocking=True) ## shuffled targets 256

        if _use_cutmixup:
            mix_img, mix_lbl = cutmixup_fn(images, target)
        else:
            mix_img = images
            mix_lbl = target

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):

            # logits: Tensor = model(mix_img) ## logits: B * 10 (current_task_classes)
            current_device = model.module.device
            sh_ce_loss, sp_ce_loss = torch.tensor(0.), torch.tensor(0.)
            if args.sh_loss:
                ## extract shared features = F(P_sh, X)
                sh_feats, sh_logits, _ = model(mix_img, mode='shared', taskid=taskid)
                ## cross_entropy for shared prompts
                sh_ce_loss = criterion(sh_logits / temperature, mix_lbl) 

            if args.sp_loss:   
                ## extract shared features = F(P_sh, X, A_t)
                sp_feats, sp_logits, _ = model(mix_img, 
                        mode=args.train_tool+'_train', 
                        taskid=taskid, 
                        max_taskid=taskid)
                ## cross_entropy for specific adapters
                sp_ce_loss = criterion(sp_logits / temperature, mix_lbl) 

            ca_loss = torch.tensor(0.)
            if args.ca_loss:

                if taskid >= 1:
                    num_sample_pre_class = 5
                    num_pre_classes = taskid * GVM.cl_mngr.num_classes_per_task
                    sh_premean, sh_precov =\
                    GVM.cache_dict['p_shared']['mean'].to(current_device), GVM.cache_dict['p_shared']['cov'].to(current_device)

                    sh_presamples = []
                    for i in range(num_pre_classes):
                        m = MultivariateNormal(sh_premean[i].float(), sh_precov[i].float())
                        sh_presamples.append(m.sample(sample_shape=(num_sample_pre_class,)))
                    sh_presamples = torch.cat(sh_presamples, dim=0).to(current_device)

                    delta_feats = (sp_feats - sh_feats.detach())
                    G = knn_graph(feat1=sh_presamples, feat2=sh_feats, 
                                k=args.simulation_k, temp=args.simulation_temperature)
                    curr_presamples = sh_presamples + G.detach().mm(delta_feats)

                    pre_feats = torch.cat([sh_presamples, curr_presamples], dim=0).detach().to(torch.float)
                    pre_lbl = torch.arange(sh_presamples.shape[0]).to(current_device) // num_sample_pre_class
                    pre_lbl_ = torch.cat([pre_lbl, pre_lbl], dim=0).to(torch.int64)

                    curr_feats = torch.cat([sh_feats, sp_feats], dim=0)
                    curr_lbl = mix_lbl + taskid * GVM.cl_mngr.num_classes_per_task
                    curr_lbl_ = torch.cat([curr_lbl, curr_lbl], dim=0).to(torch.int64)

                    input_feats = torch.cat([pre_feats, curr_feats], dim=0).detach()
                    input_lbl = torch.cat([pre_lbl_, curr_lbl_], dim=0)

                else:
                    input_feats = torch.cat([sh_feats, sp_feats], dim=0)
                    input_lbl = torch.cat([mix_lbl, mix_lbl], dim=0)

                logits = model.module.task_head[taskid](input_feats)[:, :(taskid+1) * GVM.cl_mngr.num_classes_per_task]
                ca_loss = criterion(logits / temperature, input_lbl)

            fa_loss = torch.tensor(0.).to(current_device)
            if args.fa_loss:
                if taskid >= 1:
                    curr_lbl = mix_lbl + taskid * GVM.cl_mngr.num_classes_per_task
                    sample_feats, sample_lbls = [], []
                    if args.num_tasks > 20 and taskid>20:
                        task_list = np.array(list(range(taskid)))
                        sample_task = np.random.choice(task_list, size=(20,), replace=False)
                    else:
                        sample_task = list(range(taskid))
                    for t in sample_task:
                        sample_idx = torch.randint(low=0, high=current_in_previous_list[t][-1], size=(80,))
                        sample_feat, sample_lbl = current_in_previous_list[t][0][sample_idx], current_in_previous_list[t][1][sample_idx]
                        sample_feats.append(sample_feat.to(current_device))
                        sample_lbls.append(sample_lbl.to(current_device))
                    sample_feats, sample_lbls = torch.cat(sample_feats,0), torch.cat(sample_lbls,0)
                    
                    fa_loss = criterion_cl(sp_feats, sample_feats, sp_feats, curr_lbl, sample_lbls, curr_lbl) 

        if args.sh_loss:
            if i_batch == 1:
                if args.seperate_head:
                    assert sh_logits.shape[1] == len(GVM.cl_mngr.current_task_classes)
                else:
                    assert sh_logits.shape[1] == len(GVM.cl_mngr.sofar_task_classes)

        training_loss = sh_ce_loss + sp_ce_loss + ca_loss + fa_loss

        LN_mean_loss = torch.zeros_like(training_loss)
        LN_std_loss = torch.zeros_like(training_loss)

        ## If taskid > 0 then execute Null Space Projection
        if GVM.cl_mngr.current_taskid > 0:
            _dst_tt = GVM.cl_mngr.current_taskid - 1
            for _n0, _p0 in GVM.param_dict[f'task_params_{_dst_tt}'].items():
                if 'sh_prompt' in _n0 or 't_prompt' in _n0:  ## L2 regularization for shared prompts
                    _p0 = _p0.detach()
                    _pt = model.get_parameter(_n0)
                    _mpt, _mp0 = _pt.mean(-1), _p0.mean(-1)
                    LN_mean_loss += F.l1_loss(_mpt, _mp0)
                    _spt, _sp0 = _pt.std(-1, unbiased=False), _p0.std(-1, unbiased=False)
                    LN_std_loss += F.l1_loss(_spt, _sp0)
        
        null_space_loss = (LN_mean_loss + LN_std_loss) * args.ln_loss_lam

        # loss: Tensor = ce_loss + selection_loss
        loss: Tensor = training_loss + null_space_loss

        optimizer.zero_grad()
        amp_scalar.scale(loss).backward()
        amp_scalar.step(optimizer)
        amp_scalar.update()

        if args.sp_loss:
            acc_top1 = misc.calc_accuracy(sp_logits, target, topk=(1,))[0]
        else:
            acc_top1 = misc.calc_accuracy(sh_logits, target, topk=(1,))[0]
        batch_time = ttime() - _btimer

        scalar_meter.add_step_value(len(images), loss=loss.item(), acc_top1=acc_top1, batch_time=batch_time)
        _btimer = ttime()

        if i_batch % args.logger_out_step == 0:
            print('Logs_output: *** Iteration:{:3d} || Epoch:{:3d} || Current_loss:{:.4f} || Current_acc:{:.4f} || Time:{:.4f} ***'
                .format(i_batch, curr_epoch, loss, acc_top1, ttime()-tt))
            tt = ttime()

    _epoch_scalar_str = scalar_meter.format_outout(scalar_meter.update_epoch_average_value())
    return _epoch_scalar_str


def cache_state(GVM: GlobalVarsManager, taskid: int, model: VisionTransformer):
    if taskid == 0:
        base_params = OrderedDict()
    task_params = OrderedDict()
    ## save [task_params: requires_grad=True]; [base_params: requires_grad=False]
    for n, p in model.named_parameters():
        if p.requires_grad:
            task_params[n] = p.clone()
        else:
            if taskid == 0:
                base_params[n] = p.clone()

    if taskid == 0:
        GVM.param_dict['base_params'] = base_params
    GVM.param_dict[f'task_params_{taskid}'] = task_params


def train_one_task(GVM: GlobalVarsManager, taskid: int, task_classes: list[int], model: VisionTransformer, **kwargs) -> VisionTransformer:
    args = GVM.args

    _ttimer = ttime()
    _ntstr = str(GVM.cl_mngr.num_tasks) ## '10'

    ## set not-pretrained paramenters: require_grad=True & parameters.train()
    model: VisionTransformer = set_model_mode(GVM, taskid, model, training=True, training_string=GVM.cache_dict['training_string'])

    model = modify_head(GVM, model, training=True, task_classes=task_classes)
    ## create dataset: [samples:PIL, transformed labels:int], output: [transform(img), label, idx]
    dataset = define_dataset(GVM, task_classes, training=True, transform_type=args.transform_type, 
                            target_map_to_local=args.seperate_head, expand_times=args.expand_times)
                        
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, 
                            pin_memory=True, timeout=args.timeout if args.workers > 0 else 0,
                            drop_last=args.prob_cutmixup > 0, persistent_workers=args.persistent_workers)

    criterion = nn.CrossEntropyLoss().to(model.module.device)

    if args.lr_scale == 1:
        param_groups = filter(lambda p: p.requires_grad, model.parameters()) ## only requires_grad=True parameters
    else:
        param_groups = set_learning_rates(GVM, model, args.lr, args.lr_scale, args.lr_scale_patterns)

    if taskid >= 1:
        if args.copy_weight:
            tt = time.time()
            print(' -- Start copy adapter weights from the previous task -- ')
            print()
            for blk in  model.module.blocks:
                blk.task_adapter[taskid].down_proj.weight.data = copy.deepcopy(blk.task_adapter[taskid-1].down_proj.weight.data)
                blk.task_adapter[taskid].down_proj.bias.data = copy.deepcopy(blk.task_adapter[taskid-1].down_proj.bias.data)
                blk.task_adapter[taskid].up_proj.weight.data = copy.deepcopy(blk.task_adapter[taskid-1].up_proj.weight.data)
                blk.task_adapter[taskid].up_proj.bias.data = copy.deepcopy(blk.task_adapter[taskid-1].up_proj.bias.data)
            print(' -- Finish copy adapter weights from the previous task, total time:{:.4f} -- '.format(time.time() - tt))

    if taskid == 0:
        GVM.cache_dict['update_proj_dict'] = {}
        GVM.cache_dict['update_vector'] = {}
    if args.use_null_space:
        if taskid == 0:
            GVM.cache_dict['null_param_id_dict'] = get_param_id_dict(model, args.null_patterns) ## null space only for 'prompt'
            GVM.cache_dict['interm_tensor_dict'] = {}
    else:
        assert GVM.cache_dict['update_proj_dict'] == {}

    if args.optimizer == 'mod_adam':
        optimizer = ModAdam(param_groups,  ## only prompts
                            GVM.cache_dict['update_proj_dict'], 
                            arg_dict={}, lr=args.lr, 
                            weight_decay=args.weight_decay, foreach=True) ## lr = 0.01
    else:
        optimizer = create_optimizer_v2(param_groups, opt=args.optimizer, lr=args.lr, weight_decay=args.weight_decay, foreach=True)
    ## 10 epochs, milestones=[5,8]
    scheduler, num_epochs = create_scheduler_v2(optimizer, sched=args.lr_sch, num_epochs=args.epochs, decay_epochs=args.decay_epochs, decay_milestones=args.decay_milestones,
                                                decay_rate=args.decay_rate, min_lr=args.min_lr, warmup_epochs=args.warmup_epochs, warmup_lr=args.min_lr)
    assert num_epochs == args.epochs ## 10

    frozen_feature_list = None

    tt=time.time()
    print('Extract current data in previous subspaces')
    current_in_previous_list = []
    if taskid >= 1:
        for t in range(taskid):
            feats, labels = extract_class_features(GVM, t, model, mode='adapter', return_name='feature')
            current_in_previous_list.append((feats, labels, feats.shape[0]))
            # means = extract_class_features(GVM, t, model, mode='adapter', return_name='mean')
            # current_in_previous_list.append(means)
    print('Finish extract current data in previous subspaces, time:{:.4f}'.format(time.time()-tt))
    model = set_model_mode(GVM, taskid, model, training=True, training_string=GVM.cache_dict['training_string'])

    torch.cuda.empty_cache()
    for epoch in range(0, args.epochs + 1):
        if epoch > 0:
            _epoch_scalar_str = train_one_epoch(GVM, taskid, epoch, dataloader, model, criterion, optimizer, 
                                current_in_previous_list, frozen_feature_list)
            print(f"Task [{taskid+1:>{len(_ntstr)}}/{_ntstr}] Epoch [{epoch:>{len(_nestr:=str(args.epochs))}}/{_nestr}]:: {_epoch_scalar_str}")
        scheduler.step(epoch)
    ## save values of parameters (base_params & task_params) to GVM.param_dict
    cache_state(GVM, taskid, model) ## GVM.param_dict['base_params']; GVM.param_dict['task_params_0']
    ## len(GVM.cache_dict['null_param_id_dict'])=12, 'prompt'
    if args.use_null_space and taskid + 1 < GVM.cl_mngr.num_tasks:
        new_interm_tensor_dict = get_interm_tensor_dict(GVM, taskid, 'shared', model, GVM.cache_dict['null_param_id_dict']) 
        ## dict: [pid][name] extract (QW^T)^T * (QW^T) \in [768,768] & S_p^T * S_p \in [4,4]
        ## middle tensors in the current task
        GVM.cache_dict['interm_tensor_dict'] = accumulate_interm_tensor_dict(GVM, GVM.cache_dict['interm_tensor_dict'], new_interm_tensor_dict)
        ## merge middle tensors from the current task (new_interm_tensor_dict) and the old tasks (sum)
        ## update GVM.cache_dict['interm_tensor_dict']
        GVM.cache_dict['update_proj_dict'], GVM.cache_dict['update_vector'] =\
        get_update_projection_dict(GVM, GVM.cache_dict['null_param_id_dict'], GVM.cache_dict['interm_tensor_dict'])
        ## B = U^T * U [768,768] or [4,4] --> eta * B + (1-eta) * I

    if args.refine_head:
        refine_head(GVM, model)

    get_prototypes(GVM, taskid, model, mode='shared', is_return=False)
    get_prototypes(GVM, taskid, model, mode='adapter', is_return=False)                

    model.module.remove_text_features()            

    print(f"Task [{taskid+1:>{len(_ntstr)}}/{_ntstr}]:: Training time = {misc.format_duration(ttime() - _ttimer)}")

    return model


def evaluate_one_task(GVM: GlobalVarsManager, train_taskid: int, eval_taskid: int, eval_task_classes: list[int], model: VisionTransformer, full_head: list=None) -> OrderedDict[str, float]:
    use_amp: bool = GVM.args.use_amp
    _ttimer = ttime()

    dataset = define_dataset(GVM, eval_task_classes, training=False, transform_type=GVM.args.transform_type, target_map_to_local=False)
    dataloader = DataLoader(dataset, batch_size=GVM.args.eval_batch_size, shuffle=False, num_workers=GVM.args.eval_workers, pin_memory=True, timeout=GVM.args.timeout if GVM.args.eval_workers > 0 else 0)

    set_model_mode(GVM, train_taskid, model, training=False)
    scalar_meter = misc.ScalarMeter(acc_class_inc="samp_avg:>6.2%")

    torch.cuda.empty_cache()
    router_correct = 0
    router_total = 0
    # for images, target in tqdm.tqdm(dataloader, total=len(dataloader), dynamic_ncols=True, disable=not GVM.args.show_bar):

    for i, (images, target, _) in enumerate(dataloader):
        images: Tensor = images.cuda(non_blocking=True)
        target: Tensor = target.cuda(non_blocking=True) ## tensor: 100 [label(after relabel)]

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            with torch.no_grad():

                if args.eval_tool == 'shared':
                    feat_sh, logits_sh, _ = model(images, mode='shared')  ## tensor: [B, num_classes]
                    logits = logits_sh
                
                elif args.eval_tool in ['prompt', 'adapter'] and args.eval_cross_adapter_prototypes:
                    prototype_store = GVM.cache_dict.get('p_cross_adapter', {})
                    prototypes = prototype_store.get('mean')
                    if prototypes is None:
                        raise RuntimeError(
                            "Cross-adapter classification requires cached prototypes."
                        )
                    num_seen_classes = len(GVM.cl_mngr.sofar_task_classes)
                    logits = images.new_full(
                        (images.shape[0], num_seen_classes),
                        -torch.inf,
                    )
                    for adapter_id in range(train_taskid + 1):
                        adapter_weights = F.one_hot(
                            torch.full(
                                (images.shape[0],),
                                adapter_id,
                                device=images.device,
                                dtype=torch.long,
                            ),
                            train_taskid + 1,
                        ).to(dtype=images.dtype)
                        feat_adapter, _, _ = model(
                            images,
                            mode=args.train_tool + '_eval',
                            taskid=adapter_id,
                            max_taskid=train_taskid,
                            p_task=adapter_weights,
                            eval_adapter_scale=args.eval_adapter_scale,
                        )
                        local_prototypes = F.normalize(
                            prototypes[adapter_id, :num_seen_classes].to(feat_adapter.device).float(),
                            dim=1,
                        )
                        similarity = (
                            F.normalize(feat_adapter.float(), dim=1)
                            @ local_prototypes.t()
                        )
                        logits = torch.maximum(logits, similarity)

                elif args.eval_tool in ['prompt', 'adapter'] and args.eval_prototype_router:
                    prototype_store = GVM.cache_dict.get('p_adapter', {})
                    prototypes = prototype_store.get('mean')
                    if prototypes is None:
                        raise RuntimeError(
                            "Prototype routing requires cached adapter prototypes."
                        )
                    num_classes_per_task = GVM.cl_mngr.num_classes_per_task
                    num_seen_classes = len(GVM.cl_mngr.sofar_task_classes)
                    expert_logits = []
                    router_scores = []
                    for task_id in range(train_taskid + 1):
                        task_classes = GVM.cl_mngr.get_classes(task_id)
                        class_start = task_id * num_classes_per_task
                        class_end = class_start + len(task_classes)
                        adapter_weights = F.one_hot(
                            torch.full(
                                (images.shape[0],),
                                task_id,
                                device=images.device,
                                dtype=torch.long,
                            ),
                            train_taskid + 1,
                        ).to(dtype=images.dtype)
                        feat_adapter, logits_adapter, _ = model(
                            images,
                            mode=args.train_tool + '_eval',
                            taskid=task_id,
                            max_taskid=train_taskid,
                            p_task=adapter_weights,
                            eval_adapter_scale=args.eval_adapter_scale,
                        )
                        local_prototypes = F.normalize(
                            prototypes[class_start:class_end].to(feat_adapter.device).float(),
                            dim=1,
                        )
                        similarity = (
                            F.normalize(feat_adapter.float(), dim=1)
                            @ local_prototypes.t()
                        )
                        router_scores.append(similarity.max(dim=1).values)
                        expert_logits.append(logits_adapter)

                    selected_task = torch.stack(router_scores, dim=1).argmax(dim=1)
                    logits = torch.empty(
                        (images.shape[0], num_seen_classes),
                        device=images.device,
                        dtype=expert_logits[0].dtype,
                    )
                    for task_id, task_logits in enumerate(expert_logits):
                        mask = selected_task == task_id
                        if mask.any():
                            logits[mask] = task_logits[mask]

                elif args.eval_tool in ['prompt', 'adapter'] and args.eval_adapter_ensemble:
                    num_seen_classes = len(GVM.cl_mngr.sofar_task_classes)
                    ensemble_logits = images.new_full(
                        (images.shape[0], num_seen_classes),
                        -torch.inf,
                    )
                    for task_id in range(train_taskid + 1):
                        adapter_weights = F.one_hot(
                            torch.full(
                                (images.shape[0],),
                                task_id,
                                device=images.device,
                                dtype=torch.long,
                            ),
                            train_taskid + 1,
                        ).to(dtype=images.dtype)
                        _, logits_adapter, _ = model(
                            images,
                            mode=args.train_tool + '_eval',
                            taskid=task_id,
                            max_taskid=train_taskid,
                            p_task=adapter_weights,
                            eval_adapter_scale=args.eval_adapter_scale,
                        )
                        adapter_probs = F.softmax(
                            logits_adapter / args.eval_tau,
                            dim=1,
                        )
                        ensemble_logits = torch.maximum(
                            ensemble_logits,
                            adapter_probs,
                        )
                    logits = ensemble_logits

                elif args.eval_tool in ['prompt', 'adapter']:
                    
                    assert hasattr(args, 'task_detector')
                    if args.eval_trained_task_router:
                        router = GVM.cache_dict.get('task_router')
                        if router is None:
                            raise RuntimeError(
                                "Trained task routing requires a cached task router."
                            )
                        with torch.no_grad():
                            router_features = extract_adapter_router_features(
                                GVM,
                                model,
                                images,
                                train_taskid,
                            )
                        task_probabilities = router.predict_proba(
                            router_features.float().cpu().numpy()
                        )
                        p_task = torch.as_tensor(
                            task_probabilities,
                            device=images.device,
                            dtype=torch.float32,
                        )
                        selected_id = p_task.argmax(dim=1)
                        router_correct += int((selected_id == eval_taskid).sum().item())
                        router_total += selected_id.numel()
                        feat_t = images.new_empty((images.shape[0], 1))

                    elif args.task_detector == 'logit':

                        feat_t, _, _ = model.module.encode_image(images, mode='shared', pre_logits=True)
                        if full_head is not None:
                            # Eq. 8 builds the unified classifier gu; use it for
                            # Eq. 10/11 instead of the unaggregated task heads.
                            weights, bias = full_head
                            logits_t = feat_t.mm(weights.t()) + bias.unsqueeze(0)
                        else:
                            logits_t = model.module.head(feat_t)
                        tau_ = args.eval_tau

                        # Eq. 11 uses one confidence temperature. Training-only
                        # temperature must not be multiplied in a second time.
                        p_logits = torch.softmax(logits_t / tau_, dim=1)
                        if (
                            args.prototype_classifier
                            and 'p_shared' in GVM.cache_dict
                            and (prototypes := GVM.cache_dict['p_shared'].get('mean')) is not None
                            and prototypes.shape[0] == p_logits.shape[1]
                        ):
                            prototypes = F.normalize(prototypes.to(feat_t.device).float(), dim=1)
                            features = F.normalize(feat_t.float(), dim=1)
                            prototype_logits = features @ prototypes.t()
                            prototype_logits = torch.softmax(
                                prototype_logits / args.prototype_temperature, dim=1
                            )
                            weight = min(max(args.prototype_confidence_weight, 0.0), 1.0)
                            p_logits = weight * prototype_logits + (1.0 - weight) * p_logits
                        assert feat_t.dim() == 2

                        if args.nearest_class_selection:
                            fixed_k = True
                            if fixed_k:
                                nearest_num = min(args.prototype_k, p_logits.shape[1])
                            else:
                                nearest_num = int(p_logits.shape[1] * args.prototype_ratio)
                            mask = p_logits >= p_logits.sort(dim=1, descending=True)[0][:, nearest_num-1].unsqueeze(1)
                            p_logits = p_logits * torch.as_tensor(mask, dtype=torch.float)
                            p_logits = p_logits / p_logits.sum(1).unsqueeze(1)

                        p_task = torch.zeros(feat_t.shape[0], train_taskid+1).to(feat_t.device)
                        for taskid in range(train_taskid+1):
                            if args.max_logit_score:
                                intra_prediction = p_logits[:, 
                                (GVM.cl_mngr.num_classes_per_task)*taskid:(GVM.cl_mngr.num_classes_per_task)*(taskid+1)].max(-1)[0]
                            else:
                                intra_prediction = p_logits[:, 
                                (GVM.cl_mngr.num_classes_per_task)*taskid:(GVM.cl_mngr.num_classes_per_task)*(taskid+1)].sum(-1)

                            p_task[:, taskid] = intra_prediction
                            
                        p_task = p_task / p_task.sum(1).unsqueeze(1)
                        selected_id = p_task.argmax(1)

                        if args.topk_adapter_selection:
                            selection_num = min(args.adapter_k, taskid+1)
                            mask = p_task >= p_task.sort(dim=1, descending=True)[0][:, selection_num-1].unsqueeze(1)
                            p_task = p_task * torch.as_tensor(mask, dtype=torch.float)

                        assert p_task.shape[1] == train_taskid+1
                        
                    else:
                        raise ValueError(args.task_detector)

                    if args.eval_task_weight:  
                        ## mixture-of-adapters
                        assert p_task is not None
                        feat_sp, logits, _ = model(images, mode=args.train_tool+'_eval', 
                                    taskid=selected_id, max_taskid=train_taskid, p_task=p_task, 
                                    eval_adapter_scale=args.eval_adapter_scale)

                    else:
                        feat_sp, logits, _ = model(images, mode=args.train_tool+'_eval', 
                                    taskid=selected_id, max_taskid=train_taskid, p_task=None, 
                                    eval_adapter_scale=args.eval_adapter_scale)  

                    # classifier fusion (logit-level)
                    if args.eval_local_task_head:
                        local_logits = torch.full_like(logits, -torch.inf)
                        for task_id in range(train_taskid + 1):
                            selected = selected_id == task_id
                            if selected.any():
                                task_classes = GVM.cl_mngr.get_classes(task_id)
                                class_start = task_id * GVM.cl_mngr.num_classes_per_task
                                class_end = class_start + len(task_classes)
                                if (
                                    torch.linalg.norm(
                                        model.module.head.weight[class_start:class_end]
                                    )
                                    > 1e-3
                                ):
                                    task_out = model.module.head(feat_sp)
                                else:
                                    task_out = model.module.task_head[task_id](feat_sp)
                                local_logits[selected] = task_out[selected, :logits.shape[1]]
                        logits = local_logits

                    elif args.classifier_aggregation_type == 'weighted':
                        num_classes = GVM.cl_mngr.num_classes_per_task
                        sub_logits = torch.zeros_like(logits).to(feat_t.device)
                        p_task = p_task.clamp(min=1e-6)
                        for t in range(train_taskid+1):
                            fused_logits = []
                            for t_ in range(t, train_taskid+1):
                                out = model.module.task_head[t_](feat_sp)
                                fused_logits.append(out)
                            curr_p = p_task[:, t:] / p_task[:, t:].sum(1).unsqueeze(1)
                            fused_logits = (torch.stack(fused_logits).permute(1,0,2) * curr_p.unsqueeze(-1)).sum(1)
                            class_start = t * num_classes
                            class_end = min(
                                (t + 1) * num_classes,
                                sub_logits.shape[1],
                            )
                            sub_logits[:, class_start:class_end] =\
                            fused_logits[:, class_start:class_end]
                        eta_ = 1 / (train_taskid + 2)
                        logits = sub_logits[:,:logits.shape[1]] * (1-eta_) + logits * eta_ 

                    elif args.classifier_aggregation_type == 'mean':
                        assert full_head is not None
                        weights, bias = full_head[0], full_head[1]
                        sub_logits = feat_sp.mm(weights.t()) + bias.unsqueeze(0)
                        eta_ = 1 / (train_taskid + 2)
                        logits = sub_logits[:,:logits.shape[1]] * (1-eta_) + logits * eta_

                elif args.eval_tool == 'upper_bound':
                    ## taskid is avaliable
                    feat_sp, logits, _ = model(images, mode=args.train_tool+'_train', 
                                    taskid=eval_taskid, max_taskid=train_taskid)
                    if args.eval_local_task_head:
                        task_logits = model.module.task_head[eval_taskid](feat_sp)
                        class_start = eval_taskid * GVM.cl_mngr.num_classes_per_task
                        class_end = class_start + len(eval_task_classes)
                        logits = images.new_full(
                            (images.shape[0], len(GVM.cl_mngr.sofar_task_classes)),
                            -torch.inf,
                        )
                        logits[:, class_start:class_end] = task_logits[:, class_start:class_end]

        assert logits.ndim == 2
        assert logits.shape[1] == len(GVM.cl_mngr.sofar_task_classes), f"{logits.shape}, {len(GVM.cl_mngr.sofar_task_classes)}"

        _preds = logits.argmax(dim=1)
        acc_class_inc, _, _ = misc.calc_acc_topnn_dynamically(_preds, target)
        scalar_meter.add_step_value(target.shape[0], acc_class_inc=acc_class_inc)

    # assert len(dataset) == len(scalar_meter)
    result_dict = scalar_meter.update_epoch_average_value()
    if router_total:
        print(
            f"Task [{train_taskid+1}/{GVM.cl_mngr.num_tasks}]:: "
            f"Router-Eval[{eval_taskid+1}]: "
            f"{router_correct / router_total:.2%}"
        )

    print(f"Task [{train_taskid+1}/{GVM.cl_mngr.num_tasks}]:: Eval [{eval_taskid+1:>{len(_tt:=str(train_taskid+1))}}/{_tt}]: eval_time={ttime()-_ttimer:.1f}s, {scalar_meter.format_outout(result_dict)}")

    result_dict['num_samples'] = len(dataset)

    return result_dict


def cache_adapter_prototypes(GVM: GlobalVarsManager, train_taskid: int, model: VisionTransformer) -> None:
    """Rebuild frozen adapter prototypes when loading a trained checkpoint."""
    args = GVM.args
    model = set_model_mode(GVM, train_taskid, model, training=False)
    mean_by_output = {}
    count_by_output = {}
    for task_id in range(train_taskid + 1):
        task_classes = GVM.cl_mngr.get_classes(task_id)
        output_labels = [GVM.label_map_g2l[int(cls)][2] for cls in task_classes]
        dataset = define_dataset(
            GVM,
            task_classes,
            training=True,
            transform_type=args.transform_type,
            target_map_to_local=False,
            use_eval_transform=True,
            expand_times=1,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        for images, labels, _ in dataloader:
            images = images.cuda(non_blocking=True)
            adapter_weights = F.one_hot(
                torch.full(
                    (images.shape[0],),
                    task_id,
                    device=images.device,
                    dtype=torch.long,
                ),
                train_taskid + 1,
            ).to(dtype=images.dtype)
            with torch.no_grad(), torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=args.use_amp,
            ):
                features = model.module.encode_image(
                    images,
                    mode=args.train_tool + '_eval',
                    taskid=task_id,
                    max_taskid=train_taskid,
                    p_task=adapter_weights,
                    eval_adapter_scale=args.eval_adapter_scale,
                    pre_logits=True,
                )[0].float().cpu()
            labels = labels.cpu()
            for output_label in output_labels:
                mask = labels == output_label
                if not mask.any():
                    continue
                selected = features[mask]
                if output_label not in mean_by_output:
                    mean_by_output[output_label] = selected.sum(dim=0)
                    count_by_output[output_label] = selected.shape[0]
                else:
                    mean_by_output[output_label] += selected.sum(dim=0)
                    count_by_output[output_label] += selected.shape[0]

    ordered_labels = sorted(mean_by_output)
    means = torch.stack(
        [mean_by_output[label] / count_by_output[label] for label in ordered_labels]
    )
    GVM.cache_dict['p_adapter'] = {'mean': means}


def cache_cross_adapter_prototypes(
    GVM: GlobalVarsManager,
    train_taskid: int,
    model: VisionTransformer,
) -> None:
    """Compute one centroid per seen class in every frozen adapter subspace."""
    args = GVM.args
    model = set_model_mode(GVM, train_taskid, model, training=False)
    seen_classes = GVM.cl_mngr.sofar_task_classes
    num_seen_classes = len(seen_classes)
    dataset = define_dataset(
        GVM,
        seen_classes,
        training=True,
        transform_type=args.transform_type,
        target_map_to_local=False,
        use_eval_transform=True,
        expand_times=1,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    adapter_means = torch.zeros(
        train_taskid + 1,
        num_seen_classes,
        model.module.embed_dim,
    )
    adapter_counts = torch.zeros(
        train_taskid + 1,
        num_seen_classes,
    )
    for adapter_id in range(train_taskid + 1):
        for images, labels, _ in dataloader:
            images = images.cuda(non_blocking=True)
            adapter_weights = F.one_hot(
                torch.full(
                    (images.shape[0],),
                    adapter_id,
                    device=images.device,
                    dtype=torch.long,
                ),
                train_taskid + 1,
            ).to(dtype=images.dtype)
            with torch.no_grad(), torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=args.use_amp,
            ):
                features = model.module.encode_image(
                    images,
                    mode=args.train_tool + '_eval',
                    taskid=adapter_id,
                    max_taskid=train_taskid,
                    p_task=adapter_weights,
                    eval_adapter_scale=args.eval_adapter_scale,
                    pre_logits=True,
                )[0].float().cpu()
            labels = labels.cpu()
            for label in labels.unique():
                mask = labels == label
                label_index = int(label.item())
                adapter_means[adapter_id, label_index] += features[mask].sum(dim=0)
                adapter_counts[adapter_id, label_index] += int(mask.sum().item())

    valid = adapter_counts > 0
    adapter_means[valid] /= adapter_counts[valid].unsqueeze(1)
    GVM.cache_dict['p_cross_adapter'] = {'mean': adapter_means}


def extract_adapter_router_features(
    GVM: GlobalVarsManager,
    model: VisionTransformer,
    images: Tensor,
    train_taskid: int,
) -> Tensor:
    args = GVM.args
    features = []
    for task_id in range(train_taskid + 1):
        adapter_weights = F.one_hot(
            torch.full(
                (images.shape[0],),
                task_id,
                device=images.device,
                dtype=torch.long,
            ),
            train_taskid + 1,
        ).to(dtype=images.dtype)
        _, logits_adapter, _ = model(
            images,
            mode=args.train_tool + '_eval',
            taskid=task_id,
            max_taskid=train_taskid,
            p_task=adapter_weights,
            eval_adapter_scale=args.eval_adapter_scale,
        )
        features.append(logits_adapter.float())
    return torch.cat(features, dim=1)


def cache_trained_task_router(
    GVM: GlobalVarsManager,
    train_taskid: int,
    model: VisionTransformer,
) -> None:
    args = GVM.args
    model = set_model_mode(GVM, train_taskid, model, training=False)
    seen_classes = GVM.cl_mngr.sofar_task_classes
    output_to_task = {}
    for global_label, (task_id, _, output_label) in GVM.label_map_g2l.items():
        if global_label in seen_classes:
            output_to_task[output_label] = task_id

    dataset = define_dataset(
        GVM,
        seen_classes,
        training=True,
        transform_type=args.transform_type,
        target_map_to_local=False,
        use_eval_transform=True,
        expand_times=1,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    feature_batches = []
    task_labels = []
    for images, labels, _ in dataloader:
        images = images.cuda(non_blocking=True)
        with torch.no_grad(), torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=args.use_amp,
        ):
            router_features = extract_adapter_router_features(
                GVM,
                model,
                images,
                train_taskid,
            )
        feature_batches.append(router_features.cpu())
        task_labels.extend(
            output_to_task[int(label)]
            for label in labels.tolist()
        )

    router_features = torch.cat(feature_batches, dim=0).numpy()
    task_labels = np.asarray(task_labels, dtype=np.int64)
    router = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight='balanced',
        n_jobs=1,
        random_state=GVM.args.seed,
    )
    router.fit(router_features, task_labels)
    print(
        "Task router training accuracy: {:.2f}%".format(
            router.score(router_features, task_labels) * 100
        )
    )
    GVM.cache_dict['task_router'] = router


def evaluate_tasks_sofar(GVM: GlobalVarsManager, train_taskid: int, model: VisionTransformer, 
                        pretrained_model_path: str = None):

    if pretrained_model_path is not None:
        # Uneven splits such as 2/2/2/2/1 have fewer output classes than
        # num_tasks * num_classes_per_task.
        model.module.head = nn.Linear(
            model.module.embed_dim,
            len(GVM.cl_mngr.sofar_task_classes),
        )
        model.load_state_dict(torch.load(pretrained_model_path), strict=False)
    else:
        model = modify_head(GVM, model, training=False)
    if (
        args.eval_prototype_router
        and GVM.cache_dict.get('p_adapter', {}).get('mean') is None
    ):
        cache_adapter_prototypes(GVM, train_taskid, model)
    if args.eval_cross_adapter_prototypes:
        cache_cross_adapter_prototypes(GVM, train_taskid, model)
    if args.eval_trained_task_router:
        cache_trained_task_router(GVM, train_taskid, model)
    torch.cuda.empty_cache()
    average_acc_meter = misc.ScalarMeter(acc_class_inc="samp_avg:>6.2%")

    ## weight fusion
    current_device = model.module.device
    num_classes = GVM.cl_mngr.num_classes_per_task  
    with torch.no_grad():
        fused_weight = torch.zeros((train_taskid+1)*num_classes, model.module.embed_dim).to(current_device)
        fused_bias = torch.zeros((train_taskid+1)*num_classes).to(current_device)
        for taskid in range(train_taskid + 1):
            fusion_weight, fusion_bias = [], []
            for t in range(taskid, train_taskid+1):
                fusion_weight.append(model.module.task_head[t].weight)
                fusion_bias.append(model.module.task_head[t].bias)
            fused_weight[taskid*num_classes : (taskid+1)*num_classes] =\
            copy.deepcopy(torch.stack(fusion_weight).mean(0)[taskid*num_classes : (taskid+1)*num_classes].to(current_device))
            fused_bias[taskid*num_classes : (taskid+1)*num_classes] =\
            copy.deepcopy(torch.stack(fusion_bias).mean(0)[taskid*num_classes : (taskid+1)*num_classes].to(current_device))
    full_head = [fused_weight, fused_bias]

    if pretrained_model_path is not None:
        for eval_taskid in range(args.num_tasks):
            eval_task_classes = GVM.cl_mngr.get_classes(eval_taskid) ## list[gt_labels]
            one_result_dict = evaluate_one_task(GVM, train_taskid, eval_taskid, eval_task_classes, model, full_head)
            GVM.acc_mat_dict[f'AccClassIncMat'][train_taskid, eval_taskid] = one_result_dict['acc_class_inc']
            average_acc_meter.add_step_value(**one_result_dict)
        model.module.remove_text_features()
    else:
        for eval_taskid in range(GVM.cl_mngr.current_taskid + 1):
            eval_task_classes = GVM.cl_mngr.get_classes(eval_taskid) ## list[gt_labels]
            one_result_dict = evaluate_one_task(GVM, train_taskid, eval_taskid, eval_task_classes, model, full_head)
            GVM.acc_mat_dict[f'AccClassIncMat'][train_taskid, eval_taskid] = one_result_dict['acc_class_inc']
            average_acc_meter.add_step_value(**one_result_dict)
        model.module.remove_text_features()

    avg_result_dict = average_acc_meter.update_epoch_average_value()
    GVM.acc_mat_dict[f'AccClassIncList'][train_taskid] = avg_result_dict['acc_class_inc']


def task_ending_info(GVM: GlobalVarsManager):
    current_taskid = GVM.cl_mngr.current_taskid

    acc_info_dict = {
        'class_inc_last_acc': float(GVM.acc_mat_dict['AccClassIncList'][current_taskid]),
        'class_inc_avg_acc': float(GVM.acc_mat_dict['AccClassIncList'][:current_taskid+1].mean()),
        'class_inc_last_forg': misc.calc_forgetting(GVM.acc_mat_dict['AccClassIncMat'], current_taskid),
    }
    _formatter = misc.ScalarFormatter(sep=' | ', class_inc_last_acc=">6.2%", class_inc_avg_acc=">6.2%", class_inc_last_forg=">6.2%")

    print(f":: ** Results of task [{current_taskid+1}]: [ {_formatter(**acc_info_dict)} ] **")
    print(f":: ** Time so far: {misc.format_duration(ttime() - GVM.cache_dict['exp_start_time'])} **")
    for i in range(GVM.cl_mngr.current_taskid+1):
        print("{:.2f}".format(GVM.acc_mat_dict['AccClassIncList'][i]*100))


def find_not_pretrained_params(model: VisionTransformer, pretrained: bool = True, pretrained_cfg: dict[str, str] = None, extra_pretrained_params: list[str] = []) -> list[str]:
    assert isinstance(extra_pretrained_params, (list, tuple))
    if getattr(model, 'is_tabular', False):
        return [name for name, _ in model.named_parameters() if name.startswith('patch_embed.')]

    if isinstance(model, CLIP):
        model_path = osp.join(os.path.expanduser("~/.cache/clip"), osp.basename(clip.clip._MODELS['ViT-B/16']))
        assert osp.exists(model_path), model_path
        pre_state_dict: OrderedDict[str, Tensor] = torch.jit.load(model_path).state_dict()
    else:
        assert pretrained_cfg is not None

        if 'open_clip' in pretrained_cfg.get('hf_hub_filename', ''):
            _filename = timm.models._hub.HF_OPEN_CLIP_WEIGHTS_NAME
        else:
            _filename = timm.models._hub.HF_WEIGHTS_NAME
        pre_state_dict: OrderedDict[str, Tensor] = timm.models.load_state_dict_from_hf(pretrained_cfg['hf_hub_id'], _filename)

        if 'visual.class_embedding' in pre_state_dict.keys():
            pre_state_dict = timm.models.vision_transformer._convert_openai_clip(pre_state_dict, model)

    not_pretrained_params = []
    for n, p in model.named_parameters():
        if n not in pre_state_dict.keys() or not pretrained:
            not_pretrained_params.append(n)
        else:
            if p.shape != pre_state_dict[n].shape:
                not_pretrained_params.append(n)

    for n in deepcopy(not_pretrained_params):
        for _p in extra_pretrained_params:
            if _p in n:
                not_pretrained_params.remove(n)

    return not_pretrained_params


def get_param_id_dict(model: VisionTransformer, patterns: list[str]) -> dict[int, dict[Literal['name', 'shape'], str | list[int]]]:
    param_id_dict = {}
    for n, p in model.named_parameters():
        # if p.requires_grad and any([_s in n for _s in patterns]):
        if p.requires_grad and any(pattern in n for pattern in patterns):
            param_id_dict[id(p)] = {'name': n, 'shape': list(p.shape)}
    assert len(param_id_dict) > 0, f"{param_id_dict}"
    return param_id_dict


def get_text_features(GVM: GlobalVarsManager, model: CLIP, task_classes: list[int]) -> Tensor:
    dataset_name: str = GVM.args.dataset
    class_text_list: list[str] = [clip.text_prompt_dict[dataset_name]["classes"][c] for c in task_classes]
    tmpl_text_list: list[str] = clip.text_prompt_dict[dataset_name]["templates"]

    model.cuda()
    with torch.device('cuda'):
        with torch.no_grad():
            text_tokens = clip.tokenize([_t.format(_c) for _c in class_text_list for _t in tmpl_text_list]).to(next(model.parameters()).device)
            text_features = model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            text_features: Tensor = reduce(text_features, '(c p) d -> c d', p=len(tmpl_text_list), c=len(class_text_list), reduction='mean')
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            text_features = text_features.detach()
    model.cpu()
    torch.cuda.empty_cache()

    return text_features


def get_head_dim_arg_dict(GVM: GlobalVarsManager, args: argparse.Namespace) -> dict[Literal['num_classes'], int]:
    head_dim_arg_dict = {}
    head_dim_type = args.head_dim_type ## task_classes

    match args.logit_type:
        case 'sim_imgtext':
            assert head_dim_type in ('pretrained', 'text_dim')
        case 'head_out':
            assert head_dim_type in ('task_classes')

    match head_dim_type:
        case 'task_classes':
            head_dim_arg_dict['num_classes'] = len(current_task_classes) if args.seperate_head else len(GVM.cl_mngr.sofar_task_classes)
        case 'pretrained':
            pass
        case 'text_dim':
            head_dim_arg_dict['num_classes'] = 512
        case _:
            raise ValueError(head_dim_type)
    return head_dim_arg_dict


def modify_head(GVM: GlobalVarsManager, model: VisionTransformer, training: bool, **kwargs):
    args: argparse.Namespace = GVM.args
    ## get classes for current task
    if training:
        _target_classes = kwargs['task_classes'] if args.seperate_head else GVM.cl_mngr.sofar_task_classes
    else:
        _target_classes = GVM.cl_mngr.sofar_task_classes ## ground truth labels (w/o relabel)

    if args.logit_type == 'sim_imgtext':
        model.cache_text_features(get_text_features(GVM, GVM.cache_dict['clip_model'], _target_classes))

    elif args.logit_type == 'head_out':
        if args.previous_head and GVM.cl_mngr.current_taskid >= 1 and training:
            _mh = deepcopy(model.module.head)
            _mdevice = _mh.weight.device
            _mdtype = _mh.weight.dtype
            model.module.previous_head =  nn.Linear(model.module.embed_dim, GVM.cl_mngr.current_taskid * len(_target_classes)).to(model.module.device)
            _hw = torch.cat([GVM.param_dict[f'task_params_{_t}']['module.head.weight'].data.to(_mdevice, _mdtype) 
                            for _t in range(GVM.cl_mngr.current_taskid)]) ## previous_task
            if model.module.previous_head.weight.data.shape != _hw.shape:
                model.module.previous_head =  nn.Linear(model.module.embed_dim, _hw.shape[0]).to(model.module.device)
            model.module.previous_head.weight.data = _hw
            if _mh.bias is not None:
                _hb = torch.cat([GVM.param_dict[f'task_params_{_t}']['module.head.bias'].data.to(_mdevice, _mdtype) 
                                for _t in range(GVM.cl_mngr.current_taskid)])
                assert model.module.previous_head.bias.data.shape == _hb.shape
                model.module.previous_head.bias.data = _hb
            model.module.previous_head.weight.requires_grad = False
            model.module.previous_head.bias.requires_grad = False


        if model.module.head.out_features != len(_target_classes): ## taskid >= 2
            _mh = deepcopy(model.module.head)
            _mdevice = _mh.weight.device
            _mdtype = _mh.weight.dtype
            model.module.head = _mh.__class__(_mh.in_features, len(_target_classes), _mh.bias is not None, _mdevice, _mdtype)
            model.module.head.requires_grad_(_mh.weight.requires_grad)
            ## task2 : [20, 768]
            if training:
                assert model.module.head.weight.requires_grad
                    
            else:
                assert _mh.out_features == len(GVM.cl_mngr.current_task_classes), f"{_mh.out_features}, {len(GVM.cl_mngr.current_task_classes)}"
                _hw = torch.cat([GVM.param_dict[f'task_params_{_t}']['module.head.weight'].data.to(_mdevice, _mdtype) for _t in range(GVM.cl_mngr.current_taskid + 1)])
                assert model.module.head.weight.data.shape == _hw.shape
                model.module.head.weight.data = _hw ## 20, 768

                if _mh.bias is not None:
                    _hb = torch.cat([GVM.param_dict[f'task_params_{_t}']['module.head.bias'].data.to(_mdevice, _mdtype) for _t in range(GVM.cl_mngr.current_taskid + 1)])
                    assert model.module.head.bias.data.shape == _hb.shape
                    model.module.head.bias.data = _hb
    else:
        raise ValueError(args.logit_type)

    return model


def get_interm_tensor_dict(GVM: GlobalVarsManager, taskid:int, mode: str,
    model: VisionTransformer, null_param_id_dict: dict) -> dict[int, Tensor]:
    interm_tensor_dict: dict[int, dict[str, Tensor]] = {}
    if isinstance(model, nn.DataParallel):
        model = model.module

    def _forward_hook(module: nn.Module, args: tuple[Tensor], output: Tensor):
        _pre_tokens = 197
        if isinstance(module, nn.Linear):
            _interm_tensor = args[0]
            assert (_pid := id(module.weight)) in null_param_id_dict
            if _pid not in interm_tensor_dict:
                interm_tensor_dict[_pid] = []
            interm_tensor_dict[_pid].append(_interm_tensor)
            raise NotImplementedError()
        elif isinstance(module, utils.vit_builder.IntermReader):
            _pid = module.dst_param_id  ## module index
            _mname = module.module_name  ## name: interm_reader_1
            _interm_tensor: Tensor = args[0] ## tensor: [100, H, (num_patch + prompt_length), 768//H] (multi-head: H=12)

            if _mname == 'interm_reader_1':
                w_qkv = module.other_args['w_qkv'].detach() ## attn.qkv.weight
                w_k: Tensor = rearrange(w_qkv, '(n do) di -> n do di', n=3, do=768, di=768).unbind(0)[1]
                w_k = rearrange(w_k, '(h d) D -> h d D', h=12, d=64, D=768) ## W_k: 768 * 768 --> 12 * 64 * 768
                w_k = repeat(w_k, 'h d D -> b h d D', b=_interm_tensor.shape[0]) ## 12,64,768 --> B, 12, 768//H, 768

                _interm_tensor = _interm_tensor[:, :, :_pre_tokens] ## B, H, (num_patch), 768//H
                assert _interm_tensor.shape[2] == _pre_tokens, f"{_interm_tensor.shape}"
                _interm_tensor = _interm_tensor @ w_k ## B, H, num_patch, 768
                _interm_tensor = rearrange(_interm_tensor, 'b h n d -> (b h n) d') ## (B*H*Np), 768
                _interm_tensor = torch.matmul(_interm_tensor.T, _interm_tensor) / _interm_tensor.shape[0]
                ## (QW^T)^T * (QW^T) [768, 768]
            if _mname == 'interm_reader_2':
                _interm_tensor = _interm_tensor[:, :, :_pre_tokens, _pre_tokens:] ## input=[x, prompt], S_p \in [B, H, Np, Lp]
                # assert _interm_tensor.shape[-1] == GVM.args.prompt_len
                _interm_tensor = rearrange(_interm_tensor, 'b h n m -> (b h n) m')
                _interm_tensor = torch.matmul(_interm_tensor.T, _interm_tensor) / _interm_tensor.shape[0]
                ## S_p^T * S_p [4, 4]
            assert _mname in ('interm_reader_1', 'interm_reader_2')
            if _pid not in interm_tensor_dict: ## Dict for new layers
                interm_tensor_dict[_pid] = {}
            if _mname not in interm_tensor_dict[_pid]: ## New parameter (QW^T & S_p) for an existing layer
                interm_tensor_dict[_pid][_mname] = torch.zeros_like(_interm_tensor)
            interm_tensor_dict[_pid][_mname] += _interm_tensor
        else:
            raise NotImplementedError()   ## interm_tensor_dict[pid(int)][name(interm_reader_1 or interm_reader_2)]
    ## blocks.0.attn.interm_reader_1; blocks.0.attn.interm_reader_2; ...
    _handle_list: list[torch.utils.hooks.RemovableHandle] = []
    for n, m in model.named_modules(): ## module in [interm_reader1 or interm_reader2]
        if 'interm_reader' in n and isinstance(m, utils.vit_builder.IntermReader): 
            _handle_list.append(m.register_forward_hook(_forward_hook)) ## additional operations on middle layers
            ## interm_reader1 --> _forward_hook(function)
    model = set_model_mode(GVM, taskid, model, training=False) ## model.eval()
    torch.cuda.empty_cache()
    ## Extract input space (QW^T & S_p)
    args = GVM.args
    dataset = define_dataset(GVM, GVM.cl_mngr.current_task_classes, training=True, transform_type=args.transform_type, target_map_to_local=args.seperate_head, use_eval_transform=True, expand_times=1, )
    dataloader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.eval_workers, pin_memory=True, timeout=args.timeout if args.eval_workers > 0 else 0)
    ## len(dataset) = 5000, shuffle=False, batch_size=100
    for img, _, _ in dataloader:
        with torch.no_grad():
            img: Tensor
            if mode == 'shared':
                model(img.cuda(non_blocking=True), mode=mode) ## out: interm_tensor_dict[pid][name]
            elif mode == 'selection':
                model(img.cuda(non_blocking=True), mode=mode+'_evaluation')
    torch.cuda.empty_cache()

    for _h in _handle_list:
        _h.remove()

    assert len(interm_tensor_dict) > 0
    # assert list(interm_tensor_dict.keys()) == list(null_param_id_dict.keys()), f"{interm_tensor_dict.keys()}; {null_param_id_dict}"

    if (_k := 'interm_sample_list') not in GVM.cache_dict:
        GVM.cache_dict[_k] = []
    GVM.cache_dict[_k].append(len(dataloader.dataset)) ## number of samples: 5000

    model = nn.DataParallel(model, device_ids=args.device_ids)

    return interm_tensor_dict



def accumulate_interm_tensor_dict(GVM: GlobalVarsManager, cached_interm_tensor_dict: dict[int, dict[str, Tensor]], new_interm_tensor_dict: dict[int, dict[str, Tensor]]) -> dict[int, dict[str, Tensor]]:
    assert len(new_interm_tensor_dict) > 0
    args = GVM.args

    if cached_interm_tensor_dict == {}:
        merged_interm_tensor_dict = new_interm_tensor_dict ## if task_id==1: return new_interm_tensor_dict
    else:
        assert (_lc := list(cached_interm_tensor_dict.keys())) == (_ln := list(new_interm_tensor_dict.keys())), f"{_lc}, {_ln}"
        merged_interm_tensor_dict: dict[int, Tensor] = {}
        for _pid in cached_interm_tensor_dict.keys():
            merged_interm_tensor_dict[_pid] = {}
            for _mname in cached_interm_tensor_dict[_pid].keys():
                _cached_tensor = cached_interm_tensor_dict[_pid][_mname] ## tensors from the old tasks
                _new_tensor = new_interm_tensor_dict[_pid][_mname] ## tensors from the current task
                assert _cached_tensor.shape == _new_tensor.shape

                match args.null_interm_accum: ## sum
                    case 'sum':
                        merged_interm_tensor_dict[_pid][_mname] = _cached_tensor + _new_tensor
                    case 'mean':
                        _num_list: list[int] = GVM.cache_dict['interm_sample_list']
                        merged_interm_tensor_dict[_pid][_mname] = sum(_num_list[:-1]) / sum(_num_list) * _cached_tensor + _num_list[-1] / sum(_num_list) * _new_tensor
                    case _:
                        raise ValueError()
    return merged_interm_tensor_dict


def get_update_projection_dict(GVM: GlobalVarsManager, null_param_id_dict: dict, interm_tensor_dict: dict[int, dict[str, Tensor]]) -> dict[int, dict[str, Tensor]]:
    args = GVM.args

    update_proj_dict = {}
    update_vector = {}
    torch.cuda.empty_cache()

    def adaptive_threshold(svals: torch.Tensor, offset: float = 0): ## input: S \in 768
        points: np.ndarray = svals.cpu().numpy()
        assert points.ndim == 1
        if len(points) >= 128:
            fil_points = scipy.ndimage.gaussian_filter1d(points, sigma=10)
            _delta = 1
            diff_o1 = fil_points[:-_delta] - fil_points[_delta:] ## diff = S[0:766] - S[1:767] lambda(j) - lambda(j+1)
            diff_o2 = diff_o1[:-1] - diff_o1[1:] ## lambda(j) - 2*lambda(j+1) + lambda(j+2)  ## j <= 765
            _drop_ratio = 0.03
            drop_num = int(len(points) * _drop_ratio / 2) 
            assert len(points) - drop_num >= 10
            valid_o2 = diff_o2[drop_num:-drop_num]  ## len = len(points) * (1 - _drop_ratio)
            thres_val = points[np.argmax(valid_o2) + int((len(points) - len(valid_o2)) / 2)]  ## selected_idx + drop_num
        else:
            diff_o1 = points[:-1] - points[1:]
            diff_o2 = diff_o1[:-1] - diff_o1[1:]
            thres_val = points[np.argmax(diff_o2) + int((len(points) - len(diff_o2)) / 2)]
        i_thres = np.arange(len(points))[points >= thres_val].max() ## selected_idx for SVD
        if 0 <= offset < 1:
            i_thres = min(i_thres + int(offset * (len(points) - i_thres)), len(points) - 1)
        else:
            i_thres = max(min(i_thres + int(offset), len(points) - 1), 0)

        zero_idx = np.zeros(len(points), dtype=np.int64)
        zero_idx[i_thres:] = 1
        zero_idx = torch.as_tensor(torch.from_numpy(zero_idx), dtype=torch.bool, device=svals.device)
        return zero_idx

    for (layer_idx, _pid) in enumerate(interm_tensor_dict.keys()):
        update_proj_dict[_pid] = {}
        update_vector[_pid] = {}
        for _mname in interm_tensor_dict[_pid].keys():
            _, S, U_trans = torch.linalg.svd(interm_tensor_dict[_pid][_mname], full_matrices=True) ## SVD
            S: Tensor
            U_trans: Tensor

            thres_value = {'interm_reader_1': args.null_thres_value1, 'interm_reader_2': args.null_thres_value2}[_mname]
            match args.null_thres_mode: ## adaptive
                case 'times':
                    zero_idx = S <= S[-1] * int(thres_value)
                case 'adaptive': ## return zero_idx \in 768; selected null space=1
                    zero_idx = adaptive_threshold(S, offset=thres_value) ## S=[lambda1, lambda2, ...]
                case _:
                    raise ValueError(args.null_thres_mode)
            zero_idx: torch.BoolTensor
            assert torch.count_nonzero(zero_idx) > 0, f"{zero_idx}, {type(zero_idx)}, {torch.count_nonzero(zero_idx)}"

            no_zeros_idx = (zero_idx == False)
            U1, U0 = U_trans[no_zeros_idx], U_trans[zero_idx]
            B = U0.T @ U0 ## [768, 768]
            B = B / torch.norm(B)

            null_eta: float = {'interm_reader_1': args.null_eta1, 'interm_reader_2': args.null_eta2}[_mname]
            update_proj_dict[_pid][_mname] = null_eta * B.detach() +\
            (1 - null_eta) * torch.eye(B.shape[0], device=B.device, dtype=B.dtype)

            ## eta * B + (1-eta) * I
        
    return update_proj_dict, update_vector


def get_prototypes(GVM: GlobalVarsManager, taskid:int, model: VisionTransformer, 
                mode: str = 'shared', is_return: bool = False, is_cluster: bool = False) -> None:
    
    tt = time.time()
    model = set_model_mode(GVM, taskid, model, training=False)
    torch.cuda.empty_cache()

    dataset = define_dataset(GVM, GVM.cl_mngr.current_task_classes, training=True, transform_type=args.transform_type, target_map_to_local=False, use_eval_transform=True, expand_times=1)
    dataloader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.eval_workers, pin_memory=True, timeout=args.timeout if args.eval_workers > 0 else 0)

    feats = torch.empty([len(dataset), 768], dtype=torch.float32)
    label = torch.empty([len(dataset)], dtype=torch.long)
    intra_label = torch.empty([len(dataset)], dtype=torch.long)

    tt = time.time()
    smp_idx = 0
    assert mode in [None, 'shared', 'adapter']
    for img, lbl, _ in dataloader:
        with torch.no_grad():
            img: Tensor
            lbl: Tensor
            if mode == None:
                _feat = model.module.encode_image(img.cuda(non_blocking=True), mode=None, pre_logits=True)[0].cpu()
            elif mode == 'shared':
                _feat = model.module.encode_image(img.cuda(non_blocking=True), mode='shared', pre_logits=True)[0].cpu()
            elif mode == 'adapter':
                _feat = model.module.encode_image(img.cuda(non_blocking=True), 
                        mode=args.train_tool+'_train', 
                        taskid=taskid, 
                        max_taskid=taskid)[0].cpu()
                
            for _f, _l in zip(_feat, lbl):
                feats[smp_idx] = _f
                label[smp_idx] = _l
                smp_idx += 1
    assert smp_idx == len(dataset)
    torch.cuda.empty_cache()

    _mean_list = []
    _cov_list = []
    _sub_mean_list = []
    _sub_cov_list = []
    # Store prototypes in classifier-output order, not sorted global-label order.
    for _l in GVM.cl_mngr.current_task_classes:
        _output_label = GVM.label_map_g2l[int(_l)][2]
        _cls_feats = feats[label == _output_label]
        _mean = torch.mean(_cls_feats, dim=0, keepdim=False)
        _mean_list.append(_mean)

        _cov = torch.cov(torch.tensor(_cls_feats, dtype=torch.float64).T) + torch.eye(_cls_feats.shape[-1]) / _cls_feats.shape[0]
        _cov_list.append(_cov)

        kmeans = KMeans(n_clusters=1).fit(F.normalize(_cls_feats, dim=1))
        sub_lbl = torch.tensor(kmeans.labels_)
        for _ll in sub_lbl.unique():
            _sub_cls_feats = _cls_feats[sub_lbl == _ll]
            _sub_mean = torch.mean(_sub_cls_feats, dim=0, keepdim=False)
            _sub_mean_list.append(_sub_mean)

            _sub_cov = torch.cov(torch.tensor(_sub_cls_feats, dtype=torch.float64).T) +\
            torch.eye(_sub_cls_feats.shape[-1]) * 1e-4
            _sub_cov_list.append(_sub_cov)

    _mean_list = torch.stack(_mean_list)
    _cov_list = torch.stack(_cov_list)
    _sub_mean_list = torch.stack(_sub_mean_list)
    _sub_cov_list = torch.stack(_sub_cov_list)

    model = set_model_mode(GVM, taskid, model, training=True, training_string=GVM.cache_dict['training_string'])
    print('Finish extract class prototypes, mode: {}, total time: {:.4f}'.format(mode, time.time() - tt))

    if is_return:
        return _mean_list, _sub_mean_list
    
    else:
        _key = 'p_' + str(mode)
        if _key not in GVM.cache_dict:
            GVM.cache_dict[_key] =  {'mean': _mean_list, 'cov': _cov_list, 'sub_mean': _sub_mean_list, 'sub_cov': _sub_cov_list}
        else:
            GVM.cache_dict[_key]['mean'] = torch.cat([GVM.cache_dict[_key]['mean'], _mean_list])
            GVM.cache_dict[_key]['cov'] = torch.cat([GVM.cache_dict[_key]['cov'], _cov_list])
            GVM.cache_dict[_key]['sub_mean'] = torch.cat([GVM.cache_dict[_key]['sub_mean'], _sub_mean_list])
            GVM.cache_dict[_key]['sub_cov'] = torch.cat([GVM.cache_dict[_key]['sub_cov'], _sub_cov_list])

        return None





def extract_class_features(GVM: GlobalVarsManager, taskid:int, model: VisionTransformer, 
                           mode: str = 'shared', return_name: bool=None) -> None:
    model = set_model_mode(GVM, taskid, model, training=False)
    torch.cuda.empty_cache()

    dataset = define_dataset(GVM, GVM.cl_mngr.current_task_classes, training=True, transform_type=args.transform_type, target_map_to_local=False, use_eval_transform=True, expand_times=1)
    dataloader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.eval_workers, pin_memory=True, timeout=args.timeout if args.eval_workers > 0 else 0)

    feats = torch.empty([len(dataset), 768], dtype=torch.float32)
    label = torch.empty([len(dataset)], dtype=torch.long)

    # print('Start extract current features in previous adapter:')
    tt = time.time()
    smp_idx = 0
    for img, lbl, _ in dataloader:
        with torch.no_grad():
            img: Tensor
            lbl: Tensor
            if mode == None:
                _feat = model.module.encode_image(img.cuda(non_blocking=True), mode=None, pre_logits=True)[0].cpu()
            elif mode == 'shared':
                _feat = model.module.encode_image(img.cuda(non_blocking=True), mode='shared', pre_logits=True)[0].cpu()
            elif mode == 'adapter':
                _feat = model.module.encode_image(img.cuda(non_blocking=True), 
                        mode=args.train_tool+'_train', 
                        taskid=taskid, 
                        max_taskid=taskid)[0].cpu()            
            for _f, _l in zip(_feat, lbl):
                feats[smp_idx] = _f
                label[smp_idx] = _l
                smp_idx += 1
    assert smp_idx == len(dataset)
    torch.cuda.empty_cache()
    # print('Finish extract current features in previous adapter: {:.4f}'.format(time.time() - tt))

    if return_name == 'feature':
        return feats, label

    elif return_name == 'mean':
        _mean_list = []
        for _l in label.unique():
            _cls_feats = feats[label == _l]
            _mean = torch.mean(_cls_feats, dim=0, keepdim=False)
            _mean_list.append(_mean)

        return torch.stack(_mean_list)


def refine_head(GVM: GlobalVarsManager, model: VisionTransformer):
    feats_mean: Tensor = GVM.cache_dict['class_features']['mean']
    feats_cov: Tensor = GVM.cache_dict['class_features']['cov']
    feats_class: Tensor = GVM.cache_dict['class_features']['class']
    assert len(feats_class.unique()) == len(GVM.cl_mngr.sofar_task_classes)

    stat_dataset = TensorDataset(feats_mean, feats_cov, feats_class)

    model = modify_head(GVM, model, training=False)
    mhead = model.module.head

    mhead.train()
    mhead.cuda()
    mhead.requires_grad_()

    optimizer = create_optimizer_v2(mhead, opt='sgd', lr=0.001, weight_decay=1e-4, momentum=0.9)
    scheduler, num_epochs = create_scheduler_v2(optimizer, 'multistep', num_epochs=50, decay_milestones=[999,], decay_rate=0.1)
    criterion = nn.CrossEntropyLoss().cuda()
    from torch.distributions.multivariate_normal import MultivariateNormal

    torch.cuda.empty_cache()
    scalar_meter = misc.ScalarMeter(loss="samp_avg:.4f", acc_top1="samp_avg:>6.2%")
    for epoch in range(1, num_epochs + 1):
        scheduler.step(epoch)

        smp_inp = []
        smp_tgt = []
        assert len(stat_dataset) == len(GVM.cl_mngr.sofar_task_classes)
        _ns = 256
        for _cmean, _ccov, _cclass in stat_dataset:
            m = MultivariateNormal(_cmean.float(), _ccov.float())
            _smp = m.sample(sample_shape=(_ns,))
            smp_inp.append(_smp)
            smp_tgt.append(torch.as_tensor([_cclass,] * _ns, dtype=torch.long))
        smp_inp = torch.cat(smp_inp)
        smp_tgt = torch.cat(smp_tgt)

        train_data = TensorDataset(smp_inp, smp_tgt)
        assert len(train_data) == len(stat_dataset) * _ns
        dataloader = DataLoader(train_data, batch_size=256, shuffle=True)

        for inp, tgt, _ in dataloader:
            out: Tensor = mhead(inp.cuda(non_blocking=True))
            if model.logit_type == 'head_out':
                logits = out
            elif model.logit_type == 'sim_imgtext':
                logits = model.forward_logits(out)
            loss: Tensor = criterion(logits, tgt.cuda(non_blocking=True))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc_top1, = misc.calc_accuracy(logits.cpu(), tgt.cpu(), topk=(1, ))
            scalar_meter.add_step_value(len(inp), loss=loss.item(), acc_top1=acc_top1)
        if (epoch % 10 == 0 or epoch == num_epochs):
            print(f":: epoch [{epoch}/{num_epochs}]: {scalar_meter.format_outout(scalar_meter.update_epoch_average_value())}")

    torch.cuda.empty_cache()


if __name__ == "__main__":
    args = get_args()
    seed_etc_options(args.seed)
    logs_path = os.path.join(args.logs_dir, args.logs_suffix+'.txt')
    print('Logger_path:{}'.format(logs_path))
    sys.stdout = Logger(logs_path)
    if 'debug' in args.logs_suffix:
        args.expand_times = 1
        args.epochs = 1

    ## create dataset and split dataset
    GVM = GlobalVarsManager()
    GVM.init_from_args(args)

    print('***** Start training *****')
    GVM.cache_dict['exp_start_time'] = ttime()

    for taskid, current_task_classes in GVM.cl_mngr: ## task_id=0, len(current_task_classes)=10
        print(f"{'#'*30} Task: [{taskid+1}/{GVM.cl_mngr.num_tasks}] {'#'*30}")
        print(f"Current classes ({len(current_task_classes)}): {current_task_classes}")

        if not args.consecutive_training or taskid == 0:
            ## prompt_len, prompt_init, prompt_start_block, prompt_end_block
            _prompt_args_dict = misc.get_specific_args_dict(args, 'prompt_')
            _other_args_dict = misc.get_specific_args_dict(args, 'logit_')
            _head_dim_arg_dict = get_head_dim_arg_dict(GVM, args)

            if args.logit_type == 'sim_imgtext':
                _clip_model, _clip_preprocess = clip.load('ViT-B/16', device='cpu')
                GVM.cache_dict['clip_model'] = _clip_model
                GVM.cache_dict['clip_preprocess'] = _clip_preprocess

            model: VisionTransformer = timm.create_model(args.model, pretrained=True, pretrained_strict=False, 
                                                        **_head_dim_arg_dict,
                                                        prompt_args_dict=_prompt_args_dict, other_args_dict=_other_args_dict)
            if args.dataset == 'tabular':
                model = convert_vit_to_tabular(
                    model,
                    signal_length=args.tabular_window_length,
                    patch_size=args.tabular_patch_size,
                    in_chans=args.tabular_in_chans,
                )
            model = nn.DataParallel(model, device_ids=args.device_ids)
            GVM.cache_dict['pretrained_cfg'] = deepcopy(model.module.pretrained_cfg)

            if args.pretrained_type == 'iBOT':
                checkpoint = torch.load('checkpoint_teacher.pth')
                model.module.load_state_dict(checkpoint['state_dict'], strict=False)
            
        if args.consecutive_training and taskid > 0:
            pass

        _not_pretrained_params = find_not_pretrained_params(model.module, pretrained_cfg=model.module.pretrained_cfg)

        GVM.cache_dict['not_pretrained_params'] = _not_pretrained_params
        GVM.update_label_maps(taskid, current_task_classes) ## taskid:int; current_task_classes:list, len=10, [labels]
        GVM.cache_dict['training_string'] = args.training_string ## 'prompt', 'head'
        misc.check_param_training(GVM.cache_dict['not_pretrained_params'], GVM.cache_dict['training_string'])

        if args.evaluation == False:
            model = train_one_task(GVM, taskid, current_task_classes, model)
            evaluate_tasks_sofar(GVM, taskid, model)
            task_ending_info(GVM)

        else:
            if taskid == args.num_tasks-1:

                evaluate_tasks_sofar(GVM, taskid, model, 
                pretrained_model_path=osp.join('specific-shared', 'logs', args.dataset, args.evaluation_model_name+'.pkl'))
                task_ending_info(GVM)

        if args.save_model:
            print(' ----- Save model ----- ') 
            save_model_path = osp.join('specific-shared', 'logs', args.dataset, args.save_model_name+'.pkl')
            torch.save(model.state_dict(), save_model_path)
