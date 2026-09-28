# CKAA Tabular Implementation Audit

This repository uses the tabular adaptation requested for the vibration and
pressure-XYZ fault dataset. The original paper image benchmark is not the
target dataset here.

## Method Mapping

| CKAA mechanism | Code | Status |
| --- | --- | --- |
| Task-shared prompts `Psh` in every ViT block | `TaskSharedPrompt`, `Block.forward` mode `shared` | Implemented |
| Orthogonal prompt updates | `ModAdam`, `get_interm_tensor_dict`, `get_update_projection_dict` | Implemented |
| Task-specific adapter `At_sp` | `Adapter`, `Block.task_adapter` | Implemented |
| Adapter width 64 and prompt length 4 | `Adapter.middle_dim=64`, default `--prompt_len 4` | Implemented |
| Eq.3 shared/specific cross entropy | `train_one_epoch`, `sh_ce_loss`, `sp_ce_loss` | Implemented with configurable training temperature |
| Eq.4 CSFA loss | `ContrastiveLoss`, `fa_loss` | Implemented |
| Eq.5 affinity graph | `knn_graph` | Implemented |
| Eq.6 subspace-shift feature simulation | `G.detach().mm(delta_feats)` | Implemented |
| Eq.7 task-adaptive classifier loss | `model.module.task_head[taskid]`, `ca_loss` | Implemented |
| Eq.8 unified classifier aggregation | `evaluate_tasks_sofar`, `full_head` | Implemented |
| Eq.9 total objective | `sh_ce_loss + sp_ce_loss + ca_loss + fa_loss + null_space_loss` | Implemented |
| Eq.10-11 task confidence | `p_logits`, Top-Kc mask, group probability sum | Implemented |
| Eq.12 TC-MoA adapter mixture | `Block.forward` mode `adapter_eval`, weighted adapter sum | Implemented |
| Eq.13 classifier fusion | `classifier_aggregation_type=mean`, dynamic `eta=1/(tc+2)` | Implemented |
| Null-space prompt projection | `ModAdam` projection with `null_eta1/null_eta2` | Implemented |
| Image-only patch embedding | Replaced by `TabularPatchEmbed` for `[vibration, pressure-XYZ]` | Adapted, as requested |
| `fault_csv` data entry | `A_CLData/fault_csv`, dataset key `fault_csv` | Added |

## Deviations And Fixes

| Deviation | Impact | Resolution |
| --- | --- | --- |
| `get_param_id_dict` used a tuple with `in` instead of checking each pattern | Null-space projection could fail | Fixed with `any(pattern in name for pattern in patterns)` |
| Prototypes were stored in sorted global-label order | CA label ordering could be wrong for shuffled task classes | Fixed to classifier-output order |
| Eq.11 inference divided by training temperature times `tau` | Over-smoothed task confidence | Changed to division by `tau` only |
| Uneven task split `2/2/2/2/1` could create a wrong evaluation head size | Checkpoint evaluation failed | Evaluation head now uses the number of seen classes |
| Tabular tokenizer is not the paper's 2D image patch embed | Necessary for tabular fault data | Tabular 1D tokenizer produces 196 patches plus class token |
| Optional task router/local-head routing exists for the fault dataset | It is an extension, not original TC-MoA | Disabled by default; paper path remains available with `--eval-trained-task-router false --eval-local-task-head false` |

## First-Step Validation

The tabular CKAA baseline was run on `fault_csv` with 5 incremental tasks,
two classes per task except the final one-class task, and 12 epochs per task.

```text
Task 1: Last-acc 100.00%, Avg-acc 100.00%, Forgetting 0.00%
Task 2: Last-acc  99.68%, Avg-acc  99.84%, Forgetting 0.63%
Task 3: Last-acc  98.10%, Avg-acc  99.26%, Forgetting 0.16%
Task 4: Last-acc  95.81%, Avg-acc  98.40%, Forgetting 2.74%
Task 5: Last-acc  95.99%, Avg-acc  97.92%, Forgetting 2.37%
```

Final validation:

```text
Last-acc: 95.99%
Avg-acc: 97.92%
Forgetting: 2.37%
```

The result satisfies the requested tabular adaptation target of average
accuracy near or above 90% and forgetting below 10%.
