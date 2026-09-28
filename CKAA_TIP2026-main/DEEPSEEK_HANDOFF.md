---
handoff_version: 1
repository_root: "C:/Users/Administrator/Desktop/CKAA_TIP2026-main"
code_root: "C:/Users/Administrator/Desktop/CKAA_TIP2026-main/CKAA_TIP2026-main"
current_git_head: "0d33537"
task_type: "tabular fault diagnosis, continual learning, quantization-aware continual learning"
current_status: "implemented and partially validated; not yet paper-equivalent"
confidence: "medium-low"
mandatory_warning: "The current design and implementation are not guaranteed correct. Do not trust the current metrics or hypotheses without reproduction."
---

# DeepSeek Harness Handoff

## 1. Critical Warning

The current implementation is a work in progress. The following claims are
NOT established:

- RPQ is not proven to allocate a healthy mixture of 4-bit and 8-bit layers.
  In the latest run, the deployment report showed `weight_bits=[8]`, which
  suggests the allocator may have degenerated to all-high precision.
- SAOU is implemented and executes, but the quality of its covariance,
  null-space threshold, and gradient projection has not been independently
  validated.
- The current efficiency report does not count explicit attention `QK^T` and
  `attention @ V` matrix multiplications. It currently counts Linear/Conv1d
  MACs only, so GFLOPs/GBOPS are undercounted.
- The 4-bit and 2-bit results are not yet satisfactory.
- The comparison is on a custom tabular fault dataset, not the paper's
  CIFAR-100/TinyImageNet benchmarks. Absolute numbers are not comparable.

Any harness agent should treat the current design as a hypothesis, not as a
ground-truth implementation.

## 2. Final Objective

Keep CKAA and AQCL physically separated, and improve AQCL low-bit continual
learning on the CSV fault dataset until the trend is consistent with the AQCL
paper:

```text
FP32 > 8-bit >= AQCL 4-bit > AQCL 2-bit > Vanilla fixed low-bit baselines
```

The paper expects AQCL 4-bit to be close to 8-bit and substantially better
than fixed 4-bit. AQCL 2-bit should still beat the corresponding vanilla
low-bit baseline, although it will be lower than 4-bit.

Paper reference values (must not be copied as our values):

```text
CIFAR-100 T=10:
  32-bit: ACC 64.12, BWT -5.71
   8-bit: ACC 63.96, BWT -5.39
   4-bit: ACC 63.84, BWT -5.26
   2-bit: ACC 51.19, BWT  -5.88

Efficiency, ResNet-20:
  32-bit: model 1.156 MB, GBOPS 42.991
   8-bit: model 0.330 MB, GBOPS  2.687
   4-bit: model 0.193 MB, GBOPS  0.672
   2-bit: model 0.124 MB, GBOPS  0.168
  inference extra memory: 0 MB for all AQCL variants
```

## 3. Hard Constraints

1. Do not modify the CKAA behavior in `train_eval.py` for the paper-1 path.
2. AQCL changes must stay in `train_aqcl.py` and `utils/aqcl.py`.
3. Use only the current CSV-derived `fault_csv` dataset.
4. Use 5 incremental tasks with class split `2/2/2/2/1`.
5. Keep `DataLoader(num_workers=0)`.
6. Use a single CUDA device.
7. Do not introduce `pytorch-quantization`, Brevitas, or another quantization
   framework. Quantization must remain hand-written PyTorch.
8. Do not save Fisher, covariance, SVD subspaces, or other training-only
   state into the deployment checkpoint.
9. Do not alter raw CSV files. Rebuild processed data with the converter.
10. Do not claim paper-equivalent results from a short smoke test.

## 4. Dataset Contract

Raw data:

```text
data_1_S1/*.csv
  one vibration-like signal column

data_2_S1/*.csv
  first column repeats the same signal stream
  columns 1:4 are pressure X/Y/Z
```

Processed data:

```text
A_CLData/fault_csv/
  train_windows.npy
  train_labels.npy
  eval_windows.npy
  eval_labels.npy
  train_index.csv
  eval_index.csv
  meta.json
```

Current window contract:

```text
window_length = 1568
stride = 1568
channels = [vibration, pressure_x, pressure_y, pressure_z]
window tensor shape = [4, 1568]
```

The tabular tokenizer flattens/time-interleaves the 4 channels and creates
196 1D patches plus one class token, matching CKAA's 197-token implementation.

Rebuild command:

```powershell
python .\tools\prepare_tabular_dataset.py `
  --raw-root .. `
  --output-root .\A_CLData\fault_csv `
  --overwrite
```

## 5. Code Ownership

| File | Responsibility | Rule |
| --- | --- | --- |
| `train_eval.py` | CKAA paper-1 training/evaluation | Must remain AQCL-free |
| `train_aqcl.py` | AQCL paper-2 entry point | AQCL-specific orchestration only |
| `utils/aqcl.py` | Quantization, RPQ, SAOU, deployment report | Paper-2 implementation |
| `utils/dataset_builder.py` | Dataset routing and tabular adapter | Keep CKAA-compatible |
| `utils/tabular_data.py` | CSV-to-window conversion and tabular dataset | Must handle empty CSV cells as raw zero |
| `tools/prepare_tabular_dataset.py` | Dataset build CLI | Writes `A_CLData/fault_csv` |
| `tools/run_aqcl_comparison.ps1` | 5-task comparison matrix | Use `train_aqcl.py` |
| `CKAA_AUDIT.md` | CKAA implementation audit | Paper-1 record |
| `AQCL.md` | AQCL implementation/results | Paper-2 record |

## 6. Current CKAA Baseline

CKAA paper-1 path has already been modified for tabular data. The prior
validated configuration uses:

```text
5 tasks
class split 2/2/2/2/1
12 epochs per task
ViT-Base patch16 image backbone, tabular 1D tokenizer replacement
CKAA shared prompts, adapters, FA, CA, TC-MoA, null-space prompt updates
```

Recorded CKAA result:

```text
Last-acc: 95.99%
Avg-acc:  97.92%
Forgetting: 2.37%
```

This is the paper-1 reference. Do not use AQCL changes to improve or regress
this path.

## 7. Current AQCL Implementation

Implemented components:

| Paper component | Code | Current implementation |
| --- | --- | --- |
| Asymmetric uniform quantization Eq.1 | `UniformQuantizer` | Per-tensor activations; per-output-channel weights |
| Weight QAT | `QATLinear`, `QATConv1d` | Straight-through estimator |
| Fisher/Hessian trace Eq.11-12 | `AQCLContext._fisher_and_covariance` | Diagonal empirical gradient square |
| Bit allocation Eq.13 | `AQCLContext._allocate_bits` | Compare sensitivity with mean; high/low bits |
| SAOU covariance Eq.21-22 | `AQCLContext._fisher_and_covariance` | Accumulated input covariance |
| SVD/EIGH Eq.20 | `AQCLContext._fisher_and_covariance` | Symmetric eigendecomposition with jitter |
| SAOU Eq.23-24 | `AQCLContext.modulate_gradients` | Null component plus transition-subspace mask |
| Deployment report | `AQCLContext.inference_report` | Size, GFLOPs, GBOPS, inference extra memory |

Current training additions:

- `--aqcl-warmup-epochs 1`: first epoch trains in FP32, then target bits are
  enabled.
- Per-channel weight quantization to reduce adapter/head quantization error.
- AQCL entry forces:
  - `freeze_shared_prompts_after_first=True`
  - `eval_local_task_head_task_only=True`

## 8. Current Results On fault_csv

Protocol:

```text
5 tasks
class split 2/2/2/2/1
2 epochs per task
1 warm-up epoch
batch size 32
seed 2024
```

| Case | Last-acc | Avg-acc | Forgetting | Deployment bits observed |
| --- | ---: | ---: | ---: | --- |
| FP32 | 92.69% | 95.56% | 0.87% | 32 |
| Fixed 8-bit | 86.71% | 86.41% | 9.57% | 8 |
| RPQ + SAOU, target 4/8-bit | 86.78% | 92.90% | 6.80% | `[8]` only |
| Fixed 4-bit | 21.10% | 23.57% | 12.26% | 4 |

Observed interpretation:

- AQCL matches fixed 8-bit and is far better than fixed 4-bit.
- The reported RPQ allocation is degenerate: it currently chooses 8-bit
  everywhere. This means RPQ compression/efficiency is not yet reproduced.
- Fixed 4-bit and fixed 2-bit are still far below the paper-like region.

The latest 2-bit smoke test, under an older 3-task protocol, was:

```text
Fixed 2-bit: Last-acc 11.11%, Avg-acc 20.37%
```

This must be rerun under the final 5-task protocol.

## 9. Known Defects And Uncertainty

Priority order:

1. RPQ may be degenerate.
   - Check every layer sensitivity before bit allocation.
   - If all sensitivities are equal or zero, the Fisher pass or gradient
     capture is wrong.
   - Print minimum, maximum, mean, and bit counts for every task.

2. Activation quantization may be too aggressive.
   - Current activation quantization is per-tensor min/max.
   - Consider EMA clipping thresholds or learned scales, but do not silently
     replace the paper formulation.

3. SAOU quality is unverified.
   - Log `Uo` dimension, `Up` dimension, eigenvalue range, gradient cosine
     before/after modulation.
   - Verify that SAOU improves RPQ-only under identical settings.

4. Efficiency report is incomplete.
   - `profile_macs` counts Linear/Conv1d MACs only.
   - Add explicit attention `QK^T` and `attention @ V` counts.
   - Verify the paper's GBOPS relation against Table VII.

5. Evaluation routing is a confounder.
   - Current AQCL uses trained task routing plus the selected task head.
   - Compare against the pure CKAA TC-MoA path before drawing conclusions.
   - Do not use oracle task IDs to make low-bit results look better.

6. Training schedule is too short for paper-level accuracy.
   - Current final compare is 2 epochs/task.
   - Paper uses 80 epochs/task.

7. Backbone and data differ from the paper.
   - Paper: ResNet-20 and CIFAR-100/TinyImageNet.
   - Current: tabular ViT and CSV fault data.
   - Trend similarity is required; absolute values are not.

## 10. Required Next Work

### Phase 1: Instrumentation

Add logging for each task:

```text
layer name
weight sensitivity
activation sensitivity
assigned weight bits
assigned activation bits
Uo dimension
Up dimension
minimum/maximum eigenvalue
SAOU applied or skipped count
```

Do not change training behavior in this phase.

### Phase 2: Repair RPQ

- Verify that gradients are non-zero and layer differences exist.
- If all sensitivities collapse, fix the Fisher collection path.
- Ensure a genuine 4/8-bit mixture is produced.
- Accept only if accuracy remains near 8-bit while model size decreases.

### Phase 3: Improve 4-bit and 2-bit Stability

Candidate experiments, in order:

1. More warm-up epochs.
2. More task epochs.
3. EMA or percentile clipping for activations.
4. Re-check per-channel weight quantization.
5. Verify SAOU threshold `lambda`, `theta`, and `alpha`.
6. Compare:
   - Fixed 8-bit
   - Fixed 4-bit
   - Fixed 2-bit
   - RPQ-only
   - RPQ + SAOU

### Phase 4: Reproduce Paper Trend

Required trend:

```text
32-bit >= 8-bit >= AQCL 4-bit > AQCL 2-bit
AQCL 4-bit >> fixed 4-bit
AQCL 2-bit >> fixed 2-bit
```

If this trend does not hold, report the failure and the evidence. Do not
rewrite the metric or hide the failure.

## 11. Acceptance Criteria

For the CSV fault dataset:

- CKAA path remains unchanged and AQCL remains a separate entry point.
- No training-only tensors appear in deployment checkpoints.
- Bit allocation is genuinely mixed when RPQ is enabled.
- Lower fixed bit width has lower accuracy.
- AQCL 4-bit is close to 8-bit and clearly above fixed 4-bit.
- AQCL 2-bit beats fixed 2-bit if both are run.
- Model size and GBOPS decrease monotonically with lower fixed bits.
- All reported results are backed by logs in `logs/`.

## 12. Reproduction Commands

Convert CSV:

```powershell
python .\tools\prepare_tabular_dataset.py `
  --raw-root .. `
  --output-root .\A_CLData\fault_csv `
  --overwrite
```

Run CKAA only:

```powershell
python .\train_eval.py -d fault_csv -t 5 -m vit_base_patch16_224.augreg_in21k -b 32 -e 12 -jt 0 -je 0 -et 1 --eval_batch_size 64 --temperature 28.0 --lr 0.005 --lr_scale 0.2 --lr_scale_patterns patch_embed --null_eta1 0.999 --null_eta2 0.999 -tc 2.0 --eval-tool adapter --eval-trained-task-router true --eval-local-task-head true --eval-task-weight false --freeze-shared-prompts-after-first true --prototype-confidence-weight 0.0 --seed 2024 --save-model false --logs-dir logs -sf ckaa_tabular_baseline
```

Run AQCL RPQ + SAOU:

```powershell
python .\train_aqcl.py -d fault_csv -t 5 -m vit_base_patch16_224.augreg_in21k -b 32 -e 2 -jt 0 -je 0 -et 1 --eval_batch_size 64 --temperature 28.0 --lr 0.005 --lr_scale 0.2 --lr_scale_patterns patch_embed --null_eta1 0.999 --null_eta2 0.999 -tc 2.0 --eval-tool adapter --eval-trained-task-router true --eval-local-task-head true --eval-task-weight false --prototype-confidence-weight 0.0 --seed 2024 --save-model false --logs-dir logs -sf aqcl_5s_e2_taskhead --aqcl-mode rpq_saou --aqcl-low-bits 4 --aqcl-high-bits 8 --aqcl-fisher-batches 2 --aqcl-warmup-epochs 1
```

Run the comparison matrix:

```powershell
.\tools\run_aqcl_comparison.ps1
```

## 13. Evidence Locations

Use these logs before trusting any summary:

```text
logs/ckaa_tabular_baseline.txt
logs/aqcl_5s_e2_taskhead.txt
logs/fixed8_5s_e2.txt
logs/fixed4_5s_e2.txt
logs/aqcl_compare_rpq_4_8.txt
logs/aqcl_compare_rpq_saou_4_8_v2.txt
```

## 14. Handoff Statement

The current work establishes that:

1. CKAA and AQCL can be separated cleanly.
2. Hand-written QAT, RPQ, and SAOU can run on the CSV fault dataset.
3. AQCL currently matches 8-bit accuracy and beats fixed 4-bit in the
   two-epoch 5-task setting.

The current work does NOT yet establish that:

1. The RPQ allocator is healthy.
2. The 2-bit and 4-bit deployment goals are met.
3. The efficiency metrics match the paper.
4. The current design is the best or even the correct interpretation of
   AQCL for this tabular dataset.

The next agent must reproduce, instrument, and falsify the current
hypotheses before extending the method.
