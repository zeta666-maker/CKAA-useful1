$ErrorActionPreference = "Stop"

$Root = Split-Path $PSScriptRoot -Parent
$RawRoot = Split-Path $Root -Parent
$DataRoot = Join-Path $Root "A_CLData\fault_csv"

python (Join-Path $Root "tools\prepare_tabular_dataset.py") `
    --raw-root $RawRoot `
    --output-root $DataRoot

$cases = @(
    @{ Name = "fp32"; Aqcl = "false"; Mode = "fixed"; Bits = "32"; Low = "32"; High = "32" },
    @{ Name = "fixed8"; Aqcl = "true"; Mode = "fixed"; Bits = "8"; Low = "8"; High = "8" },
    @{ Name = "fixed4"; Aqcl = "true"; Mode = "fixed"; Bits = "4"; Low = "4"; High = "4" },
    @{ Name = "fixed2"; Aqcl = "true"; Mode = "fixed"; Bits = "2"; Low = "2"; High = "2" },
    @{ Name = "rpq_4_8"; Aqcl = "true"; Mode = "rpq"; Bits = "4"; Low = "4"; High = "8" },
    @{ Name = "rpq_saou_4_8"; Aqcl = "true"; Mode = "rpq_saou"; Bits = "4"; Low = "4"; High = "8" }
)

foreach ($case in $cases) {
    $suffix = "aqcl_compare_$($case.Name)"
    $arguments = @(
        (Join-Path $Root "train_aqcl.py"),
        "-d", "fault_csv",
        "-t", "3",
        "-m", "vit_base_patch16_224.augreg_in21k",
        "-b", "32",
        "-e", "1",
        "-jt", "0",
        "-je", "0",
        "-et", "1",
        "--eval_batch_size", "64",
        "--temperature", "28.0",
        "--lr", "0.005",
        "--lr_scale", "0.2",
        "--lr_scale_patterns", "patch_embed",
        "--null_eta1", "0.999",
        "--null_eta2", "0.999",
        "-tc", "2.0",
        "--eval-tool", "adapter",
        "--eval-trained-task-router", "true",
        "--eval-local-task-head", "true",
        "--eval-task-weight", "false",
        "--freeze-shared-prompts-after-first", "true",
        "--prototype-confidence-weight", "0.0",
        "--seed", "2024",
        "--save-model", "false",
        "--logs-dir", "logs",
        "-sf", $suffix,
        "--aqcl-enable", $case.Aqcl,
        "--aqcl-mode", $case.Mode,
        "--aqcl-bits", $case.Bits,
        "--aqcl-low-bits", $case.Low,
        "--aqcl-high-bits", $case.High,
        "--aqcl-fisher-batches", "1"
    )
    python @arguments
}
