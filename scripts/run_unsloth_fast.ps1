param(
    [ValidateSet("baseline", "spad", "c_spad")]
    [string]$Mode = "baseline",

    [ValidateSet("qwen3_17b", "qwen3_5_2b")]
    [string]$Model = "qwen3_17b",

    [string]$Python = ".\.conda_unsloth\python.exe",

    [ValidateSet("config", "bf16", "fp16")]
    [string]$Precision = "config",

    [int]$GenerationBatchSize = 0
)

$ErrorActionPreference = "Stop"

$configs = @{
    qwen3_17b = @{
        baseline = "configs/formal/cot_baseline_rl_only_qwen3_17b_unsloth_fast.yaml"
        spad = "configs/formal/spad_rl_only_qwen3_17b_unsloth_fast.yaml"
        c_spad = "configs/formal/c_spad_rl_only_qwen3_17b_unsloth_fast.yaml"
    }
    qwen3_5_2b = @{
        baseline = "configs/formal/cot_baseline_rl_only_unsloth_fast.yaml"
        spad = "configs/formal/spad_rl_only_unsloth_fast.yaml"
        c_spad = "configs/formal/c_spad_rl_only_unsloth_fast.yaml"
    }
}

$argsList = @(
    "grpo_spad_unsloth.py",
    "--config", $configs[$Model][$Mode],
    "--mode", $Mode
)

if ($Precision -eq "bf16") {
    $argsList += "--bf16"
} elseif ($Precision -eq "fp16") {
    $argsList += "--fp16"
}

if ($GenerationBatchSize -gt 0) {
    $argsList += @("--generation-batch-size", [string]$GenerationBatchSize)
}

& $Python @argsList
