$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Split-Path -Parent $ScriptDir
$Python = Join-Path $Root ".venv\Scripts\python.exe"

function Run-Experiment {
    param(
        [string]$Name,
        [string]$Config,
        [string]$Mode,
        [string]$OutputDir
    )

    $OutPath = Join-Path $Root $OutputDir
    New-Item -ItemType Directory -Force -Path $OutPath | Out-Null

    $Stdout = Join-Path $OutPath "train_stdout.log"
    $Stderr = Join-Path $OutPath "train_stderr.log"
    $Status = Join-Path $OutPath "train_status.json"

    $start = Get-Date
    @{ name = $Name; config = $Config; mode = $Mode; status = "running"; started_at = $start.ToString("o") } |
        ConvertTo-Json | Set-Content -Encoding UTF8 $Status

    $args = @("grpo_spad_trl.py", "--config", $Config, "--mode", $Mode)
    $proc = Start-Process -FilePath $Python -ArgumentList $args -WorkingDirectory $Root `
        -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -WindowStyle Hidden -Wait -PassThru

    $end = Get-Date
    @{
        name = $Name
        config = $Config
        mode = $Mode
        status = $(if ($proc.ExitCode -eq 0) { "completed" } else { "failed" })
        exit_code = $proc.ExitCode
        started_at = $start.ToString("o")
        ended_at = $end.ToString("o")
        runtime_seconds = [math]::Round(($end - $start).TotalSeconds, 1)
    } | ConvertTo-Json | Set-Content -Encoding UTF8 $Status
}

Run-Experiment -Name "spad_answer_first" -Config "configs/active/spad_answer_first.yaml" -Mode "spad" -OutputDir "results/main/spad_answer_first_trl"
Run-Experiment -Name "c_spad_answer_first" -Config "configs/active/c_spad_answer_first.yaml" -Mode "c_spad" -OutputDir "results/main/c_spad_answer_first_trl"
