$ErrorActionPreference = "Stop"

if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    . ".\.venv\Scripts\Activate.ps1"
}

.\scripts\run_ingest.ps1
.\scripts\run_augment.ps1
.\scripts\run_tts.ps1
.\scripts\run_asr.ps1
.\scripts\run_eval.ps1
