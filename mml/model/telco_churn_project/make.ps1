param (
    [string]$Task = "help"
)

$Python = "python"

switch ($Task.ToLower()) {
    "setup" {
        & $Python -m pip install -r requirements.txt
    }
    "train" {
        & $Python -m src.train
    }
    "evaluate" {
        & $Python -m src.evaluate
    }
    "test" {
        & $Python -m pytest tests/test_pipeline.py -v
    }
    "run" {
        & $Python -m streamlit run app.py
    }
    "clean" {
        & $Python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('.pytest_cache')]"
    }
    default {
        Write-Host "Telco Churn Prediction System Automation (PowerShell):" -ForegroundColor Cyan
        Write-Host "  .\make.ps1 setup     - Install dependencies"
        Write-Host "  .\make.ps1 train     - Run training and tuning"
        Write-Host "  .\make.ps1 evaluate  - Run model evaluation and generate plots"
        Write-Host "  .\make.ps1 test      - Run pytest test suite"
        Write-Host "  .\make.ps1 run       - Launch Streamlit dashboard"
        Write-Host "  .\make.ps1 clean     - Clean cache files"
    }
}
