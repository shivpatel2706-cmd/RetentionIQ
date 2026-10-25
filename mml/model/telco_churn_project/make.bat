@echo off
set PYTHON=python

if "%1"=="" goto help
if "%1"=="help" goto help
if "%1"=="setup" goto setup
if "%1"=="train" goto train
if "%1"=="evaluate" goto evaluate
if "%1"=="test" goto test
if "%1"=="run" goto run
if "%1"=="clean" goto clean

echo Unknown command: %1
goto help

:help
echo Telco Churn Automation commands:
echo   make setup     - Install dependencies
echo   make train     - Run training and tuning
echo   make evaluate  - Run model evaluation and generate plots
echo   make test      - Run pytest test suite
echo   make run       - Launch Streamlit dashboard
echo   make clean     - Clean temporary cache files
goto end

:setup
%PYTHON% -m pip install -r requirements.txt
goto end

:train
%PYTHON% -m src.train
goto end

:evaluate
%PYTHON% -m src.evaluate
goto end

:test
%PYTHON% -m pytest tests/test_pipeline.py -v
goto end

:run
%PYTHON% -m streamlit run app.py
goto end

:clean
%PYTHON% -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('.pytest_cache')]"
goto end

:end
