# Automation commands for Telco Customer Churn Prediction System

.PHONY: help setup train evaluate test run clean

PYTHON = python

help:
	@echo Telco Churn Prediction System Automation:
	@echo   make setup     - Install required dependencies
	@echo   make train     - Execute SMOTE-balanced 5-fold cross-validation and hyperparameter tuning
	@echo   make evaluate  - Generate test-set evaluation metrics, ROC curves, and financial ROI
	@echo   make test      - Execute pytest unit and integration test suite
	@echo   make run       - Launch the interactive Streamlit dashboard
	@echo   make clean     - Remove temporary caches and python artifacts

setup:
	$(PYTHON) -m pip install -r requirements.txt

train:
	$(PYTHON) -m src.train

evaluate:
	$(PYTHON) -m src.evaluate

test:
	$(PYTHON) -m pytest tests/test_pipeline.py -v

run:
	$(PYTHON) -m streamlit run app.py

clean:
	python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('.pytest_cache')]"
