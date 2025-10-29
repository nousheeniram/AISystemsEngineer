.PHONY: venv install train serve mlflow test clean

venv:
	python3 -m venv .venv

install:
	. .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

train:
	. .venv/bin/activate && \
	export MLFLOW_TRACKING_URI="file:./mlruns" && \
	export MLFLOW_REGISTRY_URI="file:./mlruns" && \
	python train_pipeline.py

serve:
	. .venv/bin/activate && python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000

mlflow:
	. .venv/bin/activate && mlflow ui --host 0.0.0.0 --port 5001

test:
	. .venv/bin/activate && pytest -q

clean:
	rm -rf .pytest_cache __pycache__ */__pycache__ mlruns
