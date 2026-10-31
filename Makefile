# Local developer loop. Everything here also runs unchanged inside a Databricks job via the notebooks + bundle.
PY ?= python3
export MLFLOW_DISABLE_AGENT_HINT = 1
export DG_DATA_ROOT ?= ./data
export DG_RAW_PATH ?= ./data/raw

.PHONY: install lint test fixture demo clean wheel

install:
	$(PY) -m pip install -e ".[local,dev]"

lint:
	ruff check . && ruff format --check .

test:
	pytest -q

# synthetic fleet in the Backblaze file layout (tests / dry runs only - never for reported results)
fixture:
	$(PY) scripts/make_fixture.py --out $(DG_RAW_PATH) --drives 3000 --days 400

# whole pipeline on whatever is in DG_RAW_PATH (fixture or a real subset from scripts/download_backblaze.py)
demo:
	driveguard bronze && driveguard silver && driveguard gold && driveguard train && driveguard backfill --days 45 && driveguard monitor

wheel:
	$(PY) -m build --wheel

clean:
	rm -rf data/tables data/mlflow.db data/mlartifacts dist build
