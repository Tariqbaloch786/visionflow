.PHONY: install dev test lint fmt type docker run app clean

install:
	pip install .

dev:
	pip install -e ".[app,dev]"

test:
	pytest

lint:
	ruff check src tests

fmt:
	ruff check --fix src tests
	ruff format src tests

type:
	mypy src

docker:
	docker build -t visionflow:latest .

run:
	visionflow run --config examples/sample_config.yaml

app:
	streamlit run src/visionflow/app.py

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache htmlcov coverage.xml
