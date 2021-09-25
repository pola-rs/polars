SHELL=/bin/bash

pre-commit:
	isort .
	black .
	mypy
	flake8 .

pip:
	pip install --force-reinstall -U wheels/polars-*.whl

release-build:
	maturin build --release -o wheels -i $(which python)

release-install: release-build pip

test:
	maturin develop
	pytest tests

test-build:
	maturin build -o wheels -i $(which python)

test-install: test-build pip

build-and-test: test-install test
