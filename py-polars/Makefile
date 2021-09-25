SHELL=/bin/bash

pre-commit:
	isort .
	black .
	mypy
	flake8 .

pip:
	@cd wheels
	pip install --force-reinstall -U polars-*.whl

release-build:
	maturin build --release -o wheels -i $(which python)

release-install: release-build pip

test:
	maturin develop
	pytest tests

test-build:
	maturin build -o wheels -i $(which python)

build-and-test: test-build test
