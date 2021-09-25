SHELL=/bin/bash

pre-commit:
	isort .
	black .
	mypy
	flake8 .

release-build:
	maturin build --release -o wheels -i $(which python)

release-install: release-build
	@cd wheels
	pip install --force-reinstall -U polars-*.whl

test:
	maturin develop
	pytest tests

test-build:
	maturin build -o wheels -i $(which python)
