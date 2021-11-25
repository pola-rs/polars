SHELL=/bin/bash

pre-commit:
	isort .
	black .
	mypy
	flake8 .
	make -C .. fmt_toml

pip:
	pip install --force-reinstall -U wheels/polars-*.whl

release-build:
	maturin build --release -o wheels

release-install: release-build pip

test:
	maturin develop
	pytest tests

test-with-cov:
	maturin develop
	cd tests && pytest --cov=polars --import-mode=importlib

test-build:
	maturin build -o wheels

test-install: test-build pip

build-and-test: test-install
	pytest tests

build-and-test-with-cov: test-install
	cd tests && pytest --cov=polars --import-mode=importlib
