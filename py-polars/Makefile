.DEFAULT_GOAL := help

SHELL=/bin/bash
PYTHON=venv/bin/python
PYTHON_BIN=venv/bin

.PHONY: pre-commit test pip

venv:
	@python -m venv venv
	@venv/bin/pip install -U pip
	@venv/bin/pip install -r build.requirements.txt
	@unset CONDA_PREFIX && source venv/bin/activate && maturin develop

.PHONY: clean
clean:
	@rm -rf venv/
	@rm -rf docs/build/
	@rm -rf .hypothesis/
	@rm -rf .mypy_cache/
	@rm -rf .pytest_cache/
	@rm -f .coverage
	@rm -f coverage.xml
	@find -type f -name '*.py[co]' -delete -or -type d -name __pycache__ -delete
	@cargo clean

pre-commit: venv
	$(PYTHON_BIN)/isort .
	$(PYTHON_BIN)/black .
	$(PYTHON_BIN)/blackdoc .
	$(PYTHON_BIN)/pyupgrade --py37-plus `find polars/ tests/ -name "*.py" -type f`
	$(PYTHON_BIN)/mypy
	$(PYTHON_BIN)/flake8
	make -C .. fmt_toml
	cargo fmt --all

test: venv
	$(PYTHON_BIN)/maturin develop
	$(PYTHON) -m pytest tests/unit/

test-all: venv
	$(PYTHON_BIN)/maturin develop
	$(PYTHON) -m pytest
	$(PYTHON) tests/docs/run_doc_examples.py

test-with-cov: venv
	$(PYTHON) -m pytest --cov=polars --cov-report xml

doctest:
	$(PYTHON) tests/docs/run_doc_examples.py

install-wheel:
	pip install --force-reinstall -U wheels/polars-*.whl

build-no-venv:
	maturin build -o wheels

build-no-venv-release:
	maturin build -o wheels --release

build-and-test-no-venv:
	maturin build -o wheels
	pip install --force-reinstall -U wheels/polars-*.whl
	pytest tests

install-no-venv: build-no-venv install-wheel
install-no-venv-release: build-no-venv-release install-wheel

.PHONY: help
help:  ## Display this help screen
	@echo -e '\033[1mAvailable commands:\033[0m'
	@grep -E '^[a-z.A-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
