.DEFAULT_GOAL := telp

SHELL=/bin/bash
VENV ?= venv

WHEELS_DEBUG   = wheels/debug
WHEELS_RELEASE = wheels/release

ifeq ($(OS),Windows_NT)
    VENV_BIN=$(VENV)/Scripts
else
    VENV_BIN=$(VENV)/bin
endif

ifneq ($(CONDA_EXE),)
	CONDA_DEACT = . $(shell $(CONDA_EXE) info --base -q)/etc/profile.d/conda.sh; \
		while [ -n "$$CONDA_PREFIX" ]; do conda deactivate; done;
else
	CONDA_DEACT =
endif

ifeq ($(VENV),) # no virtual environment
	VENV_BIN     =
	VENV_ACT_BIN =
	CLEAN_VENV   = @:
venv:
	:
else # use VENV
	VENV_BIN    := $(abspath $(VENV_BIN))/# no space after trailing '/'
	VENV_ACT_BIN = $(VENV_BIN)# no space after closing ')'
	CLEAN_VENV   = -rm -rf $(VENV)
ifeq ($(VENV),venv)
venv:  ## Create virtual environment for building polars (optionally with VENV=<name>)
else
.PHONY: venv
venv: $(VENV)
$(VENV):
endif
	python -m venv  $(VENV)
	$(VENV_ACT_BIN)pip install -U pip
	$(VENV_ACT_BIN)pip install -r requirements-dev.txt
	$(VENV_ACT_BIN)pip install -r requirements-lint.txt
endif

develop: $(VENV)  ## Build by running `maturin develop`
	$(VENV_ACT_BIN)maturin develop

develop-release: $(VENV)  ## Build by running `maturin develop --release`
	$(VENV_ACT_BIN)maturin develop --release

.PHONY: clean cleaner cleanest
clean:  ## Just run `cargo clean`
	-cargo clean

cleaner: clean  ## clean + remove VENV and other auto-generated dirs & files
	$(CLEAN_VENV)
	-rm -rf target
	-rm -rf docs/build
	-rm -rf docs/source/reference/api
	-rm -rf .hypothesis
	-rm -rf .mypy_cache
	-rm -rf .pytest_cache
	-rm -rf .ruff_cache
	-rm -f .coverage
	-rm -f coverage.xml
	-rm -f polars/polars.abi3.so
	-find . \( -type f -name '*.py[co]' \) -or \( -type d -name __pycache__ \) -delete

cleanest: cleaner  ## cleaner + remove the wheels
	-rm -rf $(WHEELS_DEBUG)
	-rm -rf $(WHEELS_RELEASE)
	-rm -rf wheels

.PHONY: fmt
fmt: $(VENV)  ## Run autoformatting and linting
	$(VENV_ACT_BIN)isort .
	$(VENV_ACT_BIN)black .
	$(VENV_ACT_BIN)blackdoc .
	$(VENV_ACT_BIN)pyupgrade --py37-plus `find polars docs tests -name "*.py" -type f`
	cargo fmt --all
	-dprint fmt
	-$(VENV_ACT_BIN)mypy
	-$(VENV_ACT_BIN)flake8 polars tests docs

.PHONY: clippy
clippy:  ## Run clippy
	cargo clippy -- -D warnings -A clippy::borrow_deref_ref

.PHONY: pre-commit
pre-commit: fmt clippy  ## Run all code quality checks

.PHONY: test
test: develop # develop  ## Run fast unittests
	$(VENV_ACT_BIN)pytest tests/unit/

.PHONY: doctest
doctest: $(VENV) develop  ## Run doctests
	$(VENV_ACT_BIN)python tests/docs/run_doc_examples.py

.PHONY: test-all
test-all: # develop  ## Run all tests
	$(VENV_ACT_BIN)pytest
	$(VENV_ACT_BIN)python tests/docs/run_doc_examples.py

.PHONY: coverage
coverage:  ## Run tests and report coverage
	$(VENV_ACT_BIN)pytest --cov

build-debug: $(VENV)  ## Build debug wheel
	$(VENV_ACT_BIN)maturin build -o $(WHEELS_DEBUG)

build: $(VENV)  ## Build release wheel
	$(VENV_ACT_BIN)maturin build -o $(WHEELS_RELEASE) --release -- -C codegen-units=16 -C lto=thin -C target-cpu=native

install-debug: $(VENV)  ## pip install debug wheel
	$(VENV_ACT_BIN)pip install --no-deps --force-reinstall -U $(WHEELS_DEBUG)/polars-*.whl

install: $(VENV)  ## pip install release wheel
	$(VENV_ACT_BIN)pip install --no-deps --force-reinstall -U $(WHEELS_RELEASE)/polars-*.whl

# Shortcut targets
build-install-debug: build-debug install-debug  ## Build and install debug wheel
build-install: build install  ## Build and install release wheel

.PHONY: help
help:  ## Display this help screen
	@echo "Usage:"
	@echo "* To run a make <comand> with venv, issue"
	@echo "  > make <command>"
	@echo "* To run a make <command> with alternative environement <name>, issue"
	@echo "  > make VENV=<name> <command>"
	@echo "* To run a make <command> without virtual environment (ex: conda), issue"
	@echo "  > make VENV='' <command>"
	@echo
	@echo "Currently:"
	@echo "  VENV = '$(VENV)'"
	@echo

	@echo -e '\033[1mAvailable commands:\033[0m'
	@grep -E '^[a-z.A-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' | sort

#EOF
