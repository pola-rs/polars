.DEFAULT_GOAL := help

PYTHONPATH=
SHELL=/bin/bash
VENV=.venv

ifeq ($(OS),Windows_NT)
	VENV_BIN=$(VENV)/Scripts
else
	VENV_BIN=$(VENV)/bin
endif

.venv:  ## Set up Python virtual environment and install requirements
	python3 -m venv $(VENV)
	$(MAKE) requirements

.PHONY: requirements
requirements: .venv  ## Install/refresh Python project requirements
	$(VENV_BIN)/python -m pip install --upgrade pip
	$(VENV_BIN)/pip install --upgrade -r py-polars/requirements-dev.txt
	$(VENV_BIN)/pip install --upgrade -r py-polars/requirements-lint.txt
	$(VENV_BIN)/pip install --upgrade -r py-polars/docs/requirements-docs.txt
	$(VENV_BIN)/pip install --upgrade -r docs/requirements.txt

.PHONY: build-python
build-python: .venv  ## Compile and install Python Polars for development
	@$(MAKE) -s -C py-polars build

.PHONY: clippy
clippy:  ## Run clippy with all features
	cargo clippy --workspace --all-features --locked -- -D warnings

.PHONY: fmt
fmt:  ## Run rustfmt and dprint
	cargo fmt --all
	dprint fmt

.PHONY: clean
clean:  ## Clean up caches and build artifacts
	@rm -rf .venv/
	@rm -rf target/
	@rm -f Cargo.lock
	@cargo clean
	@$(MAKE) -s -C py-polars/ $@

.PHONY: help
help:  ## Display this help screen
	@echo -e "\033[1mAvailable commands:\033[0m"
	@grep -E '^[a-z.A-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}' | sort
