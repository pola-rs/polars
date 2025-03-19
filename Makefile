.DEFAULT_GOAL := help

PYTHONPATH=
SHELL=bash
VENV=.venv

ifeq ($(OS),Windows_NT)
	VENV_BIN=$(VENV)/Scripts
else
	VENV_BIN=$(VENV)/bin
endif

# Detect CPU architecture.
ifeq ($(OS),Windows_NT)
    ifeq ($(PROCESSOR_ARCHITECTURE),AMD64)
		ARCH := amd64
	else ifeq ($(PROCESSOR_ARCHITECTURE),x86)
		ARCH := x86
	else ifeq ($(PROCESSOR_ARCHITECTURE),ARM64)
		ARCH := arm64
	else
		ARCH := unknown
    endif
else
    UNAME_P := $(shell uname -p)
    ifeq ($(UNAME_P),x86_64)
		ARCH := amd64
	else ifneq ($(filter %86,$(UNAME_P)),)
		ARCH := x86
	else ifneq ($(filter arm%,$(UNAME_P)),)
		ARCH := arm64
	else
		ARCH := unknown
    endif
endif

# Ensure boolean arguments are normalized to 1/0 to prevent surprises.
ifdef LTS_CPU
	ifeq ($(LTS_CPU),0)
	else ifeq ($(LTS_CPU),1)
	else
$(error LTS_CPU must be 0 or 1 (or undefined, default to 0))
	endif
endif

# Define RUSTFLAGS and CFLAGS appropriate for the architecture.
# Keep synchronized with .github/workflows/release-python.yml.
ifeq ($(ARCH),amd64)
	ifeq ($(LTS_CPU),1)
		FEAT_RUSTFLAGS=-C target-feature=+sse3,+ssse3,+sse4.1,+sse4.2,+popcnt,+cmpxchg16b
		FEAT_CFLAGS=-msse3 -mssse3 -msse4.1 -msse4.2 -mpopcnt -mcx16
	else
		FEAT_RUSTFLAGS=-C target-feature=+sse3,+ssse3,+sse4.1,+sse4.2,+popcnt,+cmpxchg16b,+avx,+avx2,+fma,+bmi1,+bmi2,+lzcnt,+pclmulqdq,+movbe -Z tune-cpu=skylake
		FEAT_CFLAGS=-msse3 -mssse3 -msse4.1 -msse4.2 -mpopcnt -mcx16 -mavx -mavx2 -mfma -mbmi -mbmi2 -mlzcnt -mpclmul -mmovbe -mtune=skylake
	endif
endif

override RUSTFLAGS+=$(FEAT_RUSTFLAGS)
override CFLAGS+=$(FEAT_CFLAGS)
export RUSTFLAGS
export CFLAGS

# Define command to filter pip warnings when running maturin
FILTER_PIP_WARNINGS=| grep -v "don't match your environment"; test $${PIPESTATUS[0]} -eq 0

.venv:  ## Set up Python virtual environment and install requirements
	python3 -m venv $(VENV)
	$(MAKE) requirements

.PHONY: requirements
requirements: .venv  ## Install/refresh Python project requirements
	@unset CONDA_PREFIX \
	&& $(VENV_BIN)/python -m pip install --upgrade uv \
	&& $(VENV_BIN)/uv pip install --upgrade --compile-bytecode --no-build \
	   -r py-polars/requirements-dev.txt \
	   -r py-polars/requirements-lint.txt \
	   -r py-polars/docs/requirements-docs.txt \
	   -r docs/source/requirements.txt

.PHONY: requirements-all
requirements-all: .venv  ## Install/refresh all Python requirements (including those needed for CI tests)
	$(MAKE) requirements
	$(VENV_BIN)/uv pip install --upgrade --compile-bytecode -r py-polars/requirements-ci.txt

.PHONY: build
build: .venv  ## Compile and install Python Polars for development
	@unset CONDA_PREFIX \
	&& $(VENV_BIN)/maturin develop -m py-polars/Cargo.toml $(ARGS) \
	$(FILTER_PIP_WARNINGS)

.PHONY: build-mindebug
build-mindebug: .venv  ## Same as build, but don't include full debug information
	@unset CONDA_PREFIX \
	&& $(VENV_BIN)/maturin develop -m py-polars/Cargo.toml --profile mindebug-dev $(ARGS) \
	$(FILTER_PIP_WARNINGS)

.PHONY: build-release
build-release: .venv  ## Compile and install Python Polars binary with optimizations, with minimal debug symbols
	@unset CONDA_PREFIX \
	&& $(VENV_BIN)/maturin develop -m py-polars/Cargo.toml --release $(ARGS) \
	$(FILTER_PIP_WARNINGS)

.PHONY: build-nodebug-release
build-nodebug-release: .venv  ## Same as build-release, but without any debug symbols at all (a bit faster to build)
	@unset CONDA_PREFIX \
	&& $(VENV_BIN)/maturin develop -m py-polars/Cargo.toml --profile nodebug-release $(ARGS) \
	$(FILTER_PIP_WARNINGS)

.PHONY: build-debug-release
build-debug-release: .venv  ## Same as build-release, but with full debug symbols turned on (a bit slower to build)
	@unset CONDA_PREFIX \
	&& $(VENV_BIN)/maturin develop -m py-polars/Cargo.toml --profile debug-release $(ARGS) \
	$(FILTER_PIP_WARNINGS)

.PHONY: build-dist-release
build-dist-release: .venv  ## Compile and install Python Polars binary with super slow extra optimization turned on, for distribution
	@unset CONDA_PREFIX \
	&& $(VENV_BIN)/maturin develop -m py-polars/Cargo.toml --profile dist-release $(ARGS) \
	$(FILTER_PIP_WARNINGS)

.PHONY: check
check:  ## Run cargo check with all features
	cargo check --workspace --all-targets --all-features

.PHONY: clippy
clippy:  ## Run clippy with all features
	cargo clippy --workspace --all-targets --all-features --locked -- -D warnings -D clippy::dbg_macro

.PHONY: clippy-default
clippy-default:  ## Run clippy with default features
	cargo clippy --all-targets --locked -- -D warnings -D clippy::dbg_macro

.PHONY: fmt
fmt:  ## Run autoformatting and linting
	$(VENV_BIN)/ruff check
	$(VENV_BIN)/ruff format
	cargo fmt --all
	dprint fmt
	$(VENV_BIN)/typos
	
.PHONY: fix
fix:
	cargo clippy --workspace --all-targets --all-features --fix 
	@# Good chance the fixing introduced formatting issues, best to just do a quick format.
	cargo fmt --all
	

.PHONY: pre-commit
pre-commit: fmt clippy clippy-default  ## Run all code quality checks

.PHONY: clean
clean:  ## Clean up caches, build artifacts, and the venv
	@$(MAKE) -s -C py-polars/ $@
	@rm -rf .ruff_cache/
	@rm -rf .hypothesis/
	@rm -rf .venv/
	@cargo clean

.PHONY: help
help:  ## Display this help screen
	@echo -e "\033[1mAvailable commands:\033[0m"
	@grep -E '^[a-z.A-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}' | sort
	@echo
	@echo The build commands support LTS_CPU=1 for building for older CPUs, and ARGS which is passed through to maturin.
	@echo 'For example to build without default features use: make build ARGS="--no-default-features".'
