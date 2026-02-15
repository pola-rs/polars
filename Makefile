.DEFAULT_GOAL := help

PYTHONPATH=
SHELL=bash
ifeq ($(VENV),)
VENV := .venv
endif

RUNTIME_CARGO_TOML=py-polars/runtime/polars-runtime-32/Cargo.toml

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
		_DUMMY := $(warning Unknown architecture '$(PROCESSOR_ARCHITECTURE)', defaulting to 'unknown')
		ARCH := unknown
    endif
else
    UNAME_M := $(shell uname -m)
    ifeq ($(UNAME_M),x86_64)
		ARCH := amd64
    else ifeq ($(UNAME_M),x64)
		ARCH := amd64
    else ifeq ($(UNAME_M),arm64)
		ARCH := arm64
    else ifeq ($(UNAME_M),aarch64)
		ARCH := arm64
	else ifneq ($(filter %86,$(UNAME_M)),)
		ARCH := x86
	else
		_DUMMY := $(warning Unknown architecture '$(UNAME_M)', defaulting to 'unknown')
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
	@unset CONDA_PREFIX \
	&& python3 -m venv $(VENV) \
	&& $(MAKE) requirements

# Note: Installed separately as pyiceberg does not have wheels for 3.13, causing
# --no-build to fail.
.PHONY: requirements
requirements:  ## Install/refresh Python project requirements
	@unset CONDA_PREFIX \
	&& $(VENV_BIN)/python -m venv $(VENV) --clear \
	&& $(VENV_BIN)/python -m pip install --upgrade uv \
	&& $(VENV_BIN)/uv pip install --upgrade --compile-bytecode --no-build \
	   -r py-polars/requirements-dev.txt \
	   -r py-polars/requirements-lint.txt \
	   -r py-polars/docs/requirements-docs.txt \
	   -r docs/source/requirements.txt \
	&& $(VENV_BIN)/uv pip install --upgrade --compile-bytecode "pyiceberg>=0.7.1" pyiceberg-core \
	&& $(VENV_BIN)/uv pip install --no-deps -e py-polars \
	&& $(VENV_BIN)/uv pip uninstall polars-runtime-compat polars-runtime-64  ## Uninstall runtimes which might take precedence over polars-runtime-32

.PHONY: requirements-all
requirements-all:  ## Install/refresh all Python requirements (including those needed for CI tests)
	$(VENV_BIN)/uv pip install --upgrade --compile-bytecode -r py-polars/requirements-ci.txt
	$(MAKE) requirements

.PHONY: build
build: .venv  ## Compile and install Python Polars for development
	@unset CONDA_PREFIX \
	&& $(VENV_BIN)/maturin develop -m $(RUNTIME_CARGO_TOML) --features backtrace_filter $(ARGS) --uv \
	$(FILTER_PIP_WARNINGS)

.PHONY: build-mindebug
build-mindebug: .venv  ## Same as build, but don't include full debug information
	@unset CONDA_PREFIX \
	&& $(VENV_BIN)/maturin develop -m $(RUNTIME_CARGO_TOML) --features backtrace_filter --profile mindebug-dev $(ARGS) --uv \
	$(FILTER_PIP_WARNINGS)

.PHONY: build-release
build-release: .venv  ## Compile and install Python Polars binary with optimizations, with minimal debug symbols
	@unset CONDA_PREFIX \
	&& $(VENV_BIN)/maturin develop -m $(RUNTIME_CARGO_TOML) --features backtrace_filter --release $(ARGS) --uv \
	$(FILTER_PIP_WARNINGS)

.PHONY: build-nodebug-release
build-nodebug-release: .venv  ## Same as build-release, but without any debug symbols at all (a bit faster to build)
	@unset CONDA_PREFIX \
	&& $(VENV_BIN)/maturin develop -m $(RUNTIME_CARGO_TOML) --features backtrace_filter --profile nodebug-release $(ARGS) --uv \
	$(FILTER_PIP_WARNINGS)

.PHONY: build-debug-release
build-debug-release: .venv  ## Same as build-release, but with full debug symbols turned on (a bit slower to build)
	@unset CONDA_PREFIX \
	&& $(VENV_BIN)/maturin develop -m $(RUNTIME_CARGO_TOML) --features backtrace_filter --profile debug-release $(ARGS) --uv \
	$(FILTER_PIP_WARNINGS)

.PHONY: build-dist-release
build-dist-release: .venv  ## Compile and install Python Polars binary with super slow extra optimization turned on, for distribution
	@unset CONDA_PREFIX \
	&& $(VENV_BIN)/maturin develop -m $(RUNTIME_CARGO_TOML) --features backtrace_filter --profile dist-release $(ARGS) --uv \
	$(FILTER_PIP_WARNINGS)

.PHONY: check
check:  ## Run cargo check with all features
	cargo check --workspace --all-targets --all-features

.PHONY: clippy
clippy:  ## Run clippy with all features
	python3 tools/cargo-fail-warning.py clippy --workspace --all-targets --all-features --locked -- -W clippy::dbg_macro

.PHONY: clippy-default
clippy-default:  ## Run clippy with default features
	python3 tools/cargo-fail-warning.py clippy --all-targets --locked -- -W clippy::dbg_macro

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

.PHONY: py-lint
py-lint: .venv  ## Run python lint checks (only)
	@$(MAKE) -s -C py-polars lint $(ARGS)

.PHONY: check-fixme
check-fixme:
	@cmd_exit=0; \
	rg -q -i --hidden --glob '!/.git' --glob '!/Makefile' --glob '!/.github/workflows/lint-global.yml' 'FIXME' || cmd_exit=$$?; \
	if [ $$cmd_exit -eq 0 ]; then \
		rg -i --hidden --glob '!/.git' --glob '!/Makefile' --glob '!/.github/workflows/lint-global.yml' -C 2 'FIXME'; \
		printf "\n[ERROR] Found FIXME, use TODO for things that are ok to merge.\n"; \
		exit 1; \
	fi

.PHONY: update-dsl-schema-hashes
update-dsl-schema-hashes:  ## Update the DSL schema hashes file
	cargo run --all-features --bin dsl-schema update-hashes

.PHONY: pre-commit
pre-commit: fmt py-lint clippy clippy-default  ## Run all code quality checks

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
