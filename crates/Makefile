.DEFAULT_GOAL := help

SHELL=/bin/bash
BASE ?= main

.PHONY: fmt
fmt:  ## Run rustfmt and dprint
	cargo fmt --all
	dprint fmt

.PHONY: check
check:  ## Run cargo check with all features
	cargo check --workspace --all-targets --all-features

.PHONY: clippy
clippy:  ## Run clippy with all features
	cargo clippy -p polars --all-features

.PHONY: clippy-default
clippy-default:  ## Run clippy with default features
	cargo clippy -p polars

.PHONY: pre-commit
pre-commit: fmt clippy clippy-default  ## Run autoformatting and linting

.PHONY: check-features
check-features:  ## Run cargo check for feature flag combinations (warning: slow)
	cargo hack check -p polars --each-feature --no-dev-deps

.PHONY: miri
miri:  ## Run miri
	# not tested on all features because miri does not support SIMD
	# some tests are also filtered, because miri cannot deal with the rayon threadpool
	# we ignore leaks because the thread pool of rayon is never killed.
	MIRIFLAGS="-Zmiri-disable-isolation -Zmiri-ignore-leaks -Zmiri-disable-stacked-borrows" \
	POLARS_ALLOW_EXTENSION=1 \
	cargo miri test \
	    --features object \
	    -p polars-core \
#	    -p polars-arrow

.PHONY: test
test:  ## Run tests
	cargo test --all-features \
		-p polars-core \
		-p polars-io \
		-p polars-lazy \
		-p polars-ops \
		-p polars-plan \
		-p polars-row \
		-p polars-sql \
		-p polars-time \
		-p polars-utils \
		-- \
		--test-threads=2

.PHONY: nextest
nextest:  ## Run tests with nextest
	cargo nextest run --all-features \
		-p polars-core \
		-p polars-io \
		-p polars-lazy \
		-p polars-ops \
		-p polars-plan \
		-p polars-row \
		-p polars-sql \
		-p polars-time \
		-p polars-utils \

.PHONY: integration-tests
integration-tests:  ## Run integration tests
	cargo test --all-features --test it -p polars

.PHONY: test-doc
test-doc:  ## Run doc examples
	cargo test --doc \
	    -p polars-lazy \
	    -p polars-io \
	    -p polars-core \
		-p polars-sql

.PHONY: bench-save
bench-save:  ## Run benchmark and save
	cargo bench --features=random --bench $(BENCH) -- --save-baseline $(SAVE)

.PHONY: bench-cmp
bench-cmp:  ## Run benchmark and compare
	cargo bench --features=random --bench $(BENCH) -- --load-baseline $(FEAT) --baseline $(BASE)

.PHONY: doctest
doctest:  ## Check that documentation builds
	cargo doc --no-deps --all-features -p polars-utils
	cargo doc --no-deps --features=docs-selection -p polars-core
	cargo doc --no-deps -p polars-time
	cargo doc --no-deps -p polars-ops
	cargo doc --no-deps --all-features -p polars-io
	cargo doc --no-deps --all-features -p polars-lazy
	cargo doc --no-deps --features=docs-selection -p polars
	cargo doc --no-deps --all-features -p polars-sql

.PHONY: publish
publish:  ## Publish Polars crates
	cargo publish --allow-dirty -p polars-error
	cargo publish --allow-dirty -p polars-utils
	cargo publish --allow-dirty -p polars-arrow
	cargo publish --allow-dirty -p polars-row
	cargo publish --allow-dirty -p polars-json
	cargo publish --allow-dirty -p polars-core
	cargo publish --allow-dirty -p polars-ffi
	cargo publish --allow-dirty -p polars-ops
	cargo publish --allow-dirty -p polars-time
	cargo publish --allow-dirty -p polars-io
	cargo publish --allow-dirty -p polars-plan
	cargo publish --allow-dirty -p polars-pipe
	cargo publish --allow-dirty -p polars-lazy
	cargo publish --allow-dirty -p polars-algo
	cargo publish --allow-dirty -p polars-sql
	cargo publish --allow-dirty -p polars

.PHONY: help
help:  ## Display this help screen
	@echo -e "\033[1mAvailable commands:\033[0m"
	@grep -E '^[a-z.A-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' | sort

.PHONY: check-wasm
check-wasm:  ## Check wasm build without supported features
	cargo hack check --target wasm32-unknown-unknown -p polars --no-dev-deps  \
		--each-feature                        \
		--exclude-features async              \
		--exclude-features aws                \
		--exclude-features azure              \
		--exclude-features cloud              \
		--exclude-features cloud_write        \
		--exclude-features decompress         \
		--exclude-features decompress-fast    \
		--exclude-features default            \
		--exclude-features docs-selection     \
		--exclude-features extract_jsonpath   \
		--exclude-features fmt                \
		--exclude-features gcp                \
		--exclude-features ipc                \
		--exclude-features ipc_streaming      \
		--exclude-features json               \
		--exclude-features nightly            \
		--exclude-features parquet            \
		--exclude-features performant         \
		--exclude-features streaming          \
		--exclude-features test
