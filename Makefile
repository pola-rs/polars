.DEFAULT_GOAL := help

SHELL=/bin/bash
BASE ?= main

.PHONY: fmt
fmt:
	cargo fmt --all
	dprint fmt

.PHONY: generate_test_files
generate_test_files:
	cargo run -p polars-cli "select * from read_csv('../examples/datasets/foods1.csv')" -o parquet > ../examples/datasets/foods1.parquet
	cargo run -p polars-cli "select * from read_csv('../examples/datasets/foods1.csv')" -o arrow > ../examples/datasets/foods1.ipc

.PHONY: check
check:
	cargo check --all-targets --all-features

.PHONY: clippy
clippy:
	cargo clippy --all-targets --all-features

.PHONY: clippy-default
clippy-default:
	cargo clippy

.PHONY: test
test:  ## Run tests
	cargo test --all-features \
		-p polars-lazy \
		-p polars-io \
		-p polars-core \
		-p polars-arrow \
		-p polars-time \
		-p polars-utils \
		-p polars-row \
		-p polars-sql \
		-- \
		--test-threads=2

.PHONY: integration-tests
integration-tests:
	cargo test --all-features --test it -- --test-threads=2

.PHONY: miri
miri:
	# not tested on all features because miri does not support SIMD
	# some tests are also filtered, because miri cannot deal with the rayon threadpool
	# we ignore leaks because the thread pool of rayon is never killed.
	MIRIFLAGS="-Zmiri-disable-isolation -Zmiri-ignore-leaks -Zmiri-disable-stacked-borrows" \
	POLARS_ALLOW_EXTENSION=1 \
	cargo miri test \
	    --no-default-features \
	    --features object \
	    -p polars-core \
	    -p polars-arrow

.PHONY: test-doc
test-doc:
	cargo test --doc \
	    -p polars-lazy \
	    -p polars-io \
	    -p polars-core \
	    -p polars-arrow \
		-p polars-sql

.PHONY: pre-commit
pre-commit: fmt clippy clippy-default  ## Run autoformatting and linting

.PHONY: check-features
check-features:
	cargo hack check --each-feature --no-dev-deps

.PHONY: bench-save
bench-save:
	cargo bench --features=random --bench $(BENCH) -- --save-baseline $(SAVE)

.PHONY: bench-cmp
bench-cmp:
	cargo bench --features=random --bench $(BENCH) -- --load-baseline $(FEAT) --baseline $(BASE)

.PHONY: doctest
doctest:
	cargo doc --all-features -p polars-arrow
	cargo doc --all-features -p polars-utils
	cargo doc --features=docs-selection -p polars-core
	cargo doc -p polars-time
	cargo doc -p polars-ops
	cargo doc --all-features -p polars-io
	cargo doc --all-features -p polars-lazy
	cargo doc --features=docs-selection -p polars
	cargo doc --all-features -p polars-sql

.PHONY: publish
publish:
	cargo publish --allow-dirty -p polars-error
	cargo publish --allow-dirty -p polars-utils
	cargo publish --allow-dirty -p polars-row
	cargo publish --allow-dirty -p polars-arrow
	cargo publish --allow-dirty -p polars-json
	cargo publish --allow-dirty -p polars-core
	cargo publish --allow-dirty -p polars-ops
	cargo publish --allow-dirty -p polars-time
	cargo publish --allow-dirty -p polars-io
	cargo publish --allow-dirty -p polars-plan
	cargo publish --allow-dirty -p polars-pipe
	cargo publish --allow-dirty -p polars-lazy
	cargo publish --allow-dirty -p polars-algo
	cargo publish --allow-dirty -p polars-sql
	cargo publish --allow-dirty -p polars
	cd .. && cargo publish --allow-dirty -p polars-cli

.PHONY: help
help:  ## Display this help screen
	@echo -e "\033[1mAvailable commands:\033[0m\n"
	@grep -E '^[a-z.A-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' | sort
