.DEFAULT_GOAL := help

BASE ?= master

.PHONY: fmt check check-features clippy clippy-default test test-doc integration-tests

fmt:
	cargo fmt --all
	dprint fmt

check:
	cargo check --all-features \
	    -p polars-core \
	    -p polars-io \
	    -p polars-lazy \
	    -p polars-arrow \
	    -p polars-time \
	    -p polars-ops

clippy:
	cargo clippy --all-features \
	    -p polars-core \
	    -p polars-io \
	    -p polars-lazy \
	    -p polars-arrow \
	    -p polars-utils \
	    -p polars-ops \
	    -p polars-time

clippy-default:
	cargo clippy

test:  ## Run tests
	POLARS_MAX_THREADS=4 cargo t -p polars-core test_4_threads
	cargo test --all-features \
	    -p polars-lazy \
	    -p polars-io \
	    -p polars-core \
	    -p polars-arrow \
	    -p polars-time \
	    -p polars-utils \
	    -- \
	    --test-threads=2

integration-tests:
	cargo t --all-features --test it -- --test-threads=2

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
	    -p polars-arrow \
	    --

test-doc:
	cargo test --doc \
	    -p polars-lazy \
	    -p polars-io \
	    -p polars-core \
	    -p polars-arrow

pre-commit: fmt clippy clippy-default  ## Run autoformatting and linting


check-features:
	cargo hack check --each-feature --no-dev-deps --features private

bench-save:
	cargo bench --features=random --bench $(BENCH) -- --save-baseline $(SAVE)

bench-cmp:
	cargo bench --features=random --bench $(BENCH) -- --load-baseline $(FEAT) --baseline $(BASE)

doctest:
	cargo doc --all-features -p polars-arrow
	cargo doc --all-features -p polars-utils
	cargo doc --features=docs-selection -p polars-core
	cargo doc -p polars-time
	cargo doc -p polars-ops
	cargo doc --all-features -p polars-io
	cargo doc --all-features -p polars-lazy
	cargo doc --features=docs-selection -p polars

publish:
	cargo publish --allow-dirty -p polars-arrow
	sleep 20
	cargo publish --allow-dirty -p polars-utils
	sleep 20
	cargo publish --allow-dirty -p polars-core
	sleep 20
	cargo publish --allow-dirty -p polars-ops
	sleep 20
	cargo publish --allow-dirty -p polars-time
	sleep 20
	cargo publish --allow-dirty -p polars-io
	sleep 20
	cargo publish --allow-dirty -p polars-plan
	sleep 20
	cargo publish --allow-dirty -p polars-pipe
	sleep 20
	cargo publish --allow-dirty -p polars-lazy
	sleep 20
	cargo publish --allow-dirty -p polars-algo
	sleep 20
	cargo publish --allow-dirty -p polars-sql
	sleep 20
	cargo publish --allow-dirty -p polars

.PHONY: help
help:  ## Display this help screen
	@echo -e "\033[1mAvailable commands:\033[0m\n"
	@grep -E '^[a-z.A-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' | sort
