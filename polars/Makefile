BASE ?= master

.PHONY: fmt check check-features clippy clippy-default test test-doc integration-tests

fmt:
	cargo fmt --all
	$(MAKE) -C .. fmt_toml

check:
	cargo check --all-features \
	    -p polars-core \
	    -p polars-io \
	    -p polars-lazy \
	    -p polars-arrow \
	    -p polars-time

clippy:
	cargo clippy --all-features \
	    -p polars-core \
	    -p polars-io \
	    -p polars-lazy \
	    -p polars-arrow \
	    -p polars-utils \
	    -p polars-time

clippy-default:
	cargo clippy

test:
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
	cargo t --all-features --test it

miri:
	# not tested on all features because miri does not support SIMD
	# some tests are also filtered, because miri cannot deal with the rayon threadpool
	# Miri also reports UB in prettytable.rs, so we must toggle that feature off.
	MIRIFLAGS="-Zmiri-disable-isolation" \
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

pre-commit: fmt clippy clippy-default


check-features:
	cargo hack check --each-feature --no-dev-deps --features private

bench-save:
	cargo bench --features=random --bench $(BENCH) -- --save-baseline $(SAVE)

bench-cmp:
	cargo bench --features=random --bench $(BENCH) -- --load-baseline $(FEAT) --baseline $(BASE)
