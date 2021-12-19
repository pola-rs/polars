BASE ?= master

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
	cargo clippy -Z unstable-options --all-features \
	    -p polars-core \
	    -p polars-io \
	    -p polars-lazy \
	    -p polars-arrow \
	    -p polars-time

clippy-default:
	cargo clippy -Z unstable-options

test:
	cargo test --all-features \
	    -p polars-lazy \
	    -p polars-io \
	    -p polars-core \
	    -p polars-arrow \
	    -p polars-time \
	    -- \
	    --test-threads=2

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
