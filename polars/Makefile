
fmt:
	cargo fmt --all

check:
	cargo check --all-features \
	-p polars-core \
	-p polars-io \
	-p polars-lazy \
	-p polars-arrow

clippy:
	cargo clippy -Z unstable-options --all-features \
	-p polars-core \
	-p polars-io \
	-p polars-lazy \
	-p polars-arrow

clippy-default:
	cargo clippy -Z unstable-options

test:
	cargo test --all-features \
	-p polars-lazy \
    -p polars-io \
    -p polars-core \
    -p polars-arrow -- \
    --test-threads=2

miri:
	# not tested on all features because miri does not support SIMD
	# some tests are also filtered, because miri cannot deal with the rayon threadpool
	# Miri also reports UB in prettytable.rs, so we must toggle that feature off.
	MIRIFLAGS="-Zmiri-disable-isolation" \
	cargo miri test \
	--no-default-features \
    -p polars-core \
    -p polars-arrow --


test-doc:
	cargo test -p polars-lazy -p polars-io -p polars-core -p polars-arrow --doc


pre-commit: fmt clippy clippy-default


check-features:
	cargo hack check --each-feature --no-dev-deps --features private