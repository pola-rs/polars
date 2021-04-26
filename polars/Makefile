
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

test-doc:
	cargo test -p polars-lazy -p polars-io -p polars-core -p polars-arrow --doc
