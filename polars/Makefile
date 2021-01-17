
fmt:
	cargo fmt --all

check:
	cargo c --all-features \
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

test:
	cargo t -p polars-lazy -p polars-io -p polars-core -p polars-arrow

test-doc:
	cargo t -p polars-lazy -p polars-io -p polars-core -p polars-arrow --doc
