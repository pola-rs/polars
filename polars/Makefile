
fmt:
	cargo fmt --all

check:
	cargo c --all-features \
	-p polars-core \
	-p polars-io \
	-p polars-lazy

clippy:
	cargo clippy -Z unstable-options --all-features \
	-p polars-core \
	-p polars-io \
	-p polars-lazy

test:
	cargo t -p polars-lazy -p polars-io -p polars-core

test-doc:
	cargo t -p polars-lazy -p polars-io -p polars-core --doc
