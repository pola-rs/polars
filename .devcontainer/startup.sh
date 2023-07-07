#!/bin/bash
# all of this should be into a Makefile following the example of
# https://github.com/pola-rs/r-polars
RUST_TOOLCHAIN := nightly-2023-06-23

rustup toolchain install $(RUST_TOOLCHAIN) --component miri
rustup default $(RUST_TOOLCHAIN)

rustup component add rustfmt
rustup component add clippy

# Install dprint, see https://dprint.dev/install/
# this will be slower since it builds from the source
cargo install --locked dprint
cargo install cargo-license

cd py-polars || exit
make requirements
