#!/bin/bash
cargo --version
rustc --version

# follow https://github.com/pola-rs/polars/blob/main/CONTRIBUTING.md#setting-up-your-environment
rustup toolchain install nightly --component miri

# add https://github.com/rust-lang/rustfmt
rustup component add rustfmt --toolchain nightly

# Install dprint, see https://dprint.dev/install/
# this will be slower since it builds from the source
cargo install --locked dprint
