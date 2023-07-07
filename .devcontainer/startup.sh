#!/bin/bash
cargo --version
rustc --version

toolchain="nightly-2023-06-23"

# follow https://github.com/pola-rs/polars/blob/main/CONTRIBUTING.md#setting-up-your-environment
rustup toolchain install ${toolchain}-x86_64-unknown-linux-gnu --component miri

# add https://github.com/rust-lang/rustfmt
rustup component add rustfmt --toolchain ${toolchain}-x86_64-unknown-linux-gnu
rustup component add clippy --toolchain ${toolchain}-x86_64-unknown-linux-gnu

# Install dprint, see https://dprint.dev/install/
# this will be slower since it builds from the source
cargo install --locked dprint

# install cmake
apt-get update && apt-get install -y cmake
