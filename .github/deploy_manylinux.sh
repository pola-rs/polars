#!/bin/bash

# easier debugging
set -e
pwd
ls -la

rm py-polars/README.md
cp README.md py-polars/README.md
cd py-polars
rustup override set nightly-2023-07-27
export RUSTFLAGS='-C target-feature=+fxsr,+sse,+sse2,+sse3,+ssse3,+sse4.1,+sse4.2,+popcnt,+avx,+fma'

# first the default release
maturin publish \
  --skip-existing \
  --username ritchie46

# now compile polars with bigidx feature
sed -i 's/name = "polars"/name = "polars-u64-idx"/' pyproject.toml
# a brittle hack to insert the 'bigidx' feature
sed -i 's/"dynamic_groupby",/"dynamic_groupby",\n"bigidx",/' Cargo.toml

maturin publish \
  --skip-existing \
  --username ritchie46

# https://github.com/actions/checkout/issues/760
git config --global --add safe.directory /github/workspace
# Clean up after bigidx changes
git checkout .
