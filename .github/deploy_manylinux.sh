#!/bin/bash

# easier debugging
pwd
ls -la

rm py-polars/README.md
cp README.md py-polars/README.md
cd py-polars
rustup override set nightly-2022-04-02
export RUSTFLAGS='-C target-feature=+fxsr,+sse,+sse2,+sse3,+ssse3,+sse4.1+sse4.2,+popcnt,+avx,+fma'

# first the default release
maturin publish \
  --skip-existing \
  --username ritchie46

# now compile polars with bigidx feature
sed -i 's/name = "polars"/name = "polars-64"/' pyproject.toml
# a brittle hack to insert the 'bigidx' feature
sed '/s/"dynamic_groupby",/"dynamic_groupby",\n"bigidx",/' Cargo.toml

maturin publish \
  --skip-existing \
  --username ritchie46
