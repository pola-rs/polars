#!/bin/bash

# easier debugging
pwd
ls -la

rm py-polars/README.md
cp README.md py-polars/README.md
cd py-polars
rustup override set nightly-2022-03-23
export RUSTFLAGS='-C target-feature=+fxsr,+sse,+sse2,+sse3,+ssse3,+sse4.1+sse4.2,+popcnt,+avx,+fma'
maturin publish \
  --skip-existing \
  --username ritchie46
