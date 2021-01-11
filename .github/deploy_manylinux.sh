#!/bin/bash

# easier debugging
pwd
ls -la

rustup override set nightly-2020-11-12

rm py-polars/README.md
cp README.md py-polars/README.md
cd py-polars
maturin publish \
--username ritchie46
