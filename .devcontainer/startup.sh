#!/usr/bin/env bash

echo $PATH
export $PATH="$PATH:/root/.cargo/bin"
ls /root/.cargo/bin

cd py-polars
make build
