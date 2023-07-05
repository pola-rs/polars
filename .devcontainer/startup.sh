#!/usr/bin/env bash

echo $PATH
echo whoami

export $PATH="$PATH:/home/vscode/.cargo/bin"
ls /home/vscode/.cargo/bin

cd py-polars
make build
