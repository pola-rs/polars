#!/usr/bin/env bash

echo $PATH
echo $(whoami)

cd py-polars
make build
