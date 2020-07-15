#!/usr/bin/env bash

function test-build {
  maturin build -o wheels -i $(which python)
}

function test-install {
  test-build
  cd wheels && pip install -U py* && cd ..
}

function release-build {
  maturin build --release -o wheels -i $(which python)
}

function release-install {
  release-build
  cd wheels && pip install -U py* && cd ..
}

function build-run-tests {
  test-install
  pytest tests
}

"$@"
