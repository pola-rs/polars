#!/usr/bin/env bash

function test-build {
  maturin build -o wheels -i $(which python)
}

function test-install {
  test-build
  cd wheels && pip install --force-reinstall -U polars-*.whl && cd ..
}

function release-build {
  maturin build --release -o wheels -i $(which python)
}

function release-install {
  release-build
  cd wheels && pip install --force-reinstall -U polars-*.whl && cd ..
}

function build-run-tests {
  test-install
  pytest tests
}

function test {
  maturin develop
  pytest tests
}

function build-docker-base {
  cd ..
  docker build -f py-polars/Dockerfile_base -t ritchie46/py-polars-base .
}

function build-and-push-base {
  build-docker-base
  docker push ritchie46/py-polars-base
}

function build-docker {
  docker build -t ritchie46/py-polars .
}

function build-and-push {
  build-docker
  docker push ritchie46/py-polars
}

"$@"
