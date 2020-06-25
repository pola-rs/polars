<!---
  Licensed to the Apache Software Foundation (ASF) under one
  or more contributor license agreements.  See the NOTICE file
  distributed with this work for additional information
  regarding copyright ownership.  The ASF licenses this file
  to you under the Apache License, Version 2.0 (the
  "License"); you may not use this file except in compliance
  with the License.  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing,
  software distributed under the License is distributed on an
  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  KIND, either express or implied.  See the License for the
  specific language governing permissions and limitations
  under the License.
-->

# Native Rust implementation of Apache Arrow

## The Rust implementation of Arrow consists of the following crates

| Crate     | Description | Documentation |
|-----------|-------------|---------------|
|Arrow      | Core functionality (memory layout, array builders, low level computations) | [(README)](arrow/README.md) |
|Parquet    | Parquet support | [(README)](parquet/README.md) |
|DataFusion | In-memory query engine with SQL support | [(README)](datafusion/README.md) |

## Prerequisites

Before running tests and examples it is necessary to set up the local development environment.

### Git Submodules

The tests rely on test data that is contained in git submodules.

To pull down this data run the following:

```bash
git submodule update --init
```

This populates data in two git submodules:

- `cpp/submodules/parquet_testing/data` (sourced from https://github.com/apache/parquet-testing.git)
- `testing` (sourced from https://github.com/apache/arrow-testing)

Create two new environment variables to point to these directories as follows:

```bash
export PARQUET_TEST_DATA=/path/to/arrow/cpp/submodules/parquet-testing/data
export ARROW_TEST_DATA=/path/to/arrow/testing/data/
```

It is now possible to run `cargo test` as usual.

## Code Formatting

Our CI uses `rustfmt` to check code formatting.  Although the project is
built and tested against nightly rust we use the stable version of
`rustfmt`.  So before submitting a PR be sure to run the following
and check for lint issues:

```bash
cargo +stable fmt --all -- --check
```

## CI and Dockerized builds

There are currently multiple CI systems that build the project and they all use the same docker image. It is possible to run the same build locally.

From the root of the Arrow project, run the following command to build the Docker image that the CI system uses to build the project.

```bash
docker-compose build debian-rust
```

Run the following command to build the project in the same way that the CI system will build the project. Note that this currently does cause some files to be written to your local workspace.

```bash
docker-compose run --rm debian-rust bash
```
