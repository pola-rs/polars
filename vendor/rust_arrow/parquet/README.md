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

# An Apache Parquet implementation in Rust

## Usage
Add this to your Cargo.toml:
```toml
[dependencies]
parquet = "1.0.0-SNAPSHOT"
```

and this to your crate root:
```rust
extern crate parquet;
```

Example usage of reading data:
```rust
use std::fs::File;
use std::path::Path;
use parquet::file::reader::{FileReader, SerializedFileReader};

let file = File::open(&Path::new("/path/to/file")).unwrap();
let reader = SerializedFileReader::new(file).unwrap();
let mut iter = reader.get_row_iter(None).unwrap();
while let Some(record) = iter.next() {
    println!("{}", record);
}
```
See [crate documentation](https://docs.rs/crate/parquet/1.0.0-SNAPSHOT) on available API.

## Supported Parquet Version
- Parquet-format 2.4.0

To update Parquet format to a newer version, check if [parquet-format](https://github.com/sunchao/parquet-format-rs)
version is available. Then simply update version of `parquet-format` crate in Cargo.toml.

## Features
- [X] All encodings supported
- [X] All compression codecs supported
- [X] Read support
  - [X] Primitive column value readers
  - [X] Row record reader
  - [X] Arrow record reader
- [ ] Statistics support
- [X] Write support
  - [X] Primitive column value writers
  - [ ] Row record writer
  - [ ] Arrow record writer
- [ ] Predicate pushdown
- [ ] Parquet format 2.5 support

## Requirements
- Rust nightly

See [Working with nightly Rust](https://github.com/rust-lang-nursery/rustup.rs/blob/master/README.md#working-with-nightly-rust)
to install nightly toolchain and set it as default.

Parquet requires LLVM.  Our windows CI image includes LLVM but to build the libraries locally windows
users will have to install LLVM. Follow [this](https://github.com/appveyor/ci/issues/2651) link for info.

## Build
Run `cargo build` or `cargo build --release` to build in release mode.
Some features take advantage of SSE4.2 instructions, which can be
enabled by adding `RUSTFLAGS="-C target-feature=+sse4.2"` before the
`cargo build` command.

## Test
Run `cargo test` for unit tests.

## Binaries
The following binaries are provided (use `cargo install` to install them):
- **parquet-schema** for printing Parquet file schema and metadata.
`Usage: parquet-schema <file-path> [verbose]`, where `file-path` is the path to a Parquet file,
and optional `verbose` is the boolean flag that allows to print full metadata or schema only
(when not specified only schema will be printed).

- **parquet-read** for reading records from a Parquet file.
`Usage: parquet-read <file-path> [num-records]`, where `file-path` is the path to a Parquet file,
and `num-records` is the number of records to read from a file (when not specified all records will
be printed).

- **parquet-rowcount** for reporting the number of records in one or more Parquet files.
`Usage: parquet-rowcount <file-path> ...`, where `file-path` is the path to a Parquet file, and `...`
indicates any number of additional parquet files.

If you see `Library not loaded` error, please make sure `LD_LIBRARY_PATH` is set properly:
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(rustc --print sysroot)/lib
```

## Benchmarks
Run `cargo bench` for benchmarks.

## Docs
To build documentation, run `cargo doc --no-deps`.
To compile and view in the browser, run `cargo doc --no-deps --open`.

## License
Licensed under the Apache License, Version 2.0: http://www.apache.org/licenses/LICENSE-2.0.
