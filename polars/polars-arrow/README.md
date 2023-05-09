# polars-arrow

`polars-arrow` is a submodule of Polars that provides Arrow interfaces for the Polars.

## Features

| Feature    | Description                                                                                                   |
| ---------- | ------------------------------------------------------------------------------------------------------------- |
| nightly    | Enables the use of unstable and experimental features that are only available on the nightly version of Rust. |
| strings    | string operations.                                                                                            |
| compute    | compute functions.                                                                                            |
| temporal   | support for temporal data types: `Datetime`, `Date`, and `Time`.                                              |
| bigidx     | support for large arrays and indexes in the code.                                                             |
| performant | optimizations for performance _(at the expense of compile time)_                                              |
| like       | `like` operations.                                                                                            |
| timezones  | timezone parsing                                                                                              |
| simd       | the use of SIMD instructions for faster processing.                                                           |
