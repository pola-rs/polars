# Introduction

This User Guide is an introduction to the [Polars DataFrame library](https://github.com/pola-rs/polars). Its goal is to introduce you to Polars by going through examples and comparing it to other
solutions. Some design choices are introduced here. The guide will also introduce you to optimal usage of Polars.

Even though Polars is completely written in [Rust](https://www.rust-lang.org/) (no runtime overhead!) and uses [Arrow](https://arrow.apache.org/) -- the
[native arrow2 Rust implementation](https://github.com/jorgecarleitao/arrow2) -- as its foundation, the examples presented in this guide will be mostly using its higher-level language
bindings. Higher-level bindings only serve as a thin wrapper for functionality implemented in the core library.

For [pandas](https://pandas.pydata.org/) users, our [Python package](https://pypi.org/project/polars/) will offer the easiest way to get started with Polars.

### Philosophy

The goal of Polars is to provide a lightning fast `DataFrame` library that:

- Utilizes all available cores on your machine.
- Optimizes queries to reduce unneeded work/memory allocations.
- Handles datasets much larger than your available RAM.
- Has an API that is consistent and predictable.
- Has a strict schema (data-types should be known before running the query).

Polars is written in Rust which gives it C/C++ performance and allows it to fully control performance critical parts
in a query engine.

As such Polars goes to great lengths to:

- Reduce redundant copies.
- Traverse memory cache efficiently.
- Minimize contention in parallelism.
- Process data in chunks.
- Reuse memory allocations.
