# Introduction

This getting started guide is written for new users of Polars. The goal is to provide a quick overview of the most common functionality. For a more detailed explanation, please go to the [User Guide](../index.md)

!!! rust "Rust Users Only"

    Due to historical reasons the eager API in Rust is outdated. In the future we would like to redesign it as a small wrapper around the lazy API (as is the design in Python / NodeJS). In the  examples we will use the lazy API instead with `.lazy()` and `.collect()`. For now you can ignore these two functions. If you want to know more about the lazy and eager API go [here](../concepts/lazy-vs-eager.md).

    To enable the Lazy API ensure you have the feature flag `lazy` configured when installing Polars
    ```
    # Cargo.toml
    [dependencies]
    polars = { version = "x", features = ["lazy", ...]}
    ```

    Because of the ownership ruling in Rust we can not reuse the same `DataFrame` multiple times in the examples. For simplicity reasons we call `clone()` to overcome this issue. Note that this does not duplicate the data but just increments a pointer (`Arc`).
