# polars-plugin

`polars-plugin` contains the Polars expression-plugin and `#[polars_expr]` macro.

`pyo3-polars` re-exports `polars-plugin` APIs for backwards compatibility and provides PyO3
integration and Rust/Python conversion types.

Keyword arguments use Python's pickle representation (`serde-pickle`) because that is the format
used by plugin registration. The ABI is therefore PyO3-independent, but not language-neutral.

## Direct plugin crate

```toml
[lib]
crate-type = ["cdylib"]

[dependencies]
polars-plugin = { version = "*", features = ["dtype-full"] }
serde = { version = "1", features = ["derive"] }
```

```rust
use polars_plugin::prelude::*;
use polars_plugin::polars_expr;

#[polars_expr(output_type = Int64)]
fn double(inputs: &[Series]) -> PolarsResult<Series> {
    Ok(inputs[0].i64()?.apply_values(|v| v * 2).into_series())
}
```

Register the resulting shared library from Python with `polars.plugins.register_plugin_function`. No
`pyo3`, maturin configuration, or Python extension module is needed to build this `cdylib`.

The macro accepts `output_type`, `output_type_func`, and `output_type_func_with_kwargs`. Expression
functions take `inputs: &[Series]` followed by no argument, `kwargs`, `context`, or
`context, kwargs`; the argument names are part of the macro interface. Field callbacks take input
fields, and optionally kwargs.
