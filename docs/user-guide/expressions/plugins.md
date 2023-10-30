# Expression plugins

Expression plugins are the preferred way to create user defined functions. They allow you to compile a rust function
and register that as an expression into the polars library. The polars engine will dynamically link your function at runtime
and your expression will run almost as fast as native expressions. Note that this works without any interference of python
and thus no GIL contention.

They will benefit from the same benefits default expressions have:

- Optimization
- Parallelism
- Rust native performance

To get started we will see what is needed to create a custom expression.

## Our first custom expression: Pig Latin

For our first expression we are going to create a pig latin converter. Pig latin is a silly language where in every word
the first letter is removed, added to the back and finally "ay" is added. So the word "pig" would convert to "igpay".

We could of course already do that with expressions, e.g. `col("name").str.slice(1) + col("name").str.slice(0, 1) + "ay"`,
but a specialized function for this would perform better and allows us to learn about the plugins.

### Setting up

We start with a new library as the following `Cargo.toml` file

```toml
[package]
name = "expression_lib"
version = "0.1.0"
edition = "2021"

[lib]
name = "expression_lib"
crate-type = ["cdylib"]

[dependencies]
polars = { version = "*" }
pyo3 = { version = "*", features = ["extension-module"] }
pyo3-polars = { version = "*", features = ["derive"] }
serde = { version = "*", features = ["derive"] }
```

### Writing the expression

In this library we create a helper function that converts a `&str` to pig-latin, and we create the function that we will
expose as an expression. To expose a function we must add the `#[polars_expr(output_type=DataType)]` attribute and the function
must always accept `inputs: &[Series]` as its first argument.

```rust
// src/expressions.rs
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use std::fmt::Write;

fn pig_latin_str(value: &str, output: &mut String) {
    if let Some(first_char) = value.chars().next() {
        write!(output, "{}{}ay", &value[1..], first_char).unwrap()
    }
}

#[polars_expr(output_type=Utf8)]
fn pig_latinnify(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].utf8()?;
    let out: Utf8Chunked = ca.apply_to_buffer(pig_latin_str);
    Ok(out.into_series())
}
```

This is all that is needed on the rust side. On the python side we must setup a folder with the same name as defined in
the `Cargo.toml`, in this case "expression_lib". We will create a folder in the same directory as our rust `src` folder
named `expression_lib` and we create an `expression_lib/__init__.py`. The resulting file structure should look something like this:

```
├── 📁 expression_lib/  # name must match "lib.name" in Cargo.toml
|   └── __init__.py
|
├── 📁src/
|   ├── lib.rs
|   └── expressions.rs
|
├── Cargo.toml
└── pyproject.toml
```

Then we create a new class `Language` that will hold the expressions for our new `expr.language` namespace. The function
name of our expression can be registered. Note that it is important that this name is correct, otherwise the main polars
package cannot resolve the function name. Furthermore we can set additional keyword arguments that explain to polars how
this expression behaves. In this case we tell polars that this function is elementwise. This allows polars to run this
expression in batches. Whereas for other operations this would not be allowed, think for instance of a sort, or a slice.

```python
# expression_lib/__init__.py
import polars as pl
from polars.type_aliases import IntoExpr
from polars.utils.udfs import _get_shared_lib_location

# boilerplate needed to inform polars of the location of binary wheel.
lib = _get_shared_lib_location(__file__)

@pl.api.register_expr_namespace("language")
class Language:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def pig_latinnify(self) -> pl.Expr:
        return self._expr._register_plugin(
            lib=lib,
            symbol="pig_latinnify",
            is_elementwise=True,
        )
```

We can then compile this library in our environment by installing `maturin` and running `maturin develop --release`.

And that's it. Our expression is ready to use!

```python
import polars as pl
from expression_lib import Language

df = pl.DataFrame(
    {
        "convert": ["pig", "latin", "is", "silly"],
    }
)


out = df.with_columns(
    pig_latin=pl.col("convert").language.pig_latinnify(),
)
```

## Accepting kwargs

If you want to accept `kwargs` (keyword arguments) in a polars expression, all you have to do is define a rust `struct`
and make sure that it derives `serde::Deserialize`.

```rust
/// Provide your own kwargs struct with the proper schema and accept that type
/// in your plugin expression.
#[derive(Deserialize)]
pub struct MyKwargs {
    float_arg: f64,
    integer_arg: i64,
    string_arg: String,
    boolean_arg: bool,
}

/// If you want to accept `kwargs`. You define a `kwargs` argument
/// on the second position in you plugin. You can provide any custom struct that is deserializable
/// with the pickle protocol (on the rust side).
#[polars_expr(output_type=Utf8)]
fn append_kwargs(input: &[Series], kwargs: MyKwargs) -> PolarsResult<Series> {
    let input = &input[0];
    let input = input.cast(&DataType::Utf8)?;
    let ca = input.utf8().unwrap();

    Ok(ca
        .apply_to_buffer(|val, buf| {
            write!(
                buf,
                "{}-{}-{}-{}-{}",
                val, kwargs.float_arg, kwargs.integer_arg, kwargs.string_arg, kwargs.boolean_arg
            )
                .unwrap()
        })
        .into_series())
}
```

On the python side the kwargs can be passed when we register the plugin.

```python
@pl.api.register_expr_namespace("my_expr")
class MyCustomExpr:
    def __init__(self, expr: pl.Expr):
        self._expr = expr
        
    def append_args(
            self,
            float_arg: float,
            integer_arg: int,
            string_arg: str,
            boolean_arg: bool,
    ) -> pl.Expr:
        """
        This example shows how arguments other than `Series` can be used.
        """
        return self._expr._register_plugin(
            lib=lib,
            args=[],
            kwargs={
                "float_arg": float_arg,
                "integer_arg": integer_arg,
                "string_arg": string_arg,
                "boolean_arg": boolean_arg,
            },
            symbol="append_kwargs",
            is_elementwise=True,
        )
```

## Output data types

Output data types ofcourse don't have to be fixed. They often depend on the input types of an expression. To accommodate
this you can provide the `#[polars_expr()]` macro with an `output_type_func` argument that points to a function. This
function can map input fields `&[Field]` to an output `Field` (name and data type).

In the snippet below is an example where we use the utility `FieldsMapper` to help with this mapping.

```rust
use polars_plan::dsl::FieldsMapper;

fn haversine_output(input_fields: &[Field]) -> PolarsResult<Field> {
    FieldsMapper::new(input_fields).map_to_float_dtype()
}

#[polars_expr(output_type_func=haversine_output)]
fn haversine(inputs: &[Series]) -> PolarsResult<Series> {
    let out = match inputs[0].dtype() {
        DataType::Float32 => {
            let start_lat = inputs[0].f32().unwrap();
            let start_long = inputs[1].f32().unwrap();
            let end_lat = inputs[2].f32().unwrap();
            let end_long = inputs[3].f32().unwrap();
            crate::distances::naive_haversine(start_lat, start_long, end_lat, end_long)?
                .into_series()
        }
        DataType::Float64 => {
            let start_lat = inputs[0].f64().unwrap();
            let start_long = inputs[1].f64().unwrap();
            let end_lat = inputs[2].f64().unwrap();
            let end_long = inputs[3].f64().unwrap();
            crate::distances::naive_haversine(start_lat, start_long, end_lat, end_long)?
                .into_series()
        }
        _ => polars_bail!(InvalidOperation: "only supported for float types"),
    };
    Ok(out)
}
```

That's all you need to know to get started. Take a look at this [repo](https://github.com/pola-rs/pyo3-polars/tree/main/example/derive_expression) to see how this all fits together.

## Community plugins

Here is a curated (non-exhaustive) list of community implemented plugins.

- [polars-business](https://github.com/MarcoGorelli/polars-business) Polars extension offering utilities for business day operations
