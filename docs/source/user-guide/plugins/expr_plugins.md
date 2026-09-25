# Expression Plugins

<!-- (TODO) the steps outlined here aren't clear and may not work that well -->

Expression plugins are the preferred way to create user defined functions. They allow you to compile
a Rust function and register that as an expression into the Polars library. The Polars engine will
dynamically link your function at runtime and your expression will run almost as fast as native
expressions. Note that this works without any interference of Python and thus no GIL contention.

They will benefit from the same benefits default expressions have:

- Optimization
- Parallelism
- Rust native performance

## Background: How does a plugin interact with Polars?

In short, there are three important things to always keep in mind while building a plugin:

1. The plugin contains its own copy of the core polars functionality. It communicates with the full
   Polars install you normally use.
2. Data is not copied, but re-interpreted every time you go from general polars to functionality
   defined in your plugin.
3. Anything global, such as type information, must be known by both.

A plugin is a `cdylib`. It statically links its own copy of `polars-core`. The `polars` wheel in
your environment contains another (independent) copy. Both copies are loaded into the same process,
but they are separate machine code with separate statics.

```text
python process
├── Host; polars wheel (_polars_runtime_64.so)
└── your_plugin.so
```

The **Polars wheel (host)** is responsible for the majority of work as normal: Things like reading
files, building Series, and displaying a DataFrame.

The **Plugin** is only responsible for your custom expressions. This includes conversion to- and
from the data types that they take.

The two copies never share a `Series`, a `DataType`, or a global variable. They talk over the Arrow
C data interface only: a `SeriesExport` struct holding an `ArrowSchema` and one or more `ArrowArray`
pointers. Everything that crosses the boundary is encoded on one- , and decoded on the other side.

## Our first custom expression: Pig Latin

For our first expression we are going to create a pig latin converter. Pig latin is a silly language
where in every word the first letter is removed, added to the back and finally "ay" is added. So the
word "pig" would convert to "igpay".

We could of course already do that with expressions, e.g.
`col("name").str.slice(1) + col("name").str.slice(0, 1) + "ay"`, but a specialized function for this
would perform better and allows us to learn about the plugins.

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
pyo3 = { version = "*", features = ["extension-module", "abi3-py310"] }
pyo3-polars = { version = "*", features = ["derive"] }
serde = { version = "*", features = ["derive"] }
```

### Writing the expression

In this library we create a helper function that converts a `&str` to pig-latin, and we create the
function that we will expose as an expression. To expose a function we must add the
`#[polars_expr(output_type=DataType)]` attribute from `pyo3-polars`, and the function must always
accept `inputs: &[Series]` as its first argument.

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

#[polars_expr(output_type=String)]
fn pig_latinnify(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let out: StringChunked = ca.apply_into_string_amortized(pig_latin_str);
    Ok(out.into_series())
}
```

Note that we use `apply_into_string_amortized`, as opposed to `apply_values`, to avoid allocating a
new string for each row. If your plugin takes in multiple inputs, operates elementwise, and produces
a `String` output, then you may want to look at the `binary_elementwise_into_string_amortized`
utility function in `polars::prelude::arity`.

This is all that is needed on the Rust side. On the Python side we must setup a folder with the same
name as defined in the `Cargo.toml`, in this case "expression_lib". We will create a folder in the
same directory as our Rust `src` folder named `expression_lib` and we create an
`expression_lib/__init__.py`. The resulting file structure should look something like this:

```text
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

Then we create new expressions. The function name of our expression can be registered. Note that it
is important that this name is correct, otherwise the main Polars package cannot resolve the
function name. Furthermore we can set additional keyword arguments that explain to Polars how this
expression behaves. In this case we tell Polars that this function is elementwise. This allows
Polars to run this expression in batches. Whereas for other operations this would not be allowed,
think for instance of a sort, or a slice.

```python
# expression_lib/__init__.py
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function
from polars._typing import IntoExpr

PLUGIN_PATH = Path(__file__).parent

def pig_latinnify(expr: IntoExpr) -> pl.Expr:
    """Pig-latinnify expression."""
    return register_plugin_function(
        plugin_path=PLUGIN_PATH,
        function_name="pig_latinnify",
        args=expr,
        is_elementwise=True,
    )
```

We can then compile this library in our environment by installing `maturin` and running
`maturin develop --release`.

And that's it. Our expression is ready to use!

```python
import polars as pl
from expression_lib import pig_latinnify

df = pl.DataFrame(
    {
        "convert": ["pig", "latin", "is", "silly"],
    }
)
out = df.with_columns(pig_latin=pig_latinnify("convert"))
```

Alternatively, you can
[register a custom namespace](https://docs.pola.rs/api/python/stable/reference/api/polars.api.register_expr_namespace.html#polars.api.register_expr_namespace),
which enables you to create a `Expr.language` namespace, allowing users to write:

```python
out = df.with_columns(
    pig_latin=pl.col("convert").language.pig_latinnify(),
)
```

## Accepting kwargs

If you want to accept `kwargs` (keyword arguments) in a polars expression, all you have to do is
define a Rust `struct` and make sure that it derives `serde::Deserialize`.

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
/// with the pickle protocol (on the Rust side).
#[polars_expr(output_type=String)]
fn append_kwargs(input: &[Series], kwargs: MyKwargs) -> PolarsResult<Series> {
    let input = &input[0];
    let input = input.cast(&DataType::String)?;
    let ca = input.str().unwrap();

    Ok(ca
        .apply_into_string_amortized(|val, buf| {
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

On the Python side the kwargs can be passed when we register the plugin.

```python
def append_args(
    expr: IntoExpr,
    float_arg: float,
    integer_arg: int,
    string_arg: str,
    boolean_arg: bool,
) -> pl.Expr:
    """
    This example shows how arguments other than `Series` can be used.
    """
    return register_plugin_function(
        plugin_path=PLUGIN_PATH,
        function_name="append_kwargs",
        args=expr,
        kwargs={
            "float_arg": float_arg,
            "integer_arg": integer_arg,
            "string_arg": string_arg,
            "boolean_arg": boolean_arg,
        },
        is_elementwise=True,
    )
```

## Output data types

Polars must know the result `dtype` of your expression before running anything. Output data types of
course don't have to be fixed. They often depend on the input types of an expression. This is also
the right place to reject bad input. It runs during schema resolution before any data moves, so the
user sees the error immediately.

You declare it in the `#[polars_expr()]` attribute, in one of three ways:

| Attribute                            | Use when                         |
| ------------------------------------ | -------------------------------- |
| `output_type=Float64`                | The dtype is fixed.              |
| `output_type_func=my_fn`             | The dtype depends on the inputs. |
| `output_type_func_with_kwargs=my_fn` | It also depends on the kwargs.   |

The function types are very powerful: they can map input fields `&[Field]` to an output `Field`
(name and data type).

In the snippet below is an example where we define a function, and use the utility `FieldsMapper` to
help with this mapping. This approach helps with common cases such as taking the supertype of the
inputs.

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

Alternatively, we can define a function ourselves. This is very useful if we use an arrow extension
type (which we will cover later).

```rust
fn same_point_type(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    let DataType::Extension(typ, _) = field.dtype() else {
        polars_bail!(
            SchemaMismatch: "expected a `geoarrow.point` column, got: {}", field.dtype()
        );
    };
    if (&*typ.0 as &dyn Any).downcast_ref::<PointXY>().is_none() {
        polars_bail!(
            SchemaMismatch: "expected a 2D `geoarrow.point` column, got: {}", field.dtype()
        );
    }
    Ok(field.clone())
}
```

## Registering Arrow extension types

> Registering arrow types is **unstable** This guide was written for Polars v0.55. It might not
> accurately reflect the current way of doing things!

An Arrow extension type contains the following:

- **Name**
- **Storage type** describes the layout. It is visible to every Arrow consumer, whether or not they
  know your type.
- (optional) **Metadata**, one opaque string. Usually encoded as JSON. It is only meaningful to
  something that knows your type. Put a property in metadata only when it cannot be read off the
  storage, e.g. coordinate system.

Arrow carries both the name and optional metadata in the field's metadata, under the keys
`ARROW:extension:name` and `ARROW:extension:metadata`.

Polars models this as:

```rust
DataType::Extension(ExtensionTypeInstance, Box<DataType>)
//                  ^ the type              ^ the storage
```

The `DataType::Extension` variant is behind the **`dtype-extension`** feature-flag. If you use a
struct, you might also want to use `dtype-struct`. Enable it on both `polars` and `polars-core` in
your `Cargo.toml`. If it is not enabled, an incoming extension type will silently decay to its
storage type.

Polars needs to be made aware of each extension name. This is done using a **Factory** that turns
`(name, storage, metadata)` into a type instance.

Remember from before: there are **two copies** of `polars-core`, so we need to register them twice:

- **Polars wheel (host)**: `pl.register_extension_type(name, cls)` in Python
- **Plugin**: `polars_core::datatypes::extension::register_extension_type` in Rust

If the Python side is registered and the Rust side is not, your data still round-trips, but inside
the plugin the column arrives as a plain storage type.

### The Rust side: two traits

**`ExtensionTypeFactory`** must be implemented to define the interpretation of arrow extension data
into your custom types. One factory serves one extension name, but if a name covers several shapes,
such as 2D and 3D points, you can dynamically pick here:

```rust
impl ExtensionTypeFactory for PointFactory {
    fn create_type_instance(
        &self,
        _name: &str,
        storage: &DataType,
        metadata: Option<&str>,
    ) -> Box<dyn ExtensionTypeImpl> {
        if is_xy_storage(storage) {
            Box::new(PointXY { metadata: metadata.map(str::to_owned) })
        } else {
            Box::new(UnsupportedPoint { /* ... */ })
        }
    }
}
```

You can now define a type that represents the arrow extension type, and implement
**`ExtensionTypeImpl`** for it.

```rust
impl ExtensionTypeImpl for PointXY {
    fn name(&self) -> Cow<'_, str> { Cow::Borrowed("geoarrow.point") }

    // Produces ARROW:extension:metadata. Return None to omit it.
    fn serialize_metadata(&self) -> Option<Cow<'_, str>> {
        self.metadata.as_deref().map(Cow::Borrowed)
    }

    fn dyn_clone(&self) -> Box<dyn ExtensionTypeImpl> { Box::new(self.clone()) }

    fn dyn_eq(&self, other: &dyn ExtensionTypeImpl) -> bool {
        (other as &dyn Any).downcast_ref::<Self>().is_some_and(|o| self == o)
    }

    fn dyn_hash(&self) -> u64 { /* hash self */ }

    fn dyn_display(&self) -> Cow<'_, str> { Cow::Borrowed("point[xy]") }  // shown as ext[point[xy]]
    fn dyn_debug(&self) -> Cow<'_, str> { Cow::Borrowed("PointXY") }
}
```

`ExtensionTypeImpl` requires `Any`, which is what makes `dyn_eq` work. It is also how expressions
recover the concrete type:

```rust
if (&*typ.0 as &dyn Any).downcast_ref::<PointXY>().is_none() {
    polars_bail!(SchemaMismatch: "expected a 2D point, got: {}", s.dtype());
}
```

Now we need to expose the registration hook, which must run immediately as described before. We do
that with the `PyInit_my_plugin_name_lib`. This hook is run when Python imports the extension
module.

The easiest way to hook into it is by using `#[pymodule]`. Note that the function name must match
the **module name** as seen by maturin. Depending on your setup, this might be **different** from
your crate name.

```rust
// in your lib.rs
#[pymodule]
fn my_plugin_name_lib(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    geometry::register().map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "failed to register geoarrow extension types: {e}"
        ))
    })
}

mod geometry {
    pub fn register() -> PolarsResult<()> {
        register_extension_type("geoarrow.point", Some(Arc::new(PointFactory)))
    }
}
```

### Using arrow extension types in plugin expressions

There are two helpers to move between an extension Series and its storage:

```rust
let storage: &Series = s.ext()?.storage(); 
let out: Series = struct_series.into_extension(typ);
```

Re-wrap with the type instance you were **given**, not a freshly built one. This preserves whatever
metadata came in, so a property like a coordinate system survives the operation instead of being
reset.

### The Python side: one class

Python mirrors the Rust registry with a subclass of `pl.datatypes.BaseExtension`:

```python
class PointXY(pl.datatypes.BaseExtension):
    def __init__(self) -> None:
        super().__init__(
            name="geoarrow.point",
            storage=pl.Struct({"x": pl.Float64, "y": pl.Float64}),
            metadata=None,
        )

    def _string_repr(self) -> str:
        return "point[xy]"   # shown as ext[point[xy]] in a DataFrame header

pl.register_extension_type("geoarrow.point", PointXY)
```

`register_extension_type` takes **one class per name**. When a name covers several shapes (e.g.
points of n dimensions), register a base class and override `ext_from_params`.

```python
@classmethod
def ext_from_params(cls, name, storage, metadata):
    target = _BY_DIMENSION.get(_dimension_of(storage))
    if target is None:
        raise ValueError(f"unsupported {name!r} storage: {storage!r}")

    # Bypasses __init__ on purpose, so metadata is carried through verbatim.
    slf = target.__new__(target)
    slf._name, slf._storage, slf._metadata = name, storage, metadata
    return slf
```

To register the type without a custom class, pass the built-in `pl.Extension`. To pass a name
straight through to its storage, use `as_storage=True`.

### Constructing and unwrapping from Python

Both `Series` and `Expr` have an `.ext` namespace:

```python
# label storage with the type
pl.struct(x=..., y=...).ext.to(PointXY())
# get the storage back
pl.col("geo").ext.storage()
```

`.ext.to()` only relabels. It does not convert, and the input must already be the exact storage
type. This does the operation only on the metadata. As no operations are done on your data, so this
is very cheap. This means a constructor needs no plugin call at all:

```python
def point(x, y) -> pl.Expr:
    return pl.struct(
        pl.col(x).cast(pl.Float64).alias("x"),
        pl.col(y).cast(pl.Float64).alias("y"),
    ).ext.to(PointXY())
```

---

That's all you need to know to get started. Take a look at
[this repo](https://github.com/pola-rs/pyo3-polars/tree/main/example/derive_expression) to see how
this all fits together, and at
[this tutorial](https://marcogorelli.github.io/polars-plugins-tutorial/) to gain a more thorough
understanding.
