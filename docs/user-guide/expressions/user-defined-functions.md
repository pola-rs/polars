# User-defined functions

!!! warning "Not updated for Python Polars `0.19.0`"

    This section of the user guide still needs to be updated for the latest Polars release.

You should be convinced by now that Polars expressions are so powerful and flexible that there is much less need for custom Python functions
than in other libraries.

Still, you need to have the power to be able to pass an expression's state to a third party library or apply your black box function
over data in Polars.

For this we provide the following expressions:

- `map`
- `apply`

## To `map` or to `apply`.

These functions have an important distinction in how they operate and consequently what data they will pass to the user.

A `map` passes the `Series` backed by the `expression` as is.

`map` follows the same rules in both the `select` and the `group_by` context, this will
mean that the `Series` represents a column in a `DataFrame`. Note that in the `group_by` context, that column is not yet
aggregated!

Use cases for `map` are for instance passing the `Series` in an expression to a third party library. Below we show how
we could use `map` to pass an expression column to a neural network model.

=== ":fontawesome-brands-python: Python"
[:material-api: `map`](https://pola-rs.github.io/polars/py-polars/html/reference/expressions/api/polars.map.html)

```python
df.with_columns([
    pl.col("features").map(lambda s: MyNeuralNetwork.forward(s.to_numpy())).alias("activations")
])
```

=== ":fontawesome-brands-rust: Rust"

```rust
df.with_columns([
    col("features").map(|s| Ok(my_nn.forward(s))).alias("activations")
])
```

Use cases for `map` in the `group_by` context are slim. They are only used for performance reasons, but can quite easily lead to incorrect results. Let me explain why.

{{code_block('user-guide/expressions/user-defined-functions','dataframe',['map'])}}

```python exec="on" result="text" session="user-guide/udf"
--8<-- "python/user-guide/expressions/user-defined-functions.py:setup"
--8<-- "python/user-guide/expressions/user-defined-functions.py:dataframe"
```

In the snippet above we group by the `"keys"` column. That means we have the following groups:

```c
"a" -> [10, 7]
"b" -> [1]
```

If we would then apply a `shift` operation to the right, we'd expect:

```c
"a" -> [null, 10]
"b" -> [null]
```

Now, let's print and see what we've got.

```python
print(out)
```

```
shape: (2, 3)
┌──────┬────────────┬──────────────────┐
│ keys ┆ shift_map  ┆ shift_expression │
│ ---  ┆ ---        ┆ ---              │
│ str  ┆ list[i64]  ┆ list[i64]        │
╞══════╪════════════╪══════════════════╡
│ a    ┆ [null, 10] ┆ [null, 10]       │
│ b    ┆ [7]        ┆ [null]           │
└──────┴────────────┴──────────────────┘
```

Ouch.. we clearly get the wrong results here. Group `"b"` even got a value from group `"a"` 😵.

This went horribly wrong, because the `map` applies the function before we aggregate! So that means the whole column `[10, 7, 1`\] got shifted to `[null, 10, 7]` and was then aggregated.

So my advice is to never use `map` in the `group_by` context unless you know you need it and know what you are doing.

## To `apply`

Luckily we can fix previous example with `apply`. `apply` works on the smallest logical elements for that operation.

That is:

- `select context` -> single elements
- `group by context` -> single groups

So with `apply` we should be able to fix our example:

{{code_block('user-guide/expressions/user-defined-functions','apply',['apply'])}}

```python exec="on" result="text" session="user-guide/udf"
--8<-- "python/user-guide/expressions/user-defined-functions.py:apply"
```

And observe, a valid result! 🎉

## `apply` in the `select` context

In the `select` context, the `apply` expression passes elements of the column to the python function.

_Note that you are now running Python, this will be slow._

Let's go through some examples to see what to expect. We will continue with the `DataFrame` we defined at the start of
this section and show an example with the `apply` function and a counter example where we use the expression API to
achieve the same goals.

### Adding a counter

In this example we create a global `counter` and then add the integer `1` to the global state at every element processed.
Every iteration the result of the increment will be added to the element value.

> Note, this example isn't provided in Rust. The reason is that the global `counter` value would lead to data races when this apply is evaluated in parallel. It would be possible to wrap it in a `Mutex` to protect the variable, but that would be obscuring the point of the example. This is a case where the Python Global Interpreter Lock's performance tradeoff provides some safety guarantees.

{{code_block('user-guide/expressions/user-defined-functions','counter',['apply'])}}

```python exec="on" result="text" session="user-guide/udf"
--8<-- "python/user-guide/expressions/user-defined-functions.py:counter"
```

### Combining multiple column values

If we want to have access to values of different columns in a single `apply` function call, we can create `struct` data
type. This data type collects those columns as fields in the `struct`. So if we'd create a struct from the columns
`"keys"` and `"values"`, we would get the following struct elements:

```python
[
    {"keys": "a", "values": 10},
    {"keys": "a", "values": 7},
    {"keys": "b", "values": 1},
]
```

In Python, those would be passed as `dict` to the calling python function and can thus be indexed by `field: str`. In rust, you'll get a `Series` with the `Struct` type. The fields of the struct can then be indexed and downcast.

{{code_block('user-guide/expressions/user-defined-functions','combine',['apply','struct'])}}

```python exec="on" result="text" session="user-guide/udf"
--8<-- "python/user-guide/expressions/user-defined-functions.py:combine"
```

`Structs` are covered in detail in the next section.

### Return types?

Custom python functions are black boxes for polars. We really don't know what kind of black arts you are doing, so we have
to infer and try our best to understand what you meant.

As a user it helps to understand what we do to better utilize custom functions.

The data type is automatically inferred. We do that by waiting for the first non-null value. That value will then be used
to determine the type of the `Series`.

The mapping of python types to polars data types is as follows:

- `int` -> `Int64`
- `float` -> `Float64`
- `bool` -> `Boolean`
- `str` -> `Utf8`
- `list[tp]` -> `List[tp]` (where the inner type is inferred with the same rules)
- `dict[str, [tp]]` -> `struct`
- `Any` -> `object` (Prevent this at all times)

Rust types map as follows:

- `i32` or `i64` -> `Int64`
- `f32` or `f64` -> `Float64`
- `bool` -> `Boolean`
- `String` or `str` -> `Utf8`
- `Vec<tp>` -> `List[tp]` (where the inner type is inferred with the same rules)
