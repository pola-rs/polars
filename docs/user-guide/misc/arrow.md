# Arrow producer/consumer

## Using pyarrow

Polars can move data in and out of arrow zero copy. This can be done either via pyarrow
or natively. Let's first start by showing the pyarrow solution:

{{code_block('user-guide/misc/arrow','to_arrow',[])}}

```
pyarrow.Table
foo: int64
bar: large_string
----
foo: [[1,2,3]]
bar: [["ham","spam","jam"]]
```

Or if you want to ensure the output is zero-copy:

{{code_block('user-guide/misc/arrow','to_arrow_zero',[])}}

```
pyarrow.Table
foo: int64
bar: string_view
----
foo: [[1,2,3]]
bar: [["ham","spam","jam"]]
```

Importing from pyarrow can be achieved with `pl.from_arrow`.

## Using Polars directly

Polars can also consume and export to and import from the [Arrow C Data Interface](https://arrow.apache.org/docs/format/CDataInterface.html)
directly. This is recommended for library maintainers that want to interop with Polars without requiring a pyarrow installation.

- To export `ArrowArray` C structs, Polars exposes: `Series._export_arrow_to_c`.
- To import an `ArrowArray` C struct, Polars exposes `Series._import_arrow_from_c`.
