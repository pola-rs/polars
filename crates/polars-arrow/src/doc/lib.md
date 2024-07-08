Welcome to polars_arrow's documentation. Thanks for checking it out!

This is a library for efficient in-memory data operations with
[Arrow in-memory format](https://arrow.apache.org/docs/format/Columnar.html).
It is a re-write from the bottom up of the official `arrow` crate with soundness
and type safety in mind.

Check out [the guide](https://jorgecarleitao.github.io/polars_arrow/main/guide/) for an introduction.
Below is an example of some of the things you can do with it:

```rust
use std::sync::Arc;

use polars_arrow::array::*;
use polars_arrow::datatypes::{Field, DataType, Schema};
use polars_arrow::compute::arithmetics;
use polars_arrow::error::Result;
use polars_arrow::io::parquet::write::*;
use polars_arrow::chunk::Chunk;

fn main() -> Result<()> {
    // declare arrays
    let a = Int32Array::from(&[Some(1), None, Some(3)]);
    let b = Int32Array::from(&[Some(2), None, Some(6)]);

    // compute (probably the fastest implementation of a nullable op you can find out there)
    let c = arithmetics::basic::mul_scalar(&a, &2);
    assert_eq!(c, b);

    // declare a schema with fields
    let schema = Schema::from(vec![
        Field::new("c1", DataType::Int32, true),
        Field::new("c2", DataType::Int32, true),
    ]);

    // declare chunk
    let chunk = Chunk::new(vec![a.arced(), b.arced()]);

    // write to parquet (probably the fastest implementation of writing to parquet out there)

    let options = WriteOptions {
        write_statistics: true,
        compression: CompressionOptions::Snappy,
        version: Version::V1,
        data_page_size: None,
    };

    let row_groups = RowGroupIterator::try_new(
        vec![Ok(chunk)].into_iter(),
        &schema,
        options,
        vec![vec![Encoding::Plain], vec![Encoding::Plain]],
    )?;

    // anything implementing `std::io::Write` works
    let mut file = vec![];

    let mut writer = FileWriter::try_new(file, schema, options)?;

    // Write the file.
    for group in row_groups {
        writer.write(group?)?;
    }
    let _ = writer.end(None)?;
    Ok(())
}
```

## Cargo features

This crate has a significant number of cargo features to reduce compilation
time and number of dependencies. The feature `"full"` activates most
functionality, such as:

- `io_ipc`: to interact with the Arrow IPC format
- `io_ipc_compression`: to read and write compressed Arrow IPC (v2)
- `io_flight` to read and write to Arrow's Flight protocol
- `compute` to operate on arrays (addition, sum, sort, etc.)

The feature `simd` (not part of `full`) produces more explicit SIMD instructions
via [`std::simd`](https://doc.rust-lang.org/nightly/std/simd/index.html), but requires the
nightly channel.
