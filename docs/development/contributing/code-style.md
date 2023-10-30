# Code style

This page contains some guidance on code style.

!!! info

    Additional information will be added to this page later.

## Rust

### Naming conventions

Naming conventions for variables:

```rust
let s: Series = ...
let ca: ChunkedArray = ...
let arr: ArrayRef = ...
let arr: PrimitiveArray = ...
let dtype: DataType = ...
let data_type: ArrowDataType = ...
```

### Code example

```rust
use std::ops::Add;

use polars::export::arrow::array::*;
use polars::export::arrow::compute::arity::binary;
use polars::export::arrow::types::NativeType;
use polars::prelude::*;
use polars_core::utils::{align_chunks_binary, combine_validities_or};
use polars_core::with_match_physical_numeric_polars_type;

// Prefer to do the compute closest to the arrow arrays.
// this will tend to be faster as iterators can work directly on slices and don't have
// to go through boxed traits
fn compute_kernel<T>(arr_1: &PrimitiveArray<T>, arr_2: &PrimitiveArray<T>) -> PrimitiveArray<T>
where
    T: Add<Output = T> + NativeType,
{
    // process the null data separately
    // this saves an expensive branch and bitoperation when iterating
    let validity_1 = arr_1.validity();
    let validity_2 = arr_2.validity();

    let validity = combine_validities_or(validity_1, validity_2);

    // process the numerical data as if there were no validities
    let values_1: &[T] = arr_1.values().as_slice();
    let values_2: &[T] = arr_2.values().as_slice();

    let values = values_1
        .iter()
        .zip(values_2)
        .map(|(a, b)| *a + *b)
        .collect::<Vec<_>>();

    PrimitiveArray::from_data_default(values.into(), validity)
}

// Same kernel as above, but uses the `binary` abstraction. Prefer this,
#[allow(dead_code)]
fn compute_kernel2<T>(arr_1: &PrimitiveArray<T>, arr_2: &PrimitiveArray<T>) -> PrimitiveArray<T>
where
    T: Add<Output = T> + NativeType,
{
    binary(arr_1, arr_2, arr_1.data_type().clone(), |a, b| a + b)
}

fn compute_chunked_array_2_args<T: PolarsNumericType>(
    ca_1: &ChunkedArray<T>,
    ca_2: &ChunkedArray<T>,
) -> ChunkedArray<T> {
    // This ensures both ChunkedArrays have the same number of chunks with the
    // same offset and the same length.
    let (ca_1, ca_2) = align_chunks_binary(ca_1, ca_2);
    let chunks = ca_1
        .downcast_iter()
        .zip(ca_2.downcast_iter())
        .map(|(arr_1, arr_2)| compute_kernel(arr_1, arr_2));
    ChunkedArray::from_chunk_iter(ca_1.name(), chunks)
}

pub fn compute_expr_2_args(arg_1: &Series, arg_2: &Series) -> Series {
    // Dispatch the numerical series to `compute_chunked_array_2_args`.
    with_match_physical_numeric_polars_type!(arg_1.dtype(), |$T| {
        let ca_1: &ChunkedArray<$T> = arg_1.as_ref().as_ref().as_ref();
        let ca_2: &ChunkedArray<$T> = arg_2.as_ref().as_ref().as_ref();
        compute_chunked_array_2_args(ca_1, ca_2).into_series()
    })
}
```
