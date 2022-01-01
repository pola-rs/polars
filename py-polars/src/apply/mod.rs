pub mod dataframe;
pub mod series;
use crate::prelude::ObjectValue;
use polars::chunked_array::builder::get_list_builder;
use polars::prelude::*;
use polars_core::utils::CustomIterTools;

pub trait PyArrowPrimitiveType: PolarsNumericType {}

impl PyArrowPrimitiveType for UInt8Type {}
impl PyArrowPrimitiveType for UInt16Type {}
impl PyArrowPrimitiveType for UInt32Type {}
impl PyArrowPrimitiveType for UInt64Type {}
impl PyArrowPrimitiveType for Int8Type {}
impl PyArrowPrimitiveType for Int16Type {}
impl PyArrowPrimitiveType for Int32Type {}
impl PyArrowPrimitiveType for Int64Type {}
impl PyArrowPrimitiveType for Float32Type {}
impl PyArrowPrimitiveType for Float64Type {}

fn iterator_to_primitive<T>(
    it: impl Iterator<Item = Option<T::Native>>,
    init_null_count: usize,
    first_value: Option<T::Native>,
    name: &str,
    capacity: usize,
) -> ChunkedArray<T>
where
    T: PyArrowPrimitiveType,
{
    // safety: we know the iterators len
    let mut ca: ChunkedArray<T> = unsafe {
        if init_null_count > 0 {
            (0..init_null_count)
                .map(|_| None)
                .chain(std::iter::once(first_value))
                .chain(it)
                .trust_my_length(capacity)
                .collect_trusted()
        } else if first_value.is_some() {
            std::iter::once(first_value)
                .chain(it)
                .trust_my_length(capacity)
                .collect_trusted()
        } else {
            it.collect()
        }
    };
    debug_assert_eq!(ca.len(), capacity);
    ca.rename(name);
    ca
}

fn iterator_to_bool(
    it: impl Iterator<Item = Option<bool>>,
    init_null_count: usize,
    first_value: Option<bool>,
    name: &str,
    capacity: usize,
) -> ChunkedArray<BooleanType> {
    // safety: we know the iterators len
    let mut ca: BooleanChunked = unsafe {
        if init_null_count > 0 {
            (0..init_null_count)
                .map(|_| None)
                .chain(std::iter::once(first_value))
                .chain(it)
                .trust_my_length(capacity)
                .collect_trusted()
        } else if first_value.is_some() {
            std::iter::once(first_value)
                .chain(it)
                .trust_my_length(capacity)
                .collect_trusted()
        } else {
            it.collect()
        }
    };
    debug_assert_eq!(ca.len(), capacity);
    ca.rename(name);
    ca
}

fn iterator_to_object(
    it: impl Iterator<Item = Option<ObjectValue>>,
    init_null_count: usize,
    first_value: Option<ObjectValue>,
    name: &str,
    capacity: usize,
) -> ObjectChunked<ObjectValue> {
    // safety: we know the iterators len
    let mut ca: ObjectChunked<ObjectValue> = unsafe {
        if init_null_count > 0 {
            (0..init_null_count)
                .map(|_| None)
                .chain(std::iter::once(first_value))
                .chain(it)
                .trust_my_length(capacity)
                .collect_trusted()
        } else if first_value.is_some() {
            std::iter::once(first_value)
                .chain(it)
                .trust_my_length(capacity)
                .collect_trusted()
        } else {
            it.collect()
        }
    };
    debug_assert_eq!(ca.len(), capacity);
    ca.rename(name);
    ca
}

fn iterator_to_utf8<'a>(
    it: impl Iterator<Item = Option<&'a str>>,
    init_null_count: usize,
    first_value: Option<&'a str>,
    name: &str,
    capacity: usize,
) -> Utf8Chunked {
    // safety: we know the iterators len
    let mut ca: Utf8Chunked = unsafe {
        if init_null_count > 0 {
            (0..init_null_count)
                .map(|_| None)
                .chain(std::iter::once(first_value))
                .chain(it)
                .trust_my_length(capacity)
                .collect_trusted()
        } else if first_value.is_some() {
            std::iter::once(first_value)
                .chain(it)
                .trust_my_length(capacity)
                .collect_trusted()
        } else {
            it.collect()
        }
    };
    debug_assert_eq!(ca.len(), capacity);
    ca.rename(name);
    ca
}

fn iterator_to_list(
    dt: &DataType,
    it: impl Iterator<Item = Option<Series>>,
    init_null_count: usize,
    first_value: Option<&Series>,
    name: &str,
    capacity: usize,
) -> ListChunked {
    let mut builder = get_list_builder(dt, capacity * 5, capacity, name);
    for _ in 0..init_null_count {
        builder.append_null()
    }
    builder.append_opt_series(first_value);
    for opt_val in it {
        builder.append_opt_series(opt_val.as_ref())
    }
    builder.finish()
}
