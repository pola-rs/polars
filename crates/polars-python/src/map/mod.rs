pub mod dataframe;
pub mod lazy;
pub mod series;

use std::collections::BTreeMap;

use polars::chunked_array::builder::get_list_builder;
use polars::prelude::*;
use polars_core::export::rayon::prelude::*;
use polars_core::utils::CustomIterTools;
use polars_core::POOL;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::PyDict;
use smartstring::alias::String as SmartString;

use crate::error::PyPolarsErr;
use crate::prelude::ObjectValue;
use crate::{PySeries, Wrap};

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

fn iterator_to_struct<'a>(
    it: impl Iterator<Item = Option<Bound<'a, PyAny>>>,
    init_null_count: usize,
    first_value: AnyValue<'a>,
    name: &str,
    capacity: usize,
) -> PyResult<PySeries> {
    let (vals, flds) = match &first_value {
        av @ AnyValue::Struct(_, _, flds) => (av._iter_struct_av().collect::<Vec<_>>(), &**flds),
        AnyValue::StructOwned(payload) => (payload.0.clone(), &*payload.1),
        _ => {
            return Err(crate::exceptions::ComputeError::new_err(format!(
                "expected struct got {first_value:?}",
            )))
        },
    };

    // Every item in the struct is kept as its own buffer of AnyValues.
    // So a struct with 2 items: {a, b} will have:
    // [
    //      [ a values ]
    //      [ b values ]
    // ]
    let mut struct_fields: BTreeMap<SmartString, Vec<AnyValue>> = BTreeMap::new();

    // As a BTreeMap sorts its keys, we also need to track the original
    // order of the field names.
    let mut field_names_ordered: Vec<SmartString> = Vec::with_capacity(flds.len());

    // Use the first value and the known null count to initialize the buffers
    // if we find a new key later on, we make a new entry in the BTree.
    for (value, fld) in vals.into_iter().zip(flds) {
        let mut buf = Vec::with_capacity(capacity);
        buf.extend((0..init_null_count).map(|_| AnyValue::Null));
        buf.push(value);
        field_names_ordered.push(fld.name().clone());
        struct_fields.insert(fld.name().clone(), buf);
    }

    for dict in it {
        match dict {
            None => {
                for field_items in struct_fields.values_mut() {
                    field_items.push(AnyValue::Null);
                }
            },
            Some(dict) => {
                let dict = dict.downcast::<PyDict>()?;
                let current_len = struct_fields
                    .values()
                    .next()
                    .map(|buf| buf.len())
                    .unwrap_or(0);

                // We ignore the keys of the rest of the dicts,
                // the first item determines the output name.
                for (key, val) in dict.iter() {
                    let key = key.str().unwrap().extract::<PyBackedStr>().unwrap();
                    let item = val.extract::<Wrap<AnyValue>>()?;
                    if let Some(buf) = struct_fields.get_mut(&*key) {
                        buf.push(item.0);
                    } else {
                        let mut buf = Vec::with_capacity(capacity);
                        buf.extend((0..init_null_count + current_len).map(|_| AnyValue::Null));
                        buf.push(item.0);
                        let key: SmartString = (&*key).into();
                        field_names_ordered.push(key.clone());
                        struct_fields.insert(key, buf);
                    };
                }

                // Add nulls to keys that were not in the dict.
                if dict.len() < struct_fields.len() {
                    let current_len = current_len + 1;
                    for buf in struct_fields.values_mut() {
                        if buf.len() < current_len {
                            buf.push(AnyValue::Null)
                        }
                    }
                }
            },
        }
    }

    let fields = POOL.install(|| {
        field_names_ordered
            .par_iter()
            .map(|name| Series::new(name, struct_fields.get(name).unwrap()))
            .collect::<Vec<_>>()
    });

    Ok(StructChunked::from_series(name, &fields)
        .unwrap()
        .into_series()
        .into())
}

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
    // SAFETY: we know the iterators len.
    let ca: ChunkedArray<T> = unsafe {
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
    ca.with_name(name)
}

fn iterator_to_bool(
    it: impl Iterator<Item = Option<bool>>,
    init_null_count: usize,
    first_value: Option<bool>,
    name: &str,
    capacity: usize,
) -> ChunkedArray<BooleanType> {
    // SAFETY: we know the iterators len.
    let ca: BooleanChunked = unsafe {
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
    ca.with_name(name)
}

#[cfg(feature = "object")]
fn iterator_to_object(
    it: impl Iterator<Item = Option<ObjectValue>>,
    init_null_count: usize,
    first_value: Option<ObjectValue>,
    name: &str,
    capacity: usize,
) -> ObjectChunked<ObjectValue> {
    // SAFETY: we know the iterators len.
    let ca: ObjectChunked<ObjectValue> = unsafe {
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
    ca.with_name(name)
}

fn iterator_to_string<S: AsRef<str>>(
    it: impl Iterator<Item = Option<S>>,
    init_null_count: usize,
    first_value: Option<S>,
    name: &str,
    capacity: usize,
) -> StringChunked {
    // SAFETY: we know the iterators len.
    let ca: StringChunked = unsafe {
        if init_null_count > 0 {
            (0..init_null_count)
                .map(|_| None)
                .chain(std::iter::once(first_value))
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
    ca.with_name(name)
}

fn iterator_to_list(
    dt: &DataType,
    it: impl Iterator<Item = Option<Series>>,
    init_null_count: usize,
    first_value: Option<&Series>,
    name: &str,
    capacity: usize,
) -> PyResult<ListChunked> {
    let mut builder =
        get_list_builder(dt, capacity * 5, capacity, name).map_err(PyPolarsErr::from)?;
    for _ in 0..init_null_count {
        builder.append_null()
    }
    builder
        .append_opt_series(first_value)
        .map_err(PyPolarsErr::from)?;
    for opt_val in it {
        match opt_val {
            None => builder.append_null(),
            Some(s) => {
                if s.len() == 0 && s.dtype() != dt {
                    builder
                        .append_series(&Series::full_null("", 0, dt))
                        .unwrap()
                } else {
                    builder.append_series(&s).map_err(PyPolarsErr::from)?
                }
            },
        }
    }
    Ok(builder.finish())
}
