//! APIs to read from Parquet format.

mod binview;
mod boolean;
mod categorical;
mod dictionary_encoded;
mod fixed_size_binary;
mod nested;
mod nested_utils;
mod null;
mod primitive;
mod simple;
mod utils;

use arrow::array::{Array, FixedSizeListArray, ListArray, MapArray};
use arrow::bitmap::Bitmap;
use arrow::datatypes::{ArrowDataType, Field};
use arrow::offset::Offsets;
use polars_utils::mmap::MemReader;
use simple::page_iter_to_array;

pub use self::nested_utils::{init_nested, InitNested, NestedState};
pub use self::utils::filter::{Filter, PredicateFilter};
use self::utils::freeze_validity;
use super::*;
use crate::parquet::error::ParquetResult;
use crate::parquet::read::get_page_iterator as _get_page_iterator;
use crate::parquet::schema::types::PrimitiveType;

/// Creates a new iterator of compressed pages.
pub fn get_page_iterator(
    column_metadata: &ColumnChunkMetadata,
    reader: MemReader,
    buffer: Vec<u8>,
    max_header_size: usize,
) -> PolarsResult<PageReader> {
    Ok(_get_page_iterator(
        column_metadata,
        reader,
        buffer,
        max_header_size,
    )?)
}

/// Creates a new [`ListArray`] or [`FixedSizeListArray`].
pub fn create_list(
    dtype: ArrowDataType,
    nested: &mut NestedState,
    values: Box<dyn Array>,
) -> Box<dyn Array> {
    let (length, mut offsets, validity) = nested.pop().unwrap();
    let validity = validity.and_then(freeze_validity);
    match dtype.to_logical_type() {
        ArrowDataType::List(_) => {
            offsets.push(values.len() as i64);

            let offsets = offsets.iter().map(|x| *x as i32).collect::<Vec<_>>();

            let offsets: Offsets<i32> = offsets
                .try_into()
                .expect("i64 offsets do not fit in i32 offsets");

            Box::new(ListArray::<i32>::new(
                dtype,
                offsets.into(),
                values,
                validity,
            ))
        },
        ArrowDataType::LargeList(_) => {
            offsets.push(values.len() as i64);

            Box::new(ListArray::<i64>::new(
                dtype,
                offsets.try_into().expect("List too large"),
                values,
                validity,
            ))
        },
        ArrowDataType::FixedSizeList(_, _) => {
            Box::new(FixedSizeListArray::new(dtype, length, values, validity))
        },
        _ => unreachable!(),
    }
}

/// Creates a new [`MapArray`].
pub fn create_map(
    dtype: ArrowDataType,
    nested: &mut NestedState,
    values: Box<dyn Array>,
) -> Box<dyn Array> {
    let (_, mut offsets, validity) = nested.pop().unwrap();
    match dtype.to_logical_type() {
        ArrowDataType::Map(_, _) => {
            offsets.push(values.len() as i64);
            let offsets = offsets.iter().map(|x| *x as i32).collect::<Vec<_>>();

            let offsets: Offsets<i32> = offsets
                .try_into()
                .expect("i64 offsets do not fit in i32 offsets");

            Box::new(MapArray::new(
                dtype,
                offsets.into(),
                values,
                validity.and_then(freeze_validity),
            ))
        },
        _ => unreachable!(),
    }
}

fn is_primitive(dtype: &ArrowDataType) -> bool {
    matches!(
        dtype.to_physical_type(),
        arrow::datatypes::PhysicalType::Primitive(_)
            | arrow::datatypes::PhysicalType::Null
            | arrow::datatypes::PhysicalType::Boolean
            | arrow::datatypes::PhysicalType::Utf8
            | arrow::datatypes::PhysicalType::LargeUtf8
            | arrow::datatypes::PhysicalType::Binary
            | arrow::datatypes::PhysicalType::BinaryView
            | arrow::datatypes::PhysicalType::Utf8View
            | arrow::datatypes::PhysicalType::LargeBinary
            | arrow::datatypes::PhysicalType::FixedSizeBinary
            | arrow::datatypes::PhysicalType::Dictionary(_)
    )
}

fn columns_to_iter_recursive(
    mut columns: Vec<BasicDecompressor>,
    mut types: Vec<&PrimitiveType>,
    field: Field,
    init: Vec<InitNested>,
    filter: Option<Filter>,
) -> ParquetResult<(NestedState, Box<dyn Array>, Bitmap)> {
    if init.is_empty() && is_primitive(&field.dtype) {
        let (array, pred_true_mask) =
            page_iter_to_array(columns.pop().unwrap(), types.pop().unwrap(), field, filter)?;

        return Ok((NestedState::default(), array, pred_true_mask));
    }

    nested::columns_to_iter_recursive(columns, types, field, init, filter)
}

/// Returns the number of (parquet) columns that a [`ArrowDataType`] contains.
pub fn n_columns(dtype: &ArrowDataType) -> usize {
    use arrow::datatypes::PhysicalType::*;
    match dtype.to_physical_type() {
        Null | Boolean | Primitive(_) | Binary | FixedSizeBinary | LargeBinary | Utf8
        | Dictionary(_) | LargeUtf8 | BinaryView | Utf8View => 1,
        List | FixedSizeList | LargeList => {
            let a = dtype.to_logical_type();
            if let ArrowDataType::List(inner) = a {
                n_columns(&inner.dtype)
            } else if let ArrowDataType::LargeList(inner) = a {
                n_columns(&inner.dtype)
            } else if let ArrowDataType::FixedSizeList(inner, _) = a {
                n_columns(&inner.dtype)
            } else {
                unreachable!()
            }
        },
        Map => {
            let a = dtype.to_logical_type();
            if let ArrowDataType::Map(inner, _) = a {
                n_columns(&inner.dtype)
            } else {
                unreachable!()
            }
        },
        Struct => {
            if let ArrowDataType::Struct(fields) = dtype.to_logical_type() {
                fields.iter().map(|inner| n_columns(&inner.dtype)).sum()
            } else {
                unreachable!()
            }
        },
        _ => todo!(),
    }
}

/// An iterator adapter that maps multiple iterators of [`PagesIter`] into an iterator of [`Array`]s.
///
/// For a non-nested datatypes such as [`ArrowDataType::Int32`], this function requires a single element in `columns` and `types`.
/// For nested types, `columns` must be composed by all parquet columns with associated types `types`.
///
/// The arrays are guaranteed to be at most of size `chunk_size` and data type `field.dtype`.
pub fn column_iter_to_arrays(
    columns: Vec<BasicDecompressor>,
    types: Vec<&PrimitiveType>,
    field: Field,
    filter: Option<Filter>,
) -> PolarsResult<(Box<dyn Array>, Bitmap)> {
    let (_, array, pred_true_mask) =
        columns_to_iter_recursive(columns, types, field, vec![], filter)?;
    Ok((array, pred_true_mask))
}
