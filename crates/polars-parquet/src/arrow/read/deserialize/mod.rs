//! APIs to read from Parquet format.

mod binary;
mod binview;
mod boolean;
mod dictionary;
mod fixed_size_binary;
mod nested;
mod nested_utils;
mod null;
mod primitive;
mod simple;
mod utils;

use arrow::array::{Array, DictionaryKey, FixedSizeListArray, ListArray, MapArray};
use arrow::datatypes::{ArrowDataType, Field, IntervalUnit};
use arrow::offset::Offsets;
use polars_utils::mmap::MemReader;
use simple::page_iter_to_array;

pub use self::nested_utils::{init_nested, InitNested, NestedState};
pub use self::utils::filter::Filter;
use super::*;
use crate::parquet::read::get_page_iterator as _get_page_iterator;
use crate::parquet::schema::types::PrimitiveType;

/// Creates a new iterator of compressed pages.
pub fn get_page_iterator(
    column_metadata: &ColumnChunkMetaData,
    reader: MemReader,
    pages_filter: Option<PageFilter>,
    buffer: Vec<u8>,
    max_header_size: usize,
) -> PolarsResult<PageReader> {
    Ok(_get_page_iterator(
        column_metadata,
        reader,
        pages_filter,
        buffer,
        max_header_size,
    )?)
}

/// Creates a new [`ListArray`] or [`FixedSizeListArray`].
pub fn create_list(
    data_type: ArrowDataType,
    nested: &mut NestedState,
    values: Box<dyn Array>,
) -> Box<dyn Array> {
    let (mut offsets, validity) = nested.pop().unwrap();
    match data_type.to_logical_type() {
        ArrowDataType::List(_) => {
            offsets.push(values.len() as i64);

            let offsets = offsets.iter().map(|x| *x as i32).collect::<Vec<_>>();

            let offsets: Offsets<i32> = offsets
                .try_into()
                .expect("i64 offsets do not fit in i32 offsets");

            Box::new(ListArray::<i32>::new(
                data_type,
                offsets.into(),
                values,
                validity.and_then(|x| x.into()),
            ))
        },
        ArrowDataType::LargeList(_) => {
            offsets.push(values.len() as i64);

            Box::new(ListArray::<i64>::new(
                data_type,
                offsets.try_into().expect("List too large"),
                values,
                validity.and_then(|x| x.into()),
            ))
        },
        ArrowDataType::FixedSizeList(_, _) => Box::new(FixedSizeListArray::new(
            data_type,
            values,
            validity.and_then(|x| x.into()),
        )),
        _ => unreachable!(),
    }
}

/// Creates a new [`MapArray`].
pub fn create_map(
    data_type: ArrowDataType,
    nested: &mut NestedState,
    values: Box<dyn Array>,
) -> Box<dyn Array> {
    let (mut offsets, validity) = nested.pop().unwrap();
    match data_type.to_logical_type() {
        ArrowDataType::Map(_, _) => {
            offsets.push(values.len() as i64);
            let offsets = offsets.iter().map(|x| *x as i32).collect::<Vec<_>>();

            let offsets: Offsets<i32> = offsets
                .try_into()
                .expect("i64 offsets do not fit in i32 offsets");

            Box::new(MapArray::new(
                data_type,
                offsets.into(),
                values,
                validity.and_then(|x| x.into()),
            ))
        },
        _ => unreachable!(),
    }
}

fn is_primitive(data_type: &ArrowDataType) -> bool {
    matches!(
        data_type.to_physical_type(),
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
) -> PolarsResult<(NestedState, Box<dyn Array>)> {
    if init.is_empty() && is_primitive(&field.data_type) {
        let array = page_iter_to_array(
            columns.pop().unwrap(),
            types.pop().unwrap(),
            field.data_type,
            filter,
        )?;

        return Ok((NestedState::default(), array));
    }

    nested::columns_to_iter_recursive(columns, types, field, init, filter)
}

/// Returns the number of (parquet) columns that a [`ArrowDataType`] contains.
pub fn n_columns(data_type: &ArrowDataType) -> usize {
    use arrow::datatypes::PhysicalType::*;
    match data_type.to_physical_type() {
        Null | Boolean | Primitive(_) | Binary | FixedSizeBinary | LargeBinary | Utf8
        | Dictionary(_) | LargeUtf8 | BinaryView | Utf8View => 1,
        List | FixedSizeList | LargeList => {
            let a = data_type.to_logical_type();
            if let ArrowDataType::List(inner) = a {
                n_columns(&inner.data_type)
            } else if let ArrowDataType::LargeList(inner) = a {
                n_columns(&inner.data_type)
            } else if let ArrowDataType::FixedSizeList(inner, _) = a {
                n_columns(&inner.data_type)
            } else {
                unreachable!()
            }
        },
        Map => {
            let a = data_type.to_logical_type();
            if let ArrowDataType::Map(inner, _) = a {
                n_columns(&inner.data_type)
            } else {
                unreachable!()
            }
        },
        Struct => {
            if let ArrowDataType::Struct(fields) = data_type.to_logical_type() {
                fields.iter().map(|inner| n_columns(&inner.data_type)).sum()
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
/// The arrays are guaranteed to be at most of size `chunk_size` and data type `field.data_type`.
pub fn column_iter_to_arrays<'a>(
    columns: Vec<BasicDecompressor>,
    types: Vec<&PrimitiveType>,
    field: Field,
    filter: Option<Filter>,
) -> PolarsResult<ArrayIter<'a>> {
    let (_, array) = columns_to_iter_recursive(columns, types, field, vec![], filter)?;

    Ok(Box::new(std::iter::once(Ok(array))))
}
