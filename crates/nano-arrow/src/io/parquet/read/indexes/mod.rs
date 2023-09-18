//! API to perform page-level filtering (also known as indexes)
use parquet2::error::Error as ParquetError;
use parquet2::indexes::{
    select_pages, BooleanIndex, ByteIndex, FixedLenByteIndex, Index as ParquetIndex, NativeIndex,
    PageLocation,
};
use parquet2::metadata::{ColumnChunkMetaData, RowGroupMetaData};
use parquet2::read::{read_columns_indexes as _read_columns_indexes, read_pages_locations};
use parquet2::schema::types::PhysicalType as ParquetPhysicalType;

mod binary;
mod boolean;
mod fixed_len_binary;
mod primitive;

use std::collections::VecDeque;
use std::io::{Read, Seek};

use crate::array::UInt64Array;
use crate::datatypes::{Field, PrimitiveType};
use crate::{
    array::Array,
    datatypes::{DataType, PhysicalType},
    error::Error,
};

use super::get_field_pages;

pub use parquet2::indexes::{FilteredPage, Interval};

/// Page statistics of an Arrow field.
#[derive(Debug, PartialEq)]
pub enum FieldPageStatistics {
    /// Variant used for fields with a single parquet column (e.g. primitives, dictionaries, list)
    Single(ColumnPageStatistics),
    /// Variant used for fields with multiple parquet columns (e.g. Struct, Map)
    Multiple(Vec<FieldPageStatistics>),
}

impl From<ColumnPageStatistics> for FieldPageStatistics {
    fn from(column: ColumnPageStatistics) -> Self {
        Self::Single(column)
    }
}

/// [`ColumnPageStatistics`] contains the minimum, maximum, and null_count
/// of each page of a parquet column, as an [`Array`].
/// This struct has the following invariants:
/// * `min`, `max` and `null_count` have the same length (equal to the number of pages in the column)
/// * `min`, `max` and `null_count` are guaranteed to be non-null
/// * `min` and `max` have the same logical type
#[derive(Debug, PartialEq)]
pub struct ColumnPageStatistics {
    /// The minimum values in the pages
    pub min: Box<dyn Array>,
    /// The maximum values in the pages
    pub max: Box<dyn Array>,
    /// The number of null values in the pages.
    pub null_count: UInt64Array,
}

/// Given a sequence of [`ParquetIndex`] representing the page indexes of each column in the
/// parquet file, returns the page-level statistics as a [`FieldPageStatistics`].
///
/// This function maps timestamps, decimal types, etc. accordingly.
/// # Implementation
/// This function is CPU-bounded `O(P)` where `P` is the total number of pages on all columns.
/// # Error
/// This function errors iff the value is not deserializable to arrow (e.g. invalid utf-8)
fn deserialize(
    indexes: &mut VecDeque<&Box<dyn ParquetIndex>>,
    data_type: DataType,
) -> Result<FieldPageStatistics, Error> {
    match data_type.to_physical_type() {
        PhysicalType::Boolean => {
            let index = indexes
                .pop_front()
                .unwrap()
                .as_any()
                .downcast_ref::<BooleanIndex>()
                .unwrap();
            Ok(boolean::deserialize(&index.indexes).into())
        }
        PhysicalType::Primitive(PrimitiveType::Int128) => {
            let index = indexes.pop_front().unwrap();
            match index.physical_type() {
                ParquetPhysicalType::Int32 => {
                    let index = index.as_any().downcast_ref::<NativeIndex<i32>>().unwrap();
                    Ok(primitive::deserialize_i32(&index.indexes, data_type).into())
                }
                parquet2::schema::types::PhysicalType::Int64 => {
                    let index = index.as_any().downcast_ref::<NativeIndex<i64>>().unwrap();
                    Ok(
                        primitive::deserialize_i64(
                            &index.indexes,
                            &index.primitive_type,
                            data_type,
                        )
                        .into(),
                    )
                }
                parquet2::schema::types::PhysicalType::FixedLenByteArray(_) => {
                    let index = index.as_any().downcast_ref::<FixedLenByteIndex>().unwrap();
                    Ok(fixed_len_binary::deserialize(&index.indexes, data_type).into())
                }
                other => Err(Error::nyi(format!(
                    "Deserialize {other:?} to arrow's int64"
                ))),
            }
        }
        PhysicalType::Primitive(PrimitiveType::Int256) => {
            let index = indexes.pop_front().unwrap();
            match index.physical_type() {
                ParquetPhysicalType::Int32 => {
                    let index = index.as_any().downcast_ref::<NativeIndex<i32>>().unwrap();
                    Ok(primitive::deserialize_i32(&index.indexes, data_type).into())
                }
                parquet2::schema::types::PhysicalType::Int64 => {
                    let index = index.as_any().downcast_ref::<NativeIndex<i64>>().unwrap();
                    Ok(
                        primitive::deserialize_i64(
                            &index.indexes,
                            &index.primitive_type,
                            data_type,
                        )
                        .into(),
                    )
                }
                parquet2::schema::types::PhysicalType::FixedLenByteArray(_) => {
                    let index = index.as_any().downcast_ref::<FixedLenByteIndex>().unwrap();
                    Ok(fixed_len_binary::deserialize(&index.indexes, data_type).into())
                }
                other => Err(Error::nyi(format!(
                    "Deserialize {other:?} to arrow's int64"
                ))),
            }
        }
        PhysicalType::Primitive(PrimitiveType::UInt8)
        | PhysicalType::Primitive(PrimitiveType::UInt16)
        | PhysicalType::Primitive(PrimitiveType::UInt32)
        | PhysicalType::Primitive(PrimitiveType::Int32) => {
            let index = indexes
                .pop_front()
                .unwrap()
                .as_any()
                .downcast_ref::<NativeIndex<i32>>()
                .unwrap();
            Ok(primitive::deserialize_i32(&index.indexes, data_type).into())
        }
        PhysicalType::Primitive(PrimitiveType::UInt64)
        | PhysicalType::Primitive(PrimitiveType::Int64) => {
            let index = indexes.pop_front().unwrap();
            match index.physical_type() {
                ParquetPhysicalType::Int64 => {
                    let index = index.as_any().downcast_ref::<NativeIndex<i64>>().unwrap();
                    Ok(
                        primitive::deserialize_i64(
                            &index.indexes,
                            &index.primitive_type,
                            data_type,
                        )
                        .into(),
                    )
                }
                parquet2::schema::types::PhysicalType::Int96 => {
                    let index = index
                        .as_any()
                        .downcast_ref::<NativeIndex<[u32; 3]>>()
                        .unwrap();
                    Ok(primitive::deserialize_i96(&index.indexes, data_type).into())
                }
                other => Err(Error::nyi(format!(
                    "Deserialize {other:?} to arrow's int64"
                ))),
            }
        }
        PhysicalType::Primitive(PrimitiveType::Float32) => {
            let index = indexes
                .pop_front()
                .unwrap()
                .as_any()
                .downcast_ref::<NativeIndex<f32>>()
                .unwrap();
            Ok(primitive::deserialize_id(&index.indexes, data_type).into())
        }
        PhysicalType::Primitive(PrimitiveType::Float64) => {
            let index = indexes
                .pop_front()
                .unwrap()
                .as_any()
                .downcast_ref::<NativeIndex<f64>>()
                .unwrap();
            Ok(primitive::deserialize_id(&index.indexes, data_type).into())
        }
        PhysicalType::Binary
        | PhysicalType::LargeBinary
        | PhysicalType::Utf8
        | PhysicalType::LargeUtf8 => {
            let index = indexes
                .pop_front()
                .unwrap()
                .as_any()
                .downcast_ref::<ByteIndex>()
                .unwrap();
            binary::deserialize(&index.indexes, &data_type).map(|x| x.into())
        }
        PhysicalType::FixedSizeBinary => {
            let index = indexes
                .pop_front()
                .unwrap()
                .as_any()
                .downcast_ref::<FixedLenByteIndex>()
                .unwrap();
            Ok(fixed_len_binary::deserialize(&index.indexes, data_type).into())
        }
        PhysicalType::Dictionary(_) => {
            if let DataType::Dictionary(_, inner, _) = data_type.to_logical_type() {
                deserialize(indexes, (**inner).clone())
            } else {
                unreachable!()
            }
        }
        PhysicalType::List => {
            if let DataType::List(inner) = data_type.to_logical_type() {
                deserialize(indexes, inner.data_type.clone())
            } else {
                unreachable!()
            }
        }
        PhysicalType::LargeList => {
            if let DataType::LargeList(inner) = data_type.to_logical_type() {
                deserialize(indexes, inner.data_type.clone())
            } else {
                unreachable!()
            }
        }
        PhysicalType::Map => {
            if let DataType::Map(inner, _) = data_type.to_logical_type() {
                deserialize(indexes, inner.data_type.clone())
            } else {
                unreachable!()
            }
        }
        PhysicalType::Struct => {
            let children_fields = if let DataType::Struct(children) = data_type.to_logical_type() {
                children
            } else {
                unreachable!()
            };
            let children = children_fields
                .iter()
                .map(|child| deserialize(indexes, child.data_type.clone()))
                .collect::<Result<Vec<_>, Error>>()?;

            Ok(FieldPageStatistics::Multiple(children))
        }

        other => Err(Error::nyi(format!(
            "Deserialize into arrow's {other:?} page index"
        ))),
    }
}

/// Checks whether the row group have page index information (page statistics)
pub fn has_indexes(row_group: &RowGroupMetaData) -> bool {
    row_group
        .columns()
        .iter()
        .all(|chunk| chunk.column_chunk().column_index_offset.is_some())
}

/// Reads the column indexes from the reader assuming a valid set of derived Arrow fields
/// for all parquet the columns in the file.
///
/// It returns one [`FieldPageStatistics`] per field in `fields`
///
/// This function is expected to be used to filter out parquet pages.
///
/// # Implementation
/// This function is IO-bounded and calls `reader.read_exact` exactly once.
/// # Error
/// Errors iff the indexes can't be read or their deserialization to arrow is incorrect (e.g. invalid utf-8)
pub fn read_columns_indexes<R: Read + Seek>(
    reader: &mut R,
    chunks: &[ColumnChunkMetaData],
    fields: &[Field],
) -> Result<Vec<FieldPageStatistics>, Error> {
    let indexes = _read_columns_indexes(reader, chunks)?;

    fields
        .iter()
        .map(|field| {
            let indexes = get_field_pages(chunks, &indexes, &field.name);
            let mut indexes = indexes.into_iter().collect();

            deserialize(&mut indexes, field.data_type.clone())
        })
        .collect()
}

/// Returns the set of (row) intervals of the pages.
pub fn compute_page_row_intervals(
    locations: &[PageLocation],
    num_rows: usize,
) -> Result<Vec<Interval>, ParquetError> {
    if locations.is_empty() {
        return Ok(vec![]);
    };

    let last = (|| {
        let start: usize = locations.last().unwrap().first_row_index.try_into()?;
        let length = num_rows - start;
        Result::<_, ParquetError>::Ok(Interval::new(start, length))
    })();

    let pages_lengths = locations
        .windows(2)
        .map(|x| {
            let start = usize::try_from(x[0].first_row_index)?;
            let length = usize::try_from(x[1].first_row_index - x[0].first_row_index)?;
            Ok(Interval::new(start, length))
        })
        .chain(std::iter::once(last));
    pages_lengths.collect()
}

/// Reads all page locations and index locations (IO-bounded) and uses `predicate` to compute
/// the set of [`FilteredPage`] that fulfill the predicate.
///
/// The non-trivial argument of this function is `predicate`, that controls which pages are selected.
/// Its signature contains 2 arguments:
/// * 0th argument (indexes): contains one [`ColumnPageStatistics`] (page statistics) per field.
///   Use it to evaluate the predicate against
/// * 1th argument (intervals): contains one [`Vec<Vec<Interval>>`] (row positions) per field.
///   For each field, the outermost vector corresponds to each parquet column:
///   a primitive field contains 1 column, a struct field with 2 primitive fields contain 2 columns.
///   The inner `Vec<Interval>` contains one [`Interval`] per page: its length equals the length of [`ColumnPageStatistics`].
/// It returns a single [`Vec<Interval>`] denoting the set of intervals that the predicate selects (over all columns).
///
/// This returns one item per `field`. For each field, there is one item per column (for non-nested types it returns one column)
/// and finally [`Vec<FilteredPage>`], that corresponds to the set of selected pages.
pub fn read_filtered_pages<
    R: Read + Seek,
    F: Fn(&[FieldPageStatistics], &[Vec<Vec<Interval>>]) -> Vec<Interval>,
>(
    reader: &mut R,
    row_group: &RowGroupMetaData,
    fields: &[Field],
    predicate: F,
    //is_intersection: bool,
) -> Result<Vec<Vec<Vec<FilteredPage>>>, Error> {
    let num_rows = row_group.num_rows();

    // one vec per column
    let locations = read_pages_locations(reader, row_group.columns())?;
    // one Vec<Vec<>> per field (non-nested contain a single entry on the first column)
    let locations = fields
        .iter()
        .map(|field| get_field_pages(row_group.columns(), &locations, &field.name))
        .collect::<Vec<_>>();

    // one ColumnPageStatistics per field
    let indexes = read_columns_indexes(reader, row_group.columns(), fields)?;

    let intervals = locations
        .iter()
        .map(|locations| {
            locations
                .iter()
                .map(|locations| Ok(compute_page_row_intervals(locations, num_rows)?))
                .collect::<Result<Vec<_>, Error>>()
        })
        .collect::<Result<Vec<_>, Error>>()?;

    let intervals = predicate(&indexes, &intervals);

    locations
        .into_iter()
        .map(|locations| {
            locations
                .into_iter()
                .map(|locations| Ok(select_pages(&intervals, locations, num_rows)?))
                .collect::<Result<Vec<_>, Error>>()
        })
        .collect()
}
