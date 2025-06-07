use std::ops::Range;
use std::sync::Arc;

use arrow::array::{FixedSizeBinaryArray, FixedSizeListArray, PrimitiveArray};
use arrow::datatypes::ArrowDataType;
use arrow::types::{NativeType, f16};
use polars_core::prelude::{CompatLevel, DataFrame, DataType, Field, SchemaExt, SchemaRef, Series};
use polars_error::{PolarsResult, polars_bail};
use polars_utils::pl_str::PlSmallStr;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Hash, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum Endianness {
    Little,
    Big,
}
// pub enum EndiannessOption {
//     All(Endianness),
//     Separate(Vec<Endianness>)
// }

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub struct FwfReadOptions {
    schema: SchemaRef,
    endianness: Arc<Vec<Endianness>>,
}

impl FwfReadOptions {
    pub fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
    pub fn endianness(&self) -> Arc<Vec<Endianness>> {
        self.endianness.clone()
    }
    pub fn get_row_width(&self) -> PolarsResult<usize> {
        self.schema
            .iter_fields()
            .map(|field| get_field_width(&field)) // PolarsResult<usize>
            .try_fold(0usize, |acc, width| width.map(|w| acc + w))
    }
}

pub fn get_field_width(field: &Field) -> PolarsResult<usize> {
    get_arrow_field_width(&field.dtype().to_physical().to_arrow(CompatLevel::newest()))
}

pub fn get_arrow_field_width(dtype: &ArrowDataType) -> PolarsResult<usize> {
    let size = match dtype {
        arrow::datatypes::ArrowDataType::Int8 => 1,
        arrow::datatypes::ArrowDataType::Int16 => 2,
        arrow::datatypes::ArrowDataType::Int32 => 4,
        arrow::datatypes::ArrowDataType::Int64 => 8,
        arrow::datatypes::ArrowDataType::Int128 => 16,
        arrow::datatypes::ArrowDataType::UInt8 => 1,
        arrow::datatypes::ArrowDataType::UInt16 => 2,
        arrow::datatypes::ArrowDataType::UInt32 => 4,
        arrow::datatypes::ArrowDataType::UInt64 => 8,
        arrow::datatypes::ArrowDataType::Float16 => 2,
        arrow::datatypes::ArrowDataType::Float32 => 4,
        arrow::datatypes::ArrowDataType::Float64 => 8,
        arrow::datatypes::ArrowDataType::FixedSizeBinary(size) => *size,
        arrow::datatypes::ArrowDataType::FixedSizeList(inner_field, size) => {
            get_arrow_field_width(inner_field.dtype())? * size
        },
        _ => polars_bail!(ComputeError: "Unsupported DataType in fwf get_feild_width: {dtype:?}"),
    };
    Ok(size)
}

#[allow(clippy::too_many_arguments)]
pub fn decode_fwf(
    buffer: &[u8], // or &Mmap
    schema: SchemaRef,
    range: Range<usize>,
    selected_cols: &[usize],
    offsets: &[usize],
    widths: &[usize],
    row_size: &usize,
    endians: &[Endianness],
) -> PolarsResult<DataFrame> {
    if buffer.len() % row_size != 0 {
        polars_bail!(ComputeError: "File size is not a multiple of row size")
    }
    // 3. For each selected column, decode directly from the buffer
    let mut columns = Vec::with_capacity(selected_cols.len());
    for (i, &col_idx) in selected_cols.iter().enumerate() {
        let offset = offsets[col_idx];
        let width = widths[col_idx];
        let (name, dtype) = schema.get_at_index(col_idx).unwrap();
        let endian = &endians[i];
        // Efficient: pass the full buffer, n_rows, column offset/stride, and width
        let s = decode_fwf_column(buffer, &range, offset, width, name, dtype, endian)?;
        columns.push(s.into());
    }
    DataFrame::new_with_height(range.len(), columns)
}

pub fn decode_fwf_column(
    bytes: &[u8],
    slice: &Range<usize>,
    offset: usize,
    width: usize,
    name: &PlSmallStr,
    dtype: &DataType,
    endian: &Endianness,
) -> PolarsResult<Series> {
    let n_rows = slice.len();
    macro_rules! dispatch {
        ($ty:ty) => {{
            let elem_size = std::mem::size_of::<$ty>();
            assert!(width == elem_size, "Width mismatch for primitive type.");
            let mut vec = Vec::with_capacity(n_rows);
            match endian {
                Endianness::Little => {
                    for row in slice.clone() {
                        let start = row * offset;
                        let chunk = &bytes[start..start + width];
                        let val = <$ty>::from_le_bytes(chunk.try_into().unwrap());
                        vec.push(val);
                    }
                },
                Endianness::Big => {
                    for row in slice.clone() {
                        let start = row * offset;
                        let chunk = &bytes[start..start + width];
                        let val = <$ty>::from_be_bytes(chunk.try_into().unwrap());
                        vec.push(val);
                    }
                },
            }
            let arr = PrimitiveArray::from_vec(vec);
            unsafe {
                Ok(Series::from_chunks_and_dtype_unchecked(
                    name.clone(),
                    vec![Box::new(arr)],
                    dtype,
                ))
            }
        }};
    }
    match dtype.to_physical().to_arrow(CompatLevel::newest()) {
        arrow::datatypes::ArrowDataType::Int8 => dispatch!(i8),
        arrow::datatypes::ArrowDataType::Int16 => dispatch!(i16),
        arrow::datatypes::ArrowDataType::Int32 => dispatch!(i32),
        arrow::datatypes::ArrowDataType::Int64 => dispatch!(i64),
        arrow::datatypes::ArrowDataType::Int128 => dispatch!(i128),
        arrow::datatypes::ArrowDataType::UInt8 => dispatch!(u8),
        arrow::datatypes::ArrowDataType::UInt16 => dispatch!(u16),
        arrow::datatypes::ArrowDataType::UInt32 => dispatch!(u32),
        arrow::datatypes::ArrowDataType::UInt64 => dispatch!(u64),
        arrow::datatypes::ArrowDataType::Float16 => dispatch!(f16),
        arrow::datatypes::ArrowDataType::Float32 => dispatch!(f32),
        arrow::datatypes::ArrowDataType::Float64 => dispatch!(f64),
        arrow::datatypes::ArrowDataType::FixedSizeBinary(size) => {
            let mut chunks = Vec::with_capacity(n_rows);
            assert!(width == size, "Width mismatch for fixed width binary type.");
            for row in slice.clone() {
                let start = row * offset;
                chunks.push(Some(&bytes[start..start + size]));
            }
            let arr = FixedSizeBinaryArray::from_iter(chunks, size);
            unsafe {
                Ok(Series::from_chunks_and_dtype_unchecked(
                    name.clone(),
                    vec![Box::new(arr)],
                    dtype,
                ))
            }
        },
        arrow::datatypes::ArrowDataType::FixedSizeList(inner_field, list_size) => {
            let inner_dtype = inner_field.dtype();
            let inner_width = get_arrow_field_width(inner_dtype)?;
            let arrow_dtype = dtype.to_physical().to_arrow(CompatLevel::newest());
            assert!(
                width / list_size == inner_width,
                "Width mismatch for fixed width binary type."
            );
            macro_rules! list_dispatch {
                ($ty:ty) => {{
                    let mut vec = Vec::with_capacity(n_rows);
                    match endian {
                        Endianness::Little => {
                            for row in slice.clone() {
                                let start = row * offset;
                                for i in 0..list_size {
                                    let off = start + i * inner_width;
                                    let chunk = &bytes[off..off + inner_width];
                                    let val = <$ty>::from_le_bytes(chunk.try_into().unwrap());
                                    vec.push(val);
                                }
                            }
                        },
                        Endianness::Big => {
                            for row in slice.clone() {
                                let start = row * offset;
                                for i in 0..list_size {
                                    let off = start + i * inner_width;
                                    let chunk = &bytes[off..off + inner_width];
                                    let val = <$ty>::from_be_bytes(chunk.try_into().unwrap());
                                    vec.push(val);
                                }
                            }
                        },
                    }
                    let values = PrimitiveArray::from_vec(vec);
                    let arr = FixedSizeListArray::try_new(
                        arrow_dtype,
                        list_size,
                        Box::new(values),
                        None,
                    )?;
                    unsafe {
                        Ok(Series::from_chunks_and_dtype_unchecked(
                            name.clone(),
                            vec![Box::new(arr)],
                            dtype,
                        ))
                    }
                }};
            }
            match inner_dtype {
                arrow::datatypes::ArrowDataType::Int8 => list_dispatch!(i8),
                arrow::datatypes::ArrowDataType::Int16 => list_dispatch!(i16),
                arrow::datatypes::ArrowDataType::Int32 => list_dispatch!(i32),
                arrow::datatypes::ArrowDataType::Int64 => list_dispatch!(i64),
                arrow::datatypes::ArrowDataType::Int128 => list_dispatch!(i128),
                arrow::datatypes::ArrowDataType::UInt8 => list_dispatch!(u8),
                arrow::datatypes::ArrowDataType::UInt16 => list_dispatch!(u16),
                arrow::datatypes::ArrowDataType::UInt32 => list_dispatch!(u32),
                arrow::datatypes::ArrowDataType::UInt64 => list_dispatch!(u64),
                arrow::datatypes::ArrowDataType::Float16 => list_dispatch!(f16),
                arrow::datatypes::ArrowDataType::Float32 => list_dispatch!(f32),
                arrow::datatypes::ArrowDataType::Float64 => list_dispatch!(f64),
                _ => {
                    polars_bail!(ComputeError: "Unsupported DataType in fwf decode_stream_polars (for fixed width list): {inner_dtype:?}")
                },
            }
        },
        _ => {
            polars_bail!(ComputeError: "Unsupported DataType in fwf decode_stream_polars: {dtype:?}")
        },
    }
}
