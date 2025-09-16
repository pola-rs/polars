use std::borrow::Cow;

use arrow::compute::utils::combine_validities_and_many;
use polars_row::{RowEncodingContext, RowEncodingOptions, RowsEncoded, convert_columns};
use rayon::prelude::*;

use crate::POOL;
use crate::prelude::*;
use crate::utils::_split_offsets;

pub fn encode_rows_vertical_par_unordered(by: &[Column]) -> PolarsResult<BinaryOffsetChunked> {
    let n_threads = POOL.current_num_threads();
    let len = by[0].len();
    let splits = _split_offsets(len, n_threads);

    let chunks = splits.into_par_iter().map(|(offset, len)| {
        let sliced = by
            .iter()
            .map(|s| s.slice(offset as i64, len))
            .collect::<Vec<_>>();
        let rows = _get_rows_encoded_unordered(&sliced)?;
        Ok(rows.into_array())
    });
    let chunks = POOL.install(|| chunks.collect::<PolarsResult<Vec<_>>>());

    Ok(BinaryOffsetChunked::from_chunk_iter(
        PlSmallStr::EMPTY,
        chunks?,
    ))
}

// Almost the same but broadcast nulls to the row-encoded array.
pub fn encode_rows_vertical_par_unordered_broadcast_nulls(
    by: &[Column],
) -> PolarsResult<BinaryOffsetChunked> {
    let n_threads = POOL.current_num_threads();
    let len = by[0].len();
    let splits = _split_offsets(len, n_threads);

    let chunks = splits.into_par_iter().map(|(offset, len)| {
        let sliced = by
            .iter()
            .map(|s| s.slice(offset as i64, len))
            .collect::<Vec<_>>();
        let rows = _get_rows_encoded_unordered(&sliced)?;

        let validities = sliced
            .iter()
            .flat_map(|s| {
                let s = s.rechunk();
                #[allow(clippy::unnecessary_to_owned)]
                s.as_materialized_series()
                    .chunks()
                    .to_vec()
                    .into_iter()
                    .map(|arr| arr.validity().cloned())
            })
            .collect::<Vec<_>>();

        let validity = combine_validities_and_many(&validities);
        Ok(rows.into_array().with_validity_typed(validity))
    });
    let chunks = POOL.install(|| chunks.collect::<PolarsResult<Vec<_>>>());

    Ok(BinaryOffsetChunked::from_chunk_iter(
        PlSmallStr::EMPTY,
        chunks?,
    ))
}

/// Get the [`RowEncodingContext`] for a certain [`DataType`].
///
/// This should be given the logical type in order to communicate Polars datatype information down
/// into the row encoding / decoding.
pub fn get_row_encoding_context(dtype: &DataType) -> Option<RowEncodingContext> {
    match dtype {
        DataType::Boolean
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64
        | DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::Int128
        | DataType::Float32
        | DataType::Float64
        | DataType::String
        | DataType::Binary
        | DataType::BinaryOffset
        | DataType::Null
        | DataType::Time
        | DataType::Date
        | DataType::Datetime(_, _)
        | DataType::Duration(_) => None,

        #[cfg(feature = "dtype-categorical")]
        DataType::Categorical(_, mapping) | DataType::Enum(_, mapping) => {
            use polars_row::RowEncodingCategoricalContext;

            Some(RowEncodingContext::Categorical(
                RowEncodingCategoricalContext {
                    is_enum: matches!(dtype, DataType::Enum(_, _)),
                    mapping: mapping.clone(),
                },
            ))
        },

        DataType::Unknown(_) => panic!("Unsupported in row encoding"),

        #[cfg(feature = "object")]
        DataType::Object(_) => panic!("Unsupported in row encoding"),

        #[cfg(feature = "dtype-decimal")]
        DataType::Decimal(precision, _) => {
            Some(RowEncodingContext::Decimal(precision.unwrap_or(38)))
        },

        #[cfg(feature = "dtype-array")]
        DataType::Array(dtype, _) => get_row_encoding_context(dtype),
        DataType::List(dtype) => get_row_encoding_context(dtype),
        #[cfg(feature = "dtype-struct")]
        DataType::Struct(fs) => {
            let mut ctxts = Vec::new();

            for (i, f) in fs.iter().enumerate() {
                if let Some(ctxt) = get_row_encoding_context(f.dtype()) {
                    ctxts.reserve(fs.len());
                    ctxts.extend(std::iter::repeat_n(None, i));
                    ctxts.push(Some(ctxt));
                    break;
                }
            }

            if ctxts.is_empty() {
                return None;
            }

            ctxts.extend(
                fs[ctxts.len()..]
                    .iter()
                    .map(|f| get_row_encoding_context(f.dtype())),
            );

            Some(RowEncodingContext::Struct(ctxts))
        },
    }
}

pub fn encode_rows_unordered(by: &[Column]) -> PolarsResult<BinaryOffsetChunked> {
    let rows = _get_rows_encoded_unordered(by)?;
    Ok(BinaryOffsetChunked::with_chunk(
        PlSmallStr::EMPTY,
        rows.into_array(),
    ))
}

pub fn _get_rows_encoded_unordered(by: &[Column]) -> PolarsResult<RowsEncoded> {
    let mut cols = Vec::with_capacity(by.len());
    let mut opts = Vec::with_capacity(by.len());
    let mut ctxts = Vec::with_capacity(by.len());

    // Since ZFS exists, we might not actually have any arrays and need to get the length from the
    // columns.
    let num_rows = by.first().map_or(0, |c| c.len());

    for by in by {
        debug_assert_eq!(by.len(), num_rows);

        let by = by
            .trim_lists_to_normalized_offsets()
            .map_or(Cow::Borrowed(by), Cow::Owned);
        let by = by.propagate_nulls().map_or(by, Cow::Owned);
        let by = by.as_materialized_series();
        let arr = by.to_physical_repr().rechunk().chunks()[0].to_boxed();
        let opt = RowEncodingOptions::new_unsorted();
        let ctxt = get_row_encoding_context(by.dtype());

        cols.push(arr);
        opts.push(opt);
        ctxts.push(ctxt);
    }
    Ok(convert_columns(num_rows, &cols, &opts, &ctxts))
}

pub fn _get_rows_encoded(
    by: &[Column],
    descending: &[bool],
    nulls_last: &[bool],
) -> PolarsResult<RowsEncoded> {
    debug_assert_eq!(by.len(), descending.len());
    debug_assert_eq!(by.len(), nulls_last.len());

    let mut cols = Vec::with_capacity(by.len());
    let mut opts = Vec::with_capacity(by.len());
    let mut ctxts = Vec::with_capacity(by.len());

    // Since ZFS exists, we might not actually have any arrays and need to get the length from the
    // columns.
    let num_rows = by.first().map_or(0, |c| c.len());

    for ((by, desc), null_last) in by.iter().zip(descending).zip(nulls_last) {
        debug_assert_eq!(by.len(), num_rows);

        let by = by
            .trim_lists_to_normalized_offsets()
            .map_or(Cow::Borrowed(by), Cow::Owned);
        let by = by.propagate_nulls().map_or(by, Cow::Owned);
        let by = by.as_materialized_series();
        let arr = by.to_physical_repr().rechunk().chunks()[0].to_boxed();
        let opt = RowEncodingOptions::new_sorted(*desc, *null_last);
        let ctxt = get_row_encoding_context(by.dtype());

        cols.push(arr);
        opts.push(opt);
        ctxts.push(ctxt);
    }
    Ok(convert_columns(num_rows, &cols, &opts, &ctxts))
}

pub fn _get_rows_encoded_ca(
    name: PlSmallStr,
    by: &[Column],
    descending: &[bool],
    nulls_last: &[bool],
) -> PolarsResult<BinaryOffsetChunked> {
    _get_rows_encoded(by, descending, nulls_last)
        .map(|rows| BinaryOffsetChunked::with_chunk(name, rows.into_array()))
}

pub fn _get_rows_encoded_arr(
    by: &[Column],
    descending: &[bool],
    nulls_last: &[bool],
) -> PolarsResult<BinaryArray<i64>> {
    _get_rows_encoded(by, descending, nulls_last).map(|rows| rows.into_array())
}

pub fn _get_rows_encoded_ca_unordered(
    name: PlSmallStr,
    by: &[Column],
) -> PolarsResult<BinaryOffsetChunked> {
    _get_rows_encoded_unordered(by)
        .map(|rows| BinaryOffsetChunked::with_chunk(name, rows.into_array()))
}

#[cfg(feature = "dtype-struct")]
pub fn row_encoding_decode(
    ca: &BinaryOffsetChunked,
    fields: &[Field],
    opts: &[RowEncodingOptions],
) -> PolarsResult<StructChunked> {
    let (ctxts, dtypes) = fields
        .iter()
        .map(|f| {
            (
                get_row_encoding_context(f.dtype()),
                f.dtype().to_physical().to_arrow(CompatLevel::newest()),
            )
        })
        .collect::<(Vec<_>, Vec<_>)>();

    let struct_arrow_dtype = ArrowDataType::Struct(
        fields
            .iter()
            .map(|v| v.to_physical().to_arrow(CompatLevel::newest()))
            .collect(),
    );

    let mut rows = Vec::new();
    let chunks = ca
        .downcast_iter()
        .map(|array| {
            let decoded_arrays = unsafe {
                polars_row::decode::decode_rows_from_binary(array, opts, &ctxts, &dtypes, &mut rows)
            };
            assert_eq!(decoded_arrays.len(), fields.len());

            StructArray::new(
                struct_arrow_dtype.clone(),
                array.len(),
                decoded_arrays,
                None,
            )
            .to_boxed()
        })
        .collect::<Vec<_>>();

    Ok(unsafe {
        StructChunked::from_chunks_and_dtype(
            ca.name().clone(),
            chunks,
            DataType::Struct(fields.to_vec()),
        )
    })
}
