use arrow::compute::utils::combine_validities_and_many;
use polars_row::{convert_columns, RowEncodingContext, RowEncodingOptions, RowsEncoded};
use rayon::prelude::*;

use crate::prelude::*;
use crate::utils::_split_offsets;
use crate::POOL;

pub fn encode_rows_vertical_par_unordered(by: &[Series]) -> PolarsResult<BinaryOffsetChunked> {
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
    by: &[Series],
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
                s.chunks()
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

pub fn get_row_encoding_dictionary(dtype: &DataType) -> Option<RowEncodingContext> {
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

        DataType::Unknown(_) => panic!("Unsupported in row encoding"),

        #[cfg(feature = "object")]
        DataType::Object(_, _) => panic!("Unsupported in row encoding"),

        #[cfg(feature = "dtype-decimal")]
        DataType::Decimal(precision, _) => Some(RowEncodingContext::Decimal(precision.unwrap_or(38))),

        #[cfg(feature = "dtype-array")]
        DataType::Array(dtype, _) => get_row_encoding_dictionary(dtype),
        DataType::List(dtype) => get_row_encoding_dictionary(dtype),
        #[cfg(feature = "dtype-categorical")]
        DataType::Categorical(revmap, ordering) | DataType::Enum(revmap, ordering) => {
            let revmap = revmap.as_ref().unwrap();
            Some(match ordering {
                CategoricalOrdering::Physical => RowEncodingContext::CategoricalPhysical(
                    revmap
                        .as_ref()
                        .get_categories()
                        .len()
                        .next_power_of_two()
                        .trailing_zeros() as usize
                        + 1,
                ),
                CategoricalOrdering::Lexical => RowEncodingContext::CategoricalLexical(Box::new(
                    revmap.as_ref().get_categories().clone(),
                )),
            })
        },
        #[cfg(feature = "dtype-struct")]
        DataType::Struct(fs) => {
            let mut out = Vec::new();

            for (i, f) in fs.iter().enumerate() {
                if let Some(dict) = get_row_encoding_dictionary(f.dtype()) {
                    out.reserve(fs.len());
                    out.extend(std::iter::repeat_n(None, i));
                    out.push(Some(dict));
                    break;
                }
            }

            if out.is_empty() {
                return None;
            }

            out.extend(
                fs[out.len()..]
                    .iter()
                    .map(|f| get_row_encoding_dictionary(f.dtype())),
            );

            Some(RowEncodingContext::Struct(out))
        },
    }
}

pub fn encode_rows_unordered(by: &[Series]) -> PolarsResult<BinaryOffsetChunked> {
    let rows = _get_rows_encoded_unordered(by)?;
    Ok(BinaryOffsetChunked::with_chunk(
        PlSmallStr::EMPTY,
        rows.into_array(),
    ))
}

pub fn _get_rows_encoded_unordered(by: &[Series]) -> PolarsResult<RowsEncoded> {
    let mut cols = Vec::with_capacity(by.len());
    let mut opts = Vec::with_capacity(by.len());
    let mut dicts = Vec::with_capacity(by.len());

    // Since ZFS exists, we might not actually have any arrays and need to get the length from the
    // columns.
    let num_rows = by.first().map_or(0, |c| c.len());

    for by in by {
        debug_assert_eq!(by.len(), num_rows);

        let arr = by.to_physical_repr().rechunk().chunks()[0].to_boxed();
        let opt = RowEncodingOptions::new_unsorted();
        let dict = get_row_encoding_dictionary(by.dtype());

        cols.push(arr);
        opts.push(opt);
        dicts.push(dict);
    }
    Ok(convert_columns(num_rows, &cols, &opts, &dicts))
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
    let mut dicts = Vec::with_capacity(by.len());

    // Since ZFS exists, we might not actually have any arrays and need to get the length from the
    // columns.
    let num_rows = by.first().map_or(0, |c| c.len());

    for ((by, desc), null_last) in by.iter().zip(descending).zip(nulls_last) {
        debug_assert_eq!(by.len(), num_rows);

        let by = by.as_materialized_series();
        let arr = by.to_physical_repr().rechunk().chunks()[0].to_boxed();
        let opt = RowEncodingOptions::new_sorted(*desc, *null_last);
        let dict = get_row_encoding_dictionary(by.dtype());

        cols.push(arr);
        opts.push(opt);
        dicts.push(dict);
    }
    Ok(convert_columns(num_rows, &cols, &opts, &dicts))
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
    by: &[Series],
) -> PolarsResult<BinaryOffsetChunked> {
    _get_rows_encoded_unordered(by)
        .map(|rows| BinaryOffsetChunked::with_chunk(name, rows.into_array()))
}
