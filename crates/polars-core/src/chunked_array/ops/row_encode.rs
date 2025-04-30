use std::borrow::Cow;

use arrow::compute::utils::combine_validities_and_many;
use polars_row::{
    RowEncodingCategoricalContext, RowEncodingContext, RowEncodingOptions, RowsEncoded,
    convert_columns,
};
use polars_utils::itertools::Itertools;
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
pub fn get_row_encoding_context(dtype: &DataType, ordered: bool) -> Option<RowEncodingContext> {
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
        DataType::Object(_) => panic!("Unsupported in row encoding"),

        #[cfg(feature = "dtype-decimal")]
        DataType::Decimal(precision, _) => {
            Some(RowEncodingContext::Decimal(precision.unwrap_or(38)))
        },

        #[cfg(feature = "dtype-array")]
        DataType::Array(dtype, _) => get_row_encoding_context(dtype, ordered),
        DataType::List(dtype) => get_row_encoding_context(dtype, ordered),
        #[cfg(feature = "dtype-categorical")]
        DataType::Categorical(revmap, ordering) | DataType::Enum(revmap, ordering) => {
            let is_enum = dtype.is_enum();
            let ctx = match revmap {
                Some(revmap) => {
                    let (num_known_categories, lexical_sort_idxs) = match revmap.as_ref() {
                        RevMapping::Global(map, _, _) => {
                            let num_known_categories =
                                map.keys().max().copied().map_or(0, |m| m + 1);

                            // @TODO: This should probably be cached.
                            let lexical_sort_idxs = (ordered
                                && matches!(ordering, CategoricalOrdering::Lexical))
                            .then(|| {
                                let read_map = crate::STRING_CACHE.read_map();
                                let payloads = read_map.get_current_payloads();
                                assert!(payloads.len() >= num_known_categories as usize);

                                let mut idxs = (0..num_known_categories).collect::<Vec<u32>>();
                                idxs.sort_by_key(|&k| payloads[k as usize].as_str());
                                let mut sort_idxs = vec![0; num_known_categories as usize];
                                for (i, idx) in idxs.into_iter().enumerate_u32() {
                                    sort_idxs[idx as usize] = i;
                                }
                                sort_idxs
                            });

                            (num_known_categories, lexical_sort_idxs)
                        },
                        RevMapping::Local(values, _) => {
                            // @TODO: This should probably be cached.
                            let lexical_sort_idxs = (ordered
                                && matches!(ordering, CategoricalOrdering::Lexical))
                            .then(|| {
                                assert_eq!(values.null_count(), 0);
                                let values: Vec<&str> = values.values_iter().collect();

                                let mut idxs = (0..values.len() as u32).collect::<Vec<u32>>();
                                idxs.sort_by_key(|&k| values[k as usize]);
                                let mut sort_idxs = vec![0; values.len()];
                                for (i, idx) in idxs.into_iter().enumerate_u32() {
                                    sort_idxs[idx as usize] = i;
                                }
                                sort_idxs
                            });

                            (values.len() as u32, lexical_sort_idxs)
                        },
                    };

                    RowEncodingCategoricalContext {
                        num_known_categories,
                        is_enum,
                        lexical_sort_idxs,
                    }
                },
                None => {
                    let num_known_categories = u32::MAX;

                    if matches!(ordering, CategoricalOrdering::Lexical) && ordered {
                        panic!("lexical ordering not yet supported if rev-map not given");
                    }
                    RowEncodingCategoricalContext {
                        num_known_categories,
                        is_enum,
                        lexical_sort_idxs: None,
                    }
                },
            };

            Some(RowEncodingContext::Categorical(ctx))
        },
        #[cfg(feature = "dtype-struct")]
        DataType::Struct(fs) => {
            let mut ctxts = Vec::new();

            for (i, f) in fs.iter().enumerate() {
                if let Some(ctxt) = get_row_encoding_context(f.dtype(), ordered) {
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
                    .map(|f| get_row_encoding_context(f.dtype(), ordered)),
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
        let ctxt = get_row_encoding_context(by.dtype(), false);

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
        let ctxt = get_row_encoding_context(by.dtype(), true);

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
