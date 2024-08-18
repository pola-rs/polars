use arrow::compute::utils::combine_validities_and_many;
use compare_inner::NullOrderCmp;
use polars_row::{convert_columns, EncodingField, RowsEncoded};
use polars_utils::itertools::Itertools;

use super::*;
use crate::utils::_split_offsets;

pub(crate) fn args_validate<T: PolarsDataType>(
    ca: &ChunkedArray<T>,
    other: &[Series],
    param_value: &[bool],
    param_name: &str,
) -> PolarsResult<()> {
    for s in other {
        assert_eq!(ca.len(), s.len());
    }
    polars_ensure!(other.len() == (param_value.len() - 1),
        ComputeError:
        "the length of `{}` ({}) does not match the number of series ({})",
        param_name, param_value.len(), other.len() + 1,
    );
    Ok(())
}

pub(crate) fn arg_sort_multiple_impl<T: NullOrderCmp + Send + Copy>(
    mut vals: Vec<(IdxSize, T)>,
    by: &[Series],
    options: &SortMultipleOptions,
) -> PolarsResult<IdxCa> {
    let nulls_last = &options.nulls_last;
    let descending = &options.descending;

    debug_assert_eq!(descending.len() - 1, by.len());
    debug_assert_eq!(nulls_last.len() - 1, by.len());

    let compare_inner: Vec<_> = by
        .iter()
        .map(|s| s.into_total_ord_inner())
        .collect_trusted();

    let first_descending = descending[0];
    let first_nulls_last = nulls_last[0];

    let compare = |tpl_a: &(_, T), tpl_b: &(_, T)| -> Ordering {
        match (
            first_descending,
            tpl_a
                .1
                .null_order_cmp(&tpl_b.1, first_nulls_last ^ first_descending),
        ) {
            // if ordering is equal, we check the other arrays until we find a non-equal ordering
            // if we have exhausted all arrays, we keep the equal ordering.
            (_, Ordering::Equal) => {
                let idx_a = tpl_a.0 as usize;
                let idx_b = tpl_b.0 as usize;
                unsafe {
                    ordering_other_columns(
                        &compare_inner,
                        descending.get_unchecked(1..),
                        nulls_last.get_unchecked(1..),
                        idx_a,
                        idx_b,
                    )
                }
            },
            (true, Ordering::Less) => Ordering::Greater,
            (true, Ordering::Greater) => Ordering::Less,
            (_, ord) => ord,
        }
    };

    match (options.multithreaded, options.maintain_order) {
        (true, true) => POOL.install(|| {
            vals.par_sort_by(compare);
        }),
        (true, false) => POOL.install(|| {
            vals.par_sort_unstable_by(compare);
        }),
        (false, true) => vals.sort_by(compare),
        (false, false) => vals.sort_unstable_by(compare),
    }

    let ca: NoNull<IdxCa> = vals.into_iter().map(|(idx, _v)| idx).collect_trusted();
    // Don't set to sorted. Argsort indices are not sorted.
    Ok(ca.into_inner())
}

pub fn _get_rows_encoded_compat_array(by: &Series) -> PolarsResult<ArrayRef> {
    let by = convert_sort_column_multi_sort(by)?;
    let by = by.rechunk();

    let out = match by.dtype() {
        #[cfg(feature = "dtype-categorical")]
        DataType::Categorical(_, _) | DataType::Enum(_, _) => {
            let ca = by.categorical().unwrap();
            if ca.uses_lexical_ordering() {
                by.to_arrow(0, CompatLevel::newest())
            } else {
                ca.physical().chunks[0].clone()
            }
        },
        // Take physical
        _ => by.chunks()[0].clone(),
    };
    Ok(out)
}

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

    Ok(BinaryOffsetChunked::from_chunk_iter("", chunks?))
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

    Ok(BinaryOffsetChunked::from_chunk_iter("", chunks?))
}

pub(crate) fn encode_rows_unordered(by: &[Series]) -> PolarsResult<BinaryOffsetChunked> {
    let rows = _get_rows_encoded_unordered(by)?;
    Ok(BinaryOffsetChunked::with_chunk("", rows.into_array()))
}

pub fn _get_rows_encoded_unordered(by: &[Series]) -> PolarsResult<RowsEncoded> {
    let mut cols = Vec::with_capacity(by.len());
    let mut fields = Vec::with_capacity(by.len());
    for by in by {
        let arr = _get_rows_encoded_compat_array(by)?;
        let field = EncodingField::new_unsorted();
        match arr.data_type() {
            // Flatten the struct fields.
            ArrowDataType::Struct(_) => {
                let arr = arr.as_any().downcast_ref::<StructArray>().unwrap();
                for arr in arr.values() {
                    cols.push(arr.clone() as ArrayRef);
                    fields.push(field)
                }
            },
            _ => {
                cols.push(arr);
                fields.push(field)
            },
        }
    }
    Ok(convert_columns(&cols, &fields))
}

pub fn _get_rows_encoded(
    by: &[Series],
    descending: &[bool],
    nulls_last: &[bool],
) -> PolarsResult<RowsEncoded> {
    debug_assert_eq!(by.len(), descending.len());
    debug_assert_eq!(by.len(), nulls_last.len());

    let mut cols = Vec::with_capacity(by.len());
    let mut fields = Vec::with_capacity(by.len());

    for ((by, desc), null_last) in by.iter().zip(descending).zip(nulls_last) {
        let arr = _get_rows_encoded_compat_array(by)?;
        let sort_field = EncodingField {
            descending: *desc,
            nulls_last: *null_last,
            no_order: false,
        };
        match arr.data_type() {
            // Flatten the struct fields.
            ArrowDataType::Struct(_) => {
                let arr = arr.as_any().downcast_ref::<StructArray>().unwrap();
                let arr = arr.propagate_nulls();
                for value_arr in arr.values() {
                    cols.push(value_arr.clone() as ArrayRef);
                    fields.push(sort_field);
                }
            },
            _ => {
                cols.push(arr);
                fields.push(sort_field);
            },
        }
    }
    Ok(convert_columns(&cols, &fields))
}

pub fn _get_rows_encoded_ca(
    name: &str,
    by: &[Series],
    descending: &[bool],
    nulls_last: &[bool],
) -> PolarsResult<BinaryOffsetChunked> {
    _get_rows_encoded(by, descending, nulls_last)
        .map(|rows| BinaryOffsetChunked::with_chunk(name, rows.into_array()))
}

pub fn _get_rows_encoded_arr(
    by: &[Series],
    descending: &[bool],
    nulls_last: &[bool],
) -> PolarsResult<BinaryArray<i64>> {
    _get_rows_encoded(by, descending, nulls_last).map(|rows| rows.into_array())
}

pub fn _get_rows_encoded_ca_unordered(
    name: &str,
    by: &[Series],
) -> PolarsResult<BinaryOffsetChunked> {
    _get_rows_encoded_unordered(by)
        .map(|rows| BinaryOffsetChunked::with_chunk(name, rows.into_array()))
}

pub(crate) fn argsort_multiple_row_fmt(
    by: &[Series],
    mut descending: Vec<bool>,
    mut nulls_last: Vec<bool>,
    parallel: bool,
) -> PolarsResult<IdxCa> {
    _broadcast_bools(by.len(), &mut descending);
    _broadcast_bools(by.len(), &mut nulls_last);

    let rows_encoded = _get_rows_encoded(by, &descending, &nulls_last)?;
    let mut items: Vec<_> = rows_encoded.iter().enumerate_idx().collect();

    if parallel {
        POOL.install(|| items.par_sort_by(|a, b| a.1.cmp(b.1)));
    } else {
        items.sort_by(|a, b| a.1.cmp(b.1));
    }

    let ca: NoNull<IdxCa> = items.into_iter().map(|tpl| tpl.0).collect();
    Ok(ca.into_inner())
}
