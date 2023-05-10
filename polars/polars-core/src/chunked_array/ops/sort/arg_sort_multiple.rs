use polars_arrow::data_types::IsFloat;
use polars_row::{convert_columns, RowsEncoded, SortField};
use polars_utils::iter::EnumerateIdxTrait;

use super::*;
#[cfg(feature = "dtype-struct")]
use crate::utils::_split_offsets;
use crate::POOL;

pub(crate) fn args_validate<T: PolarsDataType>(
    ca: &ChunkedArray<T>,
    other: &[Series],
    descending: &[bool],
) -> PolarsResult<()> {
    for s in other {
        assert_eq!(ca.len(), s.len());
    }
    polars_ensure!(other.len() == (descending.len() - 1),
        ComputeError:
        "the amount of ordering booleans: {} does not match the number of series: {}",
        descending.len(), other.len() + 1,
    );
    Ok(())
}

pub(crate) fn arg_sort_multiple_impl<T: PartialOrd + Send + IsFloat + Copy>(
    mut vals: Vec<(IdxSize, T)>,
    options: &SortMultipleOptions,
) -> PolarsResult<IdxCa> {
    let descending = &options.descending;
    debug_assert_eq!(descending.len() - 1, options.other.len());
    let compare_inner: Vec<_> = options
        .other
        .iter()
        .map(|s| s.into_partial_ord_inner())
        .collect_trusted();

    let first_descending = descending[0];
    POOL.install(|| {
        vals.par_sort_by(|tpl_a, tpl_b| {
            match (first_descending, compare_fn_nan_max(&tpl_a.1, &tpl_b.1)) {
                // if ordering is equal, we check the other arrays until we find a non-equal ordering
                // if we have exhausted all arrays, we keep the equal ordering.
                (_, Ordering::Equal) => {
                    let idx_a = tpl_a.0 as usize;
                    let idx_b = tpl_b.0 as usize;
                    unsafe {
                        ordering_other_columns(
                            &compare_inner,
                            descending.get_unchecked(1..),
                            idx_a,
                            idx_b,
                        )
                    }
                }
                (true, Ordering::Less) => Ordering::Greater,
                (true, Ordering::Greater) => Ordering::Less,
                (_, ord) => ord,
            }
        });
    });
    let ca: NoNull<IdxCa> = vals.into_iter().map(|(idx, _v)| idx).collect_trusted();
    // Don't set to sorted. Argsort indices are not sorted.
    Ok(ca.into_inner())
}

pub fn _get_rows_encoded_compat_array(by: &Series) -> PolarsResult<ArrayRef> {
    let by = convert_sort_column_multi_sort(by, true)?;
    let by = by.rechunk();

    let out = match by.dtype() {
        #[cfg(feature = "dtype-categorical")]
        DataType::Categorical(_) => {
            let ca = by.categorical().unwrap();
            if ca.use_lexical_sort() {
                by.to_arrow(0)
            } else {
                ca.logical().chunks[0].clone()
            }
        }
        _ => by.to_arrow(0),
    };
    Ok(out)
}

#[cfg(feature = "dtype-struct")]
pub(crate) fn encode_rows_vertical(by: &[Series]) -> PolarsResult<BinaryChunked> {
    let n_threads = POOL.current_num_threads();
    let len = by[0].len();
    let splits = _split_offsets(len, n_threads);
    let descending = vec![false; by.len()];

    let chunks = splits
        .into_par_iter()
        .map(|(offset, len)| {
            let sliced = by
                .iter()
                .map(|s| s.slice(offset as i64, len))
                .collect::<Vec<_>>();
            let rows = _get_rows_encoded(&sliced, &descending, false)?;
            Ok(Box::new(rows.into_array()) as ArrayRef)
        })
        .collect::<PolarsResult<_>>()?;

    unsafe { Ok(BinaryChunked::from_chunks("", chunks)) }
}

pub fn _get_rows_encoded(
    by: &[Series],
    descending: &[bool],
    nulls_last: bool,
) -> PolarsResult<RowsEncoded> {
    debug_assert_eq!(by.len(), descending.len());
    let mut cols = Vec::with_capacity(by.len());
    let mut fields = Vec::with_capacity(by.len());
    for (by, descending) in by.iter().zip(descending) {
        let arr = _get_rows_encoded_compat_array(by)?;

        let sort_field = SortField {
            descending: *descending,
            nulls_last,
        };
        match arr.data_type() {
            // flatten the struct fields
            ArrowDataType::Struct(_) => {
                let arr = arr.as_any().downcast_ref::<StructArray>().unwrap();
                for arr in arr.values() {
                    cols.push(arr.clone() as ArrayRef);
                    fields.push(sort_field.clone())
                }
            }
            _ => {
                cols.push(arr);
                fields.push(sort_field)
            }
        }
    }
    Ok(convert_columns(&cols, &fields))
}

pub fn _get_rows_encoded_ca(
    name: &str,
    by: &[Series],
    descending: &[bool],
    nulls_last: bool,
) -> PolarsResult<BinaryChunked> {
    _get_rows_encoded(by, descending, nulls_last)
        .map(|rows| unsafe { BinaryChunked::from_chunks(name, vec![Box::new(rows.into_array())]) })
}

pub(crate) fn argsort_multiple_row_fmt(
    by: &[Series],
    mut descending: Vec<bool>,
    nulls_last: bool,
    parallel: bool,
) -> PolarsResult<IdxCa> {
    _broadcast_descending(by.len(), &mut descending);

    let rows_encoded = _get_rows_encoded(by, &descending, nulls_last)?;
    let mut items: Vec<_> = rows_encoded.iter().enumerate_idx().collect();

    if parallel {
        POOL.install(|| items.par_sort_by(|a, b| a.1.cmp(b.1)));
    } else {
        items.sort_by(|a, b| a.1.cmp(b.1));
    }

    let ca: NoNull<IdxCa> = items.into_iter().map(|tpl| tpl.0).collect();
    Ok(ca.into_inner())
}
