use polars_core::prelude::{Column, IdxCa, Int64Chunked, ListChunked};
use polars_core::series::Series;
use polars_error::{PolarsResult, polars_bail};
use polars_utils::IdxSize;

use super::ListNameSpaceImpl;
use crate::series::convert_and_bound_idx_ca;

pub fn lst_get(ca: &ListChunked, index: &Int64Chunked, null_on_oob: bool) -> PolarsResult<Column> {
    match index.len() {
        1 => {
            let index = index.get(0);
            if let Some(index) = index {
                ca.lst_get(index, null_on_oob).map(Column::from)
            } else {
                Ok(Column::full_null(
                    ca.name().clone(),
                    ca.len(),
                    ca.inner_dtype(),
                ))
            }
        },
        len if len == ca.len() => {
            let tmp = ca.rechunk();
            // TODO(polars-array-scalar): the ranges are read off the offsets as a slice, so
            // scalar offsets are written out here rather than the one range they share being
            // indexed once.
            let arr = tmp.downcast_as_array().to_flat();
            let offsets = arr.offsets().as_slice();
            let take_by = if ca.null_count() == 0 {
                index
                    .iter()
                    .enumerate()
                    .map(|(i, opt_idx)| match opt_idx {
                        Some(idx) => {
                            let (start, end) = unsafe {
                                (
                                    *offsets.get_unchecked(i) as i64,
                                    *offsets.get_unchecked(i + 1) as i64,
                                )
                            };
                            let offset = if idx >= 0 { start + idx } else { end + idx };
                            if offset >= end || offset < start || start == end {
                                if null_on_oob {
                                    Ok(None)
                                } else {
                                    polars_bail!(ComputeError: "get index is out of bounds");
                                }
                            } else {
                                Ok(Some(offset as IdxSize))
                            }
                        },
                        None => Ok(None),
                    })
                    .collect::<Result<IdxCa, _>>()?
            } else {
                index
                    .iter()
                    .zip(arr.validity().unwrap())
                    .enumerate()
                    .map(|(i, (opt_idx, valid))| match (valid, opt_idx) {
                        (true, Some(idx)) => {
                            let (start, end) = unsafe {
                                (
                                    *offsets.get_unchecked(i) as i64,
                                    *offsets.get_unchecked(i + 1) as i64,
                                )
                            };
                            let offset = if idx >= 0 { start + idx } else { end + idx };
                            if offset >= end || offset < start || start == end {
                                if null_on_oob {
                                    Ok(None)
                                } else {
                                    polars_bail!(ComputeError: "get index is out of bounds");
                                }
                            } else {
                                Ok(Some(offset as IdxSize))
                            }
                        },
                        _ => Ok(None),
                    })
                    .collect::<Result<IdxCa, _>>()?
            };
            // The values of a list array carry no logical type; the physical inner one is what
            // `from_physical_unchecked` below turns back into the logical one.
            let s = unsafe {
                Series::from_chunks_and_dtype_unchecked(
                    ca.name().clone(),
                    vec![arr.values().to_boxed()],
                    &ca.inner_dtype().to_physical(),
                )
            };
            unsafe {
                s.take_unchecked(&take_by)
                    .from_physical_unchecked(ca.inner_dtype())
                    .map(Column::from)
            }
        },
        _ if ca.len() == 1 => {
            if let Some(list) = ca.get(0) {
                let idx = convert_and_bound_idx_ca(index, list.len(), null_on_oob)?;
                // As above: the element carries no logical type of its own.
                let s = unsafe {
                    Series::from_chunks_and_dtype_unchecked(
                        ca.name().clone(),
                        vec![list],
                        &ca.inner_dtype().to_physical(),
                    )
                };
                unsafe {
                    s.take_unchecked(&idx)
                        .from_physical_unchecked(ca.inner_dtype())
                        .map(Column::from)
                }
            } else {
                Ok(Column::full_null(
                    ca.name().clone(),
                    ca.len(),
                    ca.inner_dtype(),
                ))
            }
        },
        len => polars_bail!(
            ComputeError:
            "`list.get` expression got an index array of length {} while the list has {} elements",
            len, ca.len()
        ),
    }
}
