use polars_core::prelude::{Column, IdxCa, Int64Chunked, ListChunked};
use polars_core::series::Series;
use polars_error::{PolarsResult, polars_bail};
use polars_utils::IdxSize;

use super::ListNameSpaceImpl;

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
            let arr = tmp.downcast_as_array();
            let offsets = arr.offsets().as_slice();
            let take_by = if ca.null_count() == 0 {
                index
                    .iter()
                    .enumerate()
                    .map(|(i, opt_idx)| match opt_idx {
                        Some(idx) => {
                            let (start, end) = unsafe {
                                (*offsets.get_unchecked(i), *offsets.get_unchecked(i + 1))
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
                                (*offsets.get_unchecked(i), *offsets.get_unchecked(i + 1))
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
            let s = Series::try_from((ca.name().clone(), arr.values().clone())).unwrap();
            unsafe { s.take_unchecked(&take_by) }
                .cast(ca.inner_dtype())
                .map(Column::from)
        },
        _ if ca.len() == 1 => {
            if ca.null_count() > 0 {
                return Ok(Column::full_null(
                    ca.name().clone(),
                    index.len(),
                    ca.inner_dtype(),
                ));
            }
            let tmp = ca.rechunk();
            let arr = tmp.downcast_as_array();
            let offsets = arr.offsets().as_slice();
            let start = offsets[0];
            let end = offsets[1];
            let out_of_bounds = |offset| offset >= end || offset < start || start == end;
            let take_by: IdxCa = index
                .iter()
                .map(|opt_idx| match opt_idx {
                    Some(idx) => {
                        let offset = if idx >= 0 { start + idx } else { end + idx };
                        if out_of_bounds(offset) {
                            if null_on_oob {
                                Ok(None)
                            } else {
                                polars_bail!(ComputeError: "get index is out of bounds");
                            }
                        } else {
                            let Ok(offset) = IdxSize::try_from(offset) else {
                                polars_bail!(ComputeError: "get index is out of bounds");
                            };
                            Ok(Some(offset))
                        }
                    },
                    None => Ok(None),
                })
                .collect::<Result<IdxCa, _>>()?;

            let s = Series::try_from((ca.name().clone(), arr.values().clone())).unwrap();
            unsafe { s.take_unchecked(&take_by) }
                .cast(ca.inner_dtype())
                .map(Column::from)
        },
        len => polars_bail!(
            ComputeError:
            "`list.get` expression got an index array of length {} while the list has {} elements",
            len, ca.len()
        ),
    }
}
