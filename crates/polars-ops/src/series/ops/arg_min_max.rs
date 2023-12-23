use argminmax::ArgMinMax;
use arrow::array::Array;
use arrow::legacy::bit_util::*;
use polars_core::series::IsSorted;
use polars_core::with_match_physical_numeric_polars_type;

use super::*;

/// Argmin/ Argmax
pub trait ArgAgg {
    /// Get the index of the minimal value
    fn arg_min(&self) -> Option<usize>;
    /// Get the index of the maximal value
    fn arg_max(&self) -> Option<usize>;
}

impl ArgAgg for Series {
    fn arg_min(&self) -> Option<usize> {
        use DataType::*;
        let s = self.to_physical_repr();
        match s.dtype() {
            String => {
                let ca = s.utf8().unwrap();
                arg_min_str(ca)
            },
            Boolean => {
                let ca = s.bool().unwrap();
                arg_min_bool(ca)
            },
            dt if dt.is_numeric() => {
                with_match_physical_numeric_polars_type!(s.dtype(), |$T| {
                    let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
                    if ca.is_empty() || ca.null_count() == ca.len() { // because argminmax assumes not empty
                        None
                    } else if let Ok(vals) = ca.cont_slice() {
                        arg_min_numeric_slice(vals, ca.is_sorted_flag())
                    } else {
                        arg_min_numeric(ca)
                    }
                })
            },
            _ => None,
        }
    }

    fn arg_max(&self) -> Option<usize> {
        use DataType::*;
        let s = self.to_physical_repr();
        match s.dtype() {
            String => {
                let ca = s.utf8().unwrap();
                arg_max_str(ca)
            },
            Boolean => {
                let ca = s.bool().unwrap();
                arg_max_bool(ca)
            },
            dt if dt.is_numeric() => {
                with_match_physical_numeric_polars_type!(s.dtype(), |$T| {
                    let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
                    if ca.is_empty() || ca.null_count() == ca.len(){ // because argminmax assumes not empty
                        None
                    } else if let Ok(vals) = ca.cont_slice() {
                        arg_max_numeric_slice(vals, ca.is_sorted_flag())
                    } else {
                        arg_max_numeric(ca)
                    }
                })
            },
            _ => None,
        }
    }
}

pub(crate) fn arg_max_bool(ca: &BooleanChunked) -> Option<usize> {
    if ca.is_empty() || ca.null_count() == ca.len() {
        None
    }
    // don't check for any, that on itself is already an argmax search
    else if ca.null_count() == 0 && ca.chunks().len() == 1 {
        let arr = ca.downcast_iter().next().unwrap();
        let mask = arr.values();
        Some(first_set_bit(mask))
    } else {
        let mut first_false_idx: Option<usize> = None;
        ca.into_iter()
            .enumerate()
            .find_map(|(idx, val)| match val {
                Some(true) => Some(idx),
                Some(false) if first_false_idx.is_none() => {
                    first_false_idx = Some(idx);
                    None
                },
                _ => None,
            })
            .or(first_false_idx)
    }
}

fn arg_min_bool(ca: &BooleanChunked) -> Option<usize> {
    if ca.is_empty() || ca.null_count() == ca.len() {
        None
    } else if ca.null_count() == 0 && ca.chunks().len() == 1 {
        let arr = ca.downcast_iter().next().unwrap();
        let mask = arr.values();
        Some(first_unset_bit(mask))
    } else {
        let mut first_true_idx: Option<usize> = None;
        ca.into_iter()
            .enumerate()
            .find_map(|(idx, val)| match val {
                Some(false) => Some(idx),
                Some(true) if first_true_idx.is_none() => {
                    first_true_idx = Some(idx);
                    None
                },
                _ => None,
            })
            .or(first_true_idx)
    }
}

fn arg_min_str(ca: &StringChunked) -> Option<usize> {
    if ca.is_empty() || ca.null_count() == ca.len() {
        return None;
    }
    match ca.is_sorted_flag() {
        IsSorted::Ascending => ca.first_non_null(),
        IsSorted::Descending => ca.last_non_null(),
        IsSorted::Not => ca
            .into_iter()
            .enumerate()
            .flat_map(|(idx, val)| val.map(|val| (idx, val)))
            .reduce(|acc, (idx, val)| if acc.1 > val { (idx, val) } else { acc })
            .map(|tpl| tpl.0),
    }
}

fn arg_max_str(ca: &StringChunked) -> Option<usize> {
    if ca.is_empty() || ca.null_count() == ca.len() {
        return None;
    }
    match ca.is_sorted_flag() {
        IsSorted::Ascending => ca.last_non_null(),
        IsSorted::Descending => ca.first_non_null(),
        IsSorted::Not => ca
            .into_iter()
            .enumerate()
            .reduce(|acc, (idx, val)| if acc.1 < val { (idx, val) } else { acc })
            .map(|tpl| tpl.0),
    }
}

fn arg_min_numeric<'a, T>(ca: &'a ChunkedArray<T>) -> Option<usize>
where
    T: PolarsNumericType,
    for<'b> &'b [T::Native]: ArgMinMax,
{
    match ca.is_sorted_flag() {
        IsSorted::Ascending => ca.first_non_null(),
        IsSorted::Descending => ca.last_non_null(),
        IsSorted::Not => {
            ca.downcast_iter()
                .fold((None, None, 0), |acc, arr| {
                    if arr.len() == 0 {
                        return acc;
                    }
                    let chunk_min: Option<(usize, T::Native)> = if arr.null_count() > 0 {
                        arr.into_iter()
                            .enumerate()
                            .flat_map(|(idx, val)| val.map(|val| (idx, *val)))
                            .reduce(|acc, (idx, val)| if acc.1 > val { (idx, val) } else { acc })
                    } else {
                        // When no nulls & array not empty => we can use fast argminmax
                        let min_idx: usize = arr.values().as_slice().argmin();
                        Some((min_idx, arr.value(min_idx)))
                    };

                    let new_offset: usize = acc.2 + arr.len();
                    match acc {
                        (Some(_), Some(acc_v), offset) => match chunk_min {
                            Some((idx, val)) if val < acc_v => {
                                (Some(idx + offset), Some(val), new_offset)
                            },
                            _ => (acc.0, acc.1, new_offset),
                        },
                        (None, None, offset) => match chunk_min {
                            Some((idx, val)) => (Some(idx + offset), Some(val), new_offset),
                            None => (None, None, new_offset),
                        },
                        _ => unreachable!(),
                    }
                })
                .0
        },
    }
}

fn arg_max_numeric<'a, T>(ca: &'a ChunkedArray<T>) -> Option<usize>
where
    T: PolarsNumericType,
    for<'b> &'b [T::Native]: ArgMinMax,
{
    match ca.is_sorted_flag() {
        IsSorted::Ascending => ca.last_non_null(),
        IsSorted::Descending => ca.first_non_null(),
        IsSorted::Not => {
            ca.downcast_iter()
                .fold((None, None, 0), |acc, arr| {
                    if arr.len() == 0 {
                        return acc;
                    }
                    let chunk_max: Option<(usize, T::Native)> = if arr.null_count() > 0 {
                        // When there are nulls, we should compare Option<T::Native>
                        arr.into_iter()
                            .enumerate()
                            .flat_map(|(idx, val)| val.map(|val| (idx, *val)))
                            .reduce(|acc, (idx, val)| if acc.1 < val { (idx, val) } else { acc })
                    } else {
                        // When no nulls & array not empty => we can use fast argminmax
                        let max_idx: usize = arr.values().as_slice().argmax();
                        Some((max_idx, arr.value(max_idx)))
                    };

                    let new_offset: usize = acc.2 + arr.len();
                    match acc {
                        (Some(_), Some(acc_v), offset) => match chunk_max {
                            Some((idx, val)) if acc_v < val => {
                                (Some(idx + offset), Some(val), new_offset)
                            },
                            _ => (acc.0, acc.1, new_offset),
                        },
                        (None, None, offset) => match chunk_max {
                            Some((idx, val)) => (Some(idx + offset), Some(val), new_offset),
                            None => (None, None, new_offset),
                        },
                        _ => unreachable!(),
                    }
                })
                .0
        },
    }
}

fn arg_min_numeric_slice<T>(vals: &[T], is_sorted: IsSorted) -> Option<usize>
where
    for<'a> &'a [T]: ArgMinMax,
{
    match is_sorted {
        // all vals are not null guarded by cont_slice
        IsSorted::Ascending => Some(0),
        // all vals are not null guarded by cont_slice
        IsSorted::Descending => Some(vals.len() - 1),
        IsSorted::Not => Some(vals.argmin()), // assumes not empty
    }
}

fn arg_max_numeric_slice<T>(vals: &[T], is_sorted: IsSorted) -> Option<usize>
where
    for<'a> &'a [T]: ArgMinMax,
{
    match is_sorted {
        // all vals are not null guarded by cont_slice
        IsSorted::Ascending => Some(vals.len() - 1),
        // all vals are not null guarded by cont_slice
        IsSorted::Descending => Some(0),
        IsSorted::Not => Some(vals.argmax()), // assumes not empty
    }
}
