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
            Utf8 => {
                let ca = s.utf8().unwrap();
                arg_min(ca)
            }
            Boolean => {
                let ca = s.bool().unwrap();
                arg_min(ca)
            }
            dt if dt.is_numeric() => {
                with_match_physical_numeric_polars_type!(s.dtype(), |$T| {
                    let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
                    arg_min(ca)
                })
            }
            _ => None,
        }
    }

    fn arg_max(&self) -> Option<usize> {
        use DataType::*;
        let s = self.to_physical_repr();
        match s.dtype() {
            Utf8 => {
                let ca = s.utf8().unwrap();
                arg_max(ca)
            }
            Boolean => {
                let ca = s.bool().unwrap();
                arg_max(ca)
            }
            dt if dt.is_numeric() => {
                with_match_physical_numeric_polars_type!(s.dtype(), |$T| {
                    let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
                    arg_max(ca)
                })
            }
            _ => None,
        }
    }
}

fn arg_min<'a, T>(ca: &'a ChunkedArray<T>) -> Option<usize>
where
    T: PolarsDataType,
    &'a ChunkedArray<T>: IntoIterator,
    <&'a ChunkedArray<T> as IntoIterator>::Item: PartialOrd,
{
    match ca.is_sorted_flag2() {
        IsSorted::Ascending => Some(0),
        IsSorted::Descending => Some(ca.len()),
        IsSorted::Not => ca
            .into_iter()
            .enumerate()
            .reduce(|acc, (idx, val)| if acc.1 > val { (idx, val) } else { acc })
            .map(|tpl| tpl.0),
    }
}

pub(crate) fn arg_max<'a, T>(ca: &'a ChunkedArray<T>) -> Option<usize>
where
    T: PolarsDataType,
    &'a ChunkedArray<T>: IntoIterator,
    <&'a ChunkedArray<T> as IntoIterator>::Item: PartialOrd,
{
    match ca.is_sorted_flag2() {
        IsSorted::Ascending => Some(ca.len()),
        IsSorted::Descending => Some(0),
        IsSorted::Not => ca
            .into_iter()
            .enumerate()
            .reduce(|acc, (idx, val)| if acc.1 < val { (idx, val) } else { acc })
            .map(|tpl| tpl.0),
    }
}
