use arrow::bitmap::utils::{BitChunkIterExact, BitChunksExact};
use arrow::bitmap::Bitmap;
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
                arg_min_bool(ca)
            }
            dt if dt.is_numeric() => {
                with_match_physical_numeric_polars_type!(s.dtype(), |$T| {
                    let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
                    if let Ok(vals) = ca.cont_slice() {
                        arg_min_slice(vals, ca.is_sorted_flag2())
                    } else {
                        arg_min(ca)
                    }
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
                arg_max_bool(ca)
            }
            dt if dt.is_numeric() => {
                with_match_physical_numeric_polars_type!(s.dtype(), |$T| {
                    let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
                    if let Ok(vals) = ca.cont_slice() {
                        arg_max_slice(vals, ca.is_sorted_flag2())
                    } else {
                        arg_max(ca)
                    }
                })
            }
            _ => None,
        }
    }
}

fn arg_max_bool(ca: &BooleanChunked) -> Option<usize> {
    if ca.is_empty() {
        None
    } else if ca.null_count() == ca.len() {
        Some(0)
    }
    // don't check for any, that on itself is already an argmax search
    else if ca.null_count() == 0 && ca.chunks().len() == 1 {
        let arr = ca.downcast_iter().next().unwrap();
        let mask = arr.values();
        Some(first_set_bit(mask))
    } else {
        ca.into_iter()
            .position(|opt_val| matches!(opt_val, Some(true)))
    }
}

fn arg_min_bool(ca: &BooleanChunked) -> Option<usize> {
    if ca.is_empty() || ca.null_count() == ca.len() || ca.all() {
        Some(0)
    } else if ca.null_count() == 0 && ca.chunks().len() == 1 {
        let arr = ca.downcast_iter().next().unwrap();
        let mask = arr.values();
        Some(first_unset_bit(mask))
    } else {
        // also null as we see that as lower in ordering than a set value
        ca.into_iter()
            .position(|opt_val| matches!(opt_val, Some(false) | None))
    }
}

#[inline]
fn get_leading_zeroes(chunk: u64) -> u32 {
    if cfg!(target_endian = "little") {
        chunk.trailing_zeros()
    } else {
        chunk.leading_zeros()
    }
}

#[inline]
fn get_leading_ones(chunk: u64) -> u32 {
    if cfg!(target_endian = "little") {
        chunk.trailing_ones()
    } else {
        chunk.leading_ones()
    }
}

fn first_set_bit_impl<I>(mut mask_chunks: I) -> usize
where
    I: BitChunkIterExact<u64>,
{
    let mut total = 0usize;
    let size = 64;
    for chunk in &mut mask_chunks {
        let pos = get_leading_zeroes(chunk);
        if pos != size {
            return total + pos as usize;
        } else {
            total += size as usize
        }
    }
    if let Some(pos) = mask_chunks.remainder_iter().position(|v| v) {
        total += pos;
        return total;
    }
    // all null, return the first
    0
}

fn first_set_bit(mask: &Bitmap) -> usize {
    if mask.unset_bits() == 0 || mask.unset_bits() == mask.len() {
        return 0;
    }
    let (slice, offset, length) = mask.as_slice();
    if offset == 0 {
        let mask_chunks = BitChunksExact::<u64>::new(slice, length);
        first_set_bit_impl(mask_chunks)
    } else {
        let mask_chunks = mask.chunks::<u64>();
        first_set_bit_impl(mask_chunks)
    }
}

fn first_unset_bit_impl<I>(mut mask_chunks: I) -> usize
where
    I: BitChunkIterExact<u64>,
{
    let mut total = 0usize;
    let size = 64;
    for chunk in &mut mask_chunks {
        let pos = get_leading_ones(chunk);
        if pos != size {
            return total + pos as usize;
        } else {
            total += size as usize
        }
    }
    if let Some(pos) = mask_chunks.remainder_iter().position(|v| !v) {
        total += pos;
        return total;
    }
    // all null, return the first
    0
}

fn first_unset_bit(mask: &Bitmap) -> usize {
    if mask.unset_bits() == 0 || mask.unset_bits() == mask.len() {
        return 0;
    }
    let (slice, offset, length) = mask.as_slice();
    if offset == 0 {
        let mask_chunks = BitChunksExact::<u64>::new(slice, length);
        first_unset_bit_impl(mask_chunks)
    } else {
        let mask_chunks = mask.chunks::<u64>();
        first_unset_bit_impl(mask_chunks)
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

fn arg_min_slice<T>(vals: &[T], is_sorted: IsSorted) -> Option<usize>
where
    T: PartialOrd,
{
    match is_sorted {
        IsSorted::Ascending => Some(0),
        IsSorted::Descending => Some(vals.len()),
        IsSorted::Not => vals
            .iter()
            .enumerate()
            .reduce(|acc, (idx, val)| if acc.1 > val { (idx, val) } else { acc })
            .map(|tpl| tpl.0),
    }
}

fn arg_max_slice<T>(vals: &[T], is_sorted: IsSorted) -> Option<usize>
where
    T: PartialOrd,
{
    match is_sorted {
        IsSorted::Ascending => Some(0),
        IsSorted::Descending => Some(vals.len()),
        IsSorted::Not => vals
            .iter()
            .enumerate()
            .reduce(|acc, (idx, val)| if acc.1 < val { (idx, val) } else { acc })
            .map(|tpl| tpl.0),
    }
}
