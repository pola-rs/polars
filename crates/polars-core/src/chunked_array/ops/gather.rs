use arrow::bitmap::bitmask::BitMask;
use arrow::bitmap::Bitmap;
use arrow::compute::take::take_unchecked;
use polars_error::polars_ensure;
use polars_utils::index::check_bounds;

use crate::prelude::*;
use crate::series::IsSorted;

const BINARY_SEARCH_LIMIT: usize = 8;

pub fn check_bounds_nulls(idx: &PrimitiveArray<IdxSize>, len: IdxSize) -> PolarsResult<()> {
    let mask = BitMask::from_bitmap(idx.validity().unwrap());

    // We iterate in chunks to make the inner loop branch-free.
    for (block_idx, block) in idx.values().chunks(32).enumerate() {
        let mut in_bounds = 0;
        for (i, x) in block.iter().enumerate() {
            in_bounds |= ((*x < len) as u32) << i;
        }
        let m = mask.get_u32(32 * block_idx);
        polars_ensure!(m == m & in_bounds, ComputeError: "gather indices are out of bounds");
    }
    Ok(())
}

pub fn check_bounds_ca(indices: &IdxCa, len: IdxSize) -> PolarsResult<()> {
    let all_valid = indices.downcast_iter().all(|a| {
        if a.null_count() == 0 {
            check_bounds(a.values(), len).is_ok()
        } else {
            check_bounds_nulls(a, len).is_ok()
        }
    });
    polars_ensure!(all_valid, OutOfBounds: "gather indices are out of bounds");
    Ok(())
}

impl<T: PolarsDataType, I: AsRef<[IdxSize]> + ?Sized> ChunkTake<I> for ChunkedArray<T>
where
    ChunkedArray<T>: ChunkTakeUnchecked<I>,
{
    /// Gather values from ChunkedArray by index.
    fn take(&self, indices: &I) -> PolarsResult<Self> {
        check_bounds(indices.as_ref(), self.len() as IdxSize)?;

        // SAFETY: we just checked the indices are valid.
        Ok(unsafe { self.take_unchecked(indices) })
    }
}

impl<T: PolarsDataType> ChunkTake<IdxCa> for ChunkedArray<T>
where
    ChunkedArray<T>: ChunkTakeUnchecked<IdxCa>,
{
    /// Gather values from ChunkedArray by index.
    fn take(&self, indices: &IdxCa) -> PolarsResult<Self> {
        check_bounds_ca(indices, self.len() as IdxSize)?;

        // SAFETY: we just checked the indices are valid.
        Ok(unsafe { self.take_unchecked(indices) })
    }
}

/// Computes cumulative lengths for efficient branchless binary search
/// lookup. The first element is always 0, and the last length of arrs
/// is always ignored (as we already checked that all indices are
/// in-bounds we don't need to check against the last length).
fn cumulative_lengths<A: StaticArray>(arrs: &[&A]) -> [IdxSize; BINARY_SEARCH_LIMIT] {
    assert!(arrs.len() <= BINARY_SEARCH_LIMIT);
    let mut ret = [IdxSize::MAX; BINARY_SEARCH_LIMIT];
    ret[0] = 0;
    for i in 1..arrs.len() {
        ret[i] = ret[i - 1] + arrs[i - 1].len() as IdxSize;
    }
    ret
}

#[rustfmt::skip]
#[inline]
fn resolve_chunked_idx(idx: IdxSize, cumlens: &[IdxSize; BINARY_SEARCH_LIMIT]) -> (usize, usize) {
    // Branchless bitwise binary search.
    let mut chunk_idx = 0;
    chunk_idx += if idx >= cumlens[chunk_idx + 0b100] { 0b0100 } else { 0 };
    chunk_idx += if idx >= cumlens[chunk_idx + 0b010] { 0b0010 } else { 0 };
    chunk_idx += if idx >= cumlens[chunk_idx + 0b001] { 0b0001 } else { 0 };
    (chunk_idx, (idx - cumlens[chunk_idx]) as usize)
}

#[inline]
unsafe fn target_value_unchecked<'a, A: StaticArray>(
    targets: &[&'a A],
    cumlens: &[IdxSize; BINARY_SEARCH_LIMIT],
    idx: IdxSize,
) -> A::ValueT<'a> {
    let (chunk_idx, arr_idx) = resolve_chunked_idx(idx, cumlens);
    let arr = targets.get_unchecked(chunk_idx);
    arr.value_unchecked(arr_idx)
}

#[inline]
unsafe fn target_get_unchecked<'a, A: StaticArray>(
    targets: &[&'a A],
    cumlens: &[IdxSize; BINARY_SEARCH_LIMIT],
    idx: IdxSize,
) -> Option<A::ValueT<'a>> {
    let (chunk_idx, arr_idx) = resolve_chunked_idx(idx, cumlens);
    let arr = targets.get_unchecked(chunk_idx);
    arr.get_unchecked(arr_idx)
}

unsafe fn gather_idx_array_unchecked<A: StaticArray>(
    dtype: ArrowDataType,
    targets: &[&A],
    has_nulls: bool,
    indices: &[IdxSize],
) -> A {
    let it = indices.iter().copied();
    if targets.len() == 1 {
        let target = targets.first().unwrap();
        if has_nulls {
            it.map(|i| target.get_unchecked(i as usize))
                .collect_arr_trusted_with_dtype(dtype)
        } else if let Some(sl) = target.as_slice() {
            // Avoid the Arc overhead from value_unchecked.
            it.map(|i| sl.get_unchecked(i as usize).clone())
                .collect_arr_trusted_with_dtype(dtype)
        } else {
            it.map(|i| target.value_unchecked(i as usize))
                .collect_arr_trusted_with_dtype(dtype)
        }
    } else {
        let cumlens = cumulative_lengths(targets);
        if has_nulls {
            it.map(|i| target_get_unchecked(targets, &cumlens, i))
                .collect_arr_trusted_with_dtype(dtype)
        } else {
            it.map(|i| target_value_unchecked(targets, &cumlens, i))
                .collect_arr_trusted_with_dtype(dtype)
        }
    }
}

impl<T: PolarsDataType, I: AsRef<[IdxSize]> + ?Sized> ChunkTakeUnchecked<I> for ChunkedArray<T>
where
    T: PolarsDataType<HasViews = FalseT, IsStruct = FalseT>,
{
    /// Gather values from ChunkedArray by index.
    unsafe fn take_unchecked(&self, indices: &I) -> Self {
        let rechunked;
        let mut ca = self;
        if self.chunks().len() > BINARY_SEARCH_LIMIT {
            rechunked = self.rechunk();
            ca = &rechunked;
        }
        let targets: Vec<_> = ca.downcast_iter().collect();
        let arr = gather_idx_array_unchecked(
            ca.dtype().to_arrow(CompatLevel::newest()),
            &targets,
            ca.null_count() > 0,
            indices.as_ref(),
        );
        ChunkedArray::from_chunk_iter_like(ca, [arr])
    }
}

pub fn _update_gather_sorted_flag(sorted_arr: IsSorted, sorted_idx: IsSorted) -> IsSorted {
    use crate::series::IsSorted::*;
    match (sorted_arr, sorted_idx) {
        (_, Not) => Not,
        (Not, _) => Not,
        (Ascending, Ascending) => Ascending,
        (Ascending, Descending) => Descending,
        (Descending, Ascending) => Descending,
        (Descending, Descending) => Ascending,
    }
}

impl<T: PolarsDataType> ChunkTakeUnchecked<IdxCa> for ChunkedArray<T>
where
    T: PolarsDataType<HasViews = FalseT, IsStruct = FalseT>,
{
    /// Gather values from ChunkedArray by index.
    unsafe fn take_unchecked(&self, indices: &IdxCa) -> Self {
        let rechunked;
        let mut ca = self;
        if self.chunks().len() > BINARY_SEARCH_LIMIT {
            rechunked = self.rechunk();
            ca = &rechunked;
        }
        let targets_have_nulls = ca.null_count() > 0;
        let targets: Vec<_> = ca.downcast_iter().collect();

        let chunks = indices.downcast_iter().map(|idx_arr| {
            let dtype = ca.dtype().to_arrow(CompatLevel::newest());
            if idx_arr.null_count() == 0 {
                gather_idx_array_unchecked(dtype, &targets, targets_have_nulls, idx_arr.values())
            } else if targets.len() == 1 {
                let target = targets.first().unwrap();
                if targets_have_nulls {
                    idx_arr
                        .iter()
                        .map(|i| target.get_unchecked(*i? as usize))
                        .collect_arr_trusted_with_dtype(dtype)
                } else {
                    idx_arr
                        .iter()
                        .map(|i| Some(target.value_unchecked(*i? as usize)))
                        .collect_arr_trusted_with_dtype(dtype)
                }
            } else {
                let cumlens = cumulative_lengths(&targets);
                if targets_have_nulls {
                    idx_arr
                        .iter()
                        .map(|i| target_get_unchecked(&targets, &cumlens, *i?))
                        .collect_arr_trusted_with_dtype(dtype)
                } else {
                    idx_arr
                        .iter()
                        .map(|i| Some(target_value_unchecked(&targets, &cumlens, *i?)))
                        .collect_arr_trusted_with_dtype(dtype)
                }
            }
        });

        let mut out = ChunkedArray::from_chunk_iter_like(ca, chunks);
        let sorted_flag = _update_gather_sorted_flag(ca.is_sorted_flag(), indices.is_sorted_flag());

        out.set_sorted_flag(sorted_flag);
        out
    }
}

impl ChunkTakeUnchecked<IdxCa> for BinaryChunked {
    /// Gather values from ChunkedArray by index.
    unsafe fn take_unchecked(&self, indices: &IdxCa) -> Self {
        let rechunked = self.rechunk();
        let indices = indices.rechunk();
        let indices_arr = indices.downcast_iter().next().unwrap();
        let chunks = rechunked
            .chunks()
            .iter()
            .map(|arr| take_unchecked(arr.as_ref(), indices_arr))
            .collect::<Vec<_>>();

        let mut out = ChunkedArray::from_chunks(self.name(), chunks);

        let sorted_flag =
            _update_gather_sorted_flag(self.is_sorted_flag(), indices.is_sorted_flag());
        out.set_sorted_flag(sorted_flag);
        out
    }
}

impl ChunkTakeUnchecked<IdxCa> for StringChunked {
    unsafe fn take_unchecked(&self, indices: &IdxCa) -> Self {
        self.as_binary()
            .take_unchecked(indices)
            .to_string_unchecked()
    }
}

impl<I: AsRef<[IdxSize]> + ?Sized> ChunkTakeUnchecked<I> for BinaryChunked {
    /// Gather values from ChunkedArray by index.
    unsafe fn take_unchecked(&self, indices: &I) -> Self {
        let indices = IdxCa::mmap_slice("", indices.as_ref());
        self.take_unchecked(&indices)
    }
}

impl<I: AsRef<[IdxSize]> + ?Sized> ChunkTakeUnchecked<I> for StringChunked {
    /// Gather values from ChunkedArray by index.
    unsafe fn take_unchecked(&self, indices: &I) -> Self {
        self.as_binary()
            .take_unchecked(indices)
            .to_string_unchecked()
    }
}

#[cfg(feature = "dtype-struct")]
impl ChunkTakeUnchecked<IdxCa> for StructChunked {
    unsafe fn take_unchecked(&self, indices: &IdxCa) -> Self {
        let a = self.rechunk();
        let index = indices.rechunk();

        let chunks = a
            .downcast_iter()
            .zip(index.downcast_iter())
            .map(|(arr, idx)| take_unchecked(arr, idx))
            .collect::<Vec<_>>();
        self.copy_with_chunks(chunks)
    }
}

#[cfg(feature = "dtype-struct")]
impl<I: AsRef<[IdxSize]> + ?Sized> ChunkTakeUnchecked<I> for StructChunked {
    unsafe fn take_unchecked(&self, indices: &I) -> Self {
        let idx = IdxCa::mmap_slice("", indices.as_ref());
        self.take_unchecked(&idx)
    }
}

impl IdxCa {
    pub fn with_nullable_idx<T, F: FnOnce(&IdxCa) -> T>(idx: &[NullableIdxSize], f: F) -> T {
        let validity: Bitmap = idx.iter().map(|idx| !idx.is_null_idx()).collect_trusted();
        let idx = bytemuck::cast_slice::<_, IdxSize>(idx);
        let arr = unsafe { arrow::ffi::mmap::slice(idx) };
        let arr = arr.with_validity_typed(Some(validity));
        let ca = IdxCa::with_chunk("", arr);

        f(&ca)
    }
}
