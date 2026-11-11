mod access;
mod arrow;
mod cast;

pub(crate) use arrow::{union_array_to_arrow, union_from_arrow};
use polars_arrow::array::builder::{ShareStrategy, StaticArrayBuilder};
use polars_arrow::array::{LargeUnionArray, LargeUnionArrayBuilder};
use polars_compute::gather::take_unchecked;
use polars_compute::propagate_nulls::propagate_nulls_union;
use polars_compute::trim_lists_to_normalized_offsets::trim_lists_to_normalized_offsets_union;
use polars_error::constants::LENGTH_LIMIT_MSG;

use crate::chunked_array::ops::append::new_chunks_owned;
use crate::chunked_array::ops::row_encode::_get_rows_encoded_ca;
use crate::prelude::*;
use crate::runtime::RAYON;
use crate::series::IsSorted;

impl UnionChunked {
    /// The variants of this union.
    pub fn union_fields(&self) -> &UnionFields {
        let DataType::Union(fields) = self.dtype() else {
            unreachable!()
        };
        fields
    }

    /// An all-null union of the given `Union` dtype.
    pub fn full_null_with_dtype(name: PlSmallStr, length: usize, dtype: &DataType) -> Self {
        let arr = LargeUnionArray::new_null(dtype.to_physical_arrow(CompatLevel::newest()), length);
        // SAFETY: the chunk has the physical arrow dtype of `dtype`.
        unsafe { Self::from_chunks_and_dtype_unchecked(name, vec![arr.boxed()], dtype.clone()) }
    }

    pub fn append(&mut self, other: &Self) -> PolarsResult<()> {
        self.append_owned(other.clone())
    }

    pub fn append_owned(&mut self, mut other: Self) -> PolarsResult<()> {
        polars_ensure!(self.dtype() == other.dtype(), append);
        let len = self.len();
        self.length = self
            .length
            .checked_add(other.length)
            .ok_or_else(|| polars_err!(ComputeError: LENGTH_LIMIT_MSG))?;
        self.null_count += other.null_count;
        self.set_sorted_flag(IsSorted::Not);
        new_chunks_owned(&mut self.chunks, std::mem::take(&mut other.chunks), len);
        Ok(())
    }

    pub fn extend(&mut self, other: &Self) -> PolarsResult<()> {
        self.append(other)?;
        self.rechunk_mut();
        Ok(())
    }

    /// Applies `f` to every chunk, or returns `None` if it changed none.
    fn map_chunks_opt(
        &self,
        f: impl Fn(&LargeUnionArray) -> Option<LargeUnionArray>,
    ) -> Option<Self> {
        let mapped: Vec<_> = self.downcast_iter().map(&f).collect();
        if mapped.iter().all(Option::is_none) {
            return None;
        }
        let chunks = mapped
            .into_iter()
            .zip(self.downcast_iter())
            .map(|(new, old)| new.map_or_else(|| old.to_boxed(), |a| a.boxed()))
            .collect();
        // SAFETY: `f` keeps the length, the validity and the dtype.
        Some(unsafe {
            Self::new_with_dims(self.field.clone(), chunks, self.length, self.null_count)
        })
    }

    pub(crate) fn trim_lists_to_normalized_offsets(&self) -> Option<Self> {
        self.map_chunks_opt(trim_lists_to_normalized_offsets_union)
    }

    pub(crate) fn propagate_nulls(&self) -> Option<Self> {
        self.map_chunks_opt(propagate_nulls_union)
    }

    pub fn get_row_encoded(&self, options: SortOptions) -> PolarsResult<BinaryOffsetChunked> {
        _get_rows_encoded_ca(
            self.name().clone(),
            &[self.clone().into_column()],
            &[options.descending],
            &[options.nulls_last],
            false,
        )
    }
}

impl ChunkSort<UnionType> for UnionChunked {
    fn sort_with(&self, mut options: SortOptions) -> UnionChunked {
        options.multithreaded &= RAYON.current_num_threads() > 1;
        let idx = self.arg_sort(options);
        let mut out = unsafe { self.take_unchecked(&idx) };
        out.set_sorted_flag(if options.descending {
            IsSorted::Descending
        } else {
            IsSorted::Ascending
        });
        out
    }

    fn sort(&self, descending: bool) -> UnionChunked {
        self.sort_with(SortOptions::new().with_order_descending(descending))
    }

    fn arg_sort(&self, options: SortOptions) -> IdxCa {
        let bin = self.get_row_encoded(options).unwrap();
        bin.arg_sort(SortOptions {
            maintain_order: options.maintain_order,
            multithreaded: options.multithreaded,
            ..Default::default()
        })
    }
}

impl ChunkTakeUnchecked<IdxCa> for UnionChunked {
    unsafe fn take_unchecked(&self, indices: &IdxCa) -> Self {
        let a = self.rechunk();
        let index = indices.rechunk();

        let chunks = a
            .downcast_iter()
            .zip(index.downcast_iter())
            .map(|(arr, idx)| unsafe { take_unchecked(arr, idx) })
            .collect::<Vec<_>>();
        unsafe { self.copy_with_chunks(chunks) }
    }
}

impl<I: AsRef<[IdxSize]> + ?Sized> ChunkTakeUnchecked<I> for UnionChunked {
    unsafe fn take_unchecked(&self, indices: &I) -> Self {
        let idx = IdxCa::mmap_slice(PlSmallStr::EMPTY, indices.as_ref());
        unsafe { self.take_unchecked(&idx) }
    }
}

impl ChunkExpandAtIndex<UnionType> for UnionChunked {
    fn new_from_index(&self, index: usize, length: usize) -> UnionChunked {
        let (chunk_idx, idx) = self.index_to_chunked_index(index);
        let chunk = self.downcast_get(chunk_idx).unwrap();
        let mut builder = LargeUnionArrayBuilder::new(chunk.dtype().clone());
        builder.subslice_extend_each_repeated(chunk, idx, 1, length, ShareStrategy::Always);
        // SAFETY: the chunk has the dtype of self.
        unsafe { self.copy_with_chunks(vec![builder.freeze().boxed()]) }
    }
}

impl ChunkShift<UnionType> for UnionChunked {
    fn shift(&self, periods: i64) -> UnionChunked {
        let len = self.len() as i64;
        let periods = periods.clamp(-len, len);
        let fill_length = periods.unsigned_abs() as usize;
        let mut slice = self.slice((-periods).max(0), self.len() - fill_length);
        let mut fill = Self::full_null_with_dtype(self.name().clone(), fill_length, self.dtype());

        if periods < 0 {
            slice.append_owned(fill).unwrap();
            slice
        } else {
            fill.append_owned(slice).unwrap();
            fill
        }
    }
}

#[cfg(feature = "zip_with")]
impl ChunkZip<UnionType> for UnionChunked {
    fn zip_with(&self, mask: &BooleanChunked, other: &UnionChunked) -> PolarsResult<UnionChunked> {
        use polars_error::PolarsContext;
        use polars_utils::broadcast::broadcast_len;

        polars_ensure!(
            self.dtype() == other.dtype(),
            SchemaMismatch: "cannot zip unions of dtypes {} and {}", self.dtype(), other.dtype()
        );
        let length = broadcast_len([self.len(), mask.len(), other.len()])
            .context("shapes of `self`, `mask` and `other` are not suitable for `zip_with`")?;

        let mut builder =
            LargeUnionArrayBuilder::new(self.dtype().to_physical_arrow(CompatLevel::newest()));
        StaticArrayBuilder::reserve(&mut builder, length);
        let share = ShareStrategy::Always;
        let mut extend = |ca: &UnionChunked, start: usize, len: usize| {
            if ca.len() == 1 {
                let arr = ca.downcast_iter().find(|a| !a.is_empty()).unwrap();
                builder.subslice_extend_each_repeated(arr, 0, 1, len, share);
                return;
            }
            let (mut chunk, mut offset) = ca.index_to_chunked_index(start);
            let mut remaining = len;
            while remaining > 0 {
                let arr = ca.downcast_get(chunk).unwrap();
                let n = remaining.min(arr.len() - offset);
                builder.subslice_extend(arr, offset, n, share);
                remaining -= n;
                chunk += 1;
                offset = 0;
            }
        };

        // A null mask value selects `other`.
        if mask.len() == 1 {
            let src = if mask.get(0).unwrap_or(false) {
                self
            } else {
                other
            };
            extend(src, 0, length);
        } else {
            let mut run: Option<(bool, usize)> = None;
            let mut i = 0;
            for arr in mask.downcast_iter() {
                let selected = match arr.validity() {
                    Some(v) => arr.values() & v,
                    None => arr.values().clone(),
                };
                for is_true in selected.iter() {
                    match run {
                        Some((v, _)) if v == is_true => {},
                        Some((v, start)) => {
                            extend(if v { self } else { other }, start, i - start);
                            run = Some((is_true, i));
                        },
                        None => run = Some((is_true, i)),
                    }
                    i += 1;
                }
            }
            if let Some((v, start)) = run {
                extend(if v { self } else { other }, start, i - start);
            }
        }

        // SAFETY: the builder has the physical dtype of `self`.
        Ok(unsafe { self.with_chunks(vec![StaticArrayBuilder::freeze(builder).boxed()]) })
    }
}
