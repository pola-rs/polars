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
        use std::borrow::Cow;

        use polars_error::PolarsContext;
        use polars_utils::broadcast::broadcast_len;

        use crate::utils::{align_chunks_binary, align_chunks_ternary};

        polars_ensure!(
            self.dtype() == other.dtype(),
            SchemaMismatch: "cannot zip unions of dtypes {} and {}", self.dtype(), other.dtype()
        );
        let length = broadcast_len([self.len(), mask.len(), other.len()])
            .context("shapes of `self`, `mask` and `other` are not suitable for `zip_with`")?;

        // A null mask value selects `other`.
        if mask.len() == 1 {
            let src = if mask.get(0).unwrap_or(false) {
                self
            } else {
                other
            };
            return Ok(if src.len() == length {
                src.clone()
            } else {
                src.new_from_index(0, length)
            });
        }

        let l_scalar = (self.len() == 1 && length != 1).then_some(self);
        let r_scalar = (other.len() == 1 && length != 1).then_some(other);
        let (mask, l, r) = match (l_scalar, r_scalar) {
            (None, None) => {
                let (m, l, r) = align_chunks_ternary(mask, self, other);
                (m, Some(l), Some(r))
            },
            (None, Some(_)) => {
                let (m, l) = align_chunks_binary(mask, self);
                (m, Some(l), None)
            },
            (Some(_), None) => {
                let (m, r) = align_chunks_binary(mask, other);
                (m, None, Some(r))
            },
            (Some(_), Some(_)) => (Cow::Borrowed(mask), None, None),
        };
        fn side<'a>(
            col: Option<&'a UnionChunked>,
            scalar: Option<&'a UnionChunked>,
            c: usize,
        ) -> ZipSide<'a> {
            match col {
                Some(col) => ZipSide::Column(col.downcast_get(c).unwrap()),
                None => {
                    let scalar = scalar.unwrap();
                    ZipSide::Scalar(scalar.downcast_iter().find(|a| !a.is_empty()).unwrap())
                },
            }
        }
        let arrow_dtype = self.dtype().to_physical_arrow(CompatLevel::newest());
        let chunks = mask
            .downcast_iter()
            .enumerate()
            .map(|(c, m)| {
                let selected = match m.validity() {
                    Some(v) => m.values() & v,
                    None => m.values().clone(),
                };
                let sides = [
                    side(l.as_deref(), l_scalar, c),
                    side(r.as_deref(), r_scalar, c),
                ];
                zip_chunk(&arrow_dtype, &selected, sides).boxed()
            })
            .collect();
        // SAFETY: the chunks have the physical dtype of `self`.
        Ok(unsafe { self.copy_with_chunks(chunks) })
    }
}

/// One input of [`zip_chunk`]: a chunk aligned with the mask, or a single row for every row.
#[derive(Clone, Copy)]
enum ZipSide<'a> {
    Column(&'a LargeUnionArray),
    Scalar(&'a LargeUnionArray),
}

impl<'a> ZipSide<'a> {
    fn array(self) -> &'a LargeUnionArray {
        match self {
            Self::Column(a) | Self::Scalar(a) => a,
        }
    }
}

/// Takes each row from `sides[0]` where `selected` is set and from `sides[1]` elsewhere.
///
/// Each output child is built from bulk gathers: the values each side contributes, then their
/// interleaving in row order, instead of one copy per run of the mask.
fn zip_chunk(
    dtype: &ArrowDataType,
    selected: &polars_arrow::bitmap::Bitmap,
    sides: [ZipSide<'_>; 2],
) -> LargeUnionArray {
    use polars_arrow::array::ChildPositions;
    use polars_arrow::bitmap::BitmapBuilder;
    use polars_compute::gather::gather_child;

    let n = selected.len();
    let n_fields = sides[0].array().fields().len();
    let mut positions: [Vec<ChildPositions>; 2] = sides.map(|s| {
        s.array()
            .fields()
            .iter()
            .map(|c| ChildPositions::new(c.len()))
            .collect()
    });
    // The side of every value of every output child, in row order.
    let mut from_second: Vec<BitmapBuilder> = (0..n_fields).map(|_| BitmapBuilder::new()).collect();
    let mut types = Vec::with_capacity(n);
    let mut offsets = Vec::with_capacity(n);
    let mut validity = BitmapBuilder::with_capacity(n);
    for r in 0..n {
        let s = usize::from(!unsafe { selected.get_bit_unchecked(r) });
        let (arr, i) = match sides[s] {
            ZipSide::Column(arr) => (arr, r),
            ZipSide::Scalar(arr) => (arr, 0),
        };
        let (t, pos, valid) = if arr.is_valid(i) {
            let (t, pos) = unsafe { arr.index_unchecked(i) };
            (t, Some(pos), true)
        } else {
            // Null rows get a null placeholder in child 0.
            (0, None, false)
        };
        positions[s][t].push(pos);
        types.push(t as i8);
        offsets.push(from_second[t].len() as i64);
        from_second[t].push(s == 1);
        validity.push(valid);
    }

    let children = (0..n_fields)
        .map(|t| {
            let from_second = std::mem::take(&mut from_second[t]).freeze();
            let [first, second] = &mut positions;
            let take = |side: ZipSide<'_>, positions: &mut ChildPositions| {
                // SAFETY: the positions are those of valid rows of `side`, so in bounds.
                unsafe {
                    gather_child(
                        side.array().fields()[t].as_ref(),
                        std::mem::replace(positions, ChildPositions::new(0)),
                    )
                }
            };
            match from_second.set_bits() {
                0 => take(sides[0], &mut first[t]),
                k if k == from_second.len() => take(sides[1], &mut second[t]),
                _ => {
                    let a = take(sides[0], &mut first[t]);
                    let b = take(sides[1], &mut second[t]);
                    let both =
                        polars_arrow::compute::concatenate::concatenate(&[a.as_ref(), b.as_ref()])
                            .unwrap();
                    let mut order = ChildPositions::new(both.len());
                    let (mut next_a, mut next_b) = (0, a.len());
                    for second in from_second.iter() {
                        let next = if second { &mut next_b } else { &mut next_a };
                        order.push(Some(*next));
                        *next += 1;
                    }
                    // SAFETY: every position is below `a.len() + b.len()`.
                    unsafe { gather_child(both.as_ref(), order) }
                },
            }
        })
        .collect();
    let validity = validity.freeze();
    let validity = (validity.unset_bits() > 0).then_some(validity);
    // SAFETY: every row's slot points at its value, or a null for a null row, in its variant's
    // child, and the slots of each child are increasing.
    unsafe {
        LargeUnionArray::new_unchecked(
            dtype.clone(),
            types.into(),
            offsets.into(),
            children,
            validity,
        )
    }
}
