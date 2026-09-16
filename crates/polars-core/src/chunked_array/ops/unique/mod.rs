use std::hash::Hash;
use std::ops::Deref;

use arrow::bitmap::MutableBitmap;
use polars_compute::unique::BooleanUniqueKernelState;
use polars_utils::total_ord::{ToTotalOrd, TotalHash, TotalOrdWrap};

use crate::hashing::_HASHMAP_INIT_SIZE;
use crate::prelude::*;
use crate::series::IsSorted;

fn finish_is_unique_helper(
    unique_idx: Vec<IdxSize>,
    len: IdxSize,
    setter: bool,
    default: bool,
) -> BooleanChunked {
    let mut values = MutableBitmap::with_capacity(len as usize);
    values.extend_constant(len as usize, default);

    for idx in unique_idx {
        unsafe { values.set_unchecked(idx as usize, setter) }
    }
    let arr = PlBooleanArray::new(values.into(), len as usize, None);
    arr.into()
}

pub(crate) fn is_unique_helper(
    groups: &GroupPositions,
    len: IdxSize,
    unique_val: bool,
    duplicated_val: bool,
) -> BooleanChunked {
    debug_assert_ne!(unique_val, duplicated_val);

    let idx = match groups.deref() {
        GroupsType::Idx(groups) => groups
            .iter()
            .filter_map(|(first, g)| if g.len() == 1 { Some(first) } else { None })
            .collect::<Vec<_>>(),
        GroupsType::Slice { groups, .. } => groups
            .iter()
            .filter_map(|[first, len]| if *len == 1 { Some(*first) } else { None })
            .collect(),
    };
    finish_is_unique_helper(idx, len, unique_val, duplicated_val)
}

#[cfg(feature = "object")]
impl<T: PolarsObject> ChunkUnique for ObjectChunked<T> {
    fn unique(&self) -> PolarsResult<ChunkedArray<ObjectType<T>>> {
        polars_bail!(opq = unique, self.dtype());
    }

    fn arg_unique(&self) -> PolarsResult<IdxCa> {
        polars_bail!(opq = arg_unique, self.dtype());
    }

    fn unique_id(&self) -> PolarsResult<(IdxSize, Vec<IdxSize>)> {
        polars_bail!(opq = unique_id, self.dtype());
    }
}

/// Whether every element of this chunked array is the same one.
///
/// One chunk that repeats a single element is the same value throughout, or the same null
/// throughout; a column with nothing but nulls in it is the same null throughout too, however
/// many chunks they are spread over. Either way it has exactly one distinct element, and the
/// whole unique family is answered off the first of them without hashing a single one. See also
/// [`scalar_groups`](crate::frame::group_by::scalar_groups).
fn reads_as_one_element<T: PolarsDataType>(ca: &ChunkedArray<T>) -> bool {
    if ca.is_empty() {
        return false;
    }

    // Nulls are all the same element, and their count is the one the masks already carry.
    if ca.null_count() == ca.len() {
        return true;
    }

    matches!(ca.chunks().as_slice(), [chunk] if chunk.is_scalar())
}

/// [`arg_unique_chunk`] over one contiguous run of values.
///
/// It is its own function, and holds nothing but the loop, so that the hash set's `insert` has
/// room to be inlined into it: the walk costs a third more when it is left as a call.
fn arg_unique_slice<T>(
    vals: &[T],
    offset: IdxSize,
    set: &mut PlHashSet<<T as ToTotalOrd>::TotalOrdItem>,
    unique: &mut Vec<IdxSize>,
) -> IdxSize
where
    T: ToTotalOrd + Copy,
    <T as ToTotalOrd>::TotalOrdItem: Hash + Eq,
{
    for (idx, val) in vals.iter().enumerate() {
        if set.insert(val.to_total_ord()) {
            unique.push(offset + idx as IdxSize)
        }
    }
    offset + vals.len() as IdxSize
}

/// Walk one chunk, recording where each value this column has not held before first appears.
///
/// `offset` is how many elements the chunks already walked hold, and carries on into the next.
fn arg_unique_chunk<T>(
    a: impl Iterator<Item = T>,
    offset: IdxSize,
    set: &mut PlHashSet<<T as ToTotalOrd>::TotalOrdItem>,
    unique: &mut Vec<IdxSize>,
) -> IdxSize
where
    T: ToTotalOrd,
    <T as ToTotalOrd>::TotalOrdItem: Hash + Eq,
{
    // The index is a local, not a `&mut` the caller lends: the loop keeps it in a register and
    // hands the next chunk's starting point back.
    let mut idx = offset;
    a.for_each(|val| {
        if set.insert(val.to_total_ord()) {
            unique.push(idx)
        }
        idx += 1;
    });
    idx
}

macro_rules! arg_unique_ca {
    ($ca:expr) => {{
        let ca = $ca;
        if reads_as_one_element(ca) {
            vec![0]
        } else {
            // One chunk at a time, not `ca.iter()` over the column: a chunk's own iterator
            // resolves its representation once for the whole chunk in `fold`, and the flattening
            // adapters between the column and it cost more per element than they hoist -- this
            // loop carries the element index itself instead of asking `enumerate` for it.
            let mut unique = Vec::with_capacity(ca.len());
            let mut offset: IdxSize = 0;
            match ca.has_nulls() {
                false => {
                    let mut set = PlHashSet::new();
                    for arr in ca.downcast_iter() {
                        offset = arg_unique_chunk(arr.values_iter(), offset, &mut set, &mut unique);
                    }
                },
                _ => {
                    let mut set = PlHashSet::new();
                    for arr in ca.downcast_iter() {
                        offset = arg_unique_chunk(arr.iter(), offset, &mut set, &mut unique);
                    }
                },
            }
            unique
        }
    }};
}

impl<T> ChunkUnique for ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: TotalHash + TotalEq + ToTotalOrd,
    <T::Native as ToTotalOrd>::TotalOrdItem: Hash + Eq + Ord,
    ChunkedArray<T>: for<'a> ChunkCompareEq<&'a ChunkedArray<T>, Item = BooleanChunked>,
{
    fn unique(&self) -> PolarsResult<Self> {
        // prevent stackoverflow repeated sorted.unique call
        if self.is_empty() {
            return Ok(self.clone());
        }
        if reads_as_one_element(self) {
            return Ok(self.slice(0, 1));
        }
        match self.is_sorted_flag() {
            IsSorted::Ascending | IsSorted::Descending => {
                if self.null_count() > 0 {
                    let mut iter = self.iter();
                    let arr: T::Array = match iter.next() {
                        None => T::Array::new_empty(),
                        Some(first) => {
                            // The elements are sorted, so an element is unique exactly where it
                            // differs from the one before it.
                            let mut last = first.to_total_ord();
                            std::iter::once(first)
                                .chain(iter.filter(move |opt_val| {
                                    let opt_val_tot_ord = opt_val.to_total_ord();
                                    let out = opt_val_tot_ord != last;
                                    last = opt_val_tot_ord;
                                    out
                                }))
                                .collect_arr()
                        },
                    };

                    Ok(ChunkedArray::with_chunk(self.name().clone(), arr))
                } else {
                    let mask = self.not_equal_missing(&self.shift(1));
                    self.filter(&mask)
                }
            },
            IsSorted::Not => {
                let sorted = self.sort(false);
                sorted.unique()
            },
        }
    }

    fn arg_unique(&self) -> PolarsResult<IdxCa> {
        // A flat chunk with no nulls in it is a slice of values, and walking each chunk's slice
        // leaves no representation test in the loop at all -- which the column's own iterator
        // cannot promise, however well each adapter between it and the chunk forwards `fold`.
        if !reads_as_one_element(self)
            && self.null_count() == 0
            && let Some(flat) = self.as_flat()
        {
            let mut set = PlHashSet::new();
            let mut unique = Vec::with_capacity(self.len());
            let mut offset: IdxSize = 0;
            for vals in flat.chunks_flat_values() {
                offset = arg_unique_slice(vals, offset, &mut set, &mut unique);
            }
            return Ok(IdxCa::from_vec(self.name().clone(), unique));
        }

        Ok(IdxCa::from_vec(self.name().clone(), arg_unique_ca!(self)))
    }

    fn n_unique(&self) -> PolarsResult<usize> {
        // prevent stackoverflow repeated sorted.unique call
        if self.is_empty() {
            return Ok(0);
        }
        if reads_as_one_element(self) {
            return Ok(1);
        }
        match self.is_sorted_flag() {
            IsSorted::Ascending | IsSorted::Descending => {
                if self.null_count() > 0 {
                    let mut count = 0;

                    if self.is_empty() {
                        return Ok(count);
                    }

                    let mut iter = self.iter();
                    let mut last = iter.next().unwrap().to_total_ord();

                    count += 1;

                    iter.for_each(|opt_val| {
                        let opt_val = opt_val.to_total_ord();
                        if opt_val != last {
                            last = opt_val;
                            count += 1;
                        }
                    });

                    Ok(count)
                } else {
                    let mask = self.not_equal_missing(&self.shift(1));
                    Ok(mask.sum().unwrap() as usize)
                }
            },
            IsSorted::Not => {
                let sorted = self.sort(false);
                sorted.n_unique()
            },
        }
    }

    fn unique_id(&self) -> PolarsResult<(IdxSize, Vec<IdxSize>)> {
        let mut n = IdxSize::from(self.has_nulls());
        let mut indices = PlHashMap::new();
        let ids = self
            .iter()
            .map(|v| match v {
                None => 0,
                Some(v) => *indices.entry(TotalOrdWrap(v)).or_insert_with(|| {
                    let i = n;
                    n += 1;
                    i
                }),
            })
            .collect_trusted();
        Ok((n, ids))
    }
}

impl ChunkUnique for StringChunked {
    fn unique(&self) -> PolarsResult<Self> {
        let out = self.as_binary().unique()?;
        Ok(unsafe { out.to_string_unchecked() })
    }

    fn arg_unique(&self) -> PolarsResult<IdxCa> {
        self.as_binary().arg_unique()
    }

    fn n_unique(&self) -> PolarsResult<usize> {
        self.as_binary().n_unique()
    }

    fn unique_id(&self) -> PolarsResult<(IdxSize, Vec<IdxSize>)> {
        self.as_binary().unique_id()
    }
}

impl ChunkUnique for BinaryChunked {
    fn unique(&self) -> PolarsResult<Self> {
        if reads_as_one_element(self) {
            return Ok(self.slice(0, 1));
        }
        match self.null_count() {
            0 => {
                let mut set =
                    PlHashSet::with_capacity(std::cmp::min(_HASHMAP_INIT_SIZE, self.len()));
                for arr in self.downcast_iter() {
                    set.extend(arr.values_iter())
                }
                Ok(BinaryChunked::from_iter_values(
                    self.name().clone(),
                    set.iter().copied(),
                ))
            },
            _ => {
                let mut set =
                    PlHashSet::with_capacity(std::cmp::min(_HASHMAP_INIT_SIZE, self.len()));
                for arr in self.downcast_iter() {
                    set.extend(arr.iter())
                }
                Ok(BinaryChunked::from_iter_options(
                    self.name().clone(),
                    set.iter().copied(),
                ))
            },
        }
    }

    fn arg_unique(&self) -> PolarsResult<IdxCa> {
        Ok(IdxCa::from_vec(self.name().clone(), arg_unique_ca!(self)))
    }

    fn n_unique(&self) -> PolarsResult<usize> {
        if reads_as_one_element(self) {
            return Ok(1);
        }
        let mut set: PlHashSet<&[u8]> = PlHashSet::new();
        if self.null_count() > 0 {
            for arr in self.downcast_iter() {
                set.extend(arr.into_iter().flatten())
            }
            Ok(set.len() + 1)
        } else {
            for arr in self.downcast_iter() {
                set.extend(arr.values_iter())
            }
            Ok(set.len())
        }
    }

    fn unique_id(&self) -> PolarsResult<(IdxSize, Vec<IdxSize>)> {
        let mut n = IdxSize::from(self.has_nulls());
        let mut indices = PlHashMap::new();
        let ids = self
            .iter()
            .map(|v| match v {
                None => 0,
                Some(v) => *indices.entry(v).or_insert_with(|| {
                    let i = n;
                    n += 1;
                    i
                }),
            })
            .collect_trusted();
        Ok((n, ids))
    }
}

impl ChunkUnique for BinaryOffsetChunked {
    fn unique(&self) -> PolarsResult<Self> {
        if reads_as_one_element(self) {
            return Ok(self.slice(0, 1));
        }
        match self.null_count() {
            0 => {
                let mut set =
                    PlHashSet::with_capacity(std::cmp::min(_HASHMAP_INIT_SIZE, self.len()));
                for arr in self.downcast_iter() {
                    set.extend(arr.values_iter())
                }
                Ok(set.iter().copied().collect_ca(self.name().clone()))
            },
            _ => {
                let mut set =
                    PlHashSet::with_capacity(std::cmp::min(_HASHMAP_INIT_SIZE, self.len()));
                for arr in self.downcast_iter() {
                    set.extend(arr.iter())
                }
                Ok(set.iter().copied().collect_ca(self.name().clone()))
            },
        }
    }

    fn arg_unique(&self) -> PolarsResult<IdxCa> {
        Ok(IdxCa::from_vec(self.name().clone(), arg_unique_ca!(self)))
    }

    fn n_unique(&self) -> PolarsResult<usize> {
        if reads_as_one_element(self) {
            return Ok(1);
        }
        let mut set: PlHashSet<&[u8]> = PlHashSet::new();
        if self.null_count() > 0 {
            for arr in self.downcast_iter() {
                set.extend(arr.into_iter().flatten())
            }
            Ok(set.len() + 1)
        } else {
            for arr in self.downcast_iter() {
                set.extend(arr.values_iter())
            }
            Ok(set.len())
        }
    }

    fn unique_id(&self) -> PolarsResult<(IdxSize, Vec<IdxSize>)> {
        let mut n = IdxSize::from(self.has_nulls());
        let mut indices = PlHashMap::new();
        let ids = self
            .iter()
            .map(|v| match v {
                None => 0,
                Some(v) => *indices.entry(v).or_insert_with(|| {
                    let i = n;
                    n += 1;
                    i
                }),
            })
            .collect_trusted();
        Ok((n, ids))
    }
}

impl ChunkUnique for BooleanChunked {
    fn unique(&self) -> PolarsResult<Self> {
        use polars_compute::unique::RangedUniqueKernel;

        if reads_as_one_element(self) {
            return Ok(self.slice(0, 1));
        }

        let mut state = BooleanUniqueKernelState::new();

        for arr in self.downcast_iter() {
            state.append(arr);

            if state.has_seen_all() {
                break;
            }
        }

        Ok(Self::with_chunk(
            self.name().clone(),
            state.finalize_unique(),
        ))
    }

    fn arg_unique(&self) -> PolarsResult<IdxCa> {
        Ok(IdxCa::from_vec(self.name().clone(), arg_unique_ca!(self)))
    }

    fn n_unique(&self) -> PolarsResult<usize> {
        use polars_compute::unique::RangedUniqueKernel;

        // There are only ever three distinct booleans -- `false`, `true` and null -- so counting
        // them is counting the bits of each chunk, not walking its elements through a hash set
        // the way the default `arg_unique().len()` does.
        let mut state = BooleanUniqueKernelState::new();

        for arr in self.downcast_iter() {
            state.append(arr);

            if state.has_seen_all() {
                break;
            }
        }

        Ok(state.finalize_n_unique())
    }

    fn unique_id(&self) -> PolarsResult<(IdxSize, Vec<IdxSize>)> {
        let num_nulls = self.null_count();
        let num_trues = self.num_trues();

        let true_idx = IdxSize::from(num_nulls > 0);
        let false_idx = IdxSize::from(num_nulls > 0) + IdxSize::from(num_trues > 0);
        let ids = self
            .iter()
            .map(|v| match v {
                None => 0,
                Some(false) => false_idx,
                Some(true) => true_idx,
            })
            .collect_trusted();
        Ok((false_idx + 1, ids))
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn unique() {
        let ca =
            ChunkedArray::<Int32Type>::from_slice(PlSmallStr::from_static("a"), &[1, 2, 3, 2, 1]);
        assert_eq!(
            ca.unique().unwrap().sort(false).iter().collect::<Vec<_>>(),
            vec![Some(1), Some(2), Some(3)]
        );
        let ca = BooleanChunked::from_slice(PlSmallStr::from_static("a"), &[true, false, true]);
        assert_eq!(
            ca.unique().unwrap().iter().collect::<Vec<_>>(),
            vec![Some(false), Some(true)]
        );

        let ca = StringChunked::new(
            PlSmallStr::EMPTY,
            &[Some("a"), None, Some("a"), Some("b"), None],
        );
        assert_eq!(
            Vec::from(&ca.unique().unwrap().sort(false)),
            &[None, Some("a"), Some("b")]
        );
    }

    #[test]
    fn arg_unique() {
        let ca =
            ChunkedArray::<Int32Type>::from_slice(PlSmallStr::from_static("a"), &[1, 2, 1, 1, 3]);
        assert_eq!(
            ca.arg_unique().unwrap().iter().collect::<Vec<_>>(),
            vec![Some(0), Some(1), Some(4)]
        );
    }

    #[test]
    fn arg_unique_is_the_same_however_the_column_is_laid_out() {
        let name = PlSmallStr::from_static("a");
        let values = [1i32, 2, 1, 1, 3, 2, 4];
        let expected = vec![Some(0), Some(1), Some(4), Some(6)];

        // One chunk, walked as the slice it is.
        let flat = ChunkedArray::<Int32Type>::from_slice(name.clone(), &values);
        assert_eq!(
            flat.arg_unique().unwrap().iter().collect::<Vec<_>>(),
            expected
        );

        // The same elements over three chunks: the index carries on across them.
        let mut chunked = ChunkedArray::<Int32Type>::from_slice(name.clone(), &values[..2]);
        chunked
            .append(&ChunkedArray::from_slice(name.clone(), &values[2..5]))
            .unwrap();
        chunked
            .append(&ChunkedArray::from_slice(name.clone(), &values[5..]))
            .unwrap();
        assert_eq!(chunked.chunks().len(), 3);
        assert_eq!(
            chunked.arg_unique().unwrap().iter().collect::<Vec<_>>(),
            expected
        );

        // Nulls are an element of their own, and the first of them is the one that is kept.
        let with_nulls = ChunkedArray::<Int32Type>::from_slice_options(
            name,
            &[Some(1), None, Some(1), None, Some(2)],
        );
        assert_eq!(
            with_nulls.arg_unique().unwrap().iter().collect::<Vec<_>>(),
            vec![Some(0), Some(1), Some(4)]
        );
    }
}
