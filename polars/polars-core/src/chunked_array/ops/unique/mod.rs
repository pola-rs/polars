#[cfg(feature = "rank")]
pub(crate) mod rank;

use std::hash::Hash;

#[cfg(feature = "object")]
use crate::chunked_array::object::ObjectType;
use crate::datatypes::PlHashSet;
use crate::frame::groupby::hashing::HASHMAP_INIT_SIZE;
use crate::frame::groupby::GroupsProxy;
#[cfg(feature = "mode")]
use crate::frame::groupby::IntoGroupsProxy;
use crate::prelude::*;
use crate::series::IsSorted;

fn finish_is_unique_helper(
    mut unique_idx: Vec<IdxSize>,
    len: IdxSize,
    unique_val: bool,
    duplicated_val: bool,
) -> BooleanChunked {
    unique_idx.sort_unstable();
    let mut unique_idx_iter = unique_idx.into_iter();
    let mut next_unique_idx = unique_idx_iter.next();
    (0..len)
        .map(|idx| match next_unique_idx {
            Some(unique_idx) => {
                if idx == unique_idx {
                    next_unique_idx = unique_idx_iter.next();
                    unique_val
                } else {
                    duplicated_val
                }
            }
            None => duplicated_val,
        })
        .collect()
}

pub(crate) fn is_unique_helper2(
    unique_idx: Vec<IdxSize>,
    len: IdxSize,
    unique_val: bool,
    duplicated_val: bool,
) -> BooleanChunked {
    debug_assert_ne!(unique_val, duplicated_val);
    finish_is_unique_helper(unique_idx, len, unique_val, duplicated_val)
}

pub(crate) fn is_unique_helper(
    groups: GroupsProxy,
    len: IdxSize,
    unique_val: bool,
    duplicated_val: bool,
) -> BooleanChunked {
    debug_assert_ne!(unique_val, duplicated_val);

    let idx = match groups {
        GroupsProxy::Idx(groups) => groups
            .into_iter()
            .filter_map(|(first, g)| if g.len() == 1 { Some(first) } else { None })
            .collect::<Vec<_>>(),
        GroupsProxy::Slice { groups, .. } => groups
            .into_iter()
            .filter_map(|[first, len]| if len == 1 { Some(first) } else { None })
            .collect(),
    };
    finish_is_unique_helper(idx, len, unique_val, duplicated_val)
}

/// if inverse is true, this is an `is_duplicated`
/// otherwise an `is_unique`
macro_rules! is_unique_duplicated {
    ($ca:expr, $inverse:expr) => {{
        let mut idx_key = PlHashMap::new();

        // instead of grouptuples, which allocates a full vec per group, we now just toggle a boolean
        // that's false if a group has multiple entries.
        $ca.into_iter().enumerate().for_each(|(idx, key)| {
            idx_key
                .entry(key)
                .and_modify(|v: &mut (IdxSize, bool)| v.1 = false)
                .or_insert((idx as IdxSize, true));
        });

        let idx: Vec<_> = idx_key
            .into_iter()
            .filter_map(|(_k, v)| if v.1 { Some(v.0) } else { None })
            .collect();
        let mut out = is_unique_helper2(idx, $ca.len() as IdxSize, !$inverse, $inverse);
        out.rename($ca.name());
        Ok(out)
    }};
}

#[cfg(feature = "object")]
impl<T: PolarsObject> ChunkUnique<ObjectType<T>> for ObjectChunked<T> {
    fn unique(&self) -> PolarsResult<ChunkedArray<ObjectType<T>>> {
        Err(PolarsError::InvalidOperation(
            "unique not supported for object".into(),
        ))
    }

    fn arg_unique(&self) -> PolarsResult<IdxCa> {
        Err(PolarsError::InvalidOperation(
            "unique not supported for object".into(),
        ))
    }
}

fn fill_set<A>(a: impl Iterator<Item = A>) -> PlHashSet<A>
where
    A: Hash + Eq,
{
    a.collect()
}

fn arg_unique<T>(a: impl Iterator<Item = T>, capacity: usize) -> Vec<IdxSize>
where
    T: Hash + Eq,
{
    let mut set = PlHashSet::new();
    let mut unique = Vec::with_capacity(capacity);
    a.enumerate().for_each(|(idx, val)| {
        if set.insert(val) {
            unique.push(idx as IdxSize)
        }
    });
    unique
}

#[cfg(feature = "mode")]
#[allow(clippy::needless_collect)]
fn mode<T: PolarsDataType>(ca: &ChunkedArray<T>) -> ChunkedArray<T>
where
    ChunkedArray<T>: IntoGroupsProxy + ChunkTake,
{
    if ca.is_empty() {
        return ca.clone();
    }
    let mut groups = ca
        .group_tuples(true, false)
        .unwrap()
        .into_idx()
        .into_iter()
        .collect_trusted::<Vec<_>>();
    groups.sort_unstable_by_key(|k| k.1.len());
    let last = &groups.last().unwrap();

    let max_occur = last.1.len();

    // collect until we don't take with trusted len anymore
    // TODO! take directly from iter, but first remove standard trusted-length collect.
    let idx = groups
        .iter()
        .rev()
        .take_while(|v| v.1.len() == max_occur)
        .map(|v| v.0)
        .collect::<Vec<_>>();
    // Safety:
    // group indices are in bounds
    unsafe { ca.take_unchecked(idx.into_iter().map(|i| i as usize).into()) }
}

macro_rules! arg_unique_ca {
    ($ca:expr) => {{
        match $ca.has_validity() {
            false => arg_unique($ca.into_no_null_iter(), $ca.len()),
            _ => arg_unique($ca.into_iter(), $ca.len()),
        }
    }};
}

impl<T> ChunkUnique<T> for ChunkedArray<T>
where
    T: PolarsIntegerType,
    T::Native: Hash + Eq + Ord,
    ChunkedArray<T>: IntoSeries,
{
    fn unique(&self) -> PolarsResult<Self> {
        // prevent stackoverflow repeated sorted.unique call
        if self.is_empty() {
            return Ok(self.clone());
        }
        match self.is_sorted_flag2() {
            IsSorted::Ascending | IsSorted::Descending => {
                // TODO! optimize this branch
                if self.null_count() > 0 {
                    let mut arr = MutablePrimitiveArray::with_capacity(self.len());
                    let mut iter = self.into_iter();
                    let mut last = None;

                    if let Some(val) = iter.next() {
                        last = val;
                        arr.push(val)
                    };

                    #[allow(clippy::unnecessary_filter_map)]
                    let to_extend = iter.filter_map(|opt_val| {
                        if opt_val != last {
                            last = opt_val;
                            Some(opt_val)
                        } else {
                            None
                        }
                    });

                    arr.extend(to_extend);
                    let arr: PrimitiveArray<T::Native> = arr.into();

                    unsafe {
                        Ok(ChunkedArray::from_chunks(
                            self.name(),
                            vec![Box::new(arr) as ArrayRef],
                        ))
                    }
                } else {
                    let mask = self.not_equal(&self.shift(1));
                    self.filter(&mask)
                }
            }
            IsSorted::Not => {
                let sorted = self.sort(false);
                sorted.unique()
            }
        }
    }

    fn arg_unique(&self) -> PolarsResult<IdxCa> {
        Ok(IdxCa::from_vec(self.name(), arg_unique_ca!(self)))
    }

    fn is_unique(&self) -> PolarsResult<BooleanChunked> {
        is_unique_duplicated!(self, false)
    }

    fn is_duplicated(&self) -> PolarsResult<BooleanChunked> {
        is_unique_duplicated!(self, true)
    }

    fn n_unique(&self) -> PolarsResult<usize> {
        if self.null_count() > 0 {
            Ok(fill_set(self.into_iter().flatten()).len() + 1)
        } else {
            Ok(fill_set(self.into_no_null_iter()).len())
        }
    }

    #[cfg(feature = "mode")]
    fn mode(&self) -> PolarsResult<Self> {
        Ok(mode(self))
    }
}

impl ChunkUnique<Utf8Type> for Utf8Chunked {
    fn unique(&self) -> PolarsResult<Self> {
        match self.null_count() {
            0 => {
                let mut set =
                    PlHashSet::with_capacity(std::cmp::min(HASHMAP_INIT_SIZE, self.len()));
                for arr in self.downcast_iter() {
                    set.extend(arr.values_iter())
                }
                Ok(Utf8Chunked::from_iter_values(
                    self.name(),
                    set.iter().copied(),
                ))
            }
            _ => {
                let mut set =
                    PlHashSet::with_capacity(std::cmp::min(HASHMAP_INIT_SIZE, self.len()));
                for arr in self.downcast_iter() {
                    set.extend(arr.iter())
                }
                Ok(Utf8Chunked::from_iter_options(
                    self.name(),
                    set.iter().copied(),
                ))
            }
        }
    }

    fn arg_unique(&self) -> PolarsResult<IdxCa> {
        Ok(IdxCa::from_vec(self.name(), arg_unique_ca!(self)))
    }

    fn is_unique(&self) -> PolarsResult<BooleanChunked> {
        is_unique_duplicated!(self, false)
    }
    fn is_duplicated(&self) -> PolarsResult<BooleanChunked> {
        is_unique_duplicated!(self, true)
    }

    fn n_unique(&self) -> PolarsResult<usize> {
        if self.null_count() > 0 {
            Ok(fill_set(self.into_iter().flatten()).len() + 1)
        } else {
            Ok(fill_set(self.into_no_null_iter()).len())
        }
    }

    #[cfg(feature = "mode")]
    fn mode(&self) -> PolarsResult<Self> {
        Ok(mode(self))
    }
}

#[cfg(feature = "dtype-binary")]
impl ChunkUnique<BinaryType> for BinaryChunked {
    fn unique(&self) -> PolarsResult<Self> {
        match self.null_count() {
            0 => {
                let mut set =
                    PlHashSet::with_capacity(std::cmp::min(HASHMAP_INIT_SIZE, self.len()));
                for arr in self.downcast_iter() {
                    set.extend(arr.values_iter())
                }
                Ok(BinaryChunked::from_iter_values(
                    self.name(),
                    set.iter().copied(),
                ))
            }
            _ => {
                let mut set =
                    PlHashSet::with_capacity(std::cmp::min(HASHMAP_INIT_SIZE, self.len()));
                for arr in self.downcast_iter() {
                    set.extend(arr.iter())
                }
                Ok(BinaryChunked::from_iter_options(
                    self.name(),
                    set.iter().copied(),
                ))
            }
        }
    }

    fn arg_unique(&self) -> PolarsResult<IdxCa> {
        Ok(IdxCa::from_vec(self.name(), arg_unique_ca!(self)))
    }

    fn is_unique(&self) -> PolarsResult<BooleanChunked> {
        is_unique_duplicated!(self, false)
    }
    fn is_duplicated(&self) -> PolarsResult<BooleanChunked> {
        is_unique_duplicated!(self, true)
    }

    fn n_unique(&self) -> PolarsResult<usize> {
        if self.null_count() > 0 {
            Ok(fill_set(self.into_iter().flatten()).len() + 1)
        } else {
            Ok(fill_set(self.into_no_null_iter()).len())
        }
    }

    #[cfg(feature = "mode")]
    fn mode(&self) -> PolarsResult<Self> {
        Ok(mode(self))
    }
}

impl ChunkUnique<BooleanType> for BooleanChunked {
    fn unique(&self) -> PolarsResult<Self> {
        // can be None, Some(true), Some(false)
        let mut unique = Vec::with_capacity(3);
        for v in self {
            if unique.len() == 3 {
                break;
            }
            if !unique.contains(&v) {
                unique.push(v)
            }
        }
        Ok(ChunkedArray::new(self.name(), &unique))
    }

    fn arg_unique(&self) -> PolarsResult<IdxCa> {
        Ok(IdxCa::from_vec(self.name(), arg_unique_ca!(self)))
    }

    fn is_unique(&self) -> PolarsResult<BooleanChunked> {
        is_unique_duplicated!(self, false)
    }
    fn is_duplicated(&self) -> PolarsResult<BooleanChunked> {
        is_unique_duplicated!(self, true)
    }
}

impl ChunkUnique<Float32Type> for Float32Chunked {
    fn unique(&self) -> PolarsResult<ChunkedArray<Float32Type>> {
        let ca = self.bit_repr_small();
        let ca = ca.unique()?;
        Ok(ca._reinterpret_float())
    }

    fn arg_unique(&self) -> PolarsResult<IdxCa> {
        self.bit_repr_small().arg_unique()
    }

    fn is_unique(&self) -> PolarsResult<BooleanChunked> {
        self.bit_repr_small().is_unique()
    }
    fn is_duplicated(&self) -> PolarsResult<BooleanChunked> {
        self.bit_repr_small().is_duplicated()
    }
}

impl ChunkUnique<Float64Type> for Float64Chunked {
    fn unique(&self) -> PolarsResult<ChunkedArray<Float64Type>> {
        let ca = self.bit_repr_large();
        let ca = ca.unique()?;
        Ok(ca._reinterpret_float())
    }

    fn arg_unique(&self) -> PolarsResult<IdxCa> {
        self.bit_repr_large().arg_unique()
    }

    fn is_unique(&self) -> PolarsResult<BooleanChunked> {
        self.bit_repr_large().is_unique()
    }
    fn is_duplicated(&self) -> PolarsResult<BooleanChunked> {
        self.bit_repr_large().is_duplicated()
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn unique() {
        let ca = ChunkedArray::<Int32Type>::from_slice("a", &[1, 2, 3, 2, 1]);
        assert_eq!(
            ca.unique()
                .unwrap()
                .sort(false)
                .into_iter()
                .collect::<Vec<_>>(),
            vec![Some(1), Some(2), Some(3)]
        );
        let ca = BooleanChunked::from_slice("a", &[true, false, true]);
        assert_eq!(
            ca.unique().unwrap().into_iter().collect::<Vec<_>>(),
            vec![Some(true), Some(false)]
        );

        let ca = Utf8Chunked::new("", &[Some("a"), None, Some("a"), Some("b"), None]);
        assert_eq!(
            Vec::from(&ca.unique().unwrap().sort(false)),
            &[None, Some("a"), Some("b")]
        );
    }

    #[test]
    fn arg_unique() {
        let ca = ChunkedArray::<Int32Type>::from_slice("a", &[1, 2, 1, 1, 3]);
        assert_eq!(
            ca.arg_unique().unwrap().into_iter().collect::<Vec<_>>(),
            vec![Some(0), Some(1), Some(4)]
        );
    }

    #[test]
    fn is_unique() {
        let ca = Float32Chunked::from_slice("a", &[1., 2., 1., 1., 3.]);
        assert_eq!(
            Vec::from(&ca.is_unique().unwrap()),
            &[
                Some(false),
                Some(true),
                Some(false),
                Some(false),
                Some(true)
            ]
        );
    }

    #[test]
    #[cfg(feature = "mode")]
    fn mode() {
        let ca = Int32Chunked::from_slice("a", &[0, 1, 2, 3, 4, 4, 5, 6, 5, 0]);
        let mut result = Vec::from(&ca.mode().unwrap());
        result.sort_by(|a, b| a.unwrap().cmp(&b.unwrap()));
        assert_eq!(&result, &[Some(0), Some(4), Some(5)]);

        let ca2 = Int32Chunked::from_slice("b", &[1, 1]);
        let mut result2 = Vec::from(&ca2.mode().unwrap());
        result2.sort_by(|a, b| a.unwrap().cmp(&b.unwrap()));
        assert_eq!(&result2, &[Some(1)]);

        let ca3 = Int32Chunked::from_slice("c", &[]);
        let result3 = Vec::from(&ca3.mode().unwrap());
        assert_eq!(result3, &[]);
    }
}
