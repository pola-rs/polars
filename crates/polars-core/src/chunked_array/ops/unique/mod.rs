use std::hash::Hash;

use arrow::bitmap::MutableBitmap;

#[cfg(feature = "object")]
use crate::datatypes::ObjectType;
use crate::datatypes::PlHashSet;
use crate::frame::group_by::GroupsProxy;
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
    let arr = BooleanArray::from_data_default(values.into(), None);
    arr.into()
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

#[cfg(feature = "object")]
impl<T: PolarsObject> ChunkUnique<ObjectType<T>> for ObjectChunked<T> {
    fn unique(&self) -> PolarsResult<ChunkedArray<ObjectType<T>>> {
        polars_bail!(opq = unique, self.dtype());
    }

    fn arg_unique(&self) -> PolarsResult<IdxCa> {
        polars_bail!(opq = arg_unique, self.dtype());
    }
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

macro_rules! arg_unique_ca {
    ($ca:expr) => {{
        match $ca.has_validity() {
            false => arg_unique($ca.into_no_null_iter(), $ca.len()),
            _ => arg_unique($ca.iter(), $ca.len()),
        }
    }};
}

impl<T> ChunkUnique<T> for ChunkedArray<T>
where
    T: PolarsIntegerType,
    T::Native: Hash + Eq + Ord,
    ChunkedArray<T>: IntoSeries + for<'a> ChunkCompare<&'a ChunkedArray<T>, Item = BooleanChunked>,
{
    fn unique(&self) -> PolarsResult<Self> {
        // prevent stackoverflow repeated sorted.unique call
        if self.is_empty() {
            return Ok(self.clone());
        }
        match self.is_sorted_flag() {
            IsSorted::Ascending | IsSorted::Descending => {
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
                    Ok(ChunkedArray::with_chunk(self.name(), arr))
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
        Ok(IdxCa::from_vec(self.name(), arg_unique_ca!(self)))
    }

    fn n_unique(&self) -> PolarsResult<usize> {
        // prevent stackoverflow repeated sorted.unique call
        if self.is_empty() {
            return Ok(0);
        }
        match self.is_sorted_flag() {
            IsSorted::Ascending | IsSorted::Descending => {
                if self.null_count() > 0 {
                    let mut count = 0;
                    let mut iter = self.into_iter();
                    let mut last = None;

                    if let Some(val) = iter.next() {
                        last = val;
                        count += 1;
                    };

                    iter.for_each(|opt_val| {
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
}

impl ChunkUnique<StringType> for StringChunked {
    fn unique(&self) -> PolarsResult<Self> {
        let out = self.as_binary().unique()?;
        Ok(unsafe { out.to_string() })
    }

    fn arg_unique(&self) -> PolarsResult<IdxCa> {
        self.as_binary().arg_unique()
    }

    fn n_unique(&self) -> PolarsResult<usize> {
        self.as_binary().n_unique()
    }
}

impl ChunkUnique<BinaryType> for BinaryChunked {
    fn unique(&self) -> PolarsResult<Self> {
        match self.null_count() {
            0 => {
                let mut set =
                    PlHashSet::with_capacity(std::cmp::min(_HASHMAP_INIT_SIZE, self.len()));
                for arr in self.downcast_iter() {
                    set.extend(arr.values_iter())
                }
                Ok(BinaryChunked::from_iter_values(
                    self.name(),
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
                    self.name(),
                    set.iter().copied(),
                ))
            },
        }
    }

    fn arg_unique(&self) -> PolarsResult<IdxCa> {
        Ok(IdxCa::from_vec(self.name(), arg_unique_ca!(self)))
    }

    fn n_unique(&self) -> PolarsResult<usize> {
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

        let ca = StringChunked::new("", &[Some("a"), None, Some("a"), Some("b"), None]);
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
}
