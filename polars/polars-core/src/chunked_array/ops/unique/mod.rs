#[cfg(feature = "rank")]
pub(crate) mod rank;

#[cfg(feature = "object")]
use crate::chunked_array::object::ObjectType;
use crate::datatypes::PlHashSet;
use crate::frame::groupby::{GroupsProxy, IntoGroupsProxy};
use crate::prelude::*;
use rayon::prelude::*;
use std::hash::Hash;

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
        .into_iter()
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
    let idx = groups
        .into_idx()
        .into_iter()
        .filter_map(|(first, g)| if g.len() == 1 { Some(first) } else { None })
        .collect::<Vec<_>>();
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
impl<T> ChunkUnique<ObjectType<T>> for ObjectChunked<T> {
    fn unique(&self) -> Result<ChunkedArray<ObjectType<T>>> {
        Err(PolarsError::InvalidOperation(
            "unique not supported for object".into(),
        ))
    }

    fn arg_unique(&self) -> Result<IdxCa> {
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
fn mode<T>(ca: &ChunkedArray<T>) -> ChunkedArray<T>
where
    ChunkedArray<T>: IntoGroupsProxy + ChunkTake,
{
    if ca.is_empty() {
        return ca.clone();
    }
    let mut groups = ca
        .group_tuples(true, false)
        .into_idx()
        .into_iter()
        .collect_trusted::<Vec<_>>();
    groups.sort_unstable_by_key(|k| k.1.len());
    let first = &groups[0];

    let max_occur = first.1.len();

    // collect until we don't take with trusted len anymore
    // TODO! take directly from iter, but first remove standard trusted-length collect.
    let mut was_equal = true;
    let idx = groups
        .iter()
        .rev()
        .take_while(|v| {
            let current = was_equal;
            was_equal = v.1.len() == max_occur;
            current
        })
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
    T::Native: Hash + Eq,
    ChunkedArray<T>: ChunkOps + IntoSeries,
{
    fn unique(&self) -> Result<Self> {
        let set = fill_set(self.into_iter());
        Ok(Self::from_iter_options(self.name(), set.iter().copied()))
    }

    fn arg_unique(&self) -> Result<IdxCa> {
        Ok(IdxCa::from_vec(self.name(), arg_unique_ca!(self)))
    }

    fn is_unique(&self) -> Result<BooleanChunked> {
        is_unique_duplicated!(self, false)
    }

    fn is_duplicated(&self) -> Result<BooleanChunked> {
        is_unique_duplicated!(self, true)
    }

    fn n_unique(&self) -> Result<usize> {
        if self.null_count() > 0 {
            Ok(fill_set(self.into_iter().flatten()).len() + 1)
        } else {
            Ok(fill_set(self.into_no_null_iter()).len())
        }
    }

    #[cfg(feature = "mode")]
    fn mode(&self) -> Result<Self> {
        Ok(mode(self))
    }
}

impl ChunkUnique<Utf8Type> for Utf8Chunked {
    fn unique(&self) -> Result<Self> {
        let set = fill_set(self.into_iter());
        Ok(Utf8Chunked::from_iter_options(
            self.name(),
            set.iter().copied(),
        ))
    }

    fn arg_unique(&self) -> Result<IdxCa> {
        Ok(IdxCa::from_vec(self.name(), arg_unique_ca!(self)))
    }

    fn is_unique(&self) -> Result<BooleanChunked> {
        is_unique_duplicated!(self, false)
    }
    fn is_duplicated(&self) -> Result<BooleanChunked> {
        is_unique_duplicated!(self, true)
    }

    fn n_unique(&self) -> Result<usize> {
        if self.null_count() > 0 {
            Ok(fill_set(self.into_iter().flatten()).len() + 1)
        } else {
            Ok(fill_set(self.into_no_null_iter()).len())
        }
    }

    #[cfg(feature = "mode")]
    fn mode(&self) -> Result<Self> {
        Ok(mode(self))
    }
}

#[cfg(feature = "dtype-u8")]
fn dummies_helper(mut groups: Vec<IdxSize>, len: usize, name: &str) -> UInt8Chunked {
    groups.sort_unstable();

    let mut av: Vec<_> = (0..len).map(|_| 0u8).collect();

    for idx in groups {
        let elem = unsafe { av.get_unchecked_mut(idx as usize) };
        *elem = 1;
    }

    ChunkedArray::from_vec(name, av)
}

#[cfg(not(feature = "dtype-u8"))]
fn dummies_helper(mut groups: Vec<IdxSize>, len: usize, name: &str) -> Int32Chunked {
    groups.sort_unstable();

    // let mut group_member_iter = groups.into_iter();
    let mut av: Vec<_> = (0..len).map(|_| 0i32).collect();

    for idx in groups {
        let elem = unsafe { av.get_unchecked_mut(idx as usize) };
        *elem = 1;
    }

    ChunkedArray::from_vec(name, av)
}

fn sort_columns(mut columns: Vec<Series>) -> Vec<Series> {
    columns.sort_by(|a, b| a.name().partial_cmp(b.name()).unwrap());
    columns
}

impl ToDummies<Utf8Type> for Utf8Chunked {
    fn to_dummies(&self) -> Result<DataFrame> {
        let groups = self.group_tuples(true, false).into_idx();
        let col_name = self.name();
        let taker = self.take_rand();

        let columns = groups
            .into_par_iter()
            .map(|(first, groups)| {
                let name = match unsafe { taker.get_unchecked(first as usize) } {
                    Some(val) => format!("{}_{}", col_name, val),
                    None => format!("{}_null", col_name),
                };
                let ca = dummies_helper(groups, self.len(), &name);
                ca.into_series()
            })
            .collect();

        Ok(DataFrame::new_no_checks(sort_columns(columns)))
    }
}
impl<T> ToDummies<T> for ChunkedArray<T>
where
    T: PolarsIntegerType + Sync,
    T::Native: Hash + Eq,
    ChunkedArray<T>: ChunkOps + ChunkCompare<T::Native> + ChunkUnique<T>,
{
    fn to_dummies(&self) -> Result<DataFrame> {
        let groups = self.group_tuples(true, false).into_idx();
        let col_name = self.name();
        let taker = self.take_rand();

        let columns = groups
            .into_par_iter()
            .map(|(first, groups)| {
                let name = match unsafe { taker.get_unchecked(first as usize) } {
                    Some(val) => format!("{}_{}", col_name, val),
                    None => format!("{}_null", col_name),
                };

                let ca = dummies_helper(groups, self.len(), &name);
                ca.into_series()
            })
            .collect();

        Ok(DataFrame::new_no_checks(sort_columns(columns)))
    }
}

impl ToDummies<Float32Type> for Float32Chunked {}
impl ToDummies<Float64Type> for Float64Chunked {}

impl ChunkUnique<BooleanType> for BooleanChunked {
    fn unique(&self) -> Result<Self> {
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

    fn arg_unique(&self) -> Result<IdxCa> {
        Ok(IdxCa::from_vec(self.name(), arg_unique_ca!(self)))
    }

    fn is_unique(&self) -> Result<BooleanChunked> {
        is_unique_duplicated!(self, false)
    }
    fn is_duplicated(&self) -> Result<BooleanChunked> {
        is_unique_duplicated!(self, true)
    }
}

impl ChunkUnique<Float32Type> for Float32Chunked {
    fn unique(&self) -> Result<ChunkedArray<Float32Type>> {
        let ca = self.bit_repr_small();
        let set = fill_set(ca.into_iter());
        Ok(set
            .into_iter()
            .map(|opt_v| opt_v.map(f32::from_bits))
            .collect())
    }

    fn arg_unique(&self) -> Result<IdxCa> {
        self.bit_repr_small().arg_unique()
    }

    fn is_unique(&self) -> Result<BooleanChunked> {
        self.bit_repr_small().is_unique()
    }
    fn is_duplicated(&self) -> Result<BooleanChunked> {
        self.bit_repr_small().is_duplicated()
    }
}

impl ChunkUnique<Float64Type> for Float64Chunked {
    fn unique(&self) -> Result<ChunkedArray<Float64Type>> {
        let ca = self.bit_repr_large();
        let set = fill_set(ca.into_iter());
        Ok(set
            .into_iter()
            .map(|opt_v| opt_v.map(f64::from_bits))
            .collect())
    }

    fn arg_unique(&self) -> Result<IdxCa> {
        self.bit_repr_large().arg_unique()
    }

    fn is_unique(&self) -> Result<BooleanChunked> {
        self.bit_repr_large().is_unique()
    }
    fn is_duplicated(&self) -> Result<BooleanChunked> {
        self.bit_repr_large().is_duplicated()
    }
}

#[cfg(feature = "is_first")]
mod is_first {
    use super::*;
    use crate::utils::CustomIterTools;
    use arrow::array::{ArrayRef, BooleanArray};

    fn is_first<T>(ca: &ChunkedArray<T>) -> BooleanChunked
    where
        T: PolarsNumericType,
        T::Native: Hash + Eq,
    {
        let mut unique = PlHashSet::new();
        let chunks = ca
            .downcast_iter()
            .map(|arr| {
                let mask: BooleanArray = arr
                    .into_iter()
                    .map(|opt_v| unique.insert(opt_v))
                    .collect_trusted();
                Arc::new(mask) as ArrayRef
            })
            .collect();

        BooleanChunked::from_chunks(ca.name(), chunks)
    }

    impl<T> IsFirst<T> for ChunkedArray<T>
    where
        T: PolarsNumericType,
    {
        fn is_first(&self) -> Result<BooleanChunked> {
            use DataType::*;
            match self.dtype() {
                // cast types to reduce compiler bloat
                Int8 | Int16 | UInt8 | UInt16 => {
                    let s = self.cast(&DataType::Int32).unwrap();
                    s.is_first()
                }
                _ => {
                    if Self::bit_repr_is_large() {
                        let ca = self.bit_repr_large();
                        Ok(is_first(&ca))
                    } else {
                        let ca = self.bit_repr_small();
                        Ok(is_first(&ca))
                    }
                }
            }
        }
    }

    impl IsFirst<Utf8Type> for Utf8Chunked {
        fn is_first(&self) -> Result<BooleanChunked> {
            let mut unique = PlHashSet::new();
            let chunks = self
                .downcast_iter()
                .map(|arr| {
                    let mask: BooleanArray = arr
                        .into_iter()
                        .map(|opt_v| unique.insert(opt_v))
                        .collect_trusted();
                    Arc::new(mask) as ArrayRef
                })
                .collect();

            Ok(BooleanChunked::from_chunks(self.name(), chunks))
        }
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
    #[cfg(feature = "is_first")]
    fn is_first() {
        let ca = UInt32Chunked::new(
            "a",
            &[Some(1), Some(2), Some(1), Some(1), None, Some(3), None],
        );
        assert_eq!(
            Vec::from(&ca.is_first().unwrap()),
            &[
                Some(true),
                Some(true),
                Some(false),
                Some(false),
                Some(true),
                Some(true),
                Some(false)
            ]
        );
    }
}
