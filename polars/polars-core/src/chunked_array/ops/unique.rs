use crate::chunked_array::builder::categorical::RevMapping;
#[cfg(feature = "object")]
use crate::chunked_array::object::ObjectType;
use crate::datatypes::PlHashSet;
use crate::frame::groupby::{GroupTuples, IntoGroupTuples};
use crate::prelude::*;
use crate::utils::{CustomIterTools, NoNull};
use arrow::array::Array;
use itertools::Itertools;
use num::NumCast;
use rayon::prelude::*;
use std::fmt::Display;
use std::hash::Hash;

fn finish_is_unique_helper(
    mut unique_idx: Vec<u32>,
    len: u32,
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
    unique_idx: Vec<u32>,
    len: u32,
    unique_val: bool,
    duplicated_val: bool,
) -> BooleanChunked {
    debug_assert_ne!(unique_val, duplicated_val);
    finish_is_unique_helper(unique_idx, len, unique_val, duplicated_val)
}

pub(crate) fn is_unique_helper(
    groups: GroupTuples,
    len: u32,
    unique_val: bool,
    duplicated_val: bool,
) -> BooleanChunked {
    debug_assert_ne!(unique_val, duplicated_val);
    let idx = groups
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

        // instead of grouptuples, wich allocates a full vec per group, we now just toggle a boolean
        // that's false if a group has multiple entries.
        $ca.into_iter().enumerate().for_each(|(idx, key)| {
            idx_key
                .entry(key)
                .and_modify(|v: &mut (u32, bool)| v.1 = false)
                .or_insert((idx as u32, true));
        });

        let idx: Vec<_> = idx_key
            .into_iter()
            .filter_map(|(_k, v)| if v.1 { Some(v.0) } else { None })
            .collect();
        let mut out = is_unique_helper2(idx, $ca.len() as u32, !$inverse, $inverse);
        out.rename($ca.name());
        Ok(out)
    }};
}

impl ChunkUnique<ListType> for ListChunked {
    fn unique(&self) -> Result<ChunkedArray<ListType>> {
        Err(PolarsError::InvalidOperation(
            "unique not supported for list".into(),
        ))
    }

    fn arg_unique(&self) -> Result<UInt32Chunked> {
        Err(PolarsError::InvalidOperation(
            "unique not supported for list".into(),
        ))
    }
}
#[cfg(feature = "object")]
impl<T> ChunkUnique<ObjectType<T>> for ObjectChunked<T> {
    fn unique(&self) -> Result<ChunkedArray<ObjectType<T>>> {
        Err(PolarsError::InvalidOperation(
            "unique not supported for object".into(),
        ))
    }

    fn arg_unique(&self) -> Result<UInt32Chunked> {
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

fn arg_unique<T>(a: impl Iterator<Item = T>, capacity: usize) -> AlignedVec<u32>
where
    T: Hash + Eq,
{
    let mut set = PlHashSet::new();
    let mut unique = AlignedVec::with_capacity(capacity);
    a.enumerate().for_each(|(idx, val)| {
        if set.insert(val) {
            unique.push(idx as u32)
        }
    });
    unique
}

#[cfg(feature = "mode")]
#[allow(clippy::needless_collect)]
fn mode<T>(ca: &ChunkedArray<T>) -> ChunkedArray<T>
where
    ChunkedArray<T>: IntoGroupTuples + ChunkTake,
{
    if ca.is_empty() {
        return ca.clone();
    }
    let mut groups = ca.group_tuples(true);
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
        match $ca.null_count() {
            0 => arg_unique($ca.into_no_null_iter(), $ca.len()),
            _ => arg_unique($ca.into_iter(), $ca.len()),
        }
    }};
}

macro_rules! impl_value_counts {
    ($self:expr) => {{
        let group_tuples = $self.group_tuples(true);
        let values =
            unsafe { $self.take_unchecked(group_tuples.iter().map(|t| t.0 as usize).into()) };
        let mut counts: NoNull<UInt32Chunked> = group_tuples
            .into_iter()
            .map(|(_, groups)| groups.len() as u32)
            .collect();
        counts.rename("counts");
        let cols = vec![values.into_series(), counts.into_inner().into_series()];
        let df = DataFrame::new_no_checks(cols);
        df.sort("counts", true)
    }};
}

impl<T> ChunkUnique<T> for ChunkedArray<T>
where
    T: PolarsIntegerType,
    T::Native: Hash + Eq + NumCast,
    ChunkedArray<T>: ChunkOps + IntoSeries,
{
    fn unique(&self) -> Result<Self> {
        let set = fill_set(self.into_iter());
        Ok(Self::new_from_opt_iter(self.name(), set.iter().copied()))
    }

    fn arg_unique(&self) -> Result<UInt32Chunked> {
        Ok(UInt32Chunked::new_from_aligned_vec(
            self.name(),
            arg_unique_ca!(self),
        ))
    }

    fn is_unique(&self) -> Result<BooleanChunked> {
        is_unique_duplicated!(self, false)
    }

    fn is_duplicated(&self) -> Result<BooleanChunked> {
        is_unique_duplicated!(self, true)
    }

    fn value_counts(&self) -> Result<DataFrame> {
        impl_value_counts!(self)
    }

    fn n_unique(&self) -> Result<usize> {
        Ok(fill_set(self.into_iter()).len())
    }

    #[cfg(feature = "mode")]
    fn mode(&self) -> Result<Self> {
        Ok(mode(self))
    }
}

impl ChunkUnique<Utf8Type> for Utf8Chunked {
    fn unique(&self) -> Result<Self> {
        let set = fill_set(self.into_iter());
        Ok(Utf8Chunked::new_from_opt_iter(
            self.name(),
            set.iter().copied(),
        ))
    }

    fn arg_unique(&self) -> Result<UInt32Chunked> {
        Ok(UInt32Chunked::new_from_aligned_vec(
            self.name(),
            arg_unique_ca!(self),
        ))
    }

    fn is_unique(&self) -> Result<BooleanChunked> {
        is_unique_duplicated!(self, false)
    }
    fn is_duplicated(&self) -> Result<BooleanChunked> {
        is_unique_duplicated!(self, true)
    }

    fn value_counts(&self) -> Result<DataFrame> {
        impl_value_counts!(self)
    }

    fn n_unique(&self) -> Result<usize> {
        Ok(fill_set(self.into_iter()).len())
    }

    #[cfg(feature = "mode")]
    fn mode(&self) -> Result<Self> {
        Ok(mode(self))
    }
}

impl ChunkUnique<CategoricalType> for CategoricalChunked {
    fn unique(&self) -> Result<Self> {
        let cat_map = self.categorical_map.as_ref().unwrap();
        let mut ca = match &**cat_map {
            RevMapping::Local(a) => UInt32Chunked::new_from_iter(self.name(), 0..(a.len() as u32)),
            RevMapping::Global(map, _, _) => {
                UInt32Chunked::new_from_iter(self.name(), map.keys().copied())
            }
        };
        ca.categorical_map = self.categorical_map.clone();
        ca.cast()
    }

    fn arg_unique(&self) -> Result<UInt32Chunked> {
        self.cast::<UInt32Type>()?.arg_unique()
    }

    fn is_unique(&self) -> Result<BooleanChunked> {
        self.cast::<UInt32Type>()?.is_unique()
    }
    fn is_duplicated(&self) -> Result<BooleanChunked> {
        self.cast::<UInt32Type>()?.is_duplicated()
    }

    fn value_counts(&self) -> Result<DataFrame> {
        impl_value_counts!(self)
    }
    fn n_unique(&self) -> Result<usize> {
        Ok(self.categorical_map.as_ref().unwrap().len())
    }
    #[cfg(feature = "mode")]
    fn mode(&self) -> Result<Self> {
        let mut ca = self.cast::<UInt32Type>()?.mode()?;
        ca.categorical_map = self.categorical_map.clone();
        ca.cast()
    }
}

#[cfg(feature = "dtype-u8")]
fn dummies_helper(mut groups: Vec<u32>, len: usize, name: &str) -> UInt8Chunked {
    groups.sort_unstable();

    let mut av: AlignedVec<_> = (0..len).map(|_| 0u8).collect_trusted();

    for idx in groups {
        let elem = unsafe { av.inner.get_unchecked_mut(idx as usize) };
        *elem = 1;
    }

    ChunkedArray::new_from_aligned_vec(name, av)
}

#[cfg(not(feature = "dtype-u8"))]
fn dummies_helper(mut groups: Vec<u32>, len: usize, name: &str) -> Int32Chunked {
    groups.sort_unstable();

    // let mut group_member_iter = groups.into_iter();
    let mut av: AlignedVec<_> = (0..len).map(|_| 0i32).collect_trusted();

    for idx in groups {
        let elem = unsafe { av.inner.get_unchecked_mut(idx as usize) };
        *elem = 1;
    }

    ChunkedArray::new_from_aligned_vec(name, av)
}

fn sort_columns(columns: Vec<Series>) -> Vec<Series> {
    columns
        .into_iter()
        .sorted_by_key(|s| s.name().to_string())
        .collect()
}

impl ToDummies<Utf8Type> for Utf8Chunked {
    fn to_dummies(&self) -> Result<DataFrame> {
        let groups = self.group_tuples(true);
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
    T::Native: Hash + Eq + Display + NumCast,
    ChunkedArray<T>: ChunkOps + ChunkCompare<T::Native> + ChunkUnique<T>,
{
    fn to_dummies(&self) -> Result<DataFrame> {
        let groups = self.group_tuples(true);
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

impl ToDummies<ListType> for ListChunked {}
#[cfg(feature = "object")]
impl<T> ToDummies<ObjectType<T>> for ObjectChunked<T> {}
impl ToDummies<Float32Type> for Float32Chunked {}
impl ToDummies<Float64Type> for Float64Chunked {}
impl ToDummies<BooleanType> for BooleanChunked {}
impl ToDummies<CategoricalType> for CategoricalChunked {}

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
        Ok(ChunkedArray::new_from_opt_slice(self.name(), &unique))
    }

    fn arg_unique(&self) -> Result<UInt32Chunked> {
        Ok(UInt32Chunked::new_from_aligned_vec(
            self.name(),
            arg_unique_ca!(self),
        ))
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

    fn arg_unique(&self) -> Result<UInt32Chunked> {
        self.bit_repr_small().arg_unique()
    }

    fn is_unique(&self) -> Result<BooleanChunked> {
        self.bit_repr_small().is_unique()
    }
    fn is_duplicated(&self) -> Result<BooleanChunked> {
        self.bit_repr_small().is_duplicated()
    }
    fn value_counts(&self) -> Result<DataFrame> {
        impl_value_counts!(self)
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

    fn arg_unique(&self) -> Result<UInt32Chunked> {
        #[cfg(feature = "dtype-u64")]
        {
            self.bit_repr_large().arg_unique()
        }

        #[cfg(not(feature = "dtype-u64"))]
        {
            panic!("activate feature dtype-u64")
        }
    }

    fn is_unique(&self) -> Result<BooleanChunked> {
        #[cfg(feature = "dtype-u64")]
        {
            self.bit_repr_large().is_unique()
        }

        #[cfg(not(feature = "dtype-u64"))]
        {
            panic!("activate feature dtype-u64")
        }
    }
    fn is_duplicated(&self) -> Result<BooleanChunked> {
        #[cfg(feature = "dtype-u64")]
        {
            self.bit_repr_large().is_duplicated()
        }

        #[cfg(not(feature = "dtype-u64"))]
        {
            panic!("activate feature dtype-u64")
        }
    }
    fn value_counts(&self) -> Result<DataFrame> {
        impl_value_counts!(self)
    }
}

#[cfg(feature = "is_first")]
mod is_first {
    use super::*;
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

        BooleanChunked::new_from_chunks(ca.name(), chunks)
    }

    impl<T> IsFirst<T> for ChunkedArray<T>
    where
        T: PolarsNumericType,
        T::Native: NumCast,
    {
        fn is_first(&self) -> Result<BooleanChunked> {
            use DataType::*;
            match self.dtype() {
                Int8 | Int16 | UInt8 | UInt16 => {
                    let ca = self.cast::<Int32Type>().unwrap();
                    ca.is_first()
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

    impl IsFirst<CategoricalType> for CategoricalChunked {
        fn is_first(&self) -> Result<BooleanChunked> {
            let ca = self.cast::<UInt32Type>().unwrap();
            ca.is_first()
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

            Ok(BooleanChunked::new_from_chunks(self.name(), chunks))
        }
    }
    impl IsFirst<ListType> for ListChunked {}
    impl IsFirst<BooleanType> for BooleanChunked {}
}

#[cfg(test)]
mod test {
    use crate::prelude::*;
    use itertools::Itertools;

    #[test]
    fn unique() {
        let ca = ChunkedArray::<Int32Type>::new_from_slice("a", &[1, 2, 3, 2, 1]);
        assert_eq!(
            ca.unique().unwrap().sort(false).into_iter().collect_vec(),
            vec![Some(1), Some(2), Some(3)]
        );
        let ca = BooleanChunked::new_from_slice("a", &[true, false, true]);
        assert_eq!(
            ca.unique().unwrap().into_iter().collect_vec(),
            vec![Some(true), Some(false)]
        );

        let ca =
            Utf8Chunked::new_from_opt_slice("", &[Some("a"), None, Some("a"), Some("b"), None]);
        assert_eq!(
            Vec::from(&ca.unique().unwrap().sort(false)),
            &[None, Some("a"), Some("b")]
        );
    }

    #[test]
    fn arg_unique() {
        let ca = ChunkedArray::<Int32Type>::new_from_slice("a", &[1, 2, 1, 1, 3]);
        assert_eq!(
            ca.arg_unique().unwrap().into_iter().collect_vec(),
            vec![Some(0), Some(1), Some(4)]
        );
    }

    #[test]
    fn is_unique() {
        let ca = Float32Chunked::new_from_slice("a", &[1., 2., 1., 1., 3.]);
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
        let ca = UInt32Chunked::new_from_opt_slice(
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
