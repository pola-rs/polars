#[cfg(feature = "object")]
use crate::chunked_array::object::ObjectType;
use crate::datatypes::PlHashSet;
use crate::frame::groupby::{GroupTuples, IntoGroupTuples};
use crate::prelude::*;
use crate::utils::NoNull;
use itertools::Itertools;
use num::NumCast;
use rayon::prelude::*;
use std::fmt::Display;
use std::hash::Hash;

pub(crate) fn is_unique_helper(
    mut groups: GroupTuples,
    len: u32,
    unique_val: bool,
    duplicated_val: bool,
) -> BooleanChunked {
    debug_assert_ne!(unique_val, duplicated_val);
    groups.sort_unstable_by_key(|t| t.0);

    let mut unique_idx_iter = groups
        .into_iter()
        .filter(|(_, g)| g.len() == 1)
        .map(|(first, _)| first);

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

fn is_unique<T>(ca: &ChunkedArray<T>) -> BooleanChunked
where
    T: PolarsDataType,
    ChunkedArray<T>: IntoGroupTuples,
{
    let groups = ca.group_tuples(true);
    let mut out = is_unique_helper(groups, ca.len() as u32, true, false);
    out.rename(ca.name());
    out
}

fn is_duplicated<T>(ca: &ChunkedArray<T>) -> BooleanChunked
where
    T: PolarsDataType,
    ChunkedArray<T>: IntoGroupTuples,
{
    let groups = ca.group_tuples(true);
    let mut out = is_unique_helper(groups, ca.len() as u32, false, true);
    out.rename(ca.name());
    out
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
    let mut unique = AlignedVec::with_capacity_aligned(capacity);
    a.enumerate().for_each(|(idx, val)| {
        if set.insert(val) {
            unique.push(idx as u32)
        }
    });
    unique
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
        Ok(is_unique(self))
    }

    fn is_duplicated(&self) -> Result<BooleanChunked> {
        Ok(is_duplicated(self))
    }

    fn value_counts(&self) -> Result<DataFrame> {
        impl_value_counts!(self)
    }

    fn n_unique(&self) -> Result<usize> {
        Ok(fill_set(self.into_iter()).len())
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
        Ok(is_unique(self))
    }
    fn is_duplicated(&self) -> Result<BooleanChunked> {
        Ok(is_duplicated(self))
    }

    fn value_counts(&self) -> Result<DataFrame> {
        impl_value_counts!(self)
    }

    fn n_unique(&self) -> Result<usize> {
        Ok(fill_set(self.into_iter()).len())
    }
}

impl ChunkUnique<CategoricalType> for CategoricalChunked {
    fn unique(&self) -> Result<Self> {
        let set = fill_set(self.into_iter());
        let mut ca = UInt32Chunked::new_from_opt_iter(self.name(), set.iter().copied());
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
        self.cast::<UInt32Type>()?.n_unique()
    }
}

#[cfg(feature = "dtype-u8")]
fn dummies_helper(mut groups: Vec<u32>, len: usize, name: &str) -> UInt8Chunked {
    groups.sort_unstable();

    // let mut group_member_iter = groups.into_iter();
    let mut av = AlignedVec::with_capacity_aligned(len);
    for _ in 0..len {
        av.push(0u8)
    }

    for idx in groups {
        let elem = unsafe { av.inner.get_unchecked_mut(idx as usize) };
        *elem = 1;
    }

    ChunkedArray::new_from_aligned_vec(name, av)
}

#[cfg(not(feature = "dtype-u8"))]
fn dummies_helper(mut groups: Vec<u32>, len: usize, name: &str) -> Int64Chunked {
    groups.sort_unstable();

    // let mut group_member_iter = groups.into_iter();
    let mut av = AlignedVec::with_capacity_aligned(len);
    for _ in 0..len {
        av.push(0i64)
    }

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

        let columns = groups
            .into_par_iter()
            .map(|(first, groups)| {
                let val = unsafe { self.get_unchecked(first as usize) };
                let name = format!("{}_{}", col_name, val);
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

        let columns = groups
            .into_par_iter()
            .map(|(first, groups)| {
                let val = unsafe { self.get_unchecked(first as usize) };
                let name = format!("{}_{}", col_name, val);
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
        Ok(is_unique(self))
    }
    fn is_duplicated(&self) -> Result<BooleanChunked> {
        Ok(is_duplicated(self))
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
        self.bit_repr_large().arg_unique()
    }

    fn is_unique(&self) -> Result<BooleanChunked> {
        self.bit_repr_large().is_unique()
    }
    fn is_duplicated(&self) -> Result<BooleanChunked> {
        self.bit_repr_large().is_duplicated()
    }
    fn value_counts(&self) -> Result<DataFrame> {
        impl_value_counts!(self)
    }
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
}
