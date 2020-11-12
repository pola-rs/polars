use crate::prelude::*;
use crate::utils::{floating_encode_f64, integer_decode_f64, Xob};
use crate::{chunked_array::float::IntegerDecode, frame::group_by::IntoGroupTuples};
use ahash::RandomState;
use num::{NumCast, ToPrimitive};
use rayon::prelude::*;
use std::collections::HashSet;
use std::fmt::Display;
use std::hash::Hash;

pub(crate) fn is_unique_helper(
    groups: impl Iterator<Item = (usize, Vec<usize>)>,
    len: usize,
    unique_val: bool,
    duplicated_val: bool,
) -> Result<BooleanChunked> {
    debug_assert_ne!(unique_val, duplicated_val);
    let mut groups = groups.collect::<Vec<_>>();
    groups.sort_unstable_by_key(|t| t.0);

    let mut unique_idx_iter = groups
        .into_iter()
        .filter(|(_, g)| g.len() == 1)
        .map(|(first, _)| first);

    let mut next_unique_idx = unique_idx_iter.next();
    let mask = (0..len)
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
        .collect();
    Ok(mask)
}

fn is_unique<T>(ca: &ChunkedArray<T>) -> Result<BooleanChunked>
where
    T: PolarsDataType,
    ChunkedArray<T>: IntoGroupTuples,
{
    dbg!(ca.group_tuples());
    let groups = ca.group_tuples().into_iter();

    is_unique_helper(groups, ca.len(), true, false)
}

fn is_duplicated<T>(ca: &ChunkedArray<T>) -> Result<BooleanChunked>
where
    T: PolarsDataType,
    ChunkedArray<T>: IntoGroupTuples,
{
    let groups = ca.group_tuples().into_iter();
    is_unique_helper(groups, ca.len(), false, true)
}

impl ChunkUnique<ListType> for ListChunked {
    fn unique(&self) -> Result<ChunkedArray<ListType>> {
        Err(PolarsError::InvalidOperation(
            "unique not support for large list".into(),
        ))
    }

    fn arg_unique(&self) -> Result<Vec<usize>> {
        Err(PolarsError::InvalidOperation(
            "unique not support for large list".into(),
        ))
    }
}

fn fill_set<A>(a: impl Iterator<Item = A>, capacity: usize) -> HashSet<A, RandomState>
where
    A: Hash + Eq,
{
    let mut set = HashSet::with_capacity_and_hasher(capacity, RandomState::new());

    for val in a {
        set.insert(val);
    }

    set
}

fn arg_unique<T>(a: impl Iterator<Item = T>, capacity: usize) -> Vec<usize>
where
    T: Hash + Eq,
{
    let mut set = HashSet::with_capacity_and_hasher(capacity, RandomState::new());
    let mut unique = Vec::with_capacity(capacity);
    a.enumerate().for_each(|(idx, val)| {
        if set.insert(val) {
            unique.push(idx)
        }
    });

    unique
}

fn arg_unique_ca<'a, T>(ca: &'a ChunkedArray<T>) -> Result<Vec<usize>>
where
    &'a ChunkedArray<T>: IntoIterator + IntoNoNullIterator,
    T: 'a,
    <&'a ChunkedArray<T> as IntoIterator>::Item: Eq + Hash,
    <&'a ChunkedArray<T> as IntoNoNullIterator>::Item: Eq + Hash,
{
    match ca.null_count() {
        0 => Ok(arg_unique(ca.into_no_null_iter(), ca.len())),
        _ => Ok(arg_unique(ca.into_iter(), ca.len())),
    }
}

macro_rules! impl_value_counts {
    ($self:expr) => {{
        let group_tuples = $self.group_tuples();
        let mut values = unsafe {
            $self.take_unchecked(group_tuples.iter().map(|t| t.0), Some(group_tuples.len()))
        };
        values.rename("counts");
        let counts: Xob<UInt32Chunked> = group_tuples
            .into_iter()
            .map(|(_, groups)| groups.len() as u32)
            .collect();
        let cols = vec![values.into_series(), counts.into_inner().into_series()];
        let df = DataFrame::new_no_checks(cols);
        df.sort("counts", true)
    }};
}

impl<T> ChunkUnique<T> for ChunkedArray<T>
where
    T: PolarsIntegerType,
    T::Native: Hash + Eq,
    ChunkedArray<T>: ChunkOps,
{
    fn unique(&self) -> Result<Self> {
        let set = match self.null_count() {
            0 => fill_set(self.into_no_null_iter().map(Some), self.len()),
            _ => fill_set(self.into_iter(), self.len()),
        };

        Ok(Self::new_from_opt_iter(self.name(), set.iter().copied()))
    }

    fn arg_unique(&self) -> Result<Vec<usize>> {
        arg_unique_ca(self)
    }

    fn is_unique(&self) -> Result<BooleanChunked> {
        is_unique(self)
    }

    fn is_duplicated(&self) -> Result<BooleanChunked> {
        is_duplicated(self)
    }

    fn value_counts(&self) -> Result<DataFrame> {
        impl_value_counts!(self)
    }
}

impl ChunkUnique<Utf8Type> for Utf8Chunked {
    fn unique(&self) -> Result<Self> {
        let set = fill_set(self.into_iter(), self.len());
        Ok(Utf8Chunked::new_from_opt_iter(
            self.name(),
            set.iter().copied(),
        ))
    }

    fn arg_unique(&self) -> Result<Vec<usize>> {
        arg_unique_ca(self)
    }

    fn is_unique(&self) -> Result<BooleanChunked> {
        is_unique(self)
    }
    fn is_duplicated(&self) -> Result<BooleanChunked> {
        is_duplicated(self)
    }

    fn value_counts(&self) -> Result<DataFrame> {
        impl_value_counts!(self)
    }
}

impl ToDummies<Utf8Type> for Utf8Chunked {
    fn to_dummies(&self) -> Result<DataFrame> {
        let u = self.unique()?;
        let unique = u.into_iter().collect::<Vec<_>>();
        let col_name = self.name();

        let columns = unique
            .par_iter()
            .filter_map(|opt_val| {
                opt_val.map(|val| {
                    let mut ca = self.eq(val);
                    ca.rename(&format!("{}_{}", col_name, val));
                    ca.into_series()
                })
            })
            .collect();
        Ok(DataFrame::new_no_checks(columns))
    }
}
impl<T> ToDummies<T> for ChunkedArray<T>
where
    T: PolarsIntegerType + Sync,
    T::Native: Hash + Eq + Display,
    ChunkedArray<T>: ChunkOps + ChunkCompare<T::Native> + ChunkUnique<T>,
{
    fn to_dummies(&self) -> Result<DataFrame> {
        let unique = self.unique()?.into_iter().collect::<Vec<_>>();
        let col_name = self.name();

        let columns = unique
            .par_iter()
            .filter_map(|opt_val| {
                opt_val.map(|val| {
                    let mut ca = self.eq(val);
                    ca.rename(&format!("{}_{}", col_name, val));
                    ca.into_series()
                })
            })
            .collect();
        Ok(DataFrame::new_no_checks(columns))
    }
}

impl ToDummies<ListType> for ListChunked {}
impl ToDummies<Float32Type> for Float32Chunked {}
impl ToDummies<Float64Type> for Float64Chunked {}
impl ToDummies<BooleanType> for BooleanChunked {}

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

    fn arg_unique(&self) -> Result<Vec<usize>> {
        arg_unique_ca(self)
    }

    fn is_unique(&self) -> Result<BooleanChunked> {
        is_unique(self)
    }
    fn is_duplicated(&self) -> Result<BooleanChunked> {
        is_duplicated(self)
    }
}

fn float_unique<T>(ca: &ChunkedArray<T>) -> Result<ChunkedArray<T>>
where
    T: PolarsFloatType,
    T::Native: NumCast + ToPrimitive,
{
    let set = match ca.null_count() {
        0 => fill_set(
            ca.into_no_null_iter()
                .map(|v| Some(integer_decode_f64(v.to_f64().unwrap()))),
            ca.len(),
        ),
        _ => fill_set(
            ca.into_iter()
                .map(|opt_v| opt_v.map(|v| integer_decode_f64(v.to_f64().unwrap()))),
            ca.len(),
        ),
    };
    Ok(ChunkedArray::new_from_opt_iter(
        ca.name(),
        set.iter().copied().map(|opt| match opt {
            Some((mantissa, exponent, sign)) => {
                let flt = floating_encode_f64(mantissa, exponent, sign);
                let val: T::Native = NumCast::from(flt).unwrap();
                Some(val)
            }
            None => None,
        }),
    ))
}

fn float_arg_unique<T>(ca: &ChunkedArray<T>) -> Result<Vec<usize>>
where
    T: PolarsFloatType,
    T::Native: IntegerDecode,
{
    match ca.null_count() {
        0 => Ok(arg_unique(
            ca.into_no_null_iter().map(|v| v.integer_decode()),
            ca.len(),
        )),
        _ => Ok(arg_unique(
            ca.into_iter()
                .map(|opt_v| opt_v.map(|v| v.integer_decode())),
            ca.len(),
        )),
    }
}

impl ChunkUnique<Float32Type> for Float32Chunked {
    fn unique(&self) -> Result<ChunkedArray<Float32Type>> {
        float_unique(self)
    }

    fn arg_unique(&self) -> Result<Vec<usize>> {
        float_arg_unique(self)
    }

    fn is_unique(&self) -> Result<BooleanChunked> {
        is_unique(self)
    }
    fn is_duplicated(&self) -> Result<BooleanChunked> {
        is_duplicated(self)
    }
    fn value_counts(&self) -> Result<DataFrame> {
        impl_value_counts!(self)
    }
}

impl ChunkUnique<Float64Type> for Float64Chunked {
    fn unique(&self) -> Result<ChunkedArray<Float64Type>> {
        float_unique(self)
    }

    fn arg_unique(&self) -> Result<Vec<usize>> {
        float_arg_unique(self)
    }

    fn is_unique(&self) -> Result<BooleanChunked> {
        is_unique(self)
    }
    fn is_duplicated(&self) -> Result<BooleanChunked> {
        is_duplicated(self)
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
            vec![0, 1, 4]
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
