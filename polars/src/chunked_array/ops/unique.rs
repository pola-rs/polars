use crate::prelude::*;
use crate::utils::{floating_encode_f64, integer_decode_f64};
use num::{NumCast, ToPrimitive};
use seahash::SeaHasher;
use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hash};
use unsafe_unwrap::UnsafeUnwrap;

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

fn fill_set<A>(
    a: impl Iterator<Item = A>,
    capacity: usize,
) -> HashSet<A, BuildHasherDefault<SeaHasher>>
where
    A: Hash + Eq,
{
    let mut set = HashSet::with_capacity_and_hasher(capacity, BuildHasherDefault::default());

    for val in a {
        set.insert(val);
    }

    set
}

fn arg_unique<T>(a: impl Iterator<Item = T>, capacity: usize) -> Vec<usize>
where
    T: Hash + Eq,
{
    let mut set =
        HashSet::with_capacity_and_hasher(capacity, BuildHasherDefault::<SeaHasher>::default());
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

impl<T> ChunkUnique<T> for ChunkedArray<T>
where
    T: PolarsIntegerType,
    T::Native: Hash + Eq,
    ChunkedArray<T>: ChunkOps,
{
    fn unique(&self) -> Result<Self> {
        let set = match self.cont_slice() {
            Ok(slice) => fill_set(slice.iter().map(|v| Some(*v)), self.len()),
            Err(_) => fill_set(self.into_iter(), self.len()),
        };

        Ok(Self::new_from_opt_iter(self.name(), set.iter().copied()))
    }

    fn arg_unique(&self) -> Result<Vec<usize>> {
        arg_unique_ca(self)
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
}

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
}

// Use stable form of specialization using autoref
// https://github.com/dtolnay/case-studies/blob/master/autoref-specialization/README.md
impl<T> ChunkUnique<T> for &ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: NumCast + ToPrimitive,
    ChunkedArray<T>: ChunkOps,
{
    fn unique(&self) -> Result<ChunkedArray<T>> {
        let set = match self.cont_slice() {
            Ok(slice) => fill_set(
                slice
                    .iter()
                    .map(|v| Some(integer_decode_f64(v.to_f64().unwrap()))),
                self.len(),
            ),
            Err(_) => fill_set(
                self.into_iter()
                    .map(|opt_v| opt_v.map(|v| integer_decode_f64(v.to_f64().unwrap()))),
                self.len(),
            ),
        };
        Ok(ChunkedArray::new_from_opt_iter(
            self.name(),
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

    fn arg_unique(&self) -> Result<Vec<usize>> {
        match self.null_count() {
            0 => Ok(arg_unique(
                self.into_no_null_iter().map(|v| {
                    let v = v.to_f64();
                    debug_assert!(v.is_some());
                    let v = unsafe { v.unsafe_unwrap() };
                    integer_decode_f64(v)
                }),
                self.len(),
            )),
            _ => Ok(arg_unique(
                self.into_iter().map(|opt_v| {
                    opt_v.map(|v| {
                        let v = v.to_f64();
                        debug_assert!(v.is_some());
                        let v = unsafe { v.unsafe_unwrap() };
                        integer_decode_f64(v)
                    })
                }),
                self.len(),
            )),
        }
    }
}

pub trait ValueCounts<T>
where
    T: ArrowPrimitiveType,
{
    fn value_counts(&self) -> HashMap<Option<T::Native>, u32, BuildHasherDefault<SeaHasher>>;
}

fn fill_set_value_count<K>(
    a: impl Iterator<Item = K>,
    capacity: usize,
) -> HashMap<K, u32, BuildHasherDefault<SeaHasher>>
where
    K: Hash + Eq,
{
    let mut kv_store = HashMap::with_capacity_and_hasher(capacity, BuildHasherDefault::default());

    for key in a {
        let count = kv_store.entry(key).or_insert(0);
        *count += 1;
    }

    kv_store
}

impl<T> ValueCounts<T> for ChunkedArray<T>
where
    T: PolarsIntegerType,
    T::Native: Hash + Eq,
    ChunkedArray<T>: ChunkOps,
{
    fn value_counts(&self) -> HashMap<Option<T::Native>, u32, BuildHasherDefault<SeaHasher>> {
        match self.null_count() {
            0 => fill_set_value_count(self.into_no_null_iter().map(|v| Some(v)), self.len()),
            _ => fill_set_value_count(self.into_iter(), self.len()),
        }
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
}
