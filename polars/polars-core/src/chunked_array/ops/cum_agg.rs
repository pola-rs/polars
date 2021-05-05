use crate::prelude::*;
use itertools::__std_iter::FromIterator;
use num::Bounded;
use std::ops::AddAssign;

impl<T> ChunkCumAgg<T> for ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: Bounded + PartialOrd + AddAssign,
    ChunkedArray<T>: FromIterator<Option<T::Native>>,
{
    fn cum_max(&self, reverse: bool) -> ChunkedArray<T> {
        let iter: Box<dyn Iterator<Item = Option<T::Native>>> = match reverse {
            false => Box::new(self.into_iter()),
            true => Box::new(self.into_iter().rev()),
        };
        let mut ca: Self = iter
            .scan(Bounded::min_value(), |state, v| match v {
                Some(v) => {
                    if v > *state {
                        *state = v
                    }
                    Some(Some(*state))
                }
                None => Some(None),
            })
            .collect();
        ca.rename(self.name());
        if reverse {
            ca.reverse()
        } else {
            ca
        }
    }

    fn cum_min(&self, reverse: bool) -> ChunkedArray<T> {
        let iter: Box<dyn Iterator<Item = Option<T::Native>>> = match reverse {
            false => Box::new(self.into_iter()),
            true => Box::new(self.into_iter().rev()),
        };
        let mut ca: Self = iter
            .scan(Bounded::max_value(), |state, v| match v {
                Some(v) => {
                    if v < *state {
                        *state = v
                    }
                    Some(Some(*state))
                }
                None => Some(None),
            })
            .collect();
        ca.rename(self.name());
        if reverse {
            ca.reverse()
        } else {
            ca
        }
    }

    fn cum_sum(&self, reverse: bool) -> ChunkedArray<T> {
        let iter: Box<dyn Iterator<Item = Option<T::Native>>> = match reverse {
            false => Box::new(self.into_iter()),
            true => Box::new(self.into_iter().rev()),
        };
        let mut ca: Self = iter
            .scan(Bounded::min_value(), |state, v| match v {
                Some(v) => {
                    *state += v;
                    Some(Some(*state))
                }
                None => Some(None),
            })
            .collect();
        ca.rename(self.name());
        if reverse {
            ca.reverse()
        } else {
            ca
        }
    }
}

impl ChunkCumAgg<CategoricalType> for CategoricalChunked {}
impl ChunkCumAgg<Utf8Type> for Utf8Chunked {}
impl ChunkCumAgg<ListType> for ListChunked {}
impl ChunkCumAgg<BooleanType> for BooleanChunked {}

#[cfg(feature = "object")]
impl<T> ChunkCumAgg<ObjectType<T>> for ObjectChunked<T> {}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn test_cum_max() {
        let ca = UInt8Chunked::new_from_opt_slice("foo", &[None, Some(1), Some(3), None, Some(1)]);
        let out = ca.cum_max(true);
        assert_eq!(Vec::from(&out), &[None, Some(3), Some(3), None, Some(1)]);
        let out = ca.cum_max(false);
        assert_eq!(Vec::from(&out), &[None, Some(1), Some(3), None, Some(3)]);
    }

    #[test]
    fn test_cum_sum() {
        let ca = Int32Chunked::new_from_opt_slice("foo", &[None, Some(1), Some(3), None, Some(1)]);
        let out = ca.cum_sum(true);
        assert_eq!(
            Vec::from(&out),
            &[Some(5), Some(5), Some(4), Some(1), Some(1)]
        );
        let out = ca.cum_sum(false);
        assert_eq!(Vec::from(&out), &[None, Some(1), Some(4), Some(4), Some(5)]);
    }
}
