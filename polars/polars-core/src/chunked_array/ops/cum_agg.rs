use crate::prelude::*;
use itertools::__std_iter::FromIterator;
use num::Bounded;
use std::ops::{Add, AddAssign};

fn det_max<T>(state: &mut T, v: Option<T>) -> Option<Option<T>>
where
    T: Copy + PartialOrd + AddAssign + Add<Output = T>,
{
    match v {
        Some(v) => {
            if v > *state {
                *state = v
            }
            Some(Some(*state))
        }
        None => Some(None),
    }
}

fn det_min<T>(state: &mut T, v: Option<T>) -> Option<Option<T>>
where
    T: Copy + PartialOrd + AddAssign + Add<Output = T>,
{
    match v {
        Some(v) => {
            if v < *state {
                *state = v
            }
            Some(Some(*state))
        }
        None => Some(None),
    }
}

fn det_sum<T>(state: &mut Option<T>, v: Option<T>) -> Option<Option<T>>
where
    T: Copy + PartialOrd + AddAssign + Add<Output = T>,
{
    match (*state, v) {
        (Some(state_inner), Some(v)) => {
            *state = Some(state_inner + v);
            Some(*state)
        }
        (None, Some(v)) => {
            *state = Some(v);
            Some(*state)
        }
        (_, None) => Some(None),
    }
}

impl<T> ChunkCumAgg<T> for ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: Bounded + PartialOrd + AddAssign + Add<Output = T::Native>,
    ChunkedArray<T>: FromIterator<Option<T::Native>>,
{
    fn cum_max(&self, reverse: bool) -> ChunkedArray<T> {
        let init = Bounded::min_value();
        let mut ca: Self = match reverse {
            false => self.into_iter().scan(init, det_max).collect(),
            true => self.into_iter().rev().scan(init, det_max).collect(),
        };

        ca.rename(self.name());
        if reverse {
            ca.reverse()
        } else {
            ca
        }
    }

    fn cum_min(&self, reverse: bool) -> ChunkedArray<T> {
        let init = Bounded::max_value();
        let mut ca: Self = match reverse {
            false => self.into_iter().scan(init, det_min).collect(),
            true => self.into_iter().rev().scan(init, det_min).collect(),
        };

        ca.rename(self.name());
        if reverse {
            ca.reverse()
        } else {
            ca
        }
    }

    fn cum_sum(&self, reverse: bool) -> ChunkedArray<T> {
        let init = None;
        let mut ca: Self = match reverse {
            false => self.into_iter().scan(init, det_sum).collect(),
            true => self.into_iter().rev().scan(init, det_sum).collect(),
        };

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
    fn test_cum_min() {
        let ca = UInt8Chunked::new_from_opt_slice("foo", &[None, Some(1), Some(3), None, Some(2)]);
        let out = ca.cum_min(true);
        assert_eq!(Vec::from(&out), &[None, Some(1), Some(2), None, Some(2)]);
        let out = ca.cum_min(false);
        assert_eq!(Vec::from(&out), &[None, Some(1), Some(1), None, Some(1)]);
    }

    #[test]
    fn test_cum_sum() {
        let ca = Int32Chunked::new_from_opt_slice("foo", &[None, Some(1), Some(3), None, Some(1)]);
        let out = ca.cum_sum(true);
        assert_eq!(Vec::from(&out), &[None, Some(5), Some(4), None, Some(1)]);
        let out = ca.cum_sum(false);
        assert_eq!(Vec::from(&out), &[None, Some(1), Some(4), None, Some(5)]);

        // just check if the trait bounds allow for floats
        let ca = Float32Chunked::new_from_opt_slice(
            "foo",
            &[None, Some(1.0), Some(3.0), None, Some(1.0)],
        );
        let _out = ca.cum_sum(false);
    }
}
