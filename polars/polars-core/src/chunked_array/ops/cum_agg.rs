use crate::prelude::*;
use crate::utils::CustomIterTools;
use itertools::__std_iter::FromIterator;
use num::Bounded;
use std::ops::{Add, AddAssign, Mul};

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

fn det_prod<T>(state: &mut Option<T>, v: Option<T>) -> Option<Option<T>>
where
    T: Copy + PartialOrd + Mul<Output = T>,
{
    match (*state, v) {
        (Some(state_inner), Some(v)) => {
            *state = Some(state_inner * v);
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
    ChunkedArray<T>: FromIterator<Option<T::Native>>,
{
    fn cummax(&self, reverse: bool) -> ChunkedArray<T> {
        let init = Bounded::min_value();
        let mut ca: Self = match reverse {
            false => self
                .into_iter()
                .scan(init, det_max)
                .trust_my_length(self.len())
                .collect_trusted(),
            true => self
                .into_iter()
                .rev()
                .scan(init, det_max)
                .trust_my_length(self.len())
                .collect_reversed(),
        };

        ca.rename(self.name());
        ca
    }

    fn cummin(&self, reverse: bool) -> ChunkedArray<T> {
        let init = Bounded::max_value();
        let mut ca: Self = match reverse {
            false => self
                .into_iter()
                .scan(init, det_min)
                .trust_my_length(self.len())
                .collect_trusted(),
            true => self
                .into_iter()
                .rev()
                .scan(init, det_min)
                .trust_my_length(self.len())
                .collect_reversed(),
        };

        ca.rename(self.name());
        ca
    }

    fn cumsum(&self, reverse: bool) -> ChunkedArray<T> {
        let init = None;
        let mut ca: Self = match reverse {
            false => self
                .into_iter()
                .scan(init, det_sum)
                .trust_my_length(self.len())
                .collect_trusted(),
            true => self
                .into_iter()
                .rev()
                .scan(init, det_sum)
                .trust_my_length(self.len())
                .collect_reversed(),
        };

        ca.rename(self.name());
        ca
    }

    fn cumprod(&self, reverse: bool) -> ChunkedArray<T> {
        let init = None;
        let mut ca: Self = match reverse {
            false => self
                .into_iter()
                .scan(init, det_prod)
                .trust_my_length(self.len())
                .collect_trusted(),
            true => self
                .into_iter()
                .rev()
                .scan(init, det_prod)
                .trust_my_length(self.len())
                .collect_reversed(),
        };

        ca.rename(self.name());
        ca
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn test_cummax() {
        let ca = UInt8Chunked::new("foo", &[None, Some(1), Some(3), None, Some(1)]);
        let out = ca.cummax(true);
        assert_eq!(Vec::from(&out), &[None, Some(3), Some(3), None, Some(1)]);
        let out = ca.cummax(false);
        assert_eq!(Vec::from(&out), &[None, Some(1), Some(3), None, Some(3)]);
    }

    #[test]
    fn test_cummin() {
        let ca = UInt8Chunked::new("foo", &[None, Some(1), Some(3), None, Some(2)]);
        let out = ca.cummin(true);
        assert_eq!(Vec::from(&out), &[None, Some(1), Some(2), None, Some(2)]);
        let out = ca.cummin(false);
        assert_eq!(Vec::from(&out), &[None, Some(1), Some(1), None, Some(1)]);
    }

    #[test]
    fn test_cumsum() {
        let ca = Int32Chunked::new("foo", &[None, Some(1), Some(3), None, Some(1)]);
        let out = ca.cumsum(true);
        assert_eq!(Vec::from(&out), &[None, Some(5), Some(4), None, Some(1)]);
        let out = ca.cumsum(false);
        assert_eq!(Vec::from(&out), &[None, Some(1), Some(4), None, Some(5)]);

        // just check if the trait bounds allow for floats
        let ca = Float32Chunked::new("foo", &[None, Some(1.0), Some(3.0), None, Some(1.0)]);
        let _out = ca.cumsum(false);
    }
}
