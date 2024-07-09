use std::fmt::Display;

use num_traits::{NumCast, Signed, Zero};
use polars_error::{polars_err, PolarsResult};
use polars_utils::IdxSize;

use crate::array::PrimitiveArray;

pub trait IndexToUsize: Display {
    /// Translate the negative index to an offset.
    fn negative_to_usize(self, len: usize) -> Option<usize>;

    fn try_negative_to_usize(self, len: usize) -> PolarsResult<usize>
    where
        Self: Sized + Copy,
    {
        self.negative_to_usize(len)
            .ok_or_else(|| polars_err!(OutOfBounds: "index {} for length: {}", self, len))
    }
}

impl<I> IndexToUsize for I
where
    I: PartialOrd + PartialEq + NumCast + Signed + Zero + Display,
{
    #[inline]
    fn negative_to_usize(self, len: usize) -> Option<usize> {
        if self >= Zero::zero() {
            if (self.to_usize().unwrap()) < len {
                Some(self.to_usize().unwrap())
            } else {
                None
            }
        } else {
            let subtract = self.abs().to_usize().unwrap();
            if subtract > len {
                None
            } else {
                Some(len - subtract)
            }
        }
    }
}

pub fn indexes_to_usizes(idx: &[IdxSize]) -> impl Iterator<Item = usize> + '_ {
    idx.iter().map(|idx| *idx as usize)
}

pub type IdxArr = PrimitiveArray<IdxSize>;
