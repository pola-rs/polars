use std::fmt::Display;

use num_traits::{NumCast, Signed, Zero};
use polars_error::{PolarsResult, polars_err};
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
            self.to_usize().filter(|idx| *idx < len)
        } else {
            let subtract = usize::try_from(self.to_i128()?.unsigned_abs()).ok()?;
            len.checked_sub(subtract)
        }
    }
}

pub fn indexes_to_usizes(idx: &[IdxSize]) -> impl Iterator<Item = usize> + '_ {
    idx.iter().map(|idx| *idx as usize)
}

pub type IdxArr = PrimitiveArray<IdxSize>;
