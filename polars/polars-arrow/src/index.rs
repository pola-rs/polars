#[cfg(not(feature = "bigidx"))]
use arrow::array::UInt32Array;
#[cfg(feature = "bigidx")]
use arrow::array::UInt64Array;
use num::{NumCast, Signed, Zero};

pub trait IndexToUsize {
    /// Translate the negative index to an offset.
    fn negative_to_usize(self, len: usize) -> Option<usize>;
}

impl<I> IndexToUsize for I
where
    I: PartialOrd + PartialEq + NumCast + Signed + Zero,
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

/// The type used by polars to index data.
#[cfg(not(feature = "bigidx"))]
pub type IdxSize = u32;
#[cfg(feature = "bigidx")]
pub type IdxSize = u64;

#[cfg(not(feature = "bigidx"))]
pub type IdxArr = UInt32Array;
#[cfg(feature = "bigidx")]
pub type IdxArr = UInt64Array;

pub fn indexes_to_usizes(idx: &[IdxSize]) -> impl Iterator<Item = usize> + '_ {
    idx.iter().map(|idx| *idx as usize)
}
