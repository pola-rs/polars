#[cfg(not(feature = "bigidx"))]
use arrow::array::UInt32Array;
#[cfg(feature = "bigidx")]
use arrow::array::UInt64Array;

pub trait IndexToUsize {
    /// Translate the negative index to an offset.
    fn negative_to_usize(self, index: usize) -> Option<usize>;
}

impl IndexToUsize for i64 {
    fn negative_to_usize(self, index: usize) -> Option<usize> {
        if self >= 0 && (self as usize) < index {
            Some(self as usize)
        } else {
            let subtract = self.abs() as usize;
            if subtract > index {
                None
            } else {
                Some(index - subtract)
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
