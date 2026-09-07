//! The comparison kernels over a [`PlBooleanArray`], whose values are a mask of their own.
//!
//! Because the values are a [`PlBitmap`], every one of these kernels is a bitwise operation on
//! that mask, and a mask that repeats a single bit comes out of one as a single bit in turn — the
//! boolean kernels never have to write a values buffer out to answer.

use polars_array::{PlBitmap, PlBitmapRef, PlBooleanArray};

use super::{PlTotalEqKernel, PlTotalOrdKernel};

/// The answer for every element at once, held in the single bit that says it.
#[inline]
fn repeated(value: bool, length: usize) -> PlBitmap {
    PlBitmap::new_scalar(value, length)
}

/// The values of `array` as a mask of their own, in whichever representation they are in.
#[inline]
fn values(array: &PlBooleanArray) -> PlBitmap {
    PlBitmap::from(array.values())
}

impl PlTotalEqKernel for PlBooleanArray {
    type Scalar = bool;

    fn validity_mask(&self) -> Option<PlBitmapRef<'_>> {
        self.validity()
    }

    fn tot_eq_kernel(&self, other: &Self) -> PlBitmap {
        assert_eq!(self.len(), other.len());
        // Two bits are equal when they do not differ, which `xor` answers in whichever
        // representation the two sides are in.
        values(self).xor(&values(other)).not()
    }

    fn tot_ne_kernel(&self, other: &Self) -> PlBitmap {
        assert_eq!(self.len(), other.len());
        values(self).xor(&values(other))
    }

    fn tot_eq_kernel_broadcast(&self, other: &Self::Scalar) -> PlBitmap {
        // Every element equals `true` exactly where its own bit is set, and `false` where it is
        // not: the answer is the values themselves, inverted or not.
        let values = values(self);
        if *other { values } else { values.not() }
    }

    fn tot_ne_kernel_broadcast(&self, other: &Self::Scalar) -> PlBitmap {
        self.tot_eq_kernel_broadcast(&!*other)
    }
}

impl PlTotalOrdKernel for PlBooleanArray {
    type Scalar = bool;

    fn tot_lt_kernel(&self, other: &Self) -> PlBitmap {
        assert_eq!(self.len(), other.len());
        // `false < true` and nothing else, so this is where the right side is set and the left is
        // not.
        values(self).not().and(&values(other))
    }

    fn tot_le_kernel(&self, other: &Self) -> PlBitmap {
        assert_eq!(self.len(), other.len());
        values(self).not().or(&values(other))
    }

    fn tot_lt_kernel_broadcast(&self, other: &Self::Scalar) -> PlBitmap {
        if *other {
            // Only `false` is below `true`.
            values(self).not()
        } else {
            // Nothing is below `false`, which holds of every element at once.
            repeated(false, self.len())
        }
    }

    fn tot_le_kernel_broadcast(&self, other: &Self::Scalar) -> PlBitmap {
        if *other {
            // Every boolean is at or below `true`, which holds of every element at once.
            repeated(true, self.len())
        } else {
            values(self).not()
        }
    }

    fn tot_gt_kernel_broadcast(&self, other: &Self::Scalar) -> PlBitmap {
        if *other {
            repeated(false, self.len())
        } else {
            values(self)
        }
    }

    fn tot_ge_kernel_broadcast(&self, other: &Self::Scalar) -> PlBitmap {
        if *other {
            values(self)
        } else {
            repeated(true, self.len())
        }
    }
}
