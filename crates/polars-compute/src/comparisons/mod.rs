#[cfg(feature = "dtype-array")]
use arrow::bitmap::utils::count_zeros;
use arrow::bitmap::{self, Bitmap};
use polars_array::{PlBitmap, PlBitmapRef};

pub trait TotalEqKernel: Sized {
    type Scalar: ?Sized;

    // The validity mask, with one bit per element. This is what `Array::validity` hands out for
    // an Arrow array; the arrays of `polars-array` implement these kernels in their flat
    // representation, whose mask is flat in turn. An array whose mask may repeat a single bit
    // implements `PlTotalEqKernel` instead.
    fn validity_mask(&self) -> Option<&Bitmap>;

    // These kernels ignore validity entirely (results for nulls are unspecified
    // but initialized).
    fn tot_eq_kernel(&self, other: &Self) -> Bitmap;
    fn tot_ne_kernel(&self, other: &Self) -> Bitmap;
    fn tot_eq_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap;
    fn tot_ne_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap;

    // These kernels treat null as any other value equal to itself but unequal
    // to anything else.
    fn tot_eq_missing_kernel(&self, other: &Self) -> Bitmap {
        let q = self.tot_eq_kernel(other);
        match (self.validity_mask(), other.validity_mask()) {
            (None, None) => q,
            (None, Some(r)) => &q & r,
            (Some(l), None) => &q & l,
            (Some(l), Some(r)) => bitmap::ternary(&q, l, r, |q, l, r| (q & l & r) | !(l | r)),
        }
    }

    fn tot_ne_missing_kernel(&self, other: &Self) -> Bitmap {
        let q = self.tot_ne_kernel(other);
        match (self.validity_mask(), other.validity_mask()) {
            (None, None) => q,
            (None, Some(r)) => &q | &!r,
            (Some(l), None) => &q | &!l,
            (Some(l), Some(r)) => bitmap::ternary(&q, l, r, |q, l, r| (q & l & r) | (l ^ r)),
        }
    }
    fn tot_eq_missing_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
        let q = self.tot_eq_kernel_broadcast(other);
        if let Some(valid) = self.validity_mask() {
            bitmap::binary(&q, valid, |q, v| q & v)
        } else {
            q
        }
    }

    fn tot_ne_missing_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
        let q = self.tot_ne_kernel_broadcast(other);
        if let Some(valid) = self.validity_mask() {
            bitmap::binary(&q, valid, |q, v| q | !v)
        } else {
            q
        }
    }
}

// Low-level comparison kernel.
pub trait TotalOrdKernel: Sized {
    type Scalar: ?Sized;

    // These kernels ignore validity entirely (results for nulls are unspecified
    // but initialized).
    fn tot_lt_kernel(&self, other: &Self) -> Bitmap;
    fn tot_le_kernel(&self, other: &Self) -> Bitmap;
    fn tot_gt_kernel(&self, other: &Self) -> Bitmap {
        other.tot_lt_kernel(self)
    }
    fn tot_ge_kernel(&self, other: &Self) -> Bitmap {
        other.tot_le_kernel(self)
    }

    // These kernels ignore validity entirely (results for nulls are unspecified
    // but initialized).
    fn tot_lt_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap;
    fn tot_le_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap;
    fn tot_gt_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap;
    fn tot_ge_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap;
}

/// What a validity mask leaves for the missing-aware kernels of [`PlTotalEqKernel`] to combine.
enum Validity<'a> {
    /// Every element is there: no mask at all, or one repeating a set bit.
    AllValid,
    /// Every element is null, which a mask repeating an unset bit says in a single bit.
    AllNull,
    /// One bit per element.
    Flat(&'a Bitmap),
}

fn validity_of(mask: Option<PlBitmapRef<'_>>) -> Validity<'_> {
    match mask {
        None => Validity::AllValid,
        Some(mask) => match mask.scalar_value() {
            Some(true) => Validity::AllValid,
            Some(false) => Validity::AllNull,
            None => Validity::Flat(mask.flat_bitmap().unwrap()),
        },
    }
}

/// `q & mask`, where `mask` holds one bit per element.
fn and_mask(q: PlBitmap, mask: &Bitmap) -> PlBitmap {
    match q.scalar_value() {
        Some(false) => q,
        Some(true) => PlBitmap::from_bitmap(mask.clone()),
        None => PlBitmap::from_bitmap(bitmap::binary(q.flat_bitmap().unwrap(), mask, |q, m| q & m)),
    }
}

/// `q | !mask`, where `mask` holds one bit per element.
fn or_not_mask(q: PlBitmap, mask: &Bitmap) -> PlBitmap {
    match q.scalar_value() {
        Some(true) => q,
        Some(false) => PlBitmap::from_bitmap(!mask),
        None => PlBitmap::from_bitmap(bitmap::binary(q.flat_bitmap().unwrap(), mask, |q, m| {
            q | !m
        })),
    }
}

/// How many values [`PlTotalEqKernel::tot_eq_missing_all`] compares where they lie before it is
/// worth writing the comparison out instead.
///
/// The written-out path allocates a mask and counts its bits for an answer that is one bool; the
/// vectorised kernel behind it only earns that back over enough values, and the elements of a
/// nested array are usually a handful.
pub(crate) const IN_PLACE_COMPARISON_LIMIT: usize = 64;

/// The answer for every element at once, held in the single bit that says it.
#[inline]
fn repeated(value: bool, length: usize) -> PlBitmap {
    PlBitmap::new_scalar(value, length)
}

/// How an element's own bit comes off the bits of the values under it.
#[cfg(feature = "dtype-array")]
#[derive(Clone, Copy)]
enum Condense {
    /// The element's bit is set when every value's bit is: what equality asks.
    All,
    /// The element's bit is set when any value's bit is: what inequality asks.
    Any,
}

#[cfg(feature = "dtype-array")]
impl Condense {
    /// The element's bit, given how many of its `width` values are unset.
    #[inline]
    fn apply(self, zeros: usize, width: usize) -> bool {
        match self {
            Self::All => zeros == 0,
            Self::Any => zeros < width,
        }
    }
}

/// The bit `values` — the bits of the values of a single element — condenses to.
///
/// The answer is read off the bits rather than written back out, which is what [`condense`] would
/// do for the one element: a caller that condenses element by element allocates nothing per one.
#[cfg(feature = "dtype-array")]
#[inline]
fn condense_one(values: &PlBitmap, how: Condense) -> bool {
    how.apply(values.unset_bits(), values.len())
}

/// Condenses `values`, holding `width` bits per element, into one bit per element.
///
/// The two ways of condensing are dispatched between here rather than inside the loop below: the
/// element's bit is read off a zero count with a couple of instructions, and a branch on `how` at
/// every one of them costs about as much again.
#[cfg(feature = "dtype-array")]
fn condense(values: PlBitmap, length: usize, width: usize, how: Condense) -> PlBitmap {
    match how {
        Condense::All => condense_by(values, length, width, |zeros, _| zeros == 0),
        Condense::Any => condense_by(values, length, width, |zeros, width| zeros < width),
    }
}

/// [`condense`], with the bit an element's zero count condenses to given as a closure that the one
/// loop below inlines.
#[cfg(feature = "dtype-array")]
#[inline]
fn condense_by(
    values: PlBitmap,
    length: usize,
    width: usize,
    bit: impl Fn(usize, usize) -> bool,
) -> PlBitmap {
    debug_assert!(width > 0);

    // One bit says the same of every value under every element, so it says the same of every
    // element in turn — however many values each of them covers.
    if let Some(set) = values.scalar_value() {
        return repeated(bit(if set { 0 } else { width }, width), length);
    }

    let values = values.into_bitmap();
    debug_assert_eq!(values.len(), length * width);

    let (slice, offset, _len) = values.as_slice();
    PlBitmap::from_bitmap(
        (0..length)
            .map(|i| bit(count_zeros(slice, offset + i * width, width), width))
            .collect(),
    )
}

/// The equality kernels over an array whose buffers may repeat a single slot.
pub trait PlTotalEqKernel: Sized {
    type Scalar: ?Sized;

    /// The validity mask, in whichever representation it is in.
    fn validity_mask(&self) -> Option<PlBitmapRef<'_>>;

    // These kernels ignore validity entirely (results for nulls are unspecified
    // but initialized).
    fn tot_eq_kernel(&self, other: &Self) -> PlBitmap;
    fn tot_ne_kernel(&self, other: &Self) -> PlBitmap;
    fn tot_eq_kernel_broadcast(&self, other: &Self::Scalar) -> PlBitmap;
    fn tot_ne_kernel_broadcast(&self, other: &Self::Scalar) -> PlBitmap;

    // These kernels treat null as any other value equal to itself but unequal
    // to anything else.
    fn tot_eq_missing_kernel(&self, other: &Self) -> PlBitmap {
        use Validity::*;

        let q = self.tot_eq_kernel(other);
        let length = q.len();

        match (
            validity_of(self.validity_mask()),
            validity_of(other.validity_mask()),
        ) {
            (AllValid, AllValid) => q,
            // A null is equal to a null and to nothing else, so a side that is null throughout
            // answers for every element at once, with no value read on either side.
            (AllNull, AllNull) => PlBitmap::new_scalar(true, length),
            (AllValid, AllNull) | (AllNull, AllValid) => PlBitmap::new_scalar(false, length),
            // One side is null throughout, so the answer is where the other side is null too.
            (AllNull, Flat(r)) => PlBitmap::from_bitmap(!r),
            (Flat(l), AllNull) => PlBitmap::from_bitmap(!l),
            (AllValid, Flat(r)) => and_mask(q, r),
            (Flat(l), AllValid) => and_mask(q, l),
            (Flat(l), Flat(r)) => {
                PlBitmap::from_bitmap(bitmap::ternary(&q.into_bitmap(), l, r, |q, l, r| {
                    (q & l & r) | !(l | r)
                }))
            },
        }
    }

    fn tot_ne_missing_kernel(&self, other: &Self) -> PlBitmap {
        use Validity::*;

        let q = self.tot_ne_kernel(other);
        let length = q.len();

        // The complement of `tot_eq_missing_kernel`, arm for arm.
        match (
            validity_of(self.validity_mask()),
            validity_of(other.validity_mask()),
        ) {
            (AllValid, AllValid) => q,
            (AllNull, AllNull) => PlBitmap::new_scalar(false, length),
            (AllValid, AllNull) | (AllNull, AllValid) => PlBitmap::new_scalar(true, length),
            (AllNull, Flat(r)) => PlBitmap::from_bitmap(r.clone()),
            (Flat(l), AllNull) => PlBitmap::from_bitmap(l.clone()),
            (AllValid, Flat(r)) => or_not_mask(q, r),
            (Flat(l), AllValid) => or_not_mask(q, l),
            (Flat(l), Flat(r)) => {
                PlBitmap::from_bitmap(bitmap::ternary(&q.into_bitmap(), l, r, |q, l, r| {
                    (q & l & r) | (l ^ r)
                }))
            },
        }
    }

    /// Whether every element of `self` equals the one at its index in `other`, a null equalling a
    /// null and nothing else.
    ///
    /// This is what a caller that wants the single bit asks for, rather than the mask: one element
    /// of a nested array against its counterpart, say, which is a handful of values at a time and
    /// once per element of the array above. The default writes the comparison out and counts the
    /// bits of it; an array whose values can be compared where they lie overrides it, since the
    /// allocation is the whole cost at that size.
    fn tot_eq_missing_all(&self, other: &Self) -> bool {
        self.tot_eq_missing_kernel(other).unset_bits() == 0
    }

    fn tot_eq_missing_kernel_broadcast(&self, other: &Self::Scalar) -> PlBitmap {
        let q = self.tot_eq_kernel_broadcast(other);
        let length = q.len();

        match validity_of(self.validity_mask()) {
            Validity::AllValid => q,
            // The scalar is a value and every element is null, so none of them is equal to it.
            Validity::AllNull => PlBitmap::new_scalar(false, length),
            Validity::Flat(valid) => and_mask(q, valid),
        }
    }

    fn tot_ne_missing_kernel_broadcast(&self, other: &Self::Scalar) -> PlBitmap {
        let q = self.tot_ne_kernel_broadcast(other);
        let length = q.len();

        match validity_of(self.validity_mask()) {
            Validity::AllValid => q,
            Validity::AllNull => PlBitmap::new_scalar(true, length),
            Validity::Flat(valid) => or_not_mask(q, valid),
        }
    }
}

/// The ordering kernels over an array whose buffers may repeat a single slot.
pub trait PlTotalOrdKernel: Sized {
    type Scalar: ?Sized;

    // These kernels ignore validity entirely (results for nulls are unspecified
    // but initialized).
    fn tot_lt_kernel(&self, other: &Self) -> PlBitmap;
    fn tot_le_kernel(&self, other: &Self) -> PlBitmap;
    fn tot_gt_kernel(&self, other: &Self) -> PlBitmap {
        other.tot_lt_kernel(self)
    }
    fn tot_ge_kernel(&self, other: &Self) -> PlBitmap {
        other.tot_le_kernel(self)
    }

    fn tot_lt_kernel_broadcast(&self, other: &Self::Scalar) -> PlBitmap;
    fn tot_le_kernel_broadcast(&self, other: &Self::Scalar) -> PlBitmap;
    fn tot_gt_kernel_broadcast(&self, other: &Self::Scalar) -> PlBitmap;
    fn tot_ge_kernel_broadcast(&self, other: &Self::Scalar) -> PlBitmap;
}

#[cfg(feature = "dtype-array")]
mod array;
mod binary;
mod boolean;
pub(crate) mod dyn_array;
mod list;
mod null;
mod pl_array;
mod pl_primitive;
mod scalar;
mod struct_;
mod view;

#[cfg(feature = "simd")]
mod _simd_dtypes {
    use arrow::types::{days_ms, i256, months_days_ns};

    use crate::NotSimdPrimitive;

    impl NotSimdPrimitive for i256 {}
    impl NotSimdPrimitive for days_ms {}
    impl NotSimdPrimitive for months_days_ns {}
}

#[cfg(feature = "simd")]
mod simd;
