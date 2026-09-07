//! The equality kernels over the nested and degenerate arrays of `polars-array`.
//!
//! A comparison of these arrays is often settled by the shapes alone — two fixed-size lists of
//! different widths are never equal, two structs of different fields never are, and every element
//! of a null array is the same null — and the answer is then the one bit that says so for every
//! element, rather than that bit written out per element.

use arrow::bitmap::utils::count_zeros;
#[cfg(feature = "dtype-array")]
use polars_array::PlFixedSizeListArray;
use polars_array::{
    PlArray, PlBitmap, PlBitmapRef, PlFixedSizeBinaryArray, PlListArray, PlNullArray, PlStructArray,
};

use super::PlTotalEqKernel;
use super::pl_dyn_array::{pl_array_tot_eq_missing_kernel, pl_array_tot_ne_missing_kernel};

/// The answer for every element at once, held in the single bit that says it.
#[inline]
fn repeated(value: bool, length: usize) -> PlBitmap {
    PlBitmap::new_scalar(value, length)
}

/// How an element's own bit comes off the bits of the values under it.
#[derive(Clone, Copy)]
enum Condense {
    /// The element's bit is set when every value's bit is: what equality asks.
    All,
    /// The element's bit is set when any value's bit is: what inequality asks.
    Any,
}

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

/// Condenses `values`, holding `width` bits per element, into one bit per element.
fn condense(values: PlBitmap, length: usize, width: usize, how: Condense) -> PlBitmap {
    debug_assert!(width > 0);

    // One bit says the same of every value under every element, so it says the same of every
    // element in turn — however many values each of them covers.
    if let Some(bit) = values.scalar_value() {
        return repeated(how.apply(if bit { 0 } else { width }, width), length);
    }

    let values = values.into_bitmap();
    debug_assert_eq!(values.len(), length * width);

    let (slice, offset, _len) = values.as_slice();
    PlBitmap::from_bitmap(
        (0..length)
            .map(|i| how.apply(count_zeros(slice, offset + i * width, width), width))
            .collect(),
    )
}

impl PlTotalEqKernel for PlNullArray {
    type Scalar = Box<dyn PlArray>;

    fn validity_mask(&self) -> Option<PlBitmapRef<'_>> {
        // Every element of a null array is null, which its mask says in a single bit.
        Some(self.validity())
    }

    fn tot_eq_kernel(&self, other: &Self) -> PlBitmap {
        assert_eq!(self.len(), other.len());
        // There is no value under a null to read, so the answer is the same for every element.
        repeated(true, self.len())
    }

    fn tot_ne_kernel(&self, other: &Self) -> PlBitmap {
        assert_eq!(self.len(), other.len());
        repeated(false, self.len())
    }

    fn tot_eq_kernel_broadcast(&self, _other: &Self::Scalar) -> PlBitmap {
        todo!("comparison of a null array against a scalar")
    }

    fn tot_ne_kernel_broadcast(&self, _other: &Self::Scalar) -> PlBitmap {
        todo!("comparison of a null array against a scalar")
    }
}

impl PlTotalEqKernel for PlFixedSizeBinaryArray {
    type Scalar = [u8];

    fn validity_mask(&self) -> Option<PlBitmapRef<'_>> {
        self.validity()
    }

    fn tot_eq_kernel(&self, other: &Self) -> PlBitmap {
        assert_eq!(self.len(), other.len());

        // Byte strings of different widths are never equal, and ones of no bytes always are:
        // either way the widths settle it for every element without a byte being read.
        if self.width() != other.width() {
            return repeated(false, self.len());
        }
        if self.width() == 0 {
            return repeated(true, self.len());
        }

        match (self.scalar_values(), other.scalar_values()) {
            // Each side repeats one byte string, so the one comparison answers for all of them.
            (Some(l), Some(r)) => repeated(l == r, self.len()),
            _ => PlBitmap::from_iter((0..self.len()).map(|i| self.value(i) == other.value(i))),
        }
    }

    fn tot_ne_kernel(&self, other: &Self) -> PlBitmap {
        self.tot_eq_kernel(other).not()
    }

    fn tot_eq_kernel_broadcast(&self, other: &Self::Scalar) -> PlBitmap {
        if self.width() != other.len() {
            return repeated(false, self.len());
        }

        match self.scalar_values() {
            Some(l) => repeated(l == other, self.len()),
            None => PlBitmap::from_iter((0..self.len()).map(|i| self.value(i) == other)),
        }
    }

    fn tot_ne_kernel_broadcast(&self, other: &Self::Scalar) -> PlBitmap {
        self.tot_eq_kernel_broadcast(other).not()
    }
}

impl PlTotalEqKernel for PlStructArray {
    type Scalar = Box<dyn PlArray>;

    fn validity_mask(&self) -> Option<PlBitmapRef<'_>> {
        self.validity()
    }

    fn tot_eq_kernel(&self, other: &Self) -> PlBitmap {
        assert_eq!(self.len(), other.len());

        // Rows of different fields are never equal, and a row of no fields always is: either way
        // the shape settles it for every element without a field being read.
        if self.num_fields() != other.num_fields() {
            return repeated(false, self.len());
        }

        // A row is equal when every one of its fields is, so the fields fold together with `and` —
        // which keeps a field that answers the same of every row in the one bit that says it.
        let mut out = repeated(true, self.len());
        for (lhs, rhs) in self.fields().iter().zip(other.fields()) {
            if lhs.array_type() != rhs.array_type() || lhs.len() != rhs.len() {
                return repeated(false, self.len());
            }
            out = out.and(&pl_array_tot_eq_missing_kernel(&**lhs, &**rhs));

            // Nothing a later field says can set a bit this one has cleared.
            if out.scalar_value() == Some(false) {
                break;
            }
        }
        out
    }

    fn tot_ne_kernel(&self, other: &Self) -> PlBitmap {
        assert_eq!(self.len(), other.len());

        if self.num_fields() != other.num_fields() {
            return repeated(true, self.len());
        }

        // A row differs when any one of its fields does, so the fields fold together with `or`.
        let mut out = repeated(false, self.len());
        for (lhs, rhs) in self.fields().iter().zip(other.fields()) {
            if lhs.array_type() != rhs.array_type() || lhs.len() != rhs.len() {
                return repeated(true, self.len());
            }
            out = out.or(&pl_array_tot_ne_missing_kernel(&**lhs, &**rhs));

            if out.scalar_value() == Some(true) {
                break;
            }
        }
        out
    }

    fn tot_eq_kernel_broadcast(&self, _other: &Self::Scalar) -> PlBitmap {
        todo!("comparison of a struct array against a scalar")
    }

    fn tot_ne_kernel_broadcast(&self, _other: &Self::Scalar) -> PlBitmap {
        todo!("comparison of a struct array against a scalar")
    }
}

#[cfg(feature = "dtype-array")]
impl PlTotalEqKernel for PlFixedSizeListArray {
    type Scalar = Box<dyn PlArray>;

    fn validity_mask(&self) -> Option<PlBitmapRef<'_>> {
        self.validity()
    }

    fn tot_eq_kernel(&self, other: &Self) -> PlBitmap {
        assert_eq!(self.len(), other.len());
        fsl_compare_values(
            self,
            other,
            Condense::All,
            pl_array_tot_eq_missing_kernel,
            false,
        )
    }

    fn tot_ne_kernel(&self, other: &Self) -> PlBitmap {
        assert_eq!(self.len(), other.len());
        fsl_compare_values(
            self,
            other,
            Condense::Any,
            pl_array_tot_ne_missing_kernel,
            true,
        )
    }

    fn tot_eq_kernel_broadcast(&self, other: &Self::Scalar) -> PlBitmap {
        fsl_compare_scalar(
            self,
            &**other,
            Condense::All,
            pl_array_tot_eq_missing_kernel,
            false,
        )
    }

    fn tot_ne_kernel_broadcast(&self, other: &Self::Scalar) -> PlBitmap {
        fsl_compare_scalar(
            self,
            &**other,
            Condense::Any,
            pl_array_tot_ne_missing_kernel,
            true,
        )
    }
}

/// Compares the lists of `lhs` against `rhs`'s, element for element.
///
/// `mismatch` is the answer for a list whose values cannot be compared against the other side's
/// at all — a different width or value type — which holds for every element at once.
#[cfg(feature = "dtype-array")]
fn fsl_compare_values(
    lhs: &PlFixedSizeListArray,
    rhs: &PlFixedSizeListArray,
    how: Condense,
    inner: fn(&dyn PlArray, &dyn PlArray) -> PlBitmap,
    mismatch: bool,
) -> PlBitmap {
    let (length, width) = (lhs.len(), lhs.width());

    if width != rhs.width() || lhs.values().array_type() != rhs.values().array_type() {
        return repeated(mismatch, length);
    }
    // A list of no values is the same list on both sides, whatever is under it.
    if width == 0 {
        return repeated(!mismatch, length);
    }

    match (lhs.scalar_values(), rhs.scalar_values()) {
        // Each side repeats one list, so comparing those two lists once — `width` values, not
        // `length * width` of them — answers for every element.
        (Some(lhs), Some(rhs)) => {
            let bit = condense(inner(lhs, rhs), 1, width, how);
            repeated(bit.get(0), length)
        },
        // At least one side holds every element's values, so both are read that way.
        _ => {
            let (lhs, rhs) = (lhs.to_flat(), rhs.to_flat());
            let values = inner(lhs.as_array().values(), rhs.as_array().values());
            condense(values, length, width, how)
        },
    }
}

/// Compares the lists of `lhs` against the single list `rhs`.
#[cfg(feature = "dtype-array")]
fn fsl_compare_scalar(
    lhs: &PlFixedSizeListArray,
    rhs: &dyn PlArray,
    how: Condense,
    inner: fn(&dyn PlArray, &dyn PlArray) -> PlBitmap,
    mismatch: bool,
) -> PlBitmap {
    let (length, width) = (lhs.len(), lhs.width());

    if width != rhs.len() || lhs.values().array_type() != rhs.array_type() {
        return repeated(mismatch, length);
    }
    if width == 0 {
        return repeated(!mismatch, length);
    }

    // The scalar is one list, so a side that repeats one list too is a single comparison.
    if let Some(lhs) = lhs.scalar_values() {
        let bit = condense(inner(lhs, rhs), 1, width, how);
        return repeated(bit.get(0), length);
    }

    // Every element's list is compared against the scalar's, which is read again for each.
    let lhs = lhs.to_flat();
    let values = lhs.as_array().values();
    PlBitmap::from_iter((0..length).map(|i| {
        let mut element = values.to_boxed();
        element.slice(i * width, width);
        condense(inner(&*element, rhs), 1, width, how).get(0)
    }))
}

impl PlTotalEqKernel for PlListArray {
    type Scalar = Box<dyn PlArray>;

    fn validity_mask(&self) -> Option<PlBitmapRef<'_>> {
        self.validity()
    }

    fn tot_eq_kernel(&self, other: &Self) -> PlBitmap {
        assert_eq!(self.len(), other.len());
        list_compare_values(
            self,
            other,
            Condense::All,
            pl_array_tot_eq_missing_kernel,
            false,
        )
    }

    fn tot_ne_kernel(&self, other: &Self) -> PlBitmap {
        assert_eq!(self.len(), other.len());
        list_compare_values(
            self,
            other,
            Condense::Any,
            pl_array_tot_ne_missing_kernel,
            true,
        )
    }

    fn tot_eq_kernel_broadcast(&self, other: &Self::Scalar) -> PlBitmap {
        list_compare_scalar(
            self,
            &**other,
            Condense::All,
            pl_array_tot_eq_missing_kernel,
            false,
        )
    }

    fn tot_ne_kernel_broadcast(&self, other: &Self::Scalar) -> PlBitmap {
        list_compare_scalar(
            self,
            &**other,
            Condense::Any,
            pl_array_tot_ne_missing_kernel,
            true,
        )
    }
}

/// Compares the lists of `lhs` against `rhs`'s, element for element.
///
/// `mismatch` is the answer for a pair of lists that cannot be compared value for value at all —
/// a different length, or a different value type, which holds for every element at once.
fn list_compare_values(
    lhs: &PlListArray,
    rhs: &PlListArray,
    how: Condense,
    inner: fn(&dyn PlArray, &dyn PlArray) -> PlBitmap,
    mismatch: bool,
) -> PlBitmap {
    let length = lhs.len();

    if lhs.values().array_type() != rhs.values().array_type() {
        return repeated(mismatch, length);
    }

    // Both sides repeat one list, so comparing those two lists once answers for every element.
    if let (Some(lhs), Some(rhs)) = (lhs.scalar_value(), rhs.scalar_value()) {
        // A null element is one the missing-aware kernel answers for, not this.
        if let (Some(lhs), Some(rhs)) = (lhs, rhs) {
            return repeated(compare_lists(&*lhs, &*rhs, how, inner, mismatch), length);
        }
    }

    PlBitmap::from_iter((0..length).map(|i| {
        // A null element has no list to read; the missing-aware kernel answers for it.
        if lhs.is_null(i) || rhs.is_null(i) {
            return !mismatch;
        }
        compare_lists(&*lhs.value(i), &*rhs.value(i), how, inner, mismatch)
    }))
}

/// Compares the lists of `lhs` against the single list `rhs`.
fn list_compare_scalar(
    lhs: &PlListArray,
    rhs: &dyn PlArray,
    how: Condense,
    inner: fn(&dyn PlArray, &dyn PlArray) -> PlBitmap,
    mismatch: bool,
) -> PlBitmap {
    let length = lhs.len();

    if lhs.values().array_type() != rhs.array_type() {
        return repeated(mismatch, length);
    }

    if let Some(Some(lhs)) = lhs.scalar_value() {
        return repeated(compare_lists(&*lhs, rhs, how, inner, mismatch), length);
    }

    PlBitmap::from_iter((0..length).map(|i| {
        if lhs.is_null(i) {
            return !mismatch;
        }
        compare_lists(&*lhs.value(i), rhs, how, inner, mismatch)
    }))
}

/// Whether the two lists answer `how` over the values they hold, one against one.
fn compare_lists(
    lhs: &dyn PlArray,
    rhs: &dyn PlArray,
    how: Condense,
    inner: fn(&dyn PlArray, &dyn PlArray) -> PlBitmap,
    mismatch: bool,
) -> bool {
    // Lists of different lengths hold no pair of values to compare.
    if lhs.len() != rhs.len() {
        return mismatch;
    }
    // Two empty lists are the same list, whatever the value type under them.
    if lhs.is_empty() {
        return !mismatch;
    }
    condense(inner(lhs, rhs), 1, lhs.len(), how).get(0)
}
