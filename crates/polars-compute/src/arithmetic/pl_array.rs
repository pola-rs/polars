//! What an arithmetic kernel does with a chunk before it reads it.
//!
//! The kernels below this module read their operands as slices: one values slot and one validity
//! bit per element. A chunk that repeats a single value holds one slot for all of its elements,
//! and writing it out to reach a kernel costs a copy per element for an answer that is the same
//! value repeated again. [`unary`] and [`binary`] stand between the two, taking a chunk apart
//! into what a kernel actually has to read.
//!
//! Three things come out of that:
//!
//! * A validity mask of a single bit is read rather than written out. The bit is either set, in
//!   which case it marks nothing and is dropped, or unset, in which case every element is null
//!   and the answer is null throughout without a kernel running at all.
//! * A values buffer of a single slot reaches the kernel as a single element, and the answer for
//!   that element is repeated over the elements that read it. This is what keeps `-x` on a
//!   repeated value in `O(1)` memory.
//! * A binary kernel whose one operand repeats a value takes it as that value, through the
//!   `*_scalar` kernel written for a repeated operand — the same codepath a broadcast operand
//!   already took. So `a * [1, 2, 3]` multiplies by `a` without writing `a` out first, and
//!   strength reduction and the other tricks those kernels play apply to a repeated chunk too.

use arrow::bitmap::Bitmap;
use arrow::types::NativeType;
use polars_array::bitmap::combine_validities_and;
use polars_array::{PlBitmap, PlPrimitiveArray};

use super::{PArr, POut};

/// A chunk taken apart into the values a kernel reads and the mask that goes back around its
/// answer. Neither buffer is written out one slot per element to get here.
enum Split<T: NativeType> {
    /// Every element is null, so an elementwise kernel answers every element with a null and the
    /// length is all that is left of the chunk. Only a validity mask of a single unset bit
    /// reaches this arm: a flat mask is never scanned to find out whether it is empty.
    AllNull,
    /// The one value every element reads, and the mask over those elements, which holds one bit
    /// per element where it is there at all. A kernel reads the value once.
    Repeated(T, Option<PlBitmap>),
    /// The chunk itself, with one values slot per element. A mask of a single set bit has been
    /// dropped on the way here, a set bit marking nothing.
    Flat(PArr<T>),
}

impl<T: NativeType> Split<T> {
    fn of(mut arr: PlPrimitiveArray<T>) -> Self {
        // A mask of a single bit says the same thing about every element, so it is read here
        // instead of reaching a kernel as one bit per element.
        if let Some(bit) = arr.validity().and_then(|validity| validity.scalar_value()) {
            if !bit {
                return Self::AllNull;
            }
            arr = arr.without_validity();
        }

        match arr.scalar_values() {
            // The kernel is elementwise, so the one value the elements share is operated on once.
            // The mask, which is flat if it is still here, comes along to mask the answer again.
            Some(value) => Self::Repeated(value, arr.validity().map(PlBitmap::from)),
            // Both buffers hold one slot per element, so the array is flat and `to_flat` borrows
            // it; the clone that `into_owned` makes is a refcount bump per buffer.
            None => Self::Flat(arr.to_flat().into_owned()),
        }
    }
}

/// The one element `value` stands for, as a flat array a kernel can read.
///
/// The buffer is left unshared, so a kernel that writes its answer over its operand still can.
fn single<T: NativeType>(value: T) -> PArr<T> {
    // The array this borrows from is a temporary that is dropped before the caller sees the
    // clone, which leaves the one slot the clone holds to the caller alone.
    PlPrimitiveArray::new_scalar(value, 1)
        .to_flat()
        .into_owned()
}

/// A kernel's answer for the one value a chunk repeats, spread back over the `length` elements
/// that read it under the `validity` mask they had.
///
/// Neither buffer is written out: the answer stays a single slot, and the mask goes back as it
/// came. A kernel that nulls its one element out — dividing by zero, say — nulls out every
/// element that reads it, whatever the mask says.
fn repeat<O: NativeType>(out: POut<O>, length: usize, validity: Option<PlBitmap>) -> POut<O> {
    debug_assert_eq!(
        out.len(),
        1,
        "an elementwise kernel answers one element with one"
    );

    match out.scalar_value().flatten() {
        None => POut::new_full_null(length),
        Some(value) => POut::new_scalar(value, length).with_validity_broadcast(bitmap(validity)),
    }
}

/// `out` with `validity` folded into the mask it carries already.
///
/// This is how the mask of an operand a kernel never saw — the repeated one, which reached it as
/// a bare value — gets back around the answer.
fn fold_in<O: NativeType>(out: POut<O>, validity: Option<PlBitmap>) -> POut<O> {
    let Some(validity) = validity else {
        return out;
    };

    let combined = combine_validities_and(out.validity(), Some(validity.as_ref()));
    out.with_validity_broadcast(bitmap(combined))
}

/// The backing bitmap of a mask, which is flat or scalar for the elements it covers and so is
/// exactly what an array takes as its own mask.
fn bitmap(validity: Option<PlBitmap>) -> Option<Bitmap> {
    validity.map(|validity| validity.into_inner().0)
}

/// Applies `flat`, an elementwise kernel over the flat representation, to `arr`.
///
/// A values buffer of a single slot stays a single slot: the one value every element reads is
/// operated on once, and the answer is repeated in turn. See the [module docs](self).
pub(super) fn unary<I, O, F>(arr: PlPrimitiveArray<I>, flat: F) -> POut<O>
where
    I: NativeType,
    O: NativeType,
    F: FnOnce(PArr<I>) -> POut<O>,
{
    let length = arr.len();

    match Split::of(arr) {
        Split::AllNull => POut::new_full_null(length),
        Split::Repeated(value, validity) => repeat(flat(single(value)), length, validity),
        Split::Flat(arr) => flat(arr),
    }
}

/// Applies a binary elementwise kernel to `lhs` and `rhs`, in the shape that reads the least of
/// them: `flat` over two flat chunks, `scalar_lhs`/`scalar_rhs` where the side it is named for
/// repeats a single value. See the [module docs](self).
///
/// # Panics
/// Panics if the two chunks cover a different number of elements.
pub(super) fn binary<L, R, O, FF, FL, FR>(
    lhs: PlPrimitiveArray<L>,
    rhs: PlPrimitiveArray<R>,
    flat: FF,
    scalar_lhs: FL,
    scalar_rhs: FR,
) -> POut<O>
where
    L: NativeType,
    R: NativeType,
    O: NativeType,
    FF: FnOnce(PArr<L>, PArr<R>) -> POut<O>,
    FL: FnOnce(L, PArr<R>) -> POut<O>,
    FR: FnOnce(PArr<L>, R) -> POut<O>,
{
    let length = lhs.len();
    assert_eq!(
        length,
        rhs.len(),
        "cannot apply a binary kernel to chunks of different lengths"
    );

    match (Split::of(lhs), Split::of(rhs)) {
        // An operand that is null answers with a null, whatever the other one holds.
        (Split::AllNull, _) | (_, Split::AllNull) => POut::new_full_null(length),

        // Both sides repeat a value, so the kernel runs once over one element each and its answer
        // covers every element. The two masks are combined for it to sit under.
        (Split::Repeated(l, lv), Split::Repeated(r, rv)) => {
            let validity = combine_validities_and(
                lv.as_ref().map(PlBitmap::as_ref),
                rv.as_ref().map(PlBitmap::as_ref),
            );
            repeat(flat(single(l), single(r)), length, validity)
        },

        // One side repeats a value and reaches the kernel as that value; the mask over the
        // elements that read it goes back around the answer afterwards.
        (Split::Repeated(l, lv), Split::Flat(rhs)) => fold_in(scalar_lhs(l, rhs), lv),
        (Split::Flat(lhs), Split::Repeated(r, rv)) => fold_in(scalar_rhs(lhs, r), rv),

        (Split::Flat(lhs), Split::Flat(rhs)) => flat(lhs, rhs),
    }
}

#[cfg(test)]
mod tests {
    use arrow::bitmap::Bitmap;

    use super::*;
    use crate::arithmetic::ArithmeticKernel;

    /// `length` copies of `value` under `validity`, in both representations.
    fn repeated(
        value: i32,
        validity: Option<&Bitmap>,
        length: usize,
    ) -> [PlPrimitiveArray<i32>; 2] {
        let scalar =
            PlPrimitiveArray::new_scalar(value, length).with_validity_broadcast(validity.cloned());
        let flat = PlPrimitiveArray::from_vec(vec![value; length])
            .with_validity_broadcast(validity.cloned());

        assert_eq!(scalar, flat);
        [scalar, flat]
    }

    /// The masks a chunk of `length` elements can carry, the scalar ones included.
    fn masks(length: usize) -> Vec<Option<Bitmap>> {
        let mut masks = vec![
            None,
            Some(Bitmap::new_with_value(true, 1)),
            Some(Bitmap::new_with_value(false, 1)),
        ];
        masks.extend((0..=length).map(|valid| Some((0..length).map(|i| i < valid).collect())));
        masks
    }

    /// A kernel reads the one value every element of a chunk holds once, and the answer covers
    /// every element in turn: the values buffer of the answer holds a single slot, as the one it
    /// was read from did.
    #[test]
    fn a_repeated_value_is_operated_on_once() {
        for length in [1, 3, 65] {
            for validity in masks(length) {
                let [scalar, flat] = repeated(-7, validity.as_ref(), length);

                let negated = ArithmeticKernel::wrapping_neg(scalar.clone());
                assert_eq!(negated, ArithmeticKernel::wrapping_neg(flat.clone()));
                assert!(negated.values_are_scalar(), "{negated:?}");

                let doubled = ArithmeticKernel::wrapping_mul_scalar(scalar, 2);
                assert_eq!(doubled, ArithmeticKernel::wrapping_mul_scalar(flat, 2));
                assert!(doubled.values_are_scalar(), "{doubled:?}");
            }
        }
    }

    /// Which representation the two operands are in is not something a kernel's answer depends
    /// on: the same values give the same answer whether they are written out or repeated.
    #[test]
    fn the_representation_of_an_operand_does_not_change_the_answer() {
        let length = 5;
        let values: PlPrimitiveArray<i32> = PlPrimitiveArray::from_vec(vec![-4, -1, 0, 3, 7]);

        for validity in masks(length) {
            let values = values.clone().with_validity_broadcast(validity.clone());
            let [scalar, flat] = repeated(3, validity.as_ref(), length);

            for (name, op) in [
                (
                    "add",
                    ArithmeticKernel::wrapping_add as fn(_, _) -> PlPrimitiveArray<i32>,
                ),
                ("sub", ArithmeticKernel::wrapping_sub),
                ("mul", ArithmeticKernel::wrapping_mul),
                ("floor_div", ArithmeticKernel::wrapping_floor_div),
                ("trunc_div", ArithmeticKernel::wrapping_trunc_div),
                ("mod", ArithmeticKernel::wrapping_mod),
            ] {
                // The repeated operand on the right, then on the left, then on both sides.
                assert_eq!(
                    op(values.clone(), scalar.clone()),
                    op(values.clone(), flat.clone()),
                    "{name} with a repeated right operand",
                );
                assert_eq!(
                    op(scalar.clone(), values.clone()),
                    op(flat.clone(), values.clone()),
                    "{name} with a repeated left operand",
                );
                assert_eq!(
                    op(scalar.clone(), scalar.clone()),
                    op(flat.clone(), flat.clone()),
                    "{name} with two repeated operands",
                );
            }
        }
    }

    /// Two operands that each repeat a value answer with a repeated value in turn, in `O(1)`.
    #[test]
    fn two_repeated_operands_answer_with_a_repeated_value() {
        let [scalar, _] = repeated(9, None, 64);
        let [three, _] = repeated(3, None, 64);

        let sum = ArithmeticKernel::wrapping_add(scalar.clone(), three.clone());
        assert!(sum.values_are_scalar(), "{sum:?}");
        assert_eq!(sum, PlPrimitiveArray::new_scalar(12, 64));

        // A kernel that nulls its one element out nulls out every element that reads it.
        let zeroed = PlPrimitiveArray::new_scalar(0, 64);
        let divided = ArithmeticKernel::wrapping_floor_div(scalar, zeroed);
        assert_eq!(divided.null_count(), 64);
    }

    /// A validity mask of a single unset bit says every element is null, and the answer is null
    /// throughout without the values ever being read.
    #[test]
    fn a_mask_of_one_unset_bit_answers_with_nulls_throughout() {
        let none = Some(Bitmap::new_with_value(false, 1));
        let [scalar, flat] = repeated(7, none.as_ref(), 32);
        let values = PlPrimitiveArray::from_vec((0..32).collect::<Vec<i32>>());

        for null in [scalar, flat] {
            for out in [
                ArithmeticKernel::wrapping_neg(null.clone()),
                ArithmeticKernel::wrapping_add_scalar(null.clone(), 1),
                ArithmeticKernel::wrapping_mul(null.clone(), values.clone()),
                ArithmeticKernel::wrapping_sub(values.clone(), null.clone()),
            ] {
                assert_eq!(out.len(), 32);
                assert_eq!(out.null_count(), 32, "{out:?}");
            }
        }
    }

    /// A repeated operand that is divided by zero, or that divides zero, is nulled out the same
    /// way a written-out one is.
    #[test]
    fn dividing_by_a_repeated_zero_nulls_every_element() {
        let values = PlPrimitiveArray::from_vec(vec![-4i32, -1, 0, 3, 7]);
        let [scalar, flat] = repeated(0, None, 5);

        for zero in [scalar, flat] {
            let quotient = ArithmeticKernel::wrapping_floor_div(values.clone(), zero.clone());
            assert_eq!(quotient.null_count(), 5, "{quotient:?}");

            let remainder = ArithmeticKernel::wrapping_mod(values.clone(), zero.clone());
            assert_eq!(remainder.null_count(), 5, "{remainder:?}");

            // Zero divided by the values is zero, except where the divisor is zero itself.
            let quotient = ArithmeticKernel::wrapping_floor_div(zero, values.clone());
            assert_eq!(quotient.null_count(), 1, "{quotient:?}");
        }
    }

    /// The floating point kernels for a repeated operand reach for the reciprocal, so the values
    /// here are the ones it is exact for; what is under test is the mask and the length.
    #[test]
    fn a_repeated_float_operand_keeps_the_mask_of_the_elements_that_read_it() {
        let mask: Bitmap = [true, false, true, true].into_iter().collect();
        let values = PlPrimitiveArray::from_vec(vec![1.0f64, 2.0, 6.0, 8.0])
            .with_validity(Some(mask.clone()));
        let repeated = PlPrimitiveArray::new_scalar(2.0f64, 4).with_validity(Some(mask));

        let quotient = ArithmeticKernel::true_div(values.clone(), repeated.clone());
        assert_eq!(
            quotient,
            PlPrimitiveArray::from_vec(vec![0.5, 1.0, 3.0, 4.0])
                .with_validity(Some([true, false, true, true].into_iter().collect()))
        );

        // The mask of the repeated operand is folded into the answer even though the kernel
        // never saw the operand itself.
        let values = values.without_validity();
        let quotient = ArithmeticKernel::true_div(values, repeated);
        assert_eq!(quotient.null_count(), 1, "{quotient:?}");
    }
}
