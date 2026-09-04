//! The comparison kernels over a [`PlPrimitiveArray`] of unknown representation.
//!
//! A chunk whose values buffer holds a single slot compares that one value, not `length` copies of
//! it: two such chunks answer each other in `O(1)`, and the [`PlBitmap`] they answer with repeats
//! the single bit that settles every element rather than holding `length` of them. One repeated
//! chunk against a flat one becomes the broadcast kernel the flat side already has. Only once both
//! sides are known to lay one slot out per element does the work cross over to the kernel over
//! [`Flat`], which is the SIMD one where the element type has it and the element-at-a-time one
//! otherwise — see `simd` and `scalar`.
//!
//! The value kernels read no validity at all, which the traits state outright, so it is the
//! *values* representation alone that these dispatch on: a mask that repeats one bit never forces
//! the values it covers to be written out. What the mask says is folded in afterwards by the
//! missing-aware kernels of [`super`], which read it in whichever representation it is in and keep
//! the answer repeated wherever it stays the same for every element.

use arrow::bitmap::Bitmap;
use arrow::types::NativeType;
use polars_array::{ArrayRepr, Flat, PlBitmap, PlBitmapRef, PlPrimitiveArray};
use polars_buffer::Buffer;
use polars_utils::total_ord::{TotalEq, TotalOrd};

use super::{PlTotalEqKernel, PlTotalOrdKernel, TotalEqKernel, TotalOrdKernel};

/// The values of a chunk as a flat array of their own, which is what the kernels over [`Flat`]
/// take: they read the values and nothing else, so dropping the mask along the way costs nothing.
///
/// This is `O(1)` — the buffer is shared, not copied — and it is what lets a chunk with flat values
/// but a mask repeating a single bit reach the flat kernel without being written out.
#[inline]
fn flat_values<T: NativeType>(values: &Buffer<T>) -> Flat<PlPrimitiveArray<T>> {
    // An array built from a values buffer alone has one slot per element and no mask, so it is
    // already flat and `to_flat` hands the borrow straight back.
    PlPrimitiveArray::from_values(values.clone())
        .to_flat()
        .into_owned()
}

/// A mask of `length` copies of `value`, held in the single bit that says it.
#[inline]
fn repeated(value: bool, length: usize) -> PlBitmap {
    PlBitmap::new_scalar(value, length)
}

/// A mask of one bit per element, which is what a flat operand leaves.
#[inline]
fn written_out(bits: Bitmap) -> PlBitmap {
    PlBitmap::from_bitmap(bits)
}

/// Dispatches a binary kernel on the values representation of both operands.
///
/// `$flat_lhs` is the kernel to reach for when only the right side repeats, and `$flat_rhs` the one
/// for when only the left side does — which for an ordering is the comparison turned around, since
/// `l < r[i]` is `r[i] > l`.
macro_rules! binary_kernel {
    ($self:expr, $other:expr, $scalar:expr, $flat:path, $flat_lhs:path, $flat_rhs:path $(,)?) => {{
        let (lhs, rhs) = ($self, $other);
        assert!(lhs.len() == rhs.len());

        match (lhs.values_repr(), rhs.values_repr()) {
            // Every element of both sides holds the one value its own side repeats, so the one
            // comparison of those two values is the answer for all of them.
            (ArrayRepr::Scalar(l), ArrayRepr::Scalar(r)) => repeated($scalar(&l, &r), lhs.len()),
            (ArrayRepr::Scalar(l), ArrayRepr::Flat(r)) => {
                written_out($flat_rhs(&flat_values(r), &l))
            },
            (ArrayRepr::Flat(l), ArrayRepr::Scalar(r)) => {
                written_out($flat_lhs(&flat_values(l), &r))
            },
            (ArrayRepr::Flat(l), ArrayRepr::Flat(r)) => {
                written_out($flat(&flat_values(l), &flat_values(r)))
            },
        }
    }};
}

/// Dispatches a broadcast kernel on the values representation of its one operand.
macro_rules! broadcast_kernel {
    ($self:expr, $other:expr, $scalar:expr, $flat:path $(,)?) => {{
        let (lhs, rhs) = ($self, $other);

        match lhs.values_repr() {
            ArrayRepr::Scalar(l) => repeated($scalar(&l, rhs), lhs.len()),
            ArrayRepr::Flat(l) => written_out($flat(&flat_values(l), rhs)),
        }
    }};
}

impl<T> PlTotalEqKernel for PlPrimitiveArray<T>
where
    T: NativeType + TotalEq,
    Flat<PlPrimitiveArray<T>>: TotalEqKernel<Scalar = T>,
{
    type Scalar = T;

    fn validity_mask(&self) -> Option<PlBitmapRef<'_>> {
        // Whatever representation the mask is in: the missing-aware kernels resolve it themselves.
        self.validity()
    }

    fn tot_eq_kernel(&self, other: &Self) -> PlBitmap {
        // Equality is symmetric, so which side repeats its value makes no difference.
        binary_kernel!(
            self,
            other,
            TotalEq::tot_eq,
            TotalEqKernel::tot_eq_kernel,
            TotalEqKernel::tot_eq_kernel_broadcast,
            TotalEqKernel::tot_eq_kernel_broadcast,
        )
    }

    fn tot_ne_kernel(&self, other: &Self) -> PlBitmap {
        binary_kernel!(
            self,
            other,
            TotalEq::tot_ne,
            TotalEqKernel::tot_ne_kernel,
            TotalEqKernel::tot_ne_kernel_broadcast,
            TotalEqKernel::tot_ne_kernel_broadcast,
        )
    }

    fn tot_eq_kernel_broadcast(&self, other: &T) -> PlBitmap {
        broadcast_kernel!(
            self,
            other,
            TotalEq::tot_eq,
            TotalEqKernel::tot_eq_kernel_broadcast,
        )
    }

    fn tot_ne_kernel_broadcast(&self, other: &T) -> PlBitmap {
        broadcast_kernel!(
            self,
            other,
            TotalEq::tot_ne,
            TotalEqKernel::tot_ne_kernel_broadcast,
        )
    }
}

impl<T> PlTotalOrdKernel for PlPrimitiveArray<T>
where
    T: NativeType + TotalOrd,
    Flat<PlPrimitiveArray<T>>: TotalOrdKernel<Scalar = T>,
{
    type Scalar = T;

    fn tot_lt_kernel(&self, other: &Self) -> PlBitmap {
        binary_kernel!(
            self,
            other,
            TotalOrd::tot_lt,
            TotalOrdKernel::tot_lt_kernel,
            TotalOrdKernel::tot_lt_kernel_broadcast,
            // A repeated left operand turns the comparison around: `l < r[i]` is `r[i] > l`.
            TotalOrdKernel::tot_gt_kernel_broadcast,
        )
    }

    fn tot_le_kernel(&self, other: &Self) -> PlBitmap {
        binary_kernel!(
            self,
            other,
            TotalOrd::tot_le,
            TotalOrdKernel::tot_le_kernel,
            TotalOrdKernel::tot_le_kernel_broadcast,
            TotalOrdKernel::tot_ge_kernel_broadcast,
        )
    }

    fn tot_lt_kernel_broadcast(&self, other: &T) -> PlBitmap {
        broadcast_kernel!(
            self,
            other,
            TotalOrd::tot_lt,
            TotalOrdKernel::tot_lt_kernel_broadcast,
        )
    }

    fn tot_le_kernel_broadcast(&self, other: &T) -> PlBitmap {
        broadcast_kernel!(
            self,
            other,
            TotalOrd::tot_le,
            TotalOrdKernel::tot_le_kernel_broadcast,
        )
    }

    fn tot_gt_kernel_broadcast(&self, other: &T) -> PlBitmap {
        broadcast_kernel!(
            self,
            other,
            TotalOrd::tot_gt,
            TotalOrdKernel::tot_gt_kernel_broadcast,
        )
    }

    fn tot_ge_kernel_broadcast(&self, other: &T) -> PlBitmap {
        broadcast_kernel!(
            self,
            other,
            TotalOrd::tot_ge,
            TotalOrdKernel::tot_ge_kernel_broadcast,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The kernels that ignore validity, whose answers for a null element the traits leave
    /// unspecified: two representations only have to agree on these where nothing is null.
    fn value_kernels<T>(lhs: &PlPrimitiveArray<T>, rhs: &PlPrimitiveArray<T>) -> Vec<Bitmap>
    where
        T: NativeType + TotalOrd,
        PlPrimitiveArray<T>: PlTotalEqKernel<Scalar = T> + PlTotalOrdKernel<Scalar = T>,
    {
        // Written out, so that two answers are compared on what they say rather than on how they
        // say it — which representation each is in is what the tests below assert separately.
        [
            PlTotalEqKernel::tot_eq_kernel(lhs, rhs),
            PlTotalEqKernel::tot_ne_kernel(lhs, rhs),
            PlTotalOrdKernel::tot_lt_kernel(lhs, rhs),
            PlTotalOrdKernel::tot_le_kernel(lhs, rhs),
            PlTotalOrdKernel::tot_gt_kernel(lhs, rhs),
            PlTotalOrdKernel::tot_ge_kernel(lhs, rhs),
        ]
        .into_iter()
        .map(PlBitmap::into_bitmap)
        .collect()
    }

    /// The kernels that read validity, and so are determined for every element.
    fn missing_kernels<T>(
        lhs: &PlPrimitiveArray<T>,
        rhs: &PlPrimitiveArray<T>,
        scalar: T,
    ) -> Vec<Bitmap>
    where
        T: NativeType + TotalOrd,
        PlPrimitiveArray<T>: PlTotalEqKernel<Scalar = T>,
    {
        [
            lhs.tot_eq_missing_kernel(rhs),
            lhs.tot_ne_missing_kernel(rhs),
            lhs.tot_eq_missing_kernel_broadcast(&scalar),
            lhs.tot_ne_missing_kernel_broadcast(&scalar),
        ]
        .into_iter()
        .map(PlBitmap::into_bitmap)
        .collect()
    }

    /// The same array flattened: every backing buffer written out one slot per element, which is
    /// the form whose answers the representation-aware kernels must reproduce.
    fn flattened<T: NativeType>(array: &PlPrimitiveArray<T>) -> PlPrimitiveArray<T> {
        let flat = array.to_flat().into_owned().into_array();
        assert!(flat.is_flat());
        flat
    }

    /// One logical array in every representation it has: the value repeated or written out,
    /// crossed with a mask that is absent, repeats one bit, or holds one bit per element.
    fn representations(value: i32, valid: bool, length: usize) -> Vec<PlPrimitiveArray<i32>> {
        let mut out = Vec::new();

        for values in [
            PlPrimitiveArray::new_scalar(value, length),
            PlPrimitiveArray::from_vec(vec![value; length]),
        ] {
            if valid {
                out.push(values.clone());
            }
            out.push(
                values
                    .clone()
                    .with_validity(Some(PlBitmap::new_scalar(valid, length))),
            );
            out.push(
                values.with_validity(Some(PlBitmap::from_bitmap(Bitmap::new_with_value(
                    valid, length,
                )))),
            );
        }

        out
    }

    #[test]
    fn every_representation_of_a_repeated_value_answers_as_the_written_out_one() {
        for (lhs_value, lhs_valid) in [(1, true), (2, true), (1, false)] {
            for (rhs_value, rhs_valid) in [(1, true), (2, true), (1, false)] {
                let lhs = representations(lhs_value, lhs_valid, 9);
                let rhs = representations(rhs_value, rhs_valid, 9);

                // Both sides written out is the baseline every other pairing must agree with.
                let (base_lhs, base_rhs) = (flattened(&lhs[0]), flattened(&rhs[0]));
                let expected_missing = missing_kernels(&base_lhs, &base_rhs, 1);
                let expected_values = value_kernels(&base_lhs, &base_rhs);

                for l in &lhs {
                    for r in &rhs {
                        assert_eq!(
                            missing_kernels(l, r, 1),
                            expected_missing,
                            "lhs {l:?} against rhs {r:?} disagrees with the flat form",
                        );

                        // Where an element is null the value kernels answer anything at all —
                        // flattening an all-null chunk does not even write the repeated value out,
                        // since nothing reads it — so only pin them down where nothing is null.
                        if lhs_valid && rhs_valid {
                            assert_eq!(
                                value_kernels(l, r),
                                expected_values,
                                "lhs {l:?} against rhs {r:?} disagrees with the flat form",
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn a_repeated_operand_against_a_written_out_one() {
        let mixed: PlPrimitiveArray<i32> = [Some(1), None, Some(3), Some(2)].into_iter().collect();

        for repeated in representations(2, true, 4) {
            let flat = flattened(&repeated);

            assert_eq!(
                value_kernels(&repeated, &mixed),
                value_kernels(&flat, &mixed)
            );
            assert_eq!(
                value_kernels(&mixed, &repeated),
                value_kernels(&mixed, &flat)
            );
            assert_eq!(
                missing_kernels(&repeated, &mixed, 2),
                missing_kernels(&flat, &mixed, 2),
            );
            assert_eq!(
                missing_kernels(&mixed, &repeated, 2),
                missing_kernels(&mixed, &flat, 2),
            );
        }

        // The ordering kernels are the ones a repeated left operand has to turn around, so pin
        // down what they answer rather than only that the two forms agree.
        let repeated = PlPrimitiveArray::new_scalar(2i32, 4);
        let values = PlPrimitiveArray::from_vec(vec![1i32, 2, 3, 4]);

        assert_eq!(
            PlTotalOrdKernel::tot_lt_kernel(&repeated, &values)
                .iter()
                .collect::<Vec<_>>(),
            [false, false, true, true],
        );
        assert_eq!(
            PlTotalOrdKernel::tot_ge_kernel(&repeated, &values)
                .iter()
                .collect::<Vec<_>>(),
            [true, true, false, false],
        );
        assert_eq!(
            repeated
                .tot_ge_kernel_broadcast(&3)
                .iter()
                .collect::<Vec<_>>(),
            [false; 4],
        );
    }

    /// The point of the whole dispatch: an answer that is the same for every element is handed
    /// back as the one bit that says so, however long the arrays are.
    #[test]
    fn an_answer_that_is_the_same_for_every_element_is_not_written_out() {
        let length = 1_000_000_000;
        let ones = PlPrimitiveArray::new_scalar(1i32, length);
        let twos = PlPrimitiveArray::new_scalar(2i32, length);
        let all_null = PlPrimitiveArray::<i32>::new_full_null(length);

        let repeated = |mask: PlBitmap, value: bool| {
            assert!(mask.is_scalar(), "the answer was written out");
            assert_eq!(mask.len(), length);
            assert_eq!(mask.scalar_value(), Some(value));
        };

        // Two repeated values compare once, and so do their masks.
        repeated(PlTotalEqKernel::tot_eq_kernel(&ones, &twos), false);
        repeated(PlTotalEqKernel::tot_ne_kernel(&ones, &twos), true);
        repeated(PlTotalOrdKernel::tot_lt_kernel(&ones, &twos), true);
        repeated(PlTotalOrdKernel::tot_gt_kernel(&ones, &twos), false);
        repeated(ones.tot_eq_kernel_broadcast(&1), true);
        repeated(ones.tot_le_kernel_broadcast(&0), false);

        // Including through the missing-aware kernels, whose masks repeat a bit as well.
        repeated(ones.tot_eq_missing_kernel(&twos), false);
        repeated(ones.tot_ne_missing_kernel(&twos), true);
        repeated(all_null.tot_eq_missing_kernel(&all_null), true);
        repeated(all_null.tot_ne_missing_kernel(&all_null), false);
        repeated(all_null.tot_eq_missing_kernel(&ones), false);
        repeated(ones.tot_ne_missing_kernel(&all_null), true);
        repeated(all_null.tot_eq_missing_kernel_broadcast(&7), false);
        repeated(all_null.tot_ne_missing_kernel_broadcast(&7), true);

        // A repeated answer against a written-out mask is the mask, which stays flat.
        let flat_mask = PlPrimitiveArray::from_vec(vec![1i32, 1, 1]).with_validity(Some(
            PlBitmap::from_bitmap(Bitmap::from_iter([true, false, true])),
        ));
        let ones_3 = PlPrimitiveArray::new_scalar(1i32, 3);

        let eq = ones_3.tot_eq_missing_kernel(&flat_mask);
        assert!(eq.is_flat());
        assert_eq!(eq.iter().collect::<Vec<_>>(), [true, false, true]);

        let ne = ones_3.tot_ne_missing_kernel(&flat_mask);
        assert_eq!(ne.iter().collect::<Vec<_>>(), [false, true, false]);
    }

    #[test]
    fn a_mask_repeating_one_bit_is_read_without_being_written_out() {
        // A side that is null throughout answers from the other side's mask alone.
        let flat_mask = PlPrimitiveArray::from_vec(vec![1i32, 2, 3]).with_validity(Some(
            PlBitmap::from_bitmap(Bitmap::from_iter([true, false, true])),
        ));
        let null_3 = PlPrimitiveArray::<i32>::new_full_null(3);

        assert!(null_3.validity_is_scalar());
        assert_eq!(
            null_3
                .tot_eq_missing_kernel(&flat_mask)
                .iter()
                .collect::<Vec<_>>(),
            [false, true, false],
        );
        assert_eq!(
            null_3
                .tot_ne_missing_kernel(&flat_mask)
                .iter()
                .collect::<Vec<_>>(),
            [true, false, true],
        );
        assert_eq!(
            flat_mask
                .tot_eq_missing_kernel(&null_3)
                .iter()
                .collect::<Vec<_>>(),
            [false, true, false],
        );

        // A repeated mask over flat values does not force those values to be written out either.
        let values = PlPrimitiveArray::from_vec(vec![1i32, 2, 3])
            .with_validity(Some(PlBitmap::new_scalar(true, 3)));
        assert!(values.values_are_flat() && values.validity_is_scalar());
        assert_eq!(
            values
                .tot_eq_missing_kernel_broadcast(&2)
                .iter()
                .collect::<Vec<_>>(),
            [false, true, false],
        );
    }

    #[test]
    fn floats_are_compared_totally_in_either_representation() {
        let nan = f64::NAN;
        let flat = PlPrimitiveArray::from_vec(vec![nan, 0.0, -0.0, 1.0]);
        let repeated = PlPrimitiveArray::new_scalar(nan, 4);

        // A NaN is equal to itself and above everything else, which is what makes the order total.
        assert_eq!(
            PlTotalEqKernel::tot_eq_kernel(&repeated, &flat)
                .iter()
                .collect::<Vec<_>>(),
            [true, false, false, false],
        );
        assert_eq!(
            PlTotalOrdKernel::tot_lt_kernel(&repeated, &flat)
                .iter()
                .collect::<Vec<_>>(),
            [false, false, false, false],
        );
        assert_eq!(
            PlTotalOrdKernel::tot_lt_kernel(&flat, &repeated)
                .iter()
                .collect::<Vec<_>>(),
            [false, true, true, true],
        );

        // The two zeroes are one number, whichever side repeats.
        let zero = PlPrimitiveArray::new_scalar(-0.0f64, 4);
        assert_eq!(zero.tot_eq_kernel_broadcast(&0.0).unset_bits(), 0);
        assert_eq!(
            PlTotalEqKernel::tot_eq_kernel(&zero, &flat)
                .iter()
                .collect::<Vec<_>>(),
            [false, true, true, false],
        );

        // Every representation of the repeated NaN agrees with the written-out one.
        assert_eq!(
            value_kernels(&repeated, &flat),
            value_kernels(&flattened(&repeated), &flat),
        );
    }
}
