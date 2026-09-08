mod scalar;

#[cfg(feature = "dtype-categorical")]
mod categorical;

use std::ops::{BitAnd, BitOr, Not};

use arrow::bitmap::BitmapBuilder;
use num_traits::{NumCast, ToPrimitive};
use polars_array::bitmap::invert;
use polars_compute::comparisons::{
    PlTotalEqKernel, PlTotalOrdKernel, TotalEqKernel, TotalOrdKernel,
};

use crate::chunked_array::ops::bits::{false_count, true_count};
use crate::prelude::*;
use crate::series::IsSorted;
use crate::series::implementations::null::NullChunked;

/// The body of a comparison between two chunked arrays, taking the shortcut a side that repeats a
/// single value allows.
///
/// A side repeats a single value when it is a column of one element, or one whose only chunk is
/// scalar; it is then compared against that value rather than written out. `broadcast` gives that
/// comparison for either side — turned around on the right, where the comparison is not symmetric
/// — `null` names how a repeated null is answered, and `flat` compares element by element.
///
/// `bounded` names, for either side, a repeated value the comparison alone settles: the bounds of
/// an order that has them.
macro_rules! broadcast_cmp {
    (@null full_null, $ca:ident) => {
        BooleanChunked::full_null(PlSmallStr::EMPTY, $ca.len())
    };
    (@null is_null, $ca:ident) => { $ca.is_null() };
    (@null is_not_null, $ca:ident) => { $ca.is_not_null() };
    (
        $lhs:ident, $rhs:ident,
        $(bounded: [$l_bound:pat => $l_settled:expr, $r_bound:pat => $r_settled:expr $(,)?],)?
        broadcast: |$value:ident| [$l_against:expr, $r_against:expr $(,)?],
        null: $null:ident,
        flat: $flat:expr $(,)?
    ) => {{
        let length = arity::broadcast_height($lhs.len(), $rhs.len())
            .expect("cannot compare arrays of different lengths");
        match ($lhs.scalar_value(), $rhs.scalar_value()) {
            $((_, Some($l_bound)) if $lhs.len() == length => $l_settled,)?
            (_, Some(value)) if $lhs.len() == length => match value {
                Some($value) => $l_against,
                None => broadcast_cmp!(@null $null, $lhs),
            },
            $((Some($r_bound), _) => $r_settled,)?
            (Some(value), _) => match value {
                Some($value) => $r_against,
                None => broadcast_cmp!(@null $null, $rhs),
            },
            _ => $flat,
        }
    }};
}

impl<T> ChunkCompareEq<&ChunkedArray<T>> for ChunkedArray<T>
where
    T: PolarsNumericType,
    Flat<T::Array>: TotalOrdKernel<Scalar = T::Native> + TotalEqKernel<Scalar = T::Native>,
{
    type Item = BooleanChunked;

    fn equal(&self, rhs: &ChunkedArray<T>) -> BooleanChunked {
        broadcast_cmp!(
            self, rhs,
            broadcast: |value| [self.equal(value), rhs.equal(value)],
            null: full_null,
            flat: arity::binary_elementwise_mut_values_flat(
                self,
                rhs,
                |a, b| a.tot_eq_kernel(b).into(),
                PlSmallStr::EMPTY,
            ),
        )
    }

    fn equal_missing(&self, rhs: &ChunkedArray<T>) -> BooleanChunked {
        broadcast_cmp!(
            self, rhs,
            broadcast: |value| [self.equal_missing(value), rhs.equal_missing(value)],
            null: is_null,
            flat: arity::binary_elementwise_kernel_flat(
                self,
                rhs,
                |a, b| a.tot_eq_missing_kernel(b).into(),
                PlSmallStr::EMPTY,
            ),
        )
    }

    fn not_equal(&self, rhs: &ChunkedArray<T>) -> BooleanChunked {
        broadcast_cmp!(
            self, rhs,
            broadcast: |value| [self.not_equal(value), rhs.not_equal(value)],
            null: full_null,
            flat: arity::binary_elementwise_mut_values_flat(
                self,
                rhs,
                |a, b| a.tot_ne_kernel(b).into(),
                PlSmallStr::EMPTY,
            ),
        )
    }

    fn not_equal_missing(&self, rhs: &ChunkedArray<T>) -> BooleanChunked {
        broadcast_cmp!(
            self, rhs,
            broadcast: |value| [self.not_equal_missing(value), rhs.not_equal_missing(value)],
            null: is_not_null,
            flat: arity::binary_elementwise_kernel_flat(
                self,
                rhs,
                |a, b| a.tot_ne_missing_kernel(b).into(),
                PlSmallStr::EMPTY,
            ),
        )
    }
}

impl<T> ChunkCompareIneq<&ChunkedArray<T>> for ChunkedArray<T>
where
    T: PolarsNumericType,
    Flat<T::Array>: TotalOrdKernel<Scalar = T::Native> + TotalEqKernel<Scalar = T::Native>,
{
    type Item = BooleanChunked;

    fn lt(&self, rhs: &ChunkedArray<T>) -> BooleanChunked {
        broadcast_cmp!(
            self, rhs,
            broadcast: |value| [self.lt(value), rhs.gt(value)],
            null: full_null,
            flat: arity::binary_elementwise_mut_values_flat(
                self,
                rhs,
                |a, b| a.tot_lt_kernel(b).into(),
                PlSmallStr::EMPTY,
            ),
        )
    }

    fn lt_eq(&self, rhs: &ChunkedArray<T>) -> BooleanChunked {
        broadcast_cmp!(
            self, rhs,
            broadcast: |value| [self.lt_eq(value), rhs.gt_eq(value)],
            null: full_null,
            flat: arity::binary_elementwise_mut_values_flat(
                self,
                rhs,
                |a, b| a.tot_le_kernel(b).into(),
                PlSmallStr::EMPTY,
            ),
        )
    }

    fn gt(&self, rhs: &Self) -> BooleanChunked {
        rhs.lt(self)
    }

    fn gt_eq(&self, rhs: &Self) -> BooleanChunked {
        rhs.lt_eq(self)
    }
}

impl ChunkCompareEq<&NullChunked> for NullChunked {
    type Item = BooleanChunked;

    fn equal(&self, rhs: &NullChunked) -> Self::Item {
        BooleanChunked::full_null(self.name().clone(), get_broadcast_length(self, rhs))
    }

    fn equal_missing(&self, rhs: &NullChunked) -> Self::Item {
        BooleanChunked::full(self.name().clone(), true, get_broadcast_length(self, rhs))
    }

    fn not_equal(&self, rhs: &NullChunked) -> Self::Item {
        BooleanChunked::full_null(self.name().clone(), get_broadcast_length(self, rhs))
    }

    fn not_equal_missing(&self, rhs: &NullChunked) -> Self::Item {
        BooleanChunked::full(self.name().clone(), false, get_broadcast_length(self, rhs))
    }
}

impl ChunkCompareIneq<&NullChunked> for NullChunked {
    type Item = BooleanChunked;

    fn gt(&self, rhs: &NullChunked) -> Self::Item {
        BooleanChunked::full_null(self.name().clone(), get_broadcast_length(self, rhs))
    }

    fn gt_eq(&self, rhs: &NullChunked) -> Self::Item {
        BooleanChunked::full_null(self.name().clone(), get_broadcast_length(self, rhs))
    }

    fn lt(&self, rhs: &NullChunked) -> Self::Item {
        BooleanChunked::full_null(self.name().clone(), get_broadcast_length(self, rhs))
    }

    fn lt_eq(&self, rhs: &NullChunked) -> Self::Item {
        BooleanChunked::full_null(self.name().clone(), get_broadcast_length(self, rhs))
    }
}

#[inline]
fn get_broadcast_length(lhs: &NullChunked, rhs: &NullChunked) -> usize {
    match (lhs.len(), rhs.len()) {
        (1, len_r) => len_r,
        (len_l, 1) => len_l,
        (len_l, len_r) if len_l == len_r => len_l,
        _ => panic!("Cannot compare two series of different lengths."),
    }
}

/// The bits `op` leaves for every chunk of `ca`, under the validity mask `ca` came with.
fn broadcast_values<F>(ca: &BooleanChunked, op: F) -> BooleanChunked
where
    F: Fn(&PlBooleanArray) -> PlBitmap,
{
    arity::unary_mut_values(ca, |arr| PlBooleanArray::from_pl_bitmap(op(arr)))
}

/// As [`broadcast_values`], for a kernel that answers over the nulls too and so leaves no mask.
fn broadcast_missing<F>(ca: &BooleanChunked, op: F) -> BooleanChunked
where
    F: Fn(&PlBooleanArray) -> PlBitmap,
{
    arity::unary_mut_with_options(ca, |arr| PlBooleanArray::from_pl_bitmap(op(arr)))
}

/// A boolean chunked array answering `value` for every element of `ca`, under its validity.
fn repeated_answer<T: PolarsDataType>(ca: &ChunkedArray<T>, value: bool) -> BooleanChunked {
    let chunks = ca.downcast_iter().map(|arr| {
        PlBooleanArray::new_scalar(value, arr.len())
            .with_validity_typed(arr.validity().map(PlBitmap::from))
    });
    BooleanChunked::from_chunk_iter(PlSmallStr::EMPTY, chunks)
}

impl ChunkCompareEq<&BooleanChunked> for BooleanChunked {
    type Item = BooleanChunked;

    fn equal(&self, rhs: &BooleanChunked) -> BooleanChunked {
        broadcast_cmp!(
            self, rhs,
            broadcast: |value| [
                broadcast_values(self, |arr| arr.tot_eq_kernel_broadcast(&value)),
                broadcast_values(rhs, |arr| arr.tot_eq_kernel_broadcast(&value)),
            ],
            null: full_null,
            flat: arity::binary_mut_values(
                self,
                rhs,
                |a, b| PlBooleanArray::from_pl_bitmap(a.tot_eq_kernel(b)),
                PlSmallStr::EMPTY,
            ),
        )
    }

    fn equal_missing(&self, rhs: &BooleanChunked) -> BooleanChunked {
        broadcast_cmp!(
            self, rhs,
            broadcast: |value| [
                broadcast_missing(self, |arr| arr.tot_eq_missing_kernel_broadcast(&value)),
                broadcast_missing(rhs, |arr| arr.tot_eq_missing_kernel_broadcast(&value)),
            ],
            null: is_null,
            flat: arity::binary_elementwise_kernel(
                self,
                rhs,
                |a, b| PlBooleanArray::from_pl_bitmap(a.tot_eq_missing_kernel(b)),
                PlSmallStr::EMPTY,
            ),
        )
    }

    fn not_equal(&self, rhs: &BooleanChunked) -> BooleanChunked {
        broadcast_cmp!(
            self, rhs,
            broadcast: |value| [
                broadcast_values(self, |arr| arr.tot_ne_kernel_broadcast(&value)),
                broadcast_values(rhs, |arr| arr.tot_ne_kernel_broadcast(&value)),
            ],
            null: full_null,
            flat: arity::binary_mut_values(
                self,
                rhs,
                |a, b| PlBooleanArray::from_pl_bitmap(a.tot_ne_kernel(b)),
                PlSmallStr::EMPTY,
            ),
        )
    }

    fn not_equal_missing(&self, rhs: &BooleanChunked) -> BooleanChunked {
        broadcast_cmp!(
            self, rhs,
            broadcast: |value| [
                broadcast_missing(self, |arr| arr.tot_ne_missing_kernel_broadcast(&value)),
                broadcast_missing(rhs, |arr| arr.tot_ne_missing_kernel_broadcast(&value)),
            ],
            null: is_not_null,
            flat: arity::binary_elementwise_kernel(
                self,
                rhs,
                |a, b| PlBooleanArray::from_pl_bitmap(a.tot_ne_missing_kernel(b)),
                PlSmallStr::EMPTY,
            ),
        )
    }
}

impl ChunkCompareIneq<&BooleanChunked> for BooleanChunked {
    type Item = BooleanChunked;

    fn lt(&self, rhs: &BooleanChunked) -> BooleanChunked {
        broadcast_cmp!(
            self, rhs,
            // No boolean is smaller than `false`, and none is greater than `true`: the value
            // alone settles the comparison, and the answer is the one bit that says so.
            bounded: [
                Some(false) => repeated_answer(self, false),
                Some(true) => repeated_answer(rhs, false),
            ],
            broadcast: |value| [
                broadcast_values(self, |arr| arr.tot_lt_kernel_broadcast(&value)),
                broadcast_values(rhs, |arr| arr.tot_gt_kernel_broadcast(&value)),
            ],
            null: full_null,
            flat: arity::binary_mut_values(
                self,
                rhs,
                |a, b| PlBooleanArray::from_pl_bitmap(a.tot_lt_kernel(b)),
                PlSmallStr::EMPTY,
            ),
        )
    }

    fn lt_eq(&self, rhs: &BooleanChunked) -> BooleanChunked {
        broadcast_cmp!(
            self, rhs,
            // Every boolean is at most `true` and at least `false`: the value alone settles the
            // comparison, and the answer is the one bit that says so.
            bounded: [
                Some(true) => repeated_answer(self, true),
                Some(false) => repeated_answer(rhs, true),
            ],
            broadcast: |value| [
                broadcast_values(self, |arr| arr.tot_le_kernel_broadcast(&value)),
                broadcast_values(rhs, |arr| arr.tot_ge_kernel_broadcast(&value)),
            ],
            null: full_null,
            flat: arity::binary_mut_values(
                self,
                rhs,
                |a, b| PlBooleanArray::from_pl_bitmap(a.tot_le_kernel(b)),
                PlSmallStr::EMPTY,
            ),
        )
    }

    fn gt(&self, rhs: &Self) -> BooleanChunked {
        rhs.lt(self)
    }

    fn gt_eq(&self, rhs: &Self) -> BooleanChunked {
        rhs.lt_eq(self)
    }
}

impl ChunkCompareEq<&StringChunked> for StringChunked {
    type Item = BooleanChunked;

    fn equal(&self, rhs: &StringChunked) -> BooleanChunked {
        self.as_binary().equal(&rhs.as_binary())
    }

    fn equal_missing(&self, rhs: &StringChunked) -> BooleanChunked {
        self.as_binary().equal_missing(&rhs.as_binary())
    }

    fn not_equal(&self, rhs: &StringChunked) -> BooleanChunked {
        self.as_binary().not_equal(&rhs.as_binary())
    }

    fn not_equal_missing(&self, rhs: &StringChunked) -> BooleanChunked {
        self.as_binary().not_equal_missing(&rhs.as_binary())
    }
}

impl ChunkCompareIneq<&StringChunked> for StringChunked {
    type Item = BooleanChunked;

    fn gt(&self, rhs: &StringChunked) -> BooleanChunked {
        self.as_binary().gt(&rhs.as_binary())
    }

    fn gt_eq(&self, rhs: &StringChunked) -> BooleanChunked {
        self.as_binary().gt_eq(&rhs.as_binary())
    }

    fn lt(&self, rhs: &StringChunked) -> BooleanChunked {
        self.as_binary().lt(&rhs.as_binary())
    }

    fn lt_eq(&self, rhs: &StringChunked) -> BooleanChunked {
        self.as_binary().lt_eq(&rhs.as_binary())
    }
}

macro_rules! binary_eq_ineq_impl {
    ($($ca:ident),+) => {
        $(
        impl ChunkCompareEq<&$ca> for $ca {
            type Item = BooleanChunked;

            fn equal(&self, rhs: &$ca) -> BooleanChunked {
                broadcast_cmp!(
                    self, rhs,
                    broadcast: |value| [self.equal(value), rhs.equal(value)],
                    null: full_null,
                    flat: arity::binary_elementwise_mut_values_flat(
                        self,
                        rhs,
                        |a, b| a.tot_eq_kernel(b).into(),
                        PlSmallStr::EMPTY,
                    ),
                )
            }

            fn equal_missing(&self, rhs: &$ca) -> BooleanChunked {
                broadcast_cmp!(
                    self, rhs,
                    broadcast: |value| [self.equal_missing(value), rhs.equal_missing(value)],
                    null: is_null,
                    flat: arity::binary_elementwise_kernel_flat(
                        self,
                        rhs,
                        |a, b| a.tot_eq_missing_kernel(b).into(),
                        PlSmallStr::EMPTY,
                    ),
                )
            }

            fn not_equal(&self, rhs: &$ca) -> BooleanChunked {
                broadcast_cmp!(
                    self, rhs,
                    broadcast: |value| [self.not_equal(value), rhs.not_equal(value)],
                    null: full_null,
                    flat: arity::binary_elementwise_mut_values_flat(
                        self,
                        rhs,
                        |a, b| a.tot_ne_kernel(b).into(),
                        PlSmallStr::EMPTY,
                    ),
                )
            }

            fn not_equal_missing(&self, rhs: &$ca) -> BooleanChunked {
                broadcast_cmp!(
                    self, rhs,
                    broadcast: |value| [
                        self.not_equal_missing(value),
                        rhs.not_equal_missing(value),
                    ],
                    null: is_not_null,
                    flat: arity::binary_elementwise_kernel_flat(
                        self,
                        rhs,
                        |a, b| a.tot_ne_missing_kernel(b).into(),
                        PlSmallStr::EMPTY,
                    ),
                )
            }
        }

        impl ChunkCompareIneq<&$ca> for $ca {
            type Item = BooleanChunked;

            fn lt(&self, rhs: &$ca) -> BooleanChunked {
                broadcast_cmp!(
                    self, rhs,
                    broadcast: |value| [self.lt(value), rhs.gt(value)],
                    null: full_null,
                    flat: arity::binary_elementwise_mut_values_flat(
                        self,
                        rhs,
                        |a, b| a.tot_lt_kernel(b).into(),
                        PlSmallStr::EMPTY,
                    ),
                )
            }

            fn lt_eq(&self, rhs: &$ca) -> BooleanChunked {
                broadcast_cmp!(
                    self, rhs,
                    broadcast: |value| [self.lt_eq(value), rhs.gt_eq(value)],
                    null: full_null,
                    flat: arity::binary_elementwise_mut_values_flat(
                        self,
                        rhs,
                        |a, b| a.tot_le_kernel(b).into(),
                        PlSmallStr::EMPTY,
                    ),
                )
            }

            fn gt(&self, rhs: &Self) -> BooleanChunked {
                rhs.lt(self)
            }

            fn gt_eq(&self, rhs: &Self) -> BooleanChunked {
                rhs.lt_eq(self)
            }
        }
        )+
    };
}

binary_eq_ineq_impl!(BinaryChunked, BinaryOffsetChunked);

fn _list_comparison_helper<F, B>(
    lhs: &ListChunked,
    rhs: &ListChunked,
    op: F,
    broadcast_op: B,
    missing: bool,
    is_ne: bool,
) -> BooleanChunked
where
    F: Fn(&PlListArray, &PlListArray) -> PlBitmap,
    B: Fn(&PlListArray, &Box<dyn PlArray>) -> PlBitmap,
{
    // Broadcast: a side that repeats a single list is compared against that list, not written out.
    let length = arity::broadcast_height(lhs.len(), rhs.len())
        .expect("cannot compare arrays of different lengths");
    match (lhs.scalar_value(), rhs.scalar_value()) {
        (_, Some(right)) if lhs.len() == length => {
            let Some(right) = right else {
                return match (missing, is_ne) {
                    (true, true) => lhs.is_not_null(),
                    (true, false) => lhs.is_null(),
                    (false, _) => BooleanChunked::full_null(PlSmallStr::EMPTY, length),
                };
            };

            if missing {
                arity::unary_mut_with_options(lhs, |a| {
                    PlBooleanArray::from_pl_bitmap(broadcast_op(a, &right))
                })
            } else {
                arity::unary_mut_values(lhs, |a| {
                    PlBooleanArray::from_pl_bitmap(broadcast_op(a, &right))
                })
            }
        },
        (Some(left), _) => {
            let Some(left) = left else {
                return match (missing, is_ne) {
                    (true, true) => rhs.is_not_null(),
                    (true, false) => rhs.is_null(),
                    (false, _) => BooleanChunked::full_null(PlSmallStr::EMPTY, length),
                };
            };

            if missing {
                arity::unary_mut_with_options(rhs, |a| {
                    PlBooleanArray::from_pl_bitmap(broadcast_op(a, &left))
                })
            } else {
                arity::unary_mut_values(rhs, |a| {
                    PlBooleanArray::from_pl_bitmap(broadcast_op(a, &left))
                })
            }
        },
        _ => {
            if missing {
                arity::binary_elementwise_kernel(
                    lhs,
                    rhs,
                    |a, b| PlBooleanArray::from_pl_bitmap(op(a, b)),
                    PlSmallStr::EMPTY,
                )
            } else {
                arity::binary_mut_values(
                    lhs,
                    rhs,
                    |a, b| PlBooleanArray::from_pl_bitmap(op(a, b)),
                    PlSmallStr::EMPTY,
                )
            }
        },
    }
}

impl ChunkCompareEq<&ListChunked> for ListChunked {
    type Item = BooleanChunked;
    fn equal(&self, rhs: &ListChunked) -> BooleanChunked {
        _list_comparison_helper(
            self,
            rhs,
            PlTotalEqKernel::tot_eq_kernel,
            PlTotalEqKernel::tot_eq_kernel_broadcast,
            false,
            false,
        )
    }

    fn equal_missing(&self, rhs: &ListChunked) -> BooleanChunked {
        _list_comparison_helper(
            self,
            rhs,
            PlTotalEqKernel::tot_eq_missing_kernel,
            PlTotalEqKernel::tot_eq_missing_kernel_broadcast,
            true,
            false,
        )
    }

    fn not_equal(&self, rhs: &ListChunked) -> BooleanChunked {
        _list_comparison_helper(
            self,
            rhs,
            PlTotalEqKernel::tot_ne_kernel,
            PlTotalEqKernel::tot_ne_kernel_broadcast,
            false,
            true,
        )
    }

    fn not_equal_missing(&self, rhs: &ListChunked) -> BooleanChunked {
        _list_comparison_helper(
            self,
            rhs,
            PlTotalEqKernel::tot_ne_missing_kernel,
            PlTotalEqKernel::tot_ne_missing_kernel_broadcast,
            true,
            true,
        )
    }
}

#[cfg(feature = "dtype-struct")]
fn struct_helper<F, R>(
    a: &StructChunked,
    b: &StructChunked,
    op: F,
    reduce: R,
    op_is_ne: bool,
    is_missing: bool,
) -> BooleanChunked
where
    F: Fn(&Series, &Series) -> BooleanChunked,
    R: Fn(BooleanChunked, BooleanChunked) -> BooleanChunked,
{
    let len_a = a.len();
    let len_b = b.len();
    let broadcasts = len_a == 1 || len_b == 1;
    assert!(a.struct_fields().len() == b.struct_fields().len());
    assert!(a.len() == b.len() || broadcasts);

    let mut out = a
        .fields_as_series()
        .iter()
        .zip(b.fields_as_series().iter())
        .map(|(l, r)| op(l, r))
        .reduce(&reduce)
        .unwrap_or_else(|| BooleanChunked::full(PlSmallStr::EMPTY, !op_is_ne, a.len()));

    if is_missing && (a.has_nulls() || b.has_nulls()) {
        // Do some allocations so that we can use the Series dispatch, it otherwise
        // gets complicated dealing with combinations of ==, != and broadcasting.
        let default =
            || BooleanChunked::with_chunk(PlSmallStr::EMPTY, PlBooleanArray::from_vec(vec![true]));
        let validity_to_ca =
            |x| BooleanChunked::with_chunk(PlSmallStr::EMPTY, PlBooleanArray::from_pl_bitmap(x));

        let a_s = a.rechunk_validity().map_or_else(default, validity_to_ca);
        let b_s = b.rechunk_validity().map_or_else(default, validity_to_ca);

        let shared_validity = (&a_s).bitand(&b_s);
        let valid_nested = if op_is_ne {
            (shared_validity).bitand(out)
        } else {
            (!shared_validity).bitor(out)
        };
        out = reduce(op(&a_s.into_series(), &b_s.into_series()), valid_nested);
    }

    if !is_missing && (a.has_nulls() || b.has_nulls()) {
        use polars_array::bitmap::combine_validities_and;

        // A side of a single element stands for every element of the output, and so does the one
        // bit its mask holds: the two masks are put over the output before they are combined.
        let length = out.len();
        let over_output = |validity: Option<PlBitmap>| match validity {
            Some(v) if v.len() == 1 && length != 1 => Some(PlBitmap::new_scalar(v.get(0), length)),
            validity => validity,
        };

        let av = over_output(a.rechunk_validity());
        let bv = over_output(b.rechunk_validity());
        out.set_validity(combine_validities_and(
            av.as_ref().map(PlBitmap::as_ref),
            bv.as_ref().map(PlBitmap::as_ref),
        ));
    }

    out
}

#[cfg(feature = "dtype-struct")]
impl ChunkCompareEq<&StructChunked> for StructChunked {
    type Item = BooleanChunked;
    fn equal(&self, rhs: &StructChunked) -> BooleanChunked {
        struct_helper(
            self,
            rhs,
            |l, r| l.equal_missing(r).unwrap(),
            |a, b| a.bitand(b),
            false,
            false,
        )
    }

    fn equal_missing(&self, rhs: &StructChunked) -> BooleanChunked {
        struct_helper(
            self,
            rhs,
            |l, r| l.equal_missing(r).unwrap(),
            |a, b| a.bitand(b),
            false,
            true,
        )
    }

    fn not_equal(&self, rhs: &StructChunked) -> BooleanChunked {
        struct_helper(
            self,
            rhs,
            |l, r| l.not_equal_missing(r).unwrap(),
            |a, b| a.bitor(b),
            true,
            false,
        )
    }

    fn not_equal_missing(&self, rhs: &StructChunked) -> BooleanChunked {
        struct_helper(
            self,
            rhs,
            |l, r| l.not_equal_missing(r).unwrap(),
            |a, b| a.bitor(b),
            true,
            true,
        )
    }
}

#[cfg(feature = "dtype-array")]
fn _array_comparison_helper<F, B>(
    lhs: &ArrayChunked,
    rhs: &ArrayChunked,
    op: F,
    broadcast_op: B,
    missing: bool,
    is_ne: bool,
) -> BooleanChunked
where
    F: Fn(&PlFixedSizeListArray, &PlFixedSizeListArray) -> PlBitmap,
    B: Fn(&PlFixedSizeListArray, &Box<dyn PlArray>) -> PlBitmap,
{
    // Broadcast: see [`_list_comparison_helper`], which dispatches the same way.
    let length = arity::broadcast_height(lhs.len(), rhs.len())
        .expect("cannot compare arrays of different lengths");
    match (lhs.scalar_value(), rhs.scalar_value()) {
        (_, Some(right)) if lhs.len() == length => {
            let Some(right) = right else {
                return match (missing, is_ne) {
                    (true, true) => lhs.is_not_null(),
                    (true, false) => lhs.is_null(),
                    (false, _) => BooleanChunked::full_null(PlSmallStr::EMPTY, length),
                };
            };

            if missing {
                arity::unary_mut_with_options(lhs, |a| {
                    PlBooleanArray::from_pl_bitmap(broadcast_op(a, &right))
                })
            } else {
                arity::unary_mut_values(lhs, |a| {
                    PlBooleanArray::from_pl_bitmap(broadcast_op(a, &right))
                })
            }
        },
        (Some(left), _) => {
            let Some(left) = left else {
                return match (missing, is_ne) {
                    (true, true) => rhs.is_not_null(),
                    (true, false) => rhs.is_null(),
                    (false, _) => BooleanChunked::full_null(PlSmallStr::EMPTY, length),
                };
            };

            if missing {
                arity::unary_mut_with_options(rhs, |a| {
                    PlBooleanArray::from_pl_bitmap(broadcast_op(a, &left))
                })
            } else {
                arity::unary_mut_values(rhs, |a| {
                    PlBooleanArray::from_pl_bitmap(broadcast_op(a, &left))
                })
            }
        },
        _ => {
            if missing {
                arity::binary_elementwise_kernel(
                    lhs,
                    rhs,
                    |a, b| PlBooleanArray::from_pl_bitmap(op(a, b)),
                    PlSmallStr::EMPTY,
                )
            } else {
                arity::binary_mut_values(
                    lhs,
                    rhs,
                    |a, b| PlBooleanArray::from_pl_bitmap(op(a, b)),
                    PlSmallStr::EMPTY,
                )
            }
        },
    }
}

#[cfg(feature = "dtype-array")]
impl ChunkCompareEq<&ArrayChunked> for ArrayChunked {
    type Item = BooleanChunked;
    fn equal(&self, rhs: &ArrayChunked) -> BooleanChunked {
        _array_comparison_helper(
            self,
            rhs,
            PlTotalEqKernel::tot_eq_kernel,
            PlTotalEqKernel::tot_eq_kernel_broadcast,
            false,
            false,
        )
    }

    fn equal_missing(&self, rhs: &ArrayChunked) -> BooleanChunked {
        _array_comparison_helper(
            self,
            rhs,
            PlTotalEqKernel::tot_eq_missing_kernel,
            PlTotalEqKernel::tot_eq_missing_kernel_broadcast,
            true,
            false,
        )
    }

    fn not_equal(&self, rhs: &ArrayChunked) -> BooleanChunked {
        _array_comparison_helper(
            self,
            rhs,
            PlTotalEqKernel::tot_ne_kernel,
            PlTotalEqKernel::tot_ne_kernel_broadcast,
            false,
            true,
        )
    }

    fn not_equal_missing(&self, rhs: &ArrayChunked) -> Self::Item {
        _array_comparison_helper(
            self,
            rhs,
            PlTotalEqKernel::tot_ne_missing_kernel,
            PlTotalEqKernel::tot_ne_missing_kernel_broadcast,
            true,
            true,
        )
    }
}

impl Not for &BooleanChunked {
    type Output = BooleanChunked;

    fn not(self) -> Self::Output {
        // Inverting a scalar values buffer is inverting the one bit it holds, so a chunk that
        // repeats a value stays `O(1)`.
        let chunks = self.downcast_iter().map(|arr| {
            PlBooleanArray::from_pl_bitmap(invert(arr.values()))
                .with_validity(arr.validity().map(PlBitmap::from))
        });
        ChunkedArray::from_chunk_iter(self.name().clone(), chunks)
    }
}

impl Not for BooleanChunked {
    type Output = BooleanChunked;

    fn not(self) -> Self::Output {
        (&self).not()
    }
}

impl BooleanChunked {
    /// Returns whether any of the values in the column are `true`.
    ///
    /// Null values are ignored.
    pub fn any(&self) -> bool {
        self.downcast_iter().any(|arr| true_count(arr) > 0)
    }

    /// Returns whether all values in the array are `true`.
    ///
    /// Null values are ignored.
    pub fn all(&self) -> bool {
        self.downcast_iter().all(|arr| false_count(arr) == 0)
    }

    /// Returns whether any of the values in the column are `true`.
    ///
    /// The output is unknown (`None`) if the array contains any null values and
    /// no `true` values.
    pub fn any_kleene(&self) -> Option<bool> {
        for arr in self.downcast_iter() {
            if true_count(arr) > 0 {
                return Some(true);
            }
        }
        if self.has_nulls() { None } else { Some(false) }
    }

    /// Returns whether all values in the column are `true`.
    ///
    /// The output is unknown (`None`) if the array contains any null values and
    /// no `false` values.
    pub fn all_kleene(&self) -> Option<bool> {
        for arr in self.downcast_iter() {
            if false_count(arr) > 0 {
                return Some(false);
            }
        }
        if self.has_nulls() { None } else { Some(true) }
    }
}

#[cfg(test)]
#[cfg_attr(feature = "nightly", allow(clippy::manual_repeat_n))] // remove once stable
mod test {
    use std::iter::repeat_n;

    use super::super::test::get_chunked_array;
    use crate::prelude::*;

    pub(crate) fn create_two_chunked() -> (Int32Chunked, Int32Chunked) {
        let mut a1 = Int32Chunked::new(PlSmallStr::from_static("a"), &[1, 2, 3]);
        let a2 = Int32Chunked::new(PlSmallStr::from_static("a"), &[4, 5, 6]);
        let a3 = Int32Chunked::new(PlSmallStr::from_static("a"), &[1, 2, 3, 4, 5, 6]);
        a1.append(&a2).unwrap();
        (a1, a3)
    }

    #[test]
    fn test_bitwise_ops() {
        let a = BooleanChunked::new(PlSmallStr::from_static("a"), &[true, false, false]);
        let b = BooleanChunked::new(
            PlSmallStr::from_static("b"),
            &[Some(true), Some(true), None],
        );
        assert_eq!(Vec::from(&a | &b), &[Some(true), Some(true), None]);
        assert_eq!(Vec::from(&a & &b), &[Some(true), Some(false), Some(false)]);
        assert_eq!(Vec::from(!b), &[Some(false), Some(false), None]);
    }

    #[test]
    fn test_compare_chunk_diff() {
        let (a1, a2) = create_two_chunked();

        assert_eq!(
            a1.equal(&a2).iter().collect::<Vec<_>>(),
            repeat_n(Some(true), 6).collect::<Vec<_>>()
        );
        assert_eq!(
            a2.equal(&a1).iter().collect::<Vec<_>>(),
            repeat_n(Some(true), 6).collect::<Vec<_>>()
        );
        assert_eq!(
            a1.not_equal(&a2).iter().collect::<Vec<_>>(),
            repeat_n(Some(false), 6).collect::<Vec<_>>()
        );
        assert_eq!(
            a2.not_equal(&a1).iter().collect::<Vec<_>>(),
            repeat_n(Some(false), 6).collect::<Vec<_>>()
        );
        assert_eq!(
            a1.gt(&a2).iter().collect::<Vec<_>>(),
            repeat_n(Some(false), 6).collect::<Vec<_>>()
        );
        assert_eq!(
            a2.gt(&a1).iter().collect::<Vec<_>>(),
            repeat_n(Some(false), 6).collect::<Vec<_>>()
        );
        assert_eq!(
            a1.gt_eq(&a2).iter().collect::<Vec<_>>(),
            repeat_n(Some(true), 6).collect::<Vec<_>>()
        );
        assert_eq!(
            a2.gt_eq(&a1).iter().collect::<Vec<_>>(),
            repeat_n(Some(true), 6).collect::<Vec<_>>()
        );
        assert_eq!(
            a1.lt_eq(&a2).iter().collect::<Vec<_>>(),
            repeat_n(Some(true), 6).collect::<Vec<_>>()
        );
        assert_eq!(
            a2.lt_eq(&a1).iter().collect::<Vec<_>>(),
            repeat_n(Some(true), 6).collect::<Vec<_>>()
        );
        assert_eq!(
            a1.lt(&a2).iter().collect::<Vec<_>>(),
            repeat_n(Some(false), 6).collect::<Vec<_>>()
        );
        assert_eq!(
            a2.lt(&a1).iter().collect::<Vec<_>>(),
            repeat_n(Some(false), 6).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_equal_chunks() {
        let a1 = get_chunked_array();
        let a2 = get_chunked_array();

        assert_eq!(
            a1.equal(&a2).iter().collect::<Vec<_>>(),
            repeat_n(Some(true), 3).collect::<Vec<_>>()
        );
        assert_eq!(
            a2.equal(&a1).iter().collect::<Vec<_>>(),
            repeat_n(Some(true), 3).collect::<Vec<_>>()
        );
        assert_eq!(
            a1.not_equal(&a2).iter().collect::<Vec<_>>(),
            repeat_n(Some(false), 3).collect::<Vec<_>>()
        );
        assert_eq!(
            a2.not_equal(&a1).iter().collect::<Vec<_>>(),
            repeat_n(Some(false), 3).collect::<Vec<_>>()
        );
        assert_eq!(
            a1.gt(&a2).iter().collect::<Vec<_>>(),
            repeat_n(Some(false), 3).collect::<Vec<_>>()
        );
        assert_eq!(
            a2.gt(&a1).iter().collect::<Vec<_>>(),
            repeat_n(Some(false), 3).collect::<Vec<_>>()
        );
        assert_eq!(
            a1.gt_eq(&a2).iter().collect::<Vec<_>>(),
            repeat_n(Some(true), 3).collect::<Vec<_>>()
        );
        assert_eq!(
            a2.gt_eq(&a1).iter().collect::<Vec<_>>(),
            repeat_n(Some(true), 3).collect::<Vec<_>>()
        );
        assert_eq!(
            a1.lt_eq(&a2).iter().collect::<Vec<_>>(),
            repeat_n(Some(true), 3).collect::<Vec<_>>()
        );
        assert_eq!(
            a2.lt_eq(&a1).iter().collect::<Vec<_>>(),
            repeat_n(Some(true), 3).collect::<Vec<_>>()
        );
        assert_eq!(
            a1.lt(&a2).iter().collect::<Vec<_>>(),
            repeat_n(Some(false), 3).collect::<Vec<_>>()
        );
        assert_eq!(
            a2.lt(&a1).iter().collect::<Vec<_>>(),
            repeat_n(Some(false), 3).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_null_handling() {
        // assert we comply with arrows way of handling null data
        // we check comparison on two arrays with one chunk and verify it is equal to a differently
        // chunked array comparison.

        // two same chunked arrays
        let a1: Int32Chunked = [Some(1), None, Some(3)].iter().copied().collect();
        let a2: Int32Chunked = [Some(1), Some(2), Some(3)].iter().copied().collect();

        let mut a2_2chunks: Int32Chunked = [Some(1), Some(2)].iter().copied().collect();
        a2_2chunks
            .append(&[Some(3)].iter().copied().collect())
            .unwrap();

        assert_eq!(
            a1.equal(&a2).iter().collect::<Vec<_>>(),
            a1.equal(&a2_2chunks).iter().collect::<Vec<_>>()
        );

        assert_eq!(
            a1.not_equal(&a2).iter().collect::<Vec<_>>(),
            a1.not_equal(&a2_2chunks).iter().collect::<Vec<_>>()
        );
        assert_eq!(
            a1.not_equal(&a2).iter().collect::<Vec<_>>(),
            a2_2chunks.not_equal(&a1).iter().collect::<Vec<_>>()
        );

        assert_eq!(
            a1.gt(&a2).iter().collect::<Vec<_>>(),
            a1.gt(&a2_2chunks).iter().collect::<Vec<_>>()
        );
        assert_eq!(
            a1.gt(&a2).iter().collect::<Vec<_>>(),
            a2_2chunks.gt(&a1).iter().collect::<Vec<_>>()
        );

        assert_eq!(
            a1.gt_eq(&a2).iter().collect::<Vec<_>>(),
            a1.gt_eq(&a2_2chunks).iter().collect::<Vec<_>>()
        );
        assert_eq!(
            a1.gt_eq(&a2).iter().collect::<Vec<_>>(),
            a2_2chunks.gt_eq(&a1).iter().collect::<Vec<_>>()
        );

        assert_eq!(
            a1.lt_eq(&a2).iter().collect::<Vec<_>>(),
            a1.lt_eq(&a2_2chunks).iter().collect::<Vec<_>>()
        );
        assert_eq!(
            a1.lt_eq(&a2).iter().collect::<Vec<_>>(),
            a2_2chunks.lt_eq(&a1).iter().collect::<Vec<_>>()
        );

        assert_eq!(
            a1.lt(&a2).iter().collect::<Vec<_>>(),
            a1.lt(&a2_2chunks).iter().collect::<Vec<_>>()
        );
        assert_eq!(
            a1.lt(&a2).iter().collect::<Vec<_>>(),
            a2_2chunks.lt(&a1).iter().collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_left_right() {
        // This failed with arrow comparisons.
        // sliced
        let a1: Int32Chunked = [Some(1), Some(2)].iter().copied().collect();
        let a1 = a1.slice(1, 1);
        let a2: Int32Chunked = [Some(2)].iter().copied().collect();
        assert_eq!(a1.equal(&a2).sum(), a2.equal(&a1).sum());
        assert_eq!(a1.not_equal(&a2).sum(), a2.not_equal(&a1).sum());
        assert_eq!(a1.gt(&a2).sum(), a2.gt(&a1).sum());
        assert_eq!(a1.lt(&a2).sum(), a2.lt(&a1).sum());
        assert_eq!(a1.lt_eq(&a2).sum(), a2.lt_eq(&a1).sum());
        assert_eq!(a1.gt_eq(&a2).sum(), a2.gt_eq(&a1).sum());

        let a1: StringChunked = ["a", "b"].iter().copied().collect();
        let a1 = a1.slice(1, 1);
        let a2: StringChunked = ["b"].iter().copied().collect();
        assert_eq!(a1.equal(&a2).sum(), a2.equal(&a1).sum());
        assert_eq!(a1.not_equal(&a2).sum(), a2.not_equal(&a1).sum());
        assert_eq!(a1.gt(&a2).sum(), a2.gt(&a1).sum());
        assert_eq!(a1.lt(&a2).sum(), a2.lt(&a1).sum());
        assert_eq!(a1.lt_eq(&a2).sum(), a2.lt_eq(&a1).sum());
        assert_eq!(a1.gt_eq(&a2).sum(), a2.gt_eq(&a1).sum());
    }

    #[test]
    fn test_kleene() {
        let a = BooleanChunked::new(PlSmallStr::EMPTY, &[Some(true), Some(false), None]);
        let trues = BooleanChunked::from_slice(PlSmallStr::EMPTY, &[true, true, true]);
        let falses = BooleanChunked::from_slice(PlSmallStr::EMPTY, &[false, false, false]);

        let c = &a | &trues;
        assert_eq!(Vec::from(&c), &[Some(true), Some(true), Some(true)]);

        let c = &a | &falses;
        assert_eq!(Vec::from(&c), &[Some(true), Some(false), None])
    }

    #[test]
    fn list_broadcasting_lists() {
        let s_el = Series::new(PlSmallStr::EMPTY, &[1, 2, 3]);
        let s_lhs = Series::new(PlSmallStr::EMPTY, &[s_el.clone(), s_el.clone()]);
        let s_rhs = Series::new(PlSmallStr::EMPTY, std::slice::from_ref(&s_el));

        let result = s_lhs.list().unwrap().equal(s_rhs.list().unwrap());
        assert_eq!(result.len(), 2);
        assert!(result.all());
    }

    #[test]
    fn test_broadcasting_bools() {
        let a = BooleanChunked::from_slice(PlSmallStr::EMPTY, &[true, false, true]);
        let true_ = BooleanChunked::from_slice(PlSmallStr::EMPTY, &[true]);
        let false_ = BooleanChunked::from_slice(PlSmallStr::EMPTY, &[false]);

        let out = a.equal(&true_);
        assert_eq!(Vec::from(&out), &[Some(true), Some(false), Some(true)]);
        let out = true_.equal(&a);
        assert_eq!(Vec::from(&out), &[Some(true), Some(false), Some(true)]);
        let out = a.equal(&false_);
        assert_eq!(Vec::from(&out), &[Some(false), Some(true), Some(false)]);
        let out = false_.equal(&a);
        assert_eq!(Vec::from(&out), &[Some(false), Some(true), Some(false)]);

        let out = a.not_equal(&true_);
        assert_eq!(Vec::from(&out), &[Some(false), Some(true), Some(false)]);
        let out = true_.not_equal(&a);
        assert_eq!(Vec::from(&out), &[Some(false), Some(true), Some(false)]);
        let out = a.not_equal(&false_);
        assert_eq!(Vec::from(&out), &[Some(true), Some(false), Some(true)]);
        let out = false_.not_equal(&a);
        assert_eq!(Vec::from(&out), &[Some(true), Some(false), Some(true)]);

        let out = a.gt(&true_);
        assert_eq!(Vec::from(&out), &[Some(false), Some(false), Some(false)]);
        let out = true_.gt(&a);
        assert_eq!(Vec::from(&out), &[Some(false), Some(true), Some(false)]);
        let out = a.gt(&false_);
        assert_eq!(Vec::from(&out), &[Some(true), Some(false), Some(true)]);
        let out = false_.gt(&a);
        assert_eq!(Vec::from(&out), &[Some(false), Some(false), Some(false)]);

        let out = a.gt_eq(&true_);
        assert_eq!(Vec::from(&out), &[Some(true), Some(false), Some(true)]);
        let out = true_.gt_eq(&a);
        assert_eq!(Vec::from(&out), &[Some(true), Some(true), Some(true)]);
        let out = a.gt_eq(&false_);
        assert_eq!(Vec::from(&out), &[Some(true), Some(true), Some(true)]);
        let out = false_.gt_eq(&a);
        assert_eq!(Vec::from(&out), &[Some(false), Some(true), Some(false)]);

        let out = a.lt(&true_);
        assert_eq!(Vec::from(&out), &[Some(false), Some(true), Some(false)]);
        let out = true_.lt(&a);
        assert_eq!(Vec::from(&out), &[Some(false), Some(false), Some(false)]);
        let out = a.lt(&false_);
        assert_eq!(Vec::from(&out), &[Some(false), Some(false), Some(false)]);
        let out = false_.lt(&a);
        assert_eq!(Vec::from(&out), &[Some(true), Some(false), Some(true)]);

        let out = a.lt_eq(&true_);
        assert_eq!(Vec::from(&out), &[Some(true), Some(true), Some(true)]);
        let out = true_.lt_eq(&a);
        assert_eq!(Vec::from(&out), &[Some(true), Some(false), Some(true)]);
        let out = a.lt_eq(&false_);
        assert_eq!(Vec::from(&out), &[Some(false), Some(true), Some(false)]);
        let out = false_.lt_eq(&a);
        assert_eq!(Vec::from(&out), &[Some(true), Some(true), Some(true)]);

        let a =
            BooleanChunked::from_slice_options(PlSmallStr::EMPTY, &[Some(true), Some(false), None]);
        let all_true = BooleanChunked::from_slice(PlSmallStr::EMPTY, &[true, true, true]);
        let all_false = BooleanChunked::from_slice(PlSmallStr::EMPTY, &[false, false, false]);
        let out = a.equal(&true_);
        assert_eq!(Vec::from(&out), &[Some(true), Some(false), None]);
        let out = a.not_equal(&true_);
        assert_eq!(Vec::from(&out), &[Some(false), Some(true), None]);

        let out = a.equal(&all_true);
        assert_eq!(Vec::from(&out), &[Some(true), Some(false), None]);
        let out = a.not_equal(&all_true);
        assert_eq!(Vec::from(&out), &[Some(false), Some(true), None]);
        let out = a.equal(&false_);
        assert_eq!(Vec::from(&out), &[Some(false), Some(true), None]);
        let out = a.not_equal(&false_);
        assert_eq!(Vec::from(&out), &[Some(true), Some(false), None]);
        let out = a.equal(&all_false);
        assert_eq!(Vec::from(&out), &[Some(false), Some(true), None]);
        let out = a.not_equal(&all_false);
        assert_eq!(Vec::from(&out), &[Some(true), Some(false), None]);
    }

    #[test]
    fn test_broadcasting_numeric() {
        let a = Int32Chunked::from_slice(PlSmallStr::EMPTY, &[1, 2, 3]);
        let one = Int32Chunked::from_slice(PlSmallStr::EMPTY, &[1]);
        let three = Int32Chunked::from_slice(PlSmallStr::EMPTY, &[3]);

        let out = a.equal(&one);
        assert_eq!(Vec::from(&out), &[Some(true), Some(false), Some(false)]);
        let out = one.equal(&a);
        assert_eq!(Vec::from(&out), &[Some(true), Some(false), Some(false)]);
        let out = a.equal(&three);
        assert_eq!(Vec::from(&out), &[Some(false), Some(false), Some(true)]);
        let out = three.equal(&a);
        assert_eq!(Vec::from(&out), &[Some(false), Some(false), Some(true)]);

        let out = a.not_equal(&one);
        assert_eq!(Vec::from(&out), &[Some(false), Some(true), Some(true)]);
        let out = one.not_equal(&a);
        assert_eq!(Vec::from(&out), &[Some(false), Some(true), Some(true)]);
        let out = a.not_equal(&three);
        assert_eq!(Vec::from(&out), &[Some(true), Some(true), Some(false)]);
        let out = three.not_equal(&a);
        assert_eq!(Vec::from(&out), &[Some(true), Some(true), Some(false)]);

        let out = a.gt(&one);
        assert_eq!(Vec::from(&out), &[Some(false), Some(true), Some(true)]);
        let out = one.gt(&a);
        assert_eq!(Vec::from(&out), &[Some(false), Some(false), Some(false)]);
        let out = a.gt(&three);
        assert_eq!(Vec::from(&out), &[Some(false), Some(false), Some(false)]);
        let out = three.gt(&a);
        assert_eq!(Vec::from(&out), &[Some(true), Some(true), Some(false)]);

        let out = a.lt(&one);
        assert_eq!(Vec::from(&out), &[Some(false), Some(false), Some(false)]);
        let out = one.lt(&a);
        assert_eq!(Vec::from(&out), &[Some(false), Some(true), Some(true)]);
        let out = a.lt(&three);
        assert_eq!(Vec::from(&out), &[Some(true), Some(true), Some(false)]);
        let out = three.lt(&a);
        assert_eq!(Vec::from(&out), &[Some(false), Some(false), Some(false)]);

        let out = a.gt_eq(&one);
        assert_eq!(Vec::from(&out), &[Some(true), Some(true), Some(true)]);
        let out = one.gt_eq(&a);
        assert_eq!(Vec::from(&out), &[Some(true), Some(false), Some(false)]);
        let out = a.gt_eq(&three);
        assert_eq!(Vec::from(&out), &[Some(false), Some(false), Some(true)]);
        let out = three.gt_eq(&a);
        assert_eq!(Vec::from(&out), &[Some(true), Some(true), Some(true)]);

        let out = a.lt_eq(&one);
        assert_eq!(Vec::from(&out), &[Some(true), Some(false), Some(false)]);
        let out = one.lt_eq(&a);
        assert_eq!(Vec::from(&out), &[Some(true), Some(true), Some(true)]);
        let out = a.lt_eq(&three);
        assert_eq!(Vec::from(&out), &[Some(true), Some(true), Some(true)]);
        let out = three.lt_eq(&a);
        assert_eq!(Vec::from(&out), &[Some(false), Some(false), Some(true)]);
    }
}
