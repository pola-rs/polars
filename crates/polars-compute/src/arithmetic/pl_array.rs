//! What an arithmetic kernel does with a chunk before it reads it.

use polars_array::bitmap::combine_validities_and;
use polars_array::{PlBitmap, PlPrimitiveArray};
use polars_arrow::types::NativeType;

use super::{PArr, POut};

/// A chunk taken apart into the values a kernel reads and the mask around its answer.
enum Split<T: NativeType> {
    /// Every element is null, so the length is all that is left of the chunk.
    AllNull,
    /// The one value every element reads, and the mask over those elements.
    Repeated(T, Option<PlBitmap>),
    /// The chunk itself, with one values slot per element.
    Flat(PArr<T>),
}

impl<T: NativeType> Split<T> {
    fn of(mut arr: PlPrimitiveArray<T>) -> Self {
        if let Some(bit) = arr.validity().and_then(|validity| validity.scalar_value()) {
            if !bit {
                return Self::AllNull;
            }
            arr = arr.without_validity();
        }

        match arr.scalar_value_ignore_validity() {
            Some(value) => Self::Repeated(value, arr.validity().map(PlBitmap::from)),
            None => Self::Flat(arr.to_flat().into_owned()),
        }
    }
}

/// The one element `value` stands for, as a flat array a kernel can read.
fn single<T: NativeType>(value: T) -> PArr<T> {
    PlPrimitiveArray::new_scalar(value, 1)
        .to_flat()
        .into_owned()
}

/// A kernel's answer for the one value a chunk repeats, spread back over `length` elements.
fn repeat<O: NativeType>(out: POut<O>, length: usize, validity: Option<PlBitmap>) -> POut<O> {
    debug_assert_eq!(
        out.len(),
        1,
        "an elementwise kernel answers one element with one"
    );

    match out.scalar_value().flatten() {
        None => POut::new_full_null(length),
        Some(value) => POut::new_scalar(value, length).with_validity(validity),
    }
}

/// `out` with `validity` folded into the mask it carries already.
fn fold_in<O: NativeType>(out: POut<O>, validity: Option<PlBitmap>) -> POut<O> {
    let Some(validity) = validity else {
        return out;
    };

    let combined = combine_validities_and(out.validity(), Some(validity.as_ref()));
    out.with_validity(combined)
}

/// Applies `flat`, an elementwise kernel over the flat representation, to `arr`.
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

/// Applies a binary elementwise kernel to `lhs` and `rhs`, in the shape that reads the least.
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
        (Split::AllNull, _) | (_, Split::AllNull) => POut::new_full_null(length),

        (Split::Repeated(l, lv), Split::Repeated(r, rv)) => {
            let validity = combine_validities_and(
                lv.as_ref().map(PlBitmap::as_ref),
                rv.as_ref().map(PlBitmap::as_ref),
            );
            repeat(flat(single(l), single(r)), length, validity)
        },

        (Split::Repeated(l, lv), Split::Flat(rhs)) => fold_in(scalar_lhs(l, rhs), lv),
        (Split::Flat(lhs), Split::Repeated(r, rv)) => fold_in(scalar_rhs(lhs, r), rv),

        (Split::Flat(lhs), Split::Flat(rhs)) => flat(lhs, rhs),
    }
}
