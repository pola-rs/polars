//! The byte class of an element type, which is all a routine that only moves bytes is taken over.
//!
//! A [`PlPrimitiveArray`](super::PlPrimitiveArray) is generic in its element type, of which there
//! are seventeen. Those seventeen fall into only nine distinct pairs of size and alignment, and a
//! routine that copies, repeats, gathers or zeroes elements reads nothing of an element but those
//! two numbers: `i32`, `u32` and `f32` differ in what their bytes *mean*, which is exactly what
//! such a routine never looks at. Writing one over the byte class rather than over the element
//! type therefore has it compiled nine times instead of seventeen, and leaves the array's own
//! methods as thin adapters that reinterpret their buffers and call it.
//!
//! Which is also why the routines here are `#[inline(never)]`: nine copies of a routine only
//! stay nine copies if the optimizer is kept from pasting each one back into all seventeen
//! adapters that call it, which is what it does to a body this small left to its own judgement.
//! Nothing here is called per element — each one appends a whole array, a whole subslice or a
//! whole batch of gathered indices — so the call it costs is nothing against the code it saves,
//! and the adapters around them stay inlinable as they were.
//!
//! Reinterpreting is free and cannot fail: an element type and its byte class have the same size
//! and alignment by construction, which [`assert_same_layout`] pins down at compile time, and a
//! [`Buffer`] carries the layout its storage was allocated with, so the original allocation is
//! still freed correctly through the reinterpreted buffer.
//!
//! What must *not* be routed through here is anything that reads what the bytes mean — ordering,
//! hashing, and equality on floats above all, where `+0.0` and `-0.0` are equal numbers with
//! different bytes and a `NaN` is a number equal to nothing at all, itself included. The one
//! place this crate does compare bytes on purpose is documented where it does so.

use arrow::Either;
use arrow::types::{AlignedBytes, NativeType};
use polars_buffer::Buffer;
use polars_utils::IdxSize;
use polars_utils::vec::PushUnchecked;

use crate::broadcast::ArrayRepr;

/// The byte class of `T`: a `[u8; N]` with the size and alignment of `T`, and nothing else.
pub(crate) type Bytes<T> = <T as NativeType>::AlignedBytes;

/// The values of an array as bytes, in whichever representation the backing buffer is in.
pub(crate) type ValuesBytes<'a, B> = ArrayRepr<&'a [B], B>;

/// Fails to compile unless `T` and its byte class really do have the same layout, which every
/// reinterpretation in this module rests on.
const fn assert_same_layout<T: NativeType>() {
    assert!(size_of::<T>() == size_of::<Bytes<T>>());
    assert!(align_of::<T>() == align_of::<Bytes<T>>());
}

/// The bytes of `value`.
#[inline(always)]
pub(crate) fn to_bytes<T: NativeType>(value: T) -> Bytes<T> {
    value.into()
}

/// The element `bytes` are the bytes of.
#[inline(always)]
pub(crate) fn from_bytes<T: NativeType>(bytes: Bytes<T>) -> T {
    bytes.into()
}

/// The elements of `slice` as their bytes, which is a borrow rather than a copy.
#[inline(always)]
pub(crate) fn slice_to_bytes<T: NativeType>(slice: &[T]) -> &[Bytes<T>] {
    const { assert_same_layout::<T>() };
    bytemuck::cast_slice(slice)
}

/// The buffer of elements `bytes` holds the bytes of, which is `O(1)` and shares the allocation.
#[inline(always)]
pub(crate) fn buffer_from_bytes<T: NativeType>(bytes: Buffer<Bytes<T>>) -> Buffer<T> {
    const { assert_same_layout::<T>() };
    bytes
        .try_transmute::<T>()
        .unwrap_or_else(|_| unreachable!("a byte class has the layout of the type it stands for"))
}

/// The buffer of elements the bytes in `values` stand for.
#[inline(always)]
pub(crate) fn buffer_from_byte_vec<T: NativeType>(values: Vec<Bytes<T>>) -> Buffer<T> {
    buffer_from_bytes::<T>(Buffer::from(values))
}

/// The bytes of the elements in `values` as a `Vec` that owns them, which reuses the allocation
/// rather than copying it.
///
/// This is the inverse of [`buffer_from_byte_vec`], and it succeeds only when the buffer can give
/// its allocation up: when it is unsliced and nothing else holds a reference to it. Otherwise the
/// buffer is handed back untouched, on the left.
#[inline(always)]
pub(crate) fn byte_vec_from_buffer<T: NativeType>(
    values: Buffer<T>,
) -> Either<Buffer<T>, Vec<Bytes<T>>> {
    const { assert_same_layout::<T>() };
    // Reinterpreting first means the `Vec` that comes back is already the builder's element type,
    // so no `Vec` is ever transmuted.
    match values.try_transmute::<Bytes<T>>() {
        Ok(bytes) => match bytes.into_mut() {
            Either::Right(values) => Either::Right(values),
            Either::Left(bytes) => Either::Left(buffer_from_bytes::<T>(bytes)),
        },
        Err(values) => Either::Left(values),
    }
}

// Everything below is what this module exists for: one copy per byte class rather than one per
// element type. Leave them out of line — see the module docs.

/// A buffer of `length` copies of `value`.
#[inline(never)]
pub(crate) fn repeat<B: AlignedBytes>(value: B, length: usize) -> Buffer<B> {
    Buffer::from(vec![value; length])
}

/// A buffer of `length` slots that are never read, and so need not be written either.
#[inline(never)]
pub(crate) fn undetermined<B: AlignedBytes>(length: usize) -> Buffer<B> {
    Buffer::zeroed(length)
}

/// Appends `length` slots that are never read, and so need not be written either.
#[inline(never)]
pub(crate) fn extend_undetermined<B: AlignedBytes>(values: &mut Vec<B>, length: usize) {
    values.resize(values.len() + length, B::zeros());
}

/// Appends the `length` values of `other` starting at `start`.
#[inline(never)]
pub(crate) fn extend_subslice<B: AlignedBytes>(
    values: &mut Vec<B>,
    other: ValuesBytes<'_, B>,
    start: usize,
    length: usize,
) {
    match other {
        ArrayRepr::Flat(slice) => values.extend_from_slice(&slice[start..start + length]),
        // Every element of the array reads the same value, so which of them the subslice covers
        // makes no difference to what is appended.
        ArrayRepr::Scalar(value) => values.resize(values.len() + length, value),
    }
}

/// Appends each of the `length` values of `other` starting at `start` `repeats` times over.
#[inline(never)]
pub(crate) fn extend_subslice_each_repeated<B: AlignedBytes>(
    values: &mut Vec<B>,
    other: ValuesBytes<'_, B>,
    start: usize,
    length: usize,
    repeats: usize,
) {
    values.reserve(length * repeats);

    match other {
        ArrayRepr::Flat(slice) => {
            for value in &slice[start..start + length] {
                // SAFETY: room for every repeat of every value was just reserved.
                unsafe {
                    for _ in 0..repeats {
                        values.push_unchecked(*value);
                    }
                }
            }
        },
        // Every element repeats the same value, so which of them is repeated is immaterial.
        ArrayRepr::Scalar(value) => values.resize(values.len() + length * repeats, value),
    }
}

/// Appends the value of `other` at every index of `idxs`, in the order they are given.
///
/// # Safety
/// Every index must be in bounds of the array `other` is the values of.
#[inline(never)]
pub(crate) unsafe fn extend_gathered<B: AlignedBytes>(
    values: &mut Vec<B>,
    other: ValuesBytes<'_, B>,
    idxs: &[IdxSize],
) {
    match other {
        // SAFETY: the indices are in bounds of the array, whose values are flat.
        ArrayRepr::Flat(slice) => values.extend(
            idxs.iter()
                .map(|idx| unsafe { *slice.get_unchecked(*idx as usize) }),
        ),
        // Every index reads the one value the array holds.
        ArrayRepr::Scalar(value) => values.resize(values.len() + idxs.len(), value),
    }
}

/// Appends the value of `other` at every index of `idxs`, in the order they are given, with an
/// index that falls outside an array of `length` elements standing for a null.
#[inline(never)]
pub(crate) fn extend_opt_gathered<B: AlignedBytes>(
    values: &mut Vec<B>,
    other: ValuesBytes<'_, B>,
    length: usize,
    idxs: &[IdxSize],
) {
    values.reserve(idxs.len());

    for idx in idxs {
        let idx = *idx as usize;
        let value = if idx < length {
            match other {
                // SAFETY: the index is in bounds of the array, whose values are flat.
                ArrayRepr::Flat(slice) => unsafe { *slice.get_unchecked(idx) },
                ArrayRepr::Scalar(value) => value,
            }
        } else {
            // The value of a null element is undetermined, so anything at all does.
            B::zeros()
        };
        // SAFETY: room for one value per index was just reserved.
        unsafe { values.push_unchecked(value) };
    }
}

#[cfg(test)]
mod tests {
    use arrow::array::View;
    use arrow::types::{days_ms, i256, months_days_ns};
    use polars_utils::aliases::PlHashSet;
    use polars_utils::float16::pf16;

    use super::*;

    /// The seventeen element types this crate dispatches on fall into nine byte classes, which is
    /// the whole point of the module: it is that ratio the routines above are compiled at.
    #[test]
    fn seventeen_element_types_fall_into_nine_byte_classes() {
        fn class<T: NativeType>() -> (usize, usize) {
            const { assert_same_layout::<T>() };
            (size_of::<Bytes<T>>(), align_of::<Bytes<T>>())
        }

        let classes = [
            class::<i8>(),
            class::<i16>(),
            class::<i32>(),
            class::<i64>(),
            class::<i128>(),
            class::<i256>(),
            class::<u8>(),
            class::<u16>(),
            class::<u32>(),
            class::<u64>(),
            class::<u128>(),
            class::<pf16>(),
            class::<f32>(),
            class::<f64>(),
            class::<days_ms>(),
            class::<months_days_ns>(),
            class::<View>(),
        ];

        assert_eq!(classes.len(), 17);
        assert_eq!(classes.iter().collect::<PlHashSet<_>>().len(), 9);

        // The three that a routine over the byte class stops telling apart, which is exactly the
        // saving: they only ever differ in what their bytes mean.
        assert_eq!(class::<i32>(), class::<u32>());
        assert_eq!(class::<i32>(), class::<f32>());
    }

    #[test]
    fn buffers_are_reinterpreted_in_place() {
        let bytes = repeat(to_bytes(7i32), 3);
        let ptr = bytes.as_slice().as_ptr();

        let values = buffer_from_bytes::<i32>(bytes);

        assert_eq!(values.as_slice(), [7i32; 3]);
        assert_eq!(
            values.as_slice().as_ptr().cast::<u8>(),
            ptr.cast::<u8>(),
            "the buffer must be reinterpreted, not copied",
        );
    }

    /// The bytes of `-0.0` differ from those of `+0.0`, which is what keeps the two apart here
    /// where `PartialEq` on the floats themselves runs them together.
    #[test]
    fn the_two_zeroes_have_different_bytes() {
        assert_eq!(-0.0f64, 0.0f64);
        assert_ne!(to_bytes(-0.0f64), to_bytes(0.0f64));
        assert!(from_bytes::<f64>(to_bytes(-0.0f64)).is_sign_negative());
    }

    #[test]
    fn values_are_appended_in_either_representation() {
        let values = [1i32, 2, 3];
        let flat = ArrayRepr::Flat(slice_to_bytes(&values));
        let scalar = ArrayRepr::Scalar(to_bytes(7i32));

        let mut built = Vec::new();
        extend_subslice(&mut built, flat, 1, 2);
        extend_subslice(&mut built, scalar, 0, 2);
        extend_subslice_each_repeated(&mut built, flat, 0, 2, 2);
        extend_subslice_each_repeated(&mut built, scalar, 0, 1, 2);
        unsafe { extend_gathered(&mut built, flat, &[2, 0]) };
        unsafe { extend_gathered(&mut built, scalar, &[0]) };
        extend_opt_gathered(&mut built, flat, 3, &[1, 7]);
        extend_opt_gathered(&mut built, scalar, 3, &[2, 7]);
        extend_undetermined(&mut built, 1);

        assert_eq!(
            buffer_from_byte_vec::<i32>(built).as_slice(),
            [2, 3, 7, 7, 1, 1, 2, 2, 7, 7, 3, 1, 7, 2, 0, 7, 0, 0],
        );
    }
}
