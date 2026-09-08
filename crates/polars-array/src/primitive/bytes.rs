//! The byte class of an element type, which is all a routine that only moves bytes is taken over.

use arrow::Either;
use arrow::types::{AlignedBytes, NativeType};
use polars_buffer::Buffer;
use polars_utils::IdxSize;
use polars_utils::vec::PushUnchecked;

/// The byte class of `T`: a `[u8; N]` with the size and alignment of `T`, and nothing else.
pub(crate) type Bytes<T> = <T as NativeType>::AlignedBytes;

/// The values of an array as bytes, in whichever representation the backing buffer is in.
#[derive(Clone, Copy)]
pub(crate) enum ValuesBytes<'a, B> {
    /// The buffer holds one slot per element.
    Flat(&'a [B]),
    /// The buffer holds the single slot every element of the array reads.
    Scalar(B),
}

/// Fails to compile unless `T` and its byte class really do have the same layout.
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

/// The bytes of the elements in `values` as a `Vec`, reusing the allocation rather than copying.
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
        ValuesBytes::Flat(slice) => values.extend_from_slice(&slice[start..start + length]),
        // Every element of the array reads the same value, so which of them the subslice covers
        // makes no difference to what is appended.
        ValuesBytes::Scalar(value) => values.resize(values.len() + length, value),
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
        ValuesBytes::Flat(slice) => {
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
        ValuesBytes::Scalar(value) => values.resize(values.len() + length * repeats, value),
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
        ValuesBytes::Flat(slice) => values.extend(
            idxs.iter()
                .map(|idx| unsafe { *slice.get_unchecked(*idx as usize) }),
        ),
        // Every index reads the one value the array holds.
        ValuesBytes::Scalar(value) => values.resize(values.len() + idxs.len(), value),
    }
}

/// Appends the value of `other` at every index of `idxs`; an index past `length` is a null.
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
                ValuesBytes::Flat(slice) => unsafe { *slice.get_unchecked(idx) },
                ValuesBytes::Scalar(value) => value,
            }
        } else {
            // The value of a null element is undetermined, so anything at all does.
            B::zeros()
        };
        // SAFETY: room for one value per index was just reserved.
        unsafe { values.push_unchecked(value) };
    }
}
