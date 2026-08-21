use std::mem::MaybeUninit;

use arrow::array::{Array, PrimitiveArray};
use arrow::bitmap::utils::SlicesIterator;
use arrow::bitmap::{self, Bitmap};
use arrow::datatypes::ArrowDataType;

use crate::NotSimdPrimitive;

mod array;
mod boolean;
mod list;
mod scalar;
#[cfg(feature = "simd")]
mod simd;
mod view;

pub trait IfThenElseKernel: Sized + Array {
    type Scalar<'a>;

    fn if_then_else(mask: &Bitmap, if_true: &Self, if_false: &Self) -> Self;
    fn if_then_else_broadcast_true(
        mask: &Bitmap,
        if_true: Self::Scalar<'_>,
        if_false: &Self,
    ) -> Self;
    fn if_then_else_broadcast_false(
        mask: &Bitmap,
        if_true: &Self,
        if_false: Self::Scalar<'_>,
    ) -> Self;
    fn if_then_else_broadcast_both(
        dtype: ArrowDataType,
        mask: &Bitmap,
        if_true: Self::Scalar<'_>,
        if_false: Self::Scalar<'_>,
    ) -> Self;
}

impl<T: NotSimdPrimitive> IfThenElseKernel for PrimitiveArray<T> {
    type Scalar<'a> = T;

    fn if_then_else(mask: &Bitmap, if_true: &Self, if_false: &Self) -> Self {
        let values = if_then_else_loop(
            mask,
            if_true.values(),
            if_false.values(),
            scalar::if_then_else_scalar_rest,
            scalar::if_then_else_scalar_64,
        );
        let validity = if_then_else_validity(mask, if_true.validity(), if_false.validity());
        PrimitiveArray::from_vec(values).with_validity(validity)
    }

    fn if_then_else_broadcast_true(
        mask: &Bitmap,
        if_true: Self::Scalar<'_>,
        if_false: &Self,
    ) -> Self {
        let values = if_then_else_loop_broadcast_false(
            true,
            mask,
            if_false.values(),
            if_true,
            scalar::if_then_else_broadcast_false_scalar_64,
        );
        let validity = if_then_else_validity(mask, None, if_false.validity());
        PrimitiveArray::from_vec(values).with_validity(validity)
    }

    fn if_then_else_broadcast_false(
        mask: &Bitmap,
        if_true: &Self,
        if_false: Self::Scalar<'_>,
    ) -> Self {
        let values = if_then_else_loop_broadcast_false(
            false,
            mask,
            if_true.values(),
            if_false,
            scalar::if_then_else_broadcast_false_scalar_64,
        );
        let validity = if_then_else_validity(mask, if_true.validity(), None);
        PrimitiveArray::from_vec(values).with_validity(validity)
    }

    fn if_then_else_broadcast_both(
        _dtype: ArrowDataType,
        mask: &Bitmap,
        if_true: Self::Scalar<'_>,
        if_false: Self::Scalar<'_>,
    ) -> Self {
        let values = if_then_else_loop_broadcast_both(
            mask,
            if_true,
            if_false,
            scalar::if_then_else_broadcast_both_scalar_64,
        );
        PrimitiveArray::from_vec(values)
    }
}

pub fn if_then_else_validity(
    mask: &Bitmap,
    if_true: Option<&Bitmap>,
    if_false: Option<&Bitmap>,
) -> Option<Bitmap> {
    match (if_true, if_false) {
        (None, None) => None,
        (None, Some(f)) => Some(mask | f),
        (Some(t), None) => Some(bitmap::binary(mask, t, |m, t| !m | t)),
        (Some(t), Some(f)) => Some(bitmap::ternary(mask, t, f, |m, t, f| (m & t) | (!m & f))),
    }
}

fn if_then_else_extend<B, ET: Fn(&mut B, usize, usize), EF: Fn(&mut B, usize, usize)>(
    builder: &mut B,
    mask: &Bitmap,
    extend_true: ET,
    extend_false: EF,
) {
    let mut last_true_end = 0;
    for (start, len) in SlicesIterator::new(mask) {
        if start != last_true_end {
            extend_false(builder, last_true_end, start - last_true_end);
        };
        extend_true(builder, start, len);
        last_true_end = start + len;
    }
    if last_true_end != mask.len() {
        extend_false(builder, last_true_end, mask.len() - last_true_end)
    }
}

fn if_then_else_loop<T, F, F64>(
    mask: &Bitmap,
    if_true: &[T],
    if_false: &[T],
    process_var: F,
    process_chunk: F64,
) -> Vec<T>
where
    T: Copy,
    F: Fn(u64, &[T], &[T], &mut [MaybeUninit<T>]),
    F64: Fn(u64, &[T; 64], &[T; 64], &mut [MaybeUninit<T>; 64]),
{
    assert_eq!(mask.len(), if_true.len());
    assert_eq!(mask.len(), if_false.len());

    let mut ret = Vec::with_capacity(mask.len());
    let out = &mut ret.spare_capacity_mut()[..mask.len()];

    // Handle prefix.
    let aligned = mask.aligned::<u64>();
    let (start_true, rest_true) = if_true.split_at(aligned.prefix_bitlen());
    let (start_false, rest_false) = if_false.split_at(aligned.prefix_bitlen());
    let (start_out, rest_out) = out.split_at_mut(aligned.prefix_bitlen());
    if aligned.prefix_bitlen() > 0 {
        process_var(aligned.prefix(), start_true, start_false, start_out);
    }

    // Handle bulk.
    let (true_chunks, true_remainder) = rest_true.as_chunks::<64>();
    let (false_chunks, false_remainder) = rest_false.as_chunks::<64>();
    let (out_chunks, out_remainder) = rest_out.as_chunks_mut::<64>();
    let combined = true_chunks.iter().zip(false_chunks).zip(out_chunks);
    for (i, ((tc, fc), oc)) in combined.enumerate() {
        let m = unsafe { *aligned.bulk().get_unchecked(i) };
        process_chunk(m, tc, fc, oc);
    }

    // Handle suffix.
    if aligned.suffix_bitlen() > 0 {
        process_var(
            aligned.suffix(),
            true_remainder,
            false_remainder,
            out_remainder,
        );
    }

    unsafe {
        ret.set_len(mask.len());
    }
    ret
}

fn if_then_else_loop_broadcast_false<T, F64>(
    invert_mask: bool, // Allows code reuse for both false and true broadcasts.
    mask: &Bitmap,
    if_true: &[T],
    if_false: T,
    process_chunk: F64,
) -> Vec<T>
where
    T: Copy,
    F64: Fn(u64, &[T; 64], T, &mut [MaybeUninit<T>; 64]),
{
    assert_eq!(mask.len(), if_true.len());

    let mut ret = Vec::with_capacity(mask.len());
    let out = &mut ret.spare_capacity_mut()[..mask.len()];

    // XOR with all 1's inverts the mask.
    let xor_inverter = if invert_mask { u64::MAX } else { 0 };

    // Handle prefix.
    let aligned = mask.aligned::<u64>();
    let (start_true, rest_true) = if_true.split_at(aligned.prefix_bitlen());
    let (start_out, rest_out) = out.split_at_mut(aligned.prefix_bitlen());
    if aligned.prefix_bitlen() > 0 {
        scalar::if_then_else_broadcast_false_scalar_rest(
            aligned.prefix() ^ xor_inverter,
            start_true,
            if_false,
            start_out,
        );
    }

    // Handle bulk.
    let (true_chunks, true_remainder) = rest_true.as_chunks::<64>();
    let (out_chunks, out_remainder) = rest_out.as_chunks_mut::<64>();
    let combined = true_chunks.iter().zip(out_chunks);
    for (i, (tc, oc)) in combined.enumerate() {
        let m = unsafe { *aligned.bulk().get_unchecked(i) } ^ xor_inverter;
        process_chunk(m, tc, if_false, oc);
    }

    // Handle suffix.
    if aligned.suffix_bitlen() > 0 {
        scalar::if_then_else_broadcast_false_scalar_rest(
            aligned.suffix() ^ xor_inverter,
            true_remainder,
            if_false,
            out_remainder,
        );
    }

    unsafe {
        ret.set_len(mask.len());
    }
    ret
}

fn if_then_else_loop_broadcast_both<T, F64>(
    mask: &Bitmap,
    if_true: T,
    if_false: T,
    generate_chunk: F64,
) -> Vec<T>
where
    T: Copy,
    F64: Fn(u64, T, T, &mut [MaybeUninit<T>; 64]),
{
    let mut ret = Vec::with_capacity(mask.len());
    let out = &mut ret.spare_capacity_mut()[..mask.len()];

    // Handle prefix.
    let aligned = mask.aligned::<u64>();
    let (start_out, rest_out) = out.split_at_mut(aligned.prefix_bitlen());
    scalar::if_then_else_broadcast_both_scalar_rest(aligned.prefix(), if_true, if_false, start_out);

    // Handle bulk.
    let (out_chunks, out_remainder) = rest_out.as_chunks_mut::<64>();
    for (i, oc) in out_chunks.iter_mut().enumerate() {
        let m = unsafe { *aligned.bulk().get_unchecked(i) };
        generate_chunk(m, if_true, if_false, oc);
    }

    // Handle suffix.
    if aligned.suffix_bitlen() > 0 {
        scalar::if_then_else_broadcast_both_scalar_rest(
            aligned.suffix(),
            if_true,
            if_false,
            out_remainder,
        );
    }

    unsafe {
        ret.set_len(mask.len());
    }
    ret
}
