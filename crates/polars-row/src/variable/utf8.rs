#![allow(unsafe_op_in_unsafe_fn)]
//! Row encoding for UTF-8 strings
//!
//! This encoding is based on the fact that in UTF-8 the bytes 0xFC - 0xFF are never valid bytes.
//! To make this work with the row encoding, we add 2 to each byte which gives us two bytes which
//! never occur in UTF-8 before and after the possible byte range. The values 0x00 and 0xFF are
//! reserved for the null sentinel. The values 0x01 and 0xFE are reserved as a sequence terminator
//! byte.
//!
//! This allows the string row encoding to have a constant 1 byte overhead.
use std::mem::MaybeUninit;

use polars_arrow::array::builder::StaticArrayBuilder;
use polars_arrow::array::{Array, PrimitiveArray, Utf8ViewArray, Utf8ViewArrayBuilder, View};
use polars_arrow::bitmap::BitmapBuilder;
use polars_arrow::datatypes::ArrowDataType;
use polars_arrow::types::NativeType;
use polars_buffer::Buffer;
use polars_dtype::categorical::{CatNative, CategoricalMapping};

use super::{BLOCK_SIZE, find_byte};
use crate::row::RowEncodingOptions;

#[inline]
pub fn len_from_item(a: Option<usize>, _opt: RowEncodingOptions) -> usize {
    // Length = 1                i.f.f. str is null
    // Length = len(str) + 1     i.f.f. str is non-null
    1 + a.unwrap_or_default()
}

pub unsafe fn len_from_buffer(row: &[u8], opt: RowEncodingOptions) -> usize {
    // null
    if *row.get_unchecked(0) == opt.null_sentinel() {
        return 1;
    }

    let term = descending_mask(opt) ^ 0x01;
    let mut i = 0;
    while i + BLOCK_SIZE <= row.len() {
        let block = std::ptr::read_unaligned(row.as_ptr().add(i) as *const [u8; BLOCK_SIZE]);
        let end = find_byte(block, term);
        if end < BLOCK_SIZE {
            return i + end + 1;
        }
        i += BLOCK_SIZE;
    }
    while *row.get_unchecked(i) != term {
        i += 1;
    }
    i + 1
}

#[inline(always)]
fn transform_block(mut block: [u8; BLOCK_SIZE], xor_mask: u8) -> [u8; BLOCK_SIZE] {
    for b in &mut block {
        *b = xor_mask ^ b.wrapping_add(2);
    }
    block
}

/// Encode `len` bytes of `block` followed by the terminator. Nothing after the terminator is
/// written.
///
/// # Safety
/// `dst` must be writable for `len + 1` bytes.
#[inline(always)]
unsafe fn encode_short(
    dst: *mut MaybeUninit<u8>,
    block: [u8; BLOCK_SIZE],
    len: usize,
    xor_mask: u8,
) {
    debug_assert!(len < BLOCK_SIZE);
    let block = transform_block(block, xor_mask);
    let term = xor_mask ^ 0x01;
    let n = len + 1;
    let dst = dst as *mut u8;
    let src = block.as_ptr();

    // Two overlapping stores cover the whole value. The terminator is the last byte of the
    // second store.
    if n >= 8 {
        let lo = std::ptr::read_unaligned(src as *const u64);
        let hi = std::ptr::read_unaligned(src.add(n - 8) as *const u64);
        let hi = (hi & 0x00FF_FFFF_FFFF_FFFF) | ((term as u64) << 56);
        std::ptr::write_unaligned(dst as *mut u64, lo);
        std::ptr::write_unaligned(dst.add(n - 8) as *mut u64, hi);
    } else if n >= 4 {
        let lo = std::ptr::read_unaligned(src as *const u32);
        let hi = std::ptr::read_unaligned(src.add(n - 4) as *const u32);
        let hi = (hi & 0x00FF_FFFF) | ((term as u32) << 24);
        std::ptr::write_unaligned(dst as *mut u32, lo);
        std::ptr::write_unaligned(dst.add(n - 4) as *mut u32, hi);
    } else {
        for i in 0..len {
            *dst.add(i) = *src.add(i);
        }
        *dst.add(len) = term;
    }
}

/// Encode the bytes of a string followed by the terminator.
///
/// # Safety
/// `dst` must be writable for `s.len() + 1` bytes.
#[inline(always)]
unsafe fn encode_bytes(dst: *mut MaybeUninit<u8>, s: &[u8], xor_mask: u8) {
    if s.len() < BLOCK_SIZE {
        let mut block = [0u8; BLOCK_SIZE];
        std::ptr::copy_nonoverlapping(s.as_ptr(), block.as_mut_ptr(), s.len());
        encode_short(dst, block, s.len(), xor_mask);
    } else {
        for (i, &b) in s.iter().enumerate() {
            *dst.add(i) = MaybeUninit::new(xor_mask ^ (b + 2));
        }
        *dst.add(s.len()) = MaybeUninit::new(xor_mask ^ 0x01);
    }
}

/// # Safety
/// `dst` must be writable for `view.length + 1` bytes and `view` must point into `buffers`.
#[inline(always)]
unsafe fn encode_view(
    dst: *mut MaybeUninit<u8>,
    view: &View,
    buffers: &[Buffer<u8>],
    xor_mask: u8,
) {
    let len = view.length as usize;
    if len <= View::MAX_INLINE_SIZE as usize {
        // The inline bytes follow the little-endian length field.
        let raw = std::ptr::read(view as *const View as *const [u8; BLOCK_SIZE]);
        let mut block = [0u8; BLOCK_SIZE];
        block[..12].copy_from_slice(&raw[4..]);
        encode_short(dst, block, len, xor_mask);
    } else {
        encode_bytes(dst, view.get_external_slice_unchecked(buffers), xor_mask);
    }
}

fn descending_mask(opt: RowEncodingOptions) -> u8 {
    if opt.contains(RowEncodingOptions::DESCENDING) {
        0xFF
    } else {
        0x00
    }
}

pub unsafe fn encode_str_view(
    buffer: &mut [MaybeUninit<u8>],
    array: &Utf8ViewArray,
    opt: RowEncodingOptions,
    offsets: &mut [usize],
) {
    let null_sentinel = opt.null_sentinel();
    let xor_mask = descending_mask(opt);
    let views = array.views().as_slice();
    let buffers = array.data_buffers().as_slice();
    let out = buffer.as_mut_ptr();

    match array.validity() {
        None => {
            for (offset, view) in offsets.iter_mut().zip(views) {
                encode_view(out.add(*offset), view, buffers, xor_mask);
                *offset += 1 + view.length as usize;
            }
        },
        Some(validity) => {
            for ((offset, view), is_valid) in offsets.iter_mut().zip(views).zip(validity.iter()) {
                if is_valid {
                    encode_view(out.add(*offset), view, buffers, xor_mask);
                    *offset += 1 + view.length as usize;
                } else {
                    *out.add(*offset) = MaybeUninit::new(null_sentinel);
                    *offset += 1;
                }
            }
        },
    }
}

pub unsafe fn encode_str<'a, I: Iterator<Item = Option<&'a str>>>(
    buffer: &mut [MaybeUninit<u8>],
    input: I,
    opt: RowEncodingOptions,
    offsets: &mut [usize],
) {
    let null_sentinel = opt.null_sentinel();
    let xor_mask = descending_mask(opt);
    let out = buffer.as_mut_ptr();

    for (offset, opt_value) in offsets.iter_mut().zip(input) {
        match opt_value {
            None => {
                *out.add(*offset) = MaybeUninit::new(null_sentinel);
                *offset += 1;
            },
            Some(s) => {
                encode_bytes(out.add(*offset), s.as_bytes(), xor_mask);
                *offset += 1 + s.len();
            },
        }
    }
}

/// Decode the value at the start of `row` into `scratch` and return its length. Nulls are not
/// handled here.
///
/// # Safety
/// `row` must hold a terminator.
#[inline(always)]
unsafe fn decode_into_scratch(row: &[u8], xor_mask: u8, scratch: &mut Vec<u8>) -> usize {
    let term = xor_mask ^ 0x01;
    scratch.clear();

    let mut i = 0;
    while i + BLOCK_SIZE <= row.len() {
        let block = std::ptr::read_unaligned(row.as_ptr().add(i) as *const [u8; BLOCK_SIZE]);
        let end = find_byte(block, term);
        scratch.extend_from_slice(&decode_block(block, xor_mask)[..end]);
        if end < BLOCK_SIZE {
            return i + end;
        }
        i += BLOCK_SIZE;
    }
    while *row.get_unchecked(i) != term {
        scratch.push((*row.get_unchecked(i) ^ xor_mask).wrapping_sub(2));
        i += 1;
    }
    i
}

#[inline(always)]
fn decode_block(mut block: [u8; BLOCK_SIZE], xor_mask: u8) -> [u8; BLOCK_SIZE] {
    for b in &mut block {
        *b = (*b ^ xor_mask).wrapping_sub(2);
    }
    block
}

/// Decode one value and push it to `builder`. Returns `false` for null.
///
/// # Safety
/// `row` must start with a null sentinel or an encoded value.
#[inline(always)]
unsafe fn decode_one(
    row: &mut &[u8],
    null_sentinel: u8,
    xor_mask: u8,
    builder: &mut Utf8ViewArrayBuilder,
    scratch: &mut Vec<u8>,
) -> bool {
    if *row.get_unchecked(0) == null_sentinel {
        *row = row.get_unchecked(1..);
        return false;
    }

    // Short values become inline views straight from a block load.
    if row.len() >= BLOCK_SIZE {
        let block = std::ptr::read_unaligned(row.as_ptr() as *const [u8; BLOCK_SIZE]);
        let len = find_byte(block, xor_mask ^ 0x01);
        if len <= View::MAX_INLINE_SIZE as usize {
            let view = View::new_inline_from_block(decode_block(block, xor_mask), len);
            builder.push_inline_view_ignore_validity(view);
            *row = row.get_unchecked(1 + len..);
            return true;
        }
    }

    let len = decode_into_scratch(row, xor_mask, scratch);
    builder.push_value_ignore_validity(std::str::from_utf8_unchecked(scratch));
    *row = row.get_unchecked(1 + len..);
    true
}

pub unsafe fn decode_str(rows: &mut [&[u8]], opt: RowEncodingOptions) -> Utf8ViewArray {
    let null_sentinel = opt.null_sentinel();
    let xor_mask = descending_mask(opt);
    let num_rows = rows.len();
    let mut builder = Utf8ViewArrayBuilder::new(ArrowDataType::Utf8View);
    builder.reserve(num_rows);
    let mut scratch = Vec::new();
    let mut validity = BitmapBuilder::new();

    for row in rows.iter_mut() {
        if !decode_one(row, null_sentinel, xor_mask, &mut builder, &mut scratch) {
            validity.reserve(num_rows);
            validity.extend_constant(builder.len(), true);
            validity.push(false);
            builder.push_null_ignore_validity();
            break;
        }
    }

    if !validity.is_empty() {
        for row in rows[builder.len()..].iter_mut() {
            let is_valid = decode_one(row, null_sentinel, xor_mask, &mut builder, &mut scratch);
            validity.push(is_valid);
            if !is_valid {
                builder.push_null_ignore_validity();
            }
        }
    }

    builder.freeze_with_validity(validity.into_opt_validity())
}

/// The same as decode_str but inserts it into the given mapping, translating
/// it to physical type T.
pub unsafe fn decode_str_as_cat<T: NativeType + CatNative>(
    rows: &mut [&[u8]],
    opt: RowEncodingOptions,
    mapping: &CategoricalMapping,
) -> PrimitiveArray<T> {
    let null_sentinel = opt.null_sentinel();
    let xor_mask = descending_mask(opt);

    let num_rows = rows.len();
    let mut out = Vec::<T>::with_capacity(rows.len());

    let mut scratch = Vec::new();
    for row in rows.iter_mut() {
        let sentinel = *unsafe { row.get_unchecked(0) };
        if sentinel == null_sentinel {
            *row = unsafe { row.get_unchecked(1..) };
            break;
        }

        let len = decode_into_scratch(row, xor_mask, &mut scratch);
        *row = row.get_unchecked(1 + len..);
        let s = unsafe { std::str::from_utf8_unchecked(&scratch) };
        out.push(T::from_cat(mapping.insert_cat(s).unwrap()));
    }

    if out.len() == num_rows {
        return PrimitiveArray::from_vec(out);
    }

    let mut validity = BitmapBuilder::with_capacity(num_rows);
    validity.extend_constant(out.len(), true);
    validity.push(false);
    out.push(T::zeroed());

    for row in rows[out.len()..].iter_mut() {
        let sentinel = *unsafe { row.get_unchecked(0) };
        validity.push(sentinel != null_sentinel);
        if sentinel == null_sentinel {
            *row = unsafe { row.get_unchecked(1..) };
            out.push(T::zeroed());
            continue;
        }

        let len = decode_into_scratch(row, xor_mask, &mut scratch);
        *row = row.get_unchecked(1 + len..);
        let s = unsafe { std::str::from_utf8_unchecked(&scratch) };
        out.push(T::from_cat(mapping.insert_cat(s).unwrap()));
    }

    PrimitiveArray::from_vec(out).with_validity(validity.into_opt_validity())
}
