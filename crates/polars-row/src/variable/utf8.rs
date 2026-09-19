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

use polars_arrow::array::{Array, PrimitiveArray, Utf8ViewArray, View};
use polars_arrow::bitmap::BitmapBuilder;
use polars_arrow::datatypes::ArrowDataType;
use polars_arrow::types::NativeType;
use polars_buffer::Buffer;
use polars_dtype::categorical::{CatNative, CategoricalMapping};

use super::view_builder::ViewBuilder;
use crate::row::RowEncodingOptions;
use crate::utils::{BLOCK, find_byte, inline_view};

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

    let end = if opt.contains(RowEncodingOptions::DESCENDING) {
        unsafe { row.iter().position(|&b| b == 0xFE).unwrap_unchecked() }
    } else {
        unsafe { row.iter().position(|&b| b == 0x01).unwrap_unchecked() }
    };

    end + 1
}

#[inline(always)]
fn transform_block(mut block: [u8; BLOCK], t: u8) -> [u8; BLOCK] {
    for b in &mut block {
        *b = t ^ b.wrapping_add(2);
    }
    block
}

/// Encode `len` bytes of `block` followed by the terminator. Nothing after the terminator is
/// written.
#[inline(always)]
unsafe fn encode_short(dst: *mut MaybeUninit<u8>, block: [u8; BLOCK], len: usize, t: u8) {
    debug_assert!(len < BLOCK);
    let block = transform_block(block, t);
    let term = t ^ 0x01;
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

/// Encode a string of at least [`BLOCK`] bytes followed by the terminator.
#[inline(always)]
unsafe fn encode_long(dst: *mut MaybeUninit<u8>, src: *const u8, len: usize, t: u8) {
    debug_assert!(len >= BLOCK);
    let mut i = 0;
    while i + BLOCK <= len {
        let block = std::ptr::read_unaligned(src.add(i) as *const [u8; BLOCK]);
        std::ptr::write_unaligned(dst.add(i) as *mut [u8; BLOCK], transform_block(block, t));
        i += BLOCK;
    }
    // The last block overlaps with the previous one and writes the same bytes there.
    let block = std::ptr::read_unaligned(src.add(len - BLOCK) as *const [u8; BLOCK]);
    std::ptr::write_unaligned(
        dst.add(len - BLOCK) as *mut [u8; BLOCK],
        transform_block(block, t),
    );
    *dst.add(len) = MaybeUninit::new(t ^ 0x01);
}

#[inline(always)]
unsafe fn encode_bytes(dst: *mut MaybeUninit<u8>, s: &[u8], t: u8) {
    if s.len() >= BLOCK {
        encode_long(dst, s.as_ptr(), s.len(), t);
    } else {
        let mut block = [0u8; BLOCK];
        std::ptr::copy_nonoverlapping(s.as_ptr(), block.as_mut_ptr(), s.len());
        encode_short(dst, block, s.len(), t);
    }
}

#[inline(always)]
unsafe fn encode_view(dst: *mut MaybeUninit<u8>, view: &View, buffers: &[Buffer<u8>], t: u8) {
    let len = view.length as usize;
    if len <= View::MAX_INLINE_SIZE as usize {
        let raw = std::ptr::read(view as *const View as *const [u8; BLOCK]);
        let mut block = [0u8; BLOCK];
        block[..12].copy_from_slice(&raw[4..]);
        encode_short(dst, block, len, t);
    } else {
        let buffer = buffers.get_unchecked(view.buffer_idx as usize);
        let src = buffer.as_ptr().add(view.offset as usize);
        if len >= BLOCK {
            encode_long(dst, src, len, t);
        } else {
            let mut block = [0u8; BLOCK];
            std::ptr::copy_nonoverlapping(src, block.as_mut_ptr(), len);
            encode_short(dst, block, len, t);
        }
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
    let t = descending_mask(opt);
    let views = array.views().as_slice();
    let buffers = array.data_buffers().as_slice();
    let out = buffer.as_mut_ptr();

    match array.validity() {
        None => {
            for (offset, view) in offsets.iter_mut().zip(views) {
                encode_view(out.add(*offset), view, buffers, t);
                *offset += 1 + view.length as usize;
            }
        },
        Some(validity) => {
            for ((offset, view), is_valid) in offsets.iter_mut().zip(views).zip(validity.iter()) {
                if is_valid {
                    encode_view(out.add(*offset), view, buffers, t);
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
    let t = descending_mask(opt);
    let out = buffer.as_mut_ptr();

    for (offset, opt_value) in offsets.iter_mut().zip(input) {
        match opt_value {
            None => {
                *out.add(*offset) = MaybeUninit::new(null_sentinel);
                *offset += 1;
            },
            Some(s) => {
                encode_bytes(out.add(*offset), s.as_bytes(), t);
                *offset += 1 + s.len();
            },
        }
    }
}

/// Decode the value at the start of `row` into `scratch` and return its length. Nulls are not
/// handled here.
#[inline(always)]
unsafe fn decode_into_scratch(row: &[u8], t: u8, scratch: &mut Vec<u8>) -> usize {
    let term = t ^ 0x01;
    scratch.clear();

    let mut i = 0;
    while i + BLOCK <= row.len() {
        let block = std::ptr::read_unaligned(row.as_ptr().add(i) as *const [u8; BLOCK]);
        let end = find_byte(block, term);
        scratch.extend_from_slice(&decode_block(block, t)[..end]);
        if end < BLOCK {
            return i + end;
        }
        i += BLOCK;
    }
    while *row.get_unchecked(i) != term {
        scratch.push((*row.get_unchecked(i) ^ t).wrapping_sub(2));
        i += 1;
    }
    i
}

#[inline(always)]
fn decode_block(mut block: [u8; BLOCK], t: u8) -> [u8; BLOCK] {
    for b in &mut block {
        *b = (*b ^ t).wrapping_sub(2);
    }
    block
}

/// Decode the value at the start of `row` into `builder` and return its length. Nulls are not
/// handled here.
#[inline(always)]
unsafe fn decode_long(row: &[u8], t: u8, builder: &mut ViewBuilder) -> (usize, u32) {
    let term = t ^ 0x01;
    let mut dst = builder.start_value(BLOCK);
    let mut prefix = [0u8; 4];
    let mut i = 0;
    loop {
        if i + BLOCK <= row.len() {
            let block = std::ptr::read_unaligned(row.as_ptr().add(i) as *const [u8; BLOCK]);
            let end = find_byte(block, term);
            let decoded = decode_block(block, t);
            std::ptr::write_unaligned(dst.add(i) as *mut [u8; BLOCK], decoded);
            if i == 0 {
                prefix.copy_from_slice(&decoded[..4]);
            }
            i += end;
            if end < BLOCK {
                break;
            }
            dst = builder.grow_value(i, BLOCK);
        } else {
            dst = builder.grow_value(i, row.len() - i);
            while *row.get_unchecked(i) != term {
                let b = (*row.get_unchecked(i) ^ t).wrapping_sub(2);
                *dst.add(i) = b;
                if i < 4 {
                    prefix[i] = b;
                }
                i += 1;
            }
            break;
        }
    }
    (i, u32::from_le_bytes(prefix))
}

/// Decode one value and push it to `builder`. Returns `false` for null.
#[inline(always)]
unsafe fn decode_one(row: &mut &[u8], opt: RowEncodingOptions, builder: &mut ViewBuilder) -> bool {
    if *row.get_unchecked(0) == opt.null_sentinel() {
        *row = row.get_unchecked(1..);
        return false;
    }

    let t = descending_mask(opt);

    // Short values become inline views straight from a block load.
    if row.len() >= BLOCK {
        let block = std::ptr::read_unaligned(row.as_ptr() as *const [u8; BLOCK]);
        let len = find_byte(block, t ^ 0x01);
        if len <= View::MAX_INLINE_SIZE as usize {
            builder.push_inline(inline_view(decode_block(block, t), len));
            *row = row.get_unchecked(1 + len..);
            return true;
        }
    }

    let (len, prefix) = decode_long(row, t, builder);
    if len <= View::MAX_INLINE_SIZE as usize {
        // Only happens close to the end of the buffer.
        builder.finish_short_value(len);
    } else {
        builder.finish_value(len, prefix);
    }
    *row = row.get_unchecked(1 + len..);
    true
}

pub unsafe fn decode_str(rows: &mut [&[u8]], opt: RowEncodingOptions) -> Utf8ViewArray {
    let num_rows = rows.len();
    let mut builder = ViewBuilder::with_capacity(num_rows);
    let mut validity = BitmapBuilder::new();

    for row in rows.iter_mut() {
        if !decode_one(row, opt, &mut builder) {
            validity.reserve(num_rows);
            validity.extend_constant(builder.len(), true);
            validity.push(false);
            builder.push_null();
            break;
        }
    }

    if !validity.is_empty() {
        for row in rows[builder.len()..].iter_mut() {
            let is_valid = decode_one(row, opt, &mut builder);
            validity.push(is_valid);
            if !is_valid {
                builder.push_null();
            }
        }
    }

    let out = builder.freeze(ArrowDataType::Utf8View, validity.into_opt_validity());
    out.to_utf8view_unchecked()
}

/// The same as decode_str but inserts it into the given mapping, translating
/// it to physical type T.
pub unsafe fn decode_str_as_cat<T: NativeType + CatNative>(
    rows: &mut [&[u8]],
    opt: RowEncodingOptions,
    mapping: &CategoricalMapping,
) -> PrimitiveArray<T> {
    let null_sentinel = opt.null_sentinel();
    let t = descending_mask(opt);

    let num_rows = rows.len();
    let mut out = Vec::<T>::with_capacity(rows.len());

    let mut scratch = Vec::new();
    for row in rows.iter_mut() {
        let sentinel = *unsafe { row.get_unchecked(0) };
        if sentinel == null_sentinel {
            *row = unsafe { row.get_unchecked(1..) };
            break;
        }

        let len = decode_into_scratch(row, t, &mut scratch);
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

        let len = decode_into_scratch(row, t, &mut scratch);
        *row = row.get_unchecked(1 + len..);
        let s = unsafe { std::str::from_utf8_unchecked(&scratch) };
        out.push(T::from_cat(mapping.insert_cat(s).unwrap()));
    }

    PrimitiveArray::from_vec(out).with_validity(validity.into_opt_validity())
}
