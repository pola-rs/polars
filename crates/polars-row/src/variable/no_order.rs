#![allow(unsafe_op_in_unsafe_fn)]
/// Row encoding for variable width elements without maintaining order.
///
/// Each element is prepended by a sentinel value.
///
/// If the sentinel value is:
/// - 0xFF: the element is None
/// - 0xFE: the element's length is encoded as 4 LE bytes following the sentinel
/// - 0x00 - 0xFD: the element's length is the sentinel value
///
/// After the sentinel value (and possible length), the data is then given.
use std::mem::MaybeUninit;

use polars_arrow::array::builder::StaticArrayBuilder;
use polars_arrow::array::{
    Array, BinaryViewArray, BinaryViewArrayBuilder, BinaryViewArrayGeneric, View, ViewType,
};
use polars_arrow::bitmap::BitmapBuilder;
use polars_arrow::datatypes::ArrowDataType;
use polars_buffer::Buffer;

use super::BLOCK_SIZE;
use crate::row::RowEncodingOptions;

#[inline(always)]
pub fn len_from_item(value: Option<usize>, opt: RowEncodingOptions) -> usize {
    debug_assert!(opt.contains(RowEncodingOptions::NO_ORDER));

    match value {
        None => 1,
        Some(l) if l < 254 => l + 1,
        Some(l) => l + 5,
    }
}

pub unsafe fn len_from_buffer(buffer: &[u8], opt: RowEncodingOptions) -> usize {
    debug_assert!(opt.contains(RowEncodingOptions::NO_ORDER));

    let sentinel = *unsafe { buffer.get_unchecked(0) };

    match sentinel {
        0xFF => 1,
        0xFE => {
            5 + u32::from_le_bytes(unsafe { buffer.get_unchecked(1..5) }.try_into().unwrap())
                as usize
        },
        length => 1 + length as usize,
    }
}

/// Write the `n` bytes at `src` to `dst` with two overlapping stores.
///
/// # Safety
/// `src` must be readable and `dst` writable for `max(n, 4)` bytes when `n >= 4`, else `n`.
#[inline(always)]
unsafe fn write_short(dst: *mut u8, src: *const u8, n: usize) {
    debug_assert!(n <= BLOCK_SIZE);
    if n >= 8 {
        let lo = std::ptr::read_unaligned(src as *const u64);
        let hi = std::ptr::read_unaligned(src.add(n - 8) as *const u64);
        std::ptr::write_unaligned(dst as *mut u64, lo);
        std::ptr::write_unaligned(dst.add(n - 8) as *mut u64, hi);
    } else if n >= 4 {
        let lo = std::ptr::read_unaligned(src as *const u32);
        let hi = std::ptr::read_unaligned(src.add(n - 4) as *const u32);
        std::ptr::write_unaligned(dst as *mut u32, lo);
        std::ptr::write_unaligned(dst.add(n - 4) as *mut u32, hi);
    } else {
        for i in 0..n {
            *dst.add(i) = *src.add(i);
        }
    }
}

/// Write the header and `len` bytes at `src` to `dst`. Returns the number of bytes written.
///
/// # Safety
/// `src` must be readable for `max(len, 16)` bytes and `dst` writable for the returned length.
#[inline(always)]
unsafe fn encode_bytes(dst: *mut u8, src: *const u8, len: usize) -> usize {
    let header = if len >= 254 {
        *dst = 0xFE;
        std::ptr::write_unaligned(dst.add(1) as *mut u32, len as u32);
        5
    } else {
        *dst = len as u8;
        1
    };
    let dst = dst.add(header);

    if len >= BLOCK_SIZE {
        std::ptr::copy_nonoverlapping(src, dst, len);
    } else {
        write_short(dst, src, len);
    }
    header + len
}

/// # Safety
/// `dst` must be writable for the encoded length and `view` must point into `buffers`.
#[inline(always)]
unsafe fn encode_view(dst: *mut u8, view: &View, buffers: &[Buffer<u8>]) -> usize {
    let len = view.length as usize;
    if len <= View::MAX_INLINE_SIZE as usize {
        // The view is [len: u32 little-endian][12 inline bytes]. Byte 3 of the length is zero
        // here, so the encoding is the view from byte 3 with `len` put in that byte.
        let src = (view as *const View as *const u8).add(3);
        let n = len + 1;
        if n >= 8 {
            let lo = std::ptr::read_unaligned(src as *const u64) | len as u64;
            let hi = std::ptr::read_unaligned(src.add(n - 8) as *const u64)
                | if n == 8 { len as u64 } else { 0 };
            std::ptr::write_unaligned(dst as *mut u64, lo);
            std::ptr::write_unaligned(dst.add(n - 8) as *mut u64, hi);
        } else if n >= 4 {
            let lo = std::ptr::read_unaligned(src as *const u32) | len as u32;
            let hi = std::ptr::read_unaligned(src.add(n - 4) as *const u32)
                | if n == 4 { len as u32 } else { 0 };
            std::ptr::write_unaligned(dst as *mut u32, lo);
            std::ptr::write_unaligned(dst.add(n - 4) as *mut u32, hi);
        } else {
            *dst = len as u8;
            for i in 1..n {
                *dst.add(i) = *src.add(i);
            }
        }
        n
    } else {
        let src = view.get_external_slice_unchecked(buffers).as_ptr();
        encode_bytes(dst, src, len)
    }
}

pub unsafe fn encode_view_no_order<T: ViewType + ?Sized>(
    buffer: &mut [MaybeUninit<u8>],
    array: &BinaryViewArrayGeneric<T>,
    opt: RowEncodingOptions,
    offsets: &mut [usize],
) {
    debug_assert!(opt.contains(RowEncodingOptions::NO_ORDER));
    let views = array.views().as_slice();
    let buffers = array.data_buffers().as_slice();
    let out = buffer.as_mut_ptr() as *mut u8;

    match array.validity() {
        None => {
            for (offset, view) in offsets.iter_mut().zip(views) {
                *offset += encode_view(out.add(*offset), view, buffers);
            }
        },
        Some(validity) => {
            for ((offset, view), is_valid) in offsets.iter_mut().zip(views).zip(validity.iter()) {
                if is_valid {
                    *offset += encode_view(out.add(*offset), view, buffers);
                } else {
                    *out.add(*offset) = 0xFF;
                    *offset += 1;
                }
            }
        },
    }
}

pub unsafe fn encode_variable_no_order<'a, I: Iterator<Item = Option<&'a [u8]>>>(
    buffer: &mut [MaybeUninit<u8>],
    input: I,
    opt: RowEncodingOptions,
    offsets: &mut [usize],
) {
    debug_assert!(opt.contains(RowEncodingOptions::NO_ORDER));
    let out = buffer.as_mut_ptr() as *mut u8;

    for (offset, opt_value) in offsets.iter_mut().zip(input) {
        match opt_value {
            None => {
                *out.add(*offset) = 0xFF;
                *offset += 1;
            },
            Some(v) => {
                if v.len() >= BLOCK_SIZE {
                    *offset += encode_bytes(out.add(*offset), v.as_ptr(), v.len());
                } else {
                    let mut block = [0u8; BLOCK_SIZE];
                    std::ptr::copy_nonoverlapping(v.as_ptr(), block.as_mut_ptr(), v.len());
                    *offset += encode_bytes(out.add(*offset), block.as_ptr(), v.len());
                }
            },
        }
    }
}

/// Decode one value and push it to `builder`. Returns `false` for null.
///
/// # Safety
/// `row` must start with a null sentinel or an encoded value.
#[inline(always)]
unsafe fn decode_one(row: &mut &[u8], builder: &mut BinaryViewArrayBuilder) -> bool {
    let sentinel = *row.get_unchecked(0);
    if sentinel == 0xFF {
        *row = row.get_unchecked(1..);
        return false;
    }

    // Short values become inline views straight from a block load.
    if sentinel <= View::MAX_INLINE_SIZE as u8 && row.len() > BLOCK_SIZE {
        let len = sentinel as usize;
        let block = std::ptr::read_unaligned(row.as_ptr().add(1) as *const [u8; BLOCK_SIZE]);
        builder.push_inline_view_ignore_validity(View::new_inline_from_block(block, len));
        *row = row.get_unchecked(1 + len..);
        return true;
    }

    let (header, length) = if sentinel < 0xFE {
        (1, sentinel as usize)
    } else {
        let length = u32::from_le_bytes(row.get_unchecked(1..5).try_into().unwrap());
        (5, length as usize)
    };
    let end = header + length;
    builder.push_value_ignore_validity(row.get_unchecked(header..end));
    *row = row.get_unchecked(end..);
    true
}

pub unsafe fn decode_variable_no_order(
    rows: &mut [&[u8]],
    opt: RowEncodingOptions,
) -> BinaryViewArray {
    debug_assert!(opt.contains(RowEncodingOptions::NO_ORDER));

    let num_rows = rows.len();
    let mut builder = BinaryViewArrayBuilder::new(ArrowDataType::BinaryView);
    builder.reserve(num_rows);
    let mut validity = BitmapBuilder::new();

    for row in rows.iter_mut() {
        if !decode_one(row, &mut builder) {
            validity.reserve(num_rows);
            validity.extend_constant(builder.len(), true);
            validity.push(false);
            builder.push_null_ignore_validity();
            break;
        }
    }

    if !validity.is_empty() {
        for row in rows[builder.len()..].iter_mut() {
            let is_valid = decode_one(row, &mut builder);
            validity.push(is_valid);
            if !is_valid {
                builder.push_null_ignore_validity();
            }
        }
    }

    builder.freeze_with_validity(validity.into_opt_validity())
}
