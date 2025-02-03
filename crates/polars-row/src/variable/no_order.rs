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

use arrow::array::{BinaryViewArray, MutableBinaryViewArray};
use arrow::bitmap::BitmapBuilder;
use polars_utils::slice::Slice2Uninit;

use crate::row::RowEncodingOptions;

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

pub unsafe fn encode_variable_no_order<'a, I: Iterator<Item = Option<&'a [u8]>>>(
    buffer: &mut [MaybeUninit<u8>],
    input: I,
    opt: RowEncodingOptions,
    offsets: &mut [usize],
) {
    debug_assert!(opt.contains(RowEncodingOptions::NO_ORDER));

    for (offset, opt_value) in offsets.iter_mut().zip(input) {
        let buffer = unsafe { buffer.get_unchecked_mut(*offset..) };
        match opt_value {
            None => {
                *unsafe { buffer.get_unchecked_mut(0) } = MaybeUninit::new(0xFF);
                *offset += 1;
            },
            Some(v) => {
                if v.len() >= 254 {
                    unsafe {
                        *buffer.get_unchecked_mut(0) = MaybeUninit::new(0xFE);
                        buffer
                            .get_unchecked_mut(1..5)
                            .copy_from_slice((v.len() as u32).to_le_bytes().as_uninit());
                        buffer
                            .get_unchecked_mut(5..5 + v.len())
                            .copy_from_slice(v.as_uninit());
                    }
                    *offset += 5 + v.len();
                } else {
                    unsafe {
                        *buffer.get_unchecked_mut(0) = MaybeUninit::new(v.len() as u8);
                        buffer
                            .get_unchecked_mut(1..1 + v.len())
                            .copy_from_slice(v.as_uninit());
                    }
                    *offset += 1 + v.len();
                }
            },
        }
    }
}

pub unsafe fn decode_variable_no_order(
    rows: &mut [&[u8]],
    opt: RowEncodingOptions,
) -> BinaryViewArray {
    debug_assert!(opt.contains(RowEncodingOptions::NO_ORDER));

    let num_rows = rows.len();
    let mut array = MutableBinaryViewArray::<[u8]>::with_capacity(num_rows);
    let mut validity = BitmapBuilder::new();

    for row in rows.iter_mut() {
        let sentinel = *unsafe { row.get_unchecked(0) };
        *row = unsafe { row.get_unchecked(1..) };
        if sentinel == 0xFF {
            validity.reserve(num_rows);
            validity.extend_constant(array.len(), true);
            validity.push(false);
            array.push_value_ignore_validity("");
            break;
        }

        let length = if sentinel < 0xFE {
            sentinel as usize
        } else {
            let length = u32::from_le_bytes(unsafe { row.get_unchecked(..4) }.try_into().unwrap());
            *row = unsafe { row.get_unchecked(4..) };
            length as usize
        };

        array.push_value_ignore_validity(unsafe { row.get_unchecked(..length) });
        *row = unsafe { row.get_unchecked(length..) };
    }

    if validity.is_empty() {
        return array.into();
    }

    for row in rows[array.len()..].iter_mut() {
        let sentinel = *unsafe { row.get_unchecked(0) };
        *row = unsafe { row.get_unchecked(1..) };

        validity.push(sentinel != 0xFF);
        if sentinel == 0xFF {
            array.push_value_ignore_validity("");
            break;
        }

        let length = if sentinel < 0xFE {
            sentinel as usize
        } else {
            let length = u32::from_le_bytes(unsafe { row.get_unchecked(..4) }.try_into().unwrap());
            *row = unsafe { row.get_unchecked(4..) };
            length as usize
        };

        array.push_value_ignore_validity(unsafe { row.get_unchecked(..length) });
        *row = unsafe { row.get_unchecked(length..) };
    }

    let array = array.freeze();
    array.with_validity(validity.into_opt_validity())
}
