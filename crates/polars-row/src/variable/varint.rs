//! A variable-length integer encoding for the `polars-row` encoding.
//!
//! This compresses integers close to 0 to less bytes than the fixed size encoding. This can save
//! quite a lot of memory if most of your integers are close to zero (e.g. with dictionary keys).
//!
//! The encoding works as follows.
//!
//! Each value starts with a *sentinel` byte. This byte is build up as follows.
//!
//! +-----------------------------+
//! | b7 : b6 : b5 b4 b3 b2 b1 b0 |
//! +-----------------------------+
//!
//! * `b7` encodes the validity of the element: `0` means `missing`, `1` means `valid`.
//! * `b6` determines the meaning of `b5` to `b0`.
//! * `b5` to `b0`:
//!   * `b6 == 0`, `b5` to `b0` is a 6-bit unsigned integer that encodes the value.
//!   * `b6 == 1`, `b5` to `b0` is a 6-bit unsigned integer that encodes the additional byte-length
//!     minus 1.
//!
//! If `b6 == 1`, the additional bytes encode the entirety of the value.
//!
//! Therefore, the following holds for byte sizes with 16-bit signed and unsigned integers.
//!
//! +-----------+------------------+-----------------+
//! | Byte Size | Signed Values    | Unsigned Values |
//! | 1         | -32 to 31        | 0 to 63         |
//! | 2         | -128 to -33 &    | 64 to 255       |
//! |           | 32 to 127        |                 |
//! | 3         | -32768 to -129 & | 256 to 65535    |
//! |           | 128 to 32767     |                 |
//! +-----------+------------------+-----------------+
//!
//! This can be extrapolated to other sizes of integers.
//!
//! Note 1. For signed integers the sign bit is flipped. This fixes the sort order. For example,
//! `0` would be smaller than `-1` without this.
//! Note 2. Given values represent the values for `descending=False` and `nulls_last=False`.
use std::mem::MaybeUninit;

use arrow::array::PrimitiveArray;
use arrow::types::NativeType;
use bytemuck::Pod;
use polars_utils::slice::Slice2Uninit;

use super::get_null_sentinel;
use crate::EncodingField;

pub(crate) trait VarIntEncoding: Pod + std::fmt::Debug {
    const IS_SIGNED: bool;

    const NUM_BYTELEN_BITS: usize = size_of::<Self>().trailing_zeros() as usize;
    const FIRST_BYTE_BITS: usize = 6 - (Self::IS_SIGNED as usize) - Self::NUM_BYTELEN_BITS;

    const INLINE_MSB_THRESHOLD: usize = if Self::IS_SIGNED { 5 } else { 6 };

    fn msb(self) -> u32;

    #[inline(always)]
    fn msb_to_byte_length(msb: u32) -> usize {
        let msb = msb as usize;
        let extra_bits = if msb > Self::FIRST_BYTE_BITS + Self::NUM_BYTELEN_BITS {
            msb - Self::FIRST_BYTE_BITS
        } else {
            0
        };
        let result = 1 + extra_bits.div_ceil(8);
        debug_assert!(result <= size_of::<Self>() + 1);
        result
    }
    fn len_from_item(value: Option<Self>) -> usize {
        match value {
            None => 1,
            Some(v) => Self::msb_to_byte_length(v.msb()),
        }
    }
    unsafe fn len_from_buffer(buffer: &[u8], field: &EncodingField) -> usize {
        let mut b = *buffer.get_unchecked(0);

        if b == get_null_sentinel(field) {
            return 1;
        }

        if field.descending {
            b = !b;
        }

        let is_inline = b & (1 << (6 + u32::from(!Self::IS_SIGNED))) == 0;
        let num_bytes = b >> (6 - usize::from(Self::IS_SIGNED) - Self::NUM_BYTELEN_BITS);
        let num_bytes = num_bytes & (1 << Self::NUM_BYTELEN_BITS) - 1;
        let mut num_bytes = num_bytes as usize;

        if Self::IS_SIGNED && b & 0x40 == 0 {
            num_bytes = !num_bytes;
        }

        debug_assert_ne!(num_bytes, (1 << Self::NUM_BYTELEN_BITS) - 1);
        if is_inline {
            1
        } else {
            1 + num_bytes
        }
    }

    unsafe fn encode_one(
        value: Option<Self>,
        buffer: &mut [MaybeUninit<u8>],
        offset: &mut usize,
        field: &EncodingField,
    );
    unsafe fn decode_one(buffer: &mut &[u8], field: &EncodingField) -> Option<Self>;
}

macro_rules! implement_varint {
    ($($t:ty,)+) => {
        $(
        impl VarIntEncoding for $t {
            #[allow(unused_comparisons)]
            const IS_SIGNED: bool = Self::MIN < 0;

            fn msb(self) -> u32 {
                let mut v = self;
                #[allow(unused_comparisons)]
                if Self::IS_SIGNED && v < 0 {
                    v = !v;
                }
                Self::BITS - v.leading_zeros()

            }

            unsafe fn encode_one(
                value: Option<Self>,
                buffer: &mut [MaybeUninit<u8>],
                offset: &mut usize,
                field: &EncodingField,
            ) {
                let null_sentinel = get_null_sentinel(field);

                match value {
                    None => {
                        buffer[*offset] = MaybeUninit::new(null_sentinel);
                        *offset += 1;
                    },
                    Some(v) => {
                        let msb = v.msb();
                        let bytelen = Self::msb_to_byte_length(msb);

                        let uses_all_bytes = bytelen > size_of::<Self>();

                        let bytes = v.to_be_bytes();
                        let bytes = bytes.as_ref();

                        buffer[*offset] = MaybeUninit::new(0);
                        buffer[*offset + usize::from(uses_all_bytes)..][..bytelen - usize::from(uses_all_bytes)].copy_from_slice(&bytes.as_uninit()[size_of::<Self>().saturating_sub(bytelen)..]);

                        let sentinel_value_bit_mask = if msb == 1 {
                            (1 << (Self::FIRST_BYTE_BITS + Self::NUM_BYTELEN_BITS)) - 1
                        } else {
                            (1 << Self::FIRST_BYTE_BITS) - 1
                        };

                        let sentinel_byte = unsafe { buffer[*offset].assume_init_mut() };

                        *sentinel_byte &= sentinel_value_bit_mask;
                        if Self::IS_SIGNED {
                            *sentinel_byte |= (!bytes[0] & 0x80) >> 1;
                        }
                        *sentinel_byte |= 0x80 & !null_sentinel;
                        *sentinel_byte |= u8::from(msb == 1) << (7 - usize::from(Self::IS_SIGNED));
                        let sentinel_bytelen = (bytelen - 1) as u8;
                        #[allow(unused_comparisons)]
                        let sentinel_bytelen = if Self::IS_SIGNED && v < 0 {
                            (!sentinel_bytelen) & ((1 << Self::NUM_BYTELEN_BITS) - 1)
                        } else {
                            sentinel_bytelen
                        };
                        *sentinel_byte |= sentinel_bytelen << (6 - usize::from(Self::IS_SIGNED) - Self::NUM_BYTELEN_BITS);

                        if field.descending {
                            *sentinel_byte ^= 0x7F;
                            for i in 1..bytelen {
                                *unsafe { buffer[*offset + i].assume_init_mut() } ^= 0xFF;
                            }
                        }
                        *offset += bytelen;
                    },
                }
            }

            unsafe fn decode_one(buffer: &mut &[u8], field: &EncodingField) -> Option<Self> {
                let null_sentinel = get_null_sentinel(field);

                if buffer[0] == null_sentinel {
                    return None;
                }

                let bytelen = Self::len_from_buffer(*buffer, field);

                let value = if bytelen <= size_of::<Self>() {
                    let mut intermediate = [0u8; size_of::<Self>()];
                    intermediate[size_of::<Self>() - bytelen..].copy_from_slice(&buffer[..bytelen]);

                    let first_byte_bits = if bytelen == 1 {
                        Self::FIRST_BYTE_BITS + Self::NUM_BYTELEN_BITS
                    } else {
                        Self::FIRST_BYTE_BITS
                    };

                    if Self::IS_SIGNED {
                        intermediate[size_of::<Self>() - bytelen] = (intermediate[size_of::<Self>() - bytelen] & ((1 << Self::FIRST_BYTE_BITS) - 1)) | ((!(intermediate[size_of::<Self>() - bytelen] & 0x40)) >> (1 + Self::NUM_BYTELEN_BITS));
                    } else {
                        intermediate[size_of::<Self>() - bytelen] = intermediate[size_of::<Self>() - bytelen] & ((1 << Self::FIRST_BYTE_BITS) - 1);
                    }

                    let mut value = Self::from_be_bytes(intermediate);

                    if Self::IS_SIGNED {
                        // Sign-extend
                        value <<= (8 * size_of::<Self>() - (bytelen - 1)) + 7 - first_byte_bits;
                        value >>= (8 * size_of::<Self>() - (bytelen - 1)) + 7 - first_byte_bits;
                    }

                    value
                } else {
                    Self::from_be_bytes(unsafe { buffer.get_unchecked(1..1 + size_of::<Self>()) }.try_into().unwrap())
                };

                *buffer = &buffer[bytelen..];
                Some(value)
            }
        }
        )+
    };
}

pub(crate) unsafe fn encode_iter<T: VarIntEncoding>(
    buffer: &mut [MaybeUninit<u8>],
    iter: impl Iterator<Item = Option<T>>,
    field: &EncodingField,
    offsets: &mut [usize],
) {
    for (opt_value, offset) in iter.zip(offsets) {
        T::encode_one(opt_value, buffer, offset, field);
    }
}

pub(crate) unsafe fn decode<T: VarIntEncoding + NativeType>(
    rows: &mut [&[u8]],
    field: &EncodingField,
) -> PrimitiveArray<T> {
    PrimitiveArray::from_iter(rows.iter_mut().map(|row| T::decode_one(row, field)))
}

implement_varint![i8, i16, i32, i64, i128, u8, u16, u32, u64, usize,];
