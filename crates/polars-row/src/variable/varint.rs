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

pub(crate) trait VarIntEncoding: Pod {
    const IS_SIGNED: bool;
    const INLINE_MSB_THRESHOLD: usize = if Self::IS_SIGNED { 5 } else { 6 };

    #[inline(always)]
    fn msb_to_byte_length(msb: u32) -> usize {
        let msb = msb as usize;
        1 + if msb <= Self::INLINE_MSB_THRESHOLD {
            0
        } else {
            (msb + usize::from(Self::IS_SIGNED)).div_ceil(8)
        }
    }

    fn msb(self) -> u32;
    fn len_from_item(value: Option<Self>) -> usize {
        match value {
            None => 1,
            Some(v) => Self::msb_to_byte_length(v.msb()),
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

pub(crate) unsafe fn len_from_buffer(buffer: &[u8]) -> usize {
    let b = *buffer.get_unchecked(0);
    1 + if b & 0xC0 == 0xC0 {
        (b & 0x3F) as usize
    } else {
        0
    }
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
                        if msb as usize <= Self::INLINE_MSB_THRESHOLD {
                            let mut sentinel = null_sentinel ^ 0x80;
                            sentinel |= v.to_le_bytes()[0] & 0x3F;

                            if Self::IS_SIGNED {
                                // Flip sign bit
                                sentinel ^= 0x20;
                            }
                            if field.descending {
                                sentinel = !sentinel;
                            }

                            buffer[*offset] = MaybeUninit::new(sentinel);
                            *offset += 1;
                        } else {
                            let byte_length = Self::msb_to_byte_length(msb);
                            let additional_bytes = byte_length - 1;
                            debug_assert!(additional_bytes > 0 && additional_bytes <= 64);

                            let mut sentinel = null_sentinel ^ 0x80;
                            sentinel |= 0x40; // not inlined
                            sentinel |= (additional_bytes - 1) as u8;
                            buffer[*offset] = MaybeUninit::new(sentinel);

                            let bytes = v.to_be_bytes();
                            let bytes = &bytes.as_ref().as_uninit()[size_of::<Self>() - (byte_length - 1)..];
                            buffer[*offset + 1..*offset + byte_length].copy_from_slice( bytes);

                            if Self::IS_SIGNED {
                                // Flip sign bit
                                *buffer[*offset + 1].assume_init_mut() ^= 0x80;
                            }
                            if field.descending {
                                buffer[*offset + 1..*offset + byte_length]
                                    .iter_mut()
                                    .for_each(|v| *v = MaybeUninit::new(!*v.assume_init_ref()));
                            }

                            *offset += byte_length;
                        }
                    },
                }
            }

            unsafe fn decode_one(buffer: &mut &[u8], field: &EncodingField) -> Option<Self> {
                let null_sentinel = get_null_sentinel(field);

                let sentinel_byte = buffer[0];
                *buffer = &buffer[1..];

                if sentinel_byte == null_sentinel {
                    return None;
                }

                let sentinel_byte = if field.descending {
                    !sentinel_byte
                } else {
                    sentinel_byte
                };

                let is_inlined = sentinel_byte & 0x40 == 0;
                if is_inlined {
                    let mut value = (sentinel_byte & 0x3F) as Self;
                    if Self::IS_SIGNED {
                        // Flip sign bit
                        value ^= 0x20;

                        // Sign-extend
                        value <<= Self::BITS - 5;
                        value >>= Self::BITS - 5;
                    }
                    Some(value)
                } else {
                    let byte_length = (sentinel_byte & 0x3F) as usize + 1;

                    let mut intermediate = [0u8; size_of::<Self>()];
                    intermediate[size_of::<Self>() - byte_length..]
                        .copy_from_slice(&buffer[..byte_length]);
                    let mut v = Self::from_be_bytes(intermediate);

                    if Self::IS_SIGNED {
                        // Flip sign bit
                        v ^= 1 << (byte_length * 8 - 1);

                        // Sign-extend
                        v <<= (size_of::<Self>() - byte_length) * 8;
                        v >>= (size_of::<Self>() - byte_length) * 8;
                    }
                    if field.descending {
                        v = !v;
                    }

                    *buffer = &buffer[byte_length..];
                    Some(v)
                }
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

implement_varint![i32, u32, usize,];
