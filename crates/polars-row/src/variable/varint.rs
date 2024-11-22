use std::mem::MaybeUninit;

use arrow::array::PrimitiveArray;
use arrow::types::NativeType;
use bytemuck::Pod;
use polars_utils::slice::Slice2Uninit;

use super::get_null_sentinel;
use crate::EncodingField;

pub(crate) trait VarIntEncoding: Pod {
    const IS_SIGNED: bool;
    const INLINE_MASK: u8 = if Self::IS_SIGNED { 0x1F } else { 0x3F };
    const INLINE_THRESHOLD: usize = Self::INLINE_MASK.count_ones() as usize;

    #[inline(always)]
    fn msb_to_byte_length(msb: u32) -> usize {
        let msb = msb as usize;
        1 + if msb <= Self::INLINE_THRESHOLD {
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
                match value {
                    None => {
                        buffer[*offset] = MaybeUninit::new(get_null_sentinel(field));
                        *offset += 1;
                    },
                    Some(v) => {
                        let msb = v.msb();
                        if msb as usize <= Self::INLINE_THRESHOLD {
                            let mut b = v.to_le_bytes()[0] & 0x3F;

                            #[allow(unused_comparisons)]
                            if Self::IS_SIGNED {
                                b ^= 0x20;
                            }

                            b |= 0x80;

                            if field.descending {
                                b = !b;
                            }

                            buffer[*offset] = MaybeUninit::new(b);
                            *offset += 1;
                        } else {
                            let byte_length = Self::msb_to_byte_length(msb);
                            debug_assert!(byte_length < 64);
                            buffer[*offset] =
                                MaybeUninit::new(0xC0 | (((byte_length - 2) & 0x3F) as u8));

                            buffer[*offset + 1..*offset + byte_length].copy_from_slice(
                                &v.to_be_bytes().as_ref().as_uninit()
                                    [size_of::<Self>() - (byte_length - 1)..],
                            );

                            if Self::IS_SIGNED {
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

                if sentinel_byte & 0x40 == 0 {
                    // Inlined
                    let sentinel_byte = if field.descending {
                        !sentinel_byte
                    } else {
                        sentinel_byte
                    };

                    let mut value = (sentinel_byte & 0x3F) as Self;
                    if Self::IS_SIGNED {
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
