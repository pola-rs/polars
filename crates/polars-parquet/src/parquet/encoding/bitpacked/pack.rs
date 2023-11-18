/// Macro that generates a packing function taking the number of bits as a const generic
macro_rules! pack_impl {
    ($t:ty, $bytes:literal, $bits:tt) => {
        pub fn pack<const NUM_BITS: usize>(input: &[$t; $bits], output: &mut [u8]) {
            if NUM_BITS == 0 {
                for out in output {
                    *out = 0;
                }
                return;
            }
            assert!(NUM_BITS <= $bytes * 8);
            assert!(output.len() >= NUM_BITS * $bytes);

            let mask = match NUM_BITS {
                $bits => <$t>::MAX,
                _ => ((1 << NUM_BITS) - 1),
            };

            for i in 0..$bits {
                let start_bit = i * NUM_BITS;
                let end_bit = start_bit + NUM_BITS;

                let start_bit_offset = start_bit % $bits;
                let end_bit_offset = end_bit % $bits;
                let start_byte = start_bit / $bits;
                let end_byte = end_bit / $bits;
                if start_byte != end_byte && end_bit_offset != 0 {
                    let a = input[i] << start_bit_offset;
                    let val_a = <$t>::to_le_bytes(a);
                    for i in 0..$bytes {
                        output[start_byte * $bytes + i] |= val_a[i]
                    }

                    let b = (input[i] >> (NUM_BITS - end_bit_offset)) & mask;
                    let val_b = <$t>::to_le_bytes(b);
                    for i in 0..$bytes {
                        output[end_byte * $bytes + i] |= val_b[i]
                    }
                } else {
                    let val = (input[i] & mask) << start_bit_offset;
                    let val = <$t>::to_le_bytes(val);

                    for i in 0..$bytes {
                        output[start_byte * $bytes + i] |= val[i]
                    }
                }
            }
        }
    };
}

/// Macro that generates pack functions that accept num_bits as a parameter
macro_rules! pack {
    ($name:ident, $t:ty, $bytes:literal, $bits:tt) => {
        mod $name {
            pack_impl!($t, $bytes, $bits);
        }

        /// Pack unpacked `input` into `output` with a bit width of `num_bits`
        pub fn $name(input: &[$t; $bits], output: &mut [u8], num_bits: usize) {
            // This will get optimised into a jump table
            seq_macro::seq!(i in 0..=$bits {
                if i == num_bits {
                    return $name::pack::<i>(input, output);
                }
            });
            unreachable!("invalid num_bits {}", num_bits);
        }
    };
}

pack!(pack8, u8, 1, 8);
pack!(pack16, u16, 2, 16);
pack!(pack32, u32, 4, 32);
pack!(pack64, u64, 8, 64);

#[cfg(test)]
mod tests {
    use super::super::unpack::*;
    use super::*;

    #[test]
    fn test_basic() {
        let input = [0u16, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        for num_bits in 4..16 {
            let mut output = [0u8; 16 * 2];
            pack16(&input, &mut output, num_bits);
            let mut other = [0u16; 16];
            unpack16(&output, &mut other, num_bits);
            assert_eq!(other, input);
        }
    }

    #[test]
    fn test_u32() {
        let input = [
            0u32, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0u32, 1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, 14, 15,
        ];
        for num_bits in 4..32 {
            let mut output = [0u8; 32 * 4];
            pack32(&input, &mut output, num_bits);
            let mut other = [0u32; 32];
            unpack32(&output, &mut other, num_bits);
            assert_eq!(other, input);
        }
    }
}
