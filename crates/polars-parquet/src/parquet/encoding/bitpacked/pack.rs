/// Macro that generates a packing function taking the number of bits as a const generic
macro_rules! pack_impl {
    ($t:ty, $bytes:literal, $bits:tt, $bits_minus_one:tt) => {
        // Adapted from https://github.com/quickwit-oss/bitpacking
        pub unsafe fn pack<const NUM_BITS: usize>(input: &[$t; $bits], output: &mut [u8]) {
            if NUM_BITS == 0 {
                for out in output {
                    *out = 0;
                }
                return;
            }
            assert!(NUM_BITS <= $bits);
            assert!(output.len() >= NUM_BITS * $bytes);

            let input_ptr = input.as_ptr();
            let mut output_ptr = output.as_mut_ptr() as *mut $t;
            let mut out_register: $t = read_unaligned(input_ptr);

            if $bits == NUM_BITS {
                write_unaligned(output_ptr, out_register);
                output_ptr = output_ptr.offset(1);
            }

            // Using microbenchmark (79d1fff), unrolling this loop is over 10x
            // faster than not (>20x faster than old algorithm)
            seq_macro!(i in 1..$bits_minus_one {
                let bits_filled: usize = i * NUM_BITS;
                let inner_cursor: usize = bits_filled % $bits;
                let remaining: usize = $bits - inner_cursor;

                let offset_ptr = input_ptr.add(i);
                let in_register: $t = read_unaligned(offset_ptr);

                out_register =
                    if inner_cursor > 0 {
                        out_register | (in_register << inner_cursor)
                    } else {
                        in_register
                    };

                if remaining <= NUM_BITS {
                    write_unaligned(output_ptr, out_register);
                    output_ptr = output_ptr.offset(1);
                    if 0 < remaining && remaining < NUM_BITS {
                        out_register = in_register >> remaining
                    }
                }
            });

            let in_register: $t = read_unaligned(input_ptr.add($bits - 1));
            out_register = if $bits - NUM_BITS > 0 {
                out_register | (in_register << ($bits - NUM_BITS))
            } else {
                in_register
            };
            write_unaligned(output_ptr, out_register)
        }
    };
}

/// Macro that generates pack functions that accept num_bits as a parameter
macro_rules! pack {
    ($name:ident, $t:ty, $bytes:literal, $bits:tt, $bits_minus_one:tt) => {
        mod $name {
            use std::ptr::{read_unaligned, write_unaligned};
            pack_impl!($t, $bytes, $bits, $bits_minus_one);
        }

        /// Pack unpacked `input` into `output` with a bit width of `num_bits`
        pub fn $name(input: &[$t; $bits], output: &mut [u8], num_bits: usize) {
            // This will get optimised into a jump table
            seq_macro!(i in 0..=$bits {
                if i == num_bits {
                    unsafe {
                        return $name::pack::<i>(input, output);
                    }
                }
            });
            unreachable!("invalid num_bits {}", num_bits);
        }
    };
}

pack!(pack32, u32, 4, 32, 31);
pack!(pack64, u64, 8, 64, 63);

#[cfg(test)]
mod tests {
    use rand::distributions::{Distribution, Uniform};

    use super::super::unpack::*;
    use super::*;

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

    #[test]
    fn test_u32_random() {
        let mut rng = rand::thread_rng();
        let mut random_array = [0u32; 32];
        let between = Uniform::from(0..131_072);
        for num_bits in 17..=32 {
            for i in &mut random_array {
                *i = between.sample(&mut rng);
            }
            let mut output = [0u8; 32 * 4];
            pack32(&random_array, &mut output, num_bits);
            let mut other = [0u32; 32];
            unpack32(&output, &mut other, num_bits);
            assert_eq!(other, random_array);
        }
    }

    #[test]
    fn test_u64_random() {
        let mut rng = rand::thread_rng();
        let mut random_array = [0u64; 64];
        let between = Uniform::from(0..131_072);
        for num_bits in 17..=64 {
            for i in &mut random_array {
                *i = between.sample(&mut rng);
            }
            let mut output = [0u8; 64 * 8];
            pack64(&random_array, &mut output, num_bits);
            let mut other = [0u64; 64];
            unpack64(&output, &mut other, num_bits);
            assert_eq!(other, random_array);
        }
    }
}
