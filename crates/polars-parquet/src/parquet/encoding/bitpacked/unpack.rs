// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.
//
// Copied from https://github.com/apache/arrow-rs/blob/6859efa690d4c9530cf8a24053bc6ed81025a164/parquet/src/util/bit_pack.rs

// This implements bit unpacking. For example, for `u8` and `num_bits=3`.
// 0b001_101_110  -> 0b0000_0001, 0b0000_0101, 0b0000_0110
//
// This file is a bit insane. It unrolls all the possible num_bits vs. combinations. These are very
// highly used functions in Parquet and therefore this that been extensively unrolled and
// optimized. Attempts have been done to introduce SIMD here, but those attempts have not paid off
// in comparison to auto-vectorization.
//
// Generally, there are two code-size vs. runtime tradeoffs taken here in favor of
// runtime.
//
// 1. Each individual function unrolled to a point where all constants are known to
// the compiler. In microbenchmarks, this increases the performance by around 4.5
// to 5 times.
// 2. All functions are compiled separately and dispatch is done using a
// jumptable. In microbenchmarks, this increases the performance by around 2 to 2.5
// times.

/// Macro that generates an unpack function taking the number of bits as a const generic
macro_rules! unpack_impl {
    ($t:ty, $bytes:literal, $bits:tt) => {
        pub fn unpack<const NUM_BITS: usize>(input: &[u8], output: &mut [$t; $bits]) {
            if NUM_BITS == 0 {
                for out in output {
                    *out = 0;
                }
                return;
            }

            assert!(NUM_BITS <= $bytes * 8);

            let mask = match NUM_BITS {
                $bits => <$t>::MAX,
                _ => ((1 << NUM_BITS) - 1),
            };

            assert!(input.len() >= NUM_BITS * $bytes);

            let r = |output_idx: usize| {
                <$t>::from_le_bytes(
                    input[output_idx * $bytes..output_idx * $bytes + $bytes]
                        .try_into()
                        .unwrap(),
                )
            };

            // @NOTE
            // I was surprised too, but this macro vs. a for loop saves around 4.5 - 5x on
            // performance in a microbenchmark. Although the code it generates is completely
            // insane. There should be something we can do here to make this less code, sane code
            // and faster code.
            seq_macro!(i in 0..$bits {
                let start_bit = i * NUM_BITS;
                let end_bit = start_bit + NUM_BITS;

                let start_bit_offset = start_bit % $bits;
                let end_bit_offset = end_bit % $bits;
                let start_byte = start_bit / $bits;
                let end_byte = end_bit / $bits;
                if start_byte != end_byte && end_bit_offset != 0 {
                    let val = r(start_byte);
                    let a = val >> start_bit_offset;
                    let val = r(end_byte);
                    let b = val << (NUM_BITS - end_bit_offset);

                    output[i] = a | (b & mask);
                } else {
                    let val = r(start_byte);
                    output[i] = (val >> start_bit_offset) & mask;
                }
            });
        }
    };
}

/// Macro that generates unpack functions that accept num_bits as a parameter
macro_rules! unpack {
    ($name:ident, $t:ty, $bytes:literal, $bits:tt) => {
        mod $name {
            unpack_impl!($t, $bytes, $bits);
        }

        /// Unpack packed `input` into `output` with a bit width of `num_bits`
        pub fn $name(input: &[u8], output: &mut [$t; $bits], num_bits: usize) {
            // This will get optimised into a jump table
            //
            // @NOTE
            // This jumptable appoach saves around 2 - 2.5x on performance over no jumptable and no
            // generics.
            seq_macro!(i in 0..=$bits {
                if i == num_bits {
                    return $name::unpack::<i>(input, output);
                }
            });
            unreachable!("invalid num_bits {}", num_bits);
        }
    };
}

unpack!(unpack32, u32, 4, 32);
unpack!(unpack64, u64, 8, 64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        let input = [0xFF; 4096];

        for i in 0..=32 {
            let mut output = [0; 32];
            unpack32(&input, &mut output, i);
            for (idx, out) in output.iter().enumerate() {
                assert_eq!(out.trailing_ones() as usize, i, "out[{}] = {}", idx, out);
            }
        }

        for i in 0..=64 {
            let mut output = [0; 64];
            unpack64(&input, &mut output, i);
            for (idx, out) in output.iter().enumerate() {
                assert_eq!(out.trailing_ones() as usize, i, "out[{}] = {}", idx, out);
            }
        }
    }
}
