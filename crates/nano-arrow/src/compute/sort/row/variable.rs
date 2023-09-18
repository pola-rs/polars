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

use crate::compute::sort::SortOptions;

use super::{null_sentinel, Rows};

/// The block size of the variable length encoding
pub const BLOCK_SIZE: usize = 32;

/// The continuation token
pub const BLOCK_CONTINUATION: u8 = 0xFF;

/// Indicates an empty string
pub const EMPTY_SENTINEL: u8 = 1;

/// Indicates a non-empty string
pub const NON_EMPTY_SENTINEL: u8 = 2;

/// Returns the ceil of `value`/`divisor`
#[inline]
fn div_ceil(value: usize, divisor: usize) -> usize {
    // Rewrite as `value.div_ceil(&divisor)` after
    // https://github.com/rust-lang/rust/issues/88581 is merged.
    value / divisor + (0 != value % divisor) as usize
}

/// Returns the length of the encoded representation of a byte array, including the null byte
pub fn encoded_len(a: Option<&[u8]>) -> usize {
    match a {
        Some(a) => 1 + div_ceil(a.len(), BLOCK_SIZE) * (BLOCK_SIZE + 1),
        None => 1,
    }
}

/// Variable length values are encoded as
///
/// - single `0_u8` if null
/// - single `1_u8` if empty array
/// - `2_u8` if not empty, followed by one or more blocks
///
/// where a block is encoded as
///
/// - [`BLOCK_SIZE`] bytes of string data, padded with 0s
/// - `0xFF_u8` if this is not the last block for this string
/// - otherwise the length of the block as a `u8`
pub fn encode<'a, I: Iterator<Item = Option<&'a [u8]>>>(out: &mut Rows, i: I, opts: SortOptions) {
    for (offset, maybe_val) in out.offsets.iter_mut().skip(1).zip(i) {
        match maybe_val {
            Some(val) if val.is_empty() => {
                out.buffer[*offset] = match opts.descending {
                    true => !EMPTY_SENTINEL,
                    false => EMPTY_SENTINEL,
                };
                *offset += 1;
            }
            Some(val) => {
                let block_count = div_ceil(val.len(), BLOCK_SIZE);
                let end_offset = *offset + 1 + block_count * (BLOCK_SIZE + 1);
                let to_write = &mut out.buffer[*offset..end_offset];

                // Write `2_u8` to demarcate as non-empty, non-null string
                to_write[0] = NON_EMPTY_SENTINEL;

                let mut chunks = val.chunks_exact(BLOCK_SIZE);
                for (input, output) in chunks
                    .by_ref()
                    .zip(to_write[1..].chunks_exact_mut(BLOCK_SIZE + 1))
                {
                    let input: &[u8; BLOCK_SIZE] = input.try_into().unwrap();
                    let out_block: &mut [u8; BLOCK_SIZE] =
                        (&mut output[..BLOCK_SIZE]).try_into().unwrap();

                    *out_block = *input;

                    // Indicate that there are further blocks to follow
                    output[BLOCK_SIZE] = BLOCK_CONTINUATION;
                }

                let remainder = chunks.remainder();
                if !remainder.is_empty() {
                    let start_offset = 1 + (block_count - 1) * (BLOCK_SIZE + 1);
                    to_write[start_offset..start_offset + remainder.len()]
                        .copy_from_slice(remainder);
                    *to_write.last_mut().unwrap() = remainder.len() as u8;
                } else {
                    // We must overwrite the continuation marker written by the loop above
                    *to_write.last_mut().unwrap() = BLOCK_SIZE as u8;
                }

                *offset = end_offset;

                if opts.descending {
                    // Invert bits
                    to_write.iter_mut().for_each(|v| *v = !*v)
                }
            }
            None => {
                out.buffer[*offset] = null_sentinel(opts);
                *offset += 1;
            }
        }
    }
}
