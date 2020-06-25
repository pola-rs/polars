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

/// Unpack 32 values with bit width `num_bits` from `in_ptr`, and write to `out_ptr`.
/// Return the `in_ptr` where the starting offset points to the first byte after all the
/// bytes that were consumed.
// TODO: may be better to make these more compact using if-else conditions.
//  However, this may require const generics:
//     https://github.com/rust-lang/rust/issues/44580
//  to eliminate the branching cost.
// TODO: we should use SIMD instructions to further optimize this. I have explored
//    https://github.com/tantivy-search/bitpacking
// but the layout it uses for SIMD is different from Parquet.
// TODO: support packing as well, which is used for encoding.
pub unsafe fn unpack32(
    mut in_ptr: *const u32,
    out_ptr: *mut u32,
    num_bits: usize,
) -> *const u32 {
    in_ptr = match num_bits {
        0 => nullunpacker32(in_ptr, out_ptr),
        1 => unpack1_32(in_ptr, out_ptr),
        2 => unpack2_32(in_ptr, out_ptr),
        3 => unpack3_32(in_ptr, out_ptr),
        4 => unpack4_32(in_ptr, out_ptr),
        5 => unpack5_32(in_ptr, out_ptr),
        6 => unpack6_32(in_ptr, out_ptr),
        7 => unpack7_32(in_ptr, out_ptr),
        8 => unpack8_32(in_ptr, out_ptr),
        9 => unpack9_32(in_ptr, out_ptr),
        10 => unpack10_32(in_ptr, out_ptr),
        11 => unpack11_32(in_ptr, out_ptr),
        12 => unpack12_32(in_ptr, out_ptr),
        13 => unpack13_32(in_ptr, out_ptr),
        14 => unpack14_32(in_ptr, out_ptr),
        15 => unpack15_32(in_ptr, out_ptr),
        16 => unpack16_32(in_ptr, out_ptr),
        17 => unpack17_32(in_ptr, out_ptr),
        18 => unpack18_32(in_ptr, out_ptr),
        19 => unpack19_32(in_ptr, out_ptr),
        20 => unpack20_32(in_ptr, out_ptr),
        21 => unpack21_32(in_ptr, out_ptr),
        22 => unpack22_32(in_ptr, out_ptr),
        23 => unpack23_32(in_ptr, out_ptr),
        24 => unpack24_32(in_ptr, out_ptr),
        25 => unpack25_32(in_ptr, out_ptr),
        26 => unpack26_32(in_ptr, out_ptr),
        27 => unpack27_32(in_ptr, out_ptr),
        28 => unpack28_32(in_ptr, out_ptr),
        29 => unpack29_32(in_ptr, out_ptr),
        30 => unpack30_32(in_ptr, out_ptr),
        31 => unpack31_32(in_ptr, out_ptr),
        32 => unpack32_32(in_ptr, out_ptr),
        _ => unimplemented!(),
    };
    in_ptr
}

unsafe fn nullunpacker32(in_buf: *const u32, mut out: *mut u32) -> *const u32 {
    for _ in 0..32 {
        *out = 0;
        out = out.offset(1);
    }
    in_buf
}

unsafe fn unpack1_32(in_buf: *const u32, mut out: *mut u32) -> *const u32 {
    *out = ((*in_buf) >> 0) & 1;
    out = out.offset(1);
    *out = ((*in_buf) >> 1) & 1;
    out = out.offset(1);
    *out = ((*in_buf) >> 2) & 1;
    out = out.offset(1);
    *out = ((*in_buf) >> 3) & 1;
    out = out.offset(1);
    *out = ((*in_buf) >> 4) & 1;
    out = out.offset(1);
    *out = ((*in_buf) >> 5) & 1;
    out = out.offset(1);
    *out = ((*in_buf) >> 6) & 1;
    out = out.offset(1);
    *out = ((*in_buf) >> 7) & 1;
    out = out.offset(1);
    *out = ((*in_buf) >> 8) & 1;
    out = out.offset(1);
    *out = ((*in_buf) >> 9) & 1;
    out = out.offset(1);
    *out = ((*in_buf) >> 10) & 1;
    out = out.offset(1);
    *out = ((*in_buf) >> 11) & 1;
    out = out.offset(1);
    *out = ((*in_buf) >> 12) & 1;
    out = out.offset(1);
    *out = ((*in_buf) >> 13) & 1;
    out = out.offset(1);
    *out = ((*in_buf) >> 14) & 1;
    out = out.offset(1);
    *out = ((*in_buf) >> 15) & 1;
    out = out.offset(1);
    *out = ((*in_buf) >> 16) & 1;
    out = out.offset(1);
    *out = ((*in_buf) >> 17) & 1;
    out = out.offset(1);
    *out = ((*in_buf) >> 18) & 1;
    out = out.offset(1);
    *out = ((*in_buf) >> 19) & 1;
    out = out.offset(1);
    *out = ((*in_buf) >> 20) & 1;
    out = out.offset(1);
    *out = ((*in_buf) >> 21) & 1;
    out = out.offset(1);
    *out = ((*in_buf) >> 22) & 1;
    out = out.offset(1);
    *out = ((*in_buf) >> 23) & 1;
    out = out.offset(1);
    *out = ((*in_buf) >> 24) & 1;
    out = out.offset(1);
    *out = ((*in_buf) >> 25) & 1;
    out = out.offset(1);
    *out = ((*in_buf) >> 26) & 1;
    out = out.offset(1);
    *out = ((*in_buf) >> 27) & 1;
    out = out.offset(1);
    *out = ((*in_buf) >> 28) & 1;
    out = out.offset(1);
    *out = ((*in_buf) >> 29) & 1;
    out = out.offset(1);
    *out = ((*in_buf) >> 30) & 1;
    out = out.offset(1);
    *out = (*in_buf) >> 31;

    in_buf.offset(1)
}

unsafe fn unpack2_32(mut in_buf: *const u32, mut out: *mut u32) -> *const u32 {
    *out = ((*in_buf) >> 0) % (1u32 << 2);
    out = out.offset(1);
    *out = ((*in_buf) >> 2) % (1u32 << 2);
    out = out.offset(1);
    *out = ((*in_buf) >> 4) % (1u32 << 2);
    out = out.offset(1);
    *out = ((*in_buf) >> 6) % (1u32 << 2);
    out = out.offset(1);
    *out = ((*in_buf) >> 8) % (1u32 << 2);
    out = out.offset(1);
    *out = ((*in_buf) >> 10) % (1u32 << 2);
    out = out.offset(1);
    *out = ((*in_buf) >> 12) % (1u32 << 2);
    out = out.offset(1);
    *out = ((*in_buf) >> 14) % (1u32 << 2);
    out = out.offset(1);
    *out = ((*in_buf) >> 16) % (1u32 << 2);
    out = out.offset(1);
    *out = ((*in_buf) >> 18) % (1u32 << 2);
    out = out.offset(1);
    *out = ((*in_buf) >> 20) % (1u32 << 2);
    out = out.offset(1);
    *out = ((*in_buf) >> 22) % (1u32 << 2);
    out = out.offset(1);
    *out = ((*in_buf) >> 24) % (1u32 << 2);
    out = out.offset(1);
    *out = ((*in_buf) >> 26) % (1u32 << 2);
    out = out.offset(1);
    *out = ((*in_buf) >> 28) % (1u32 << 2);
    out = out.offset(1);
    *out = (*in_buf) >> 30;
    out = out.offset(1);
    in_buf = in_buf.offset(1);
    *out = ((*in_buf) >> 0) % (1u32 << 2);
    out = out.offset(1);
    *out = ((*in_buf) >> 2) % (1u32 << 2);
    out = out.offset(1);
    *out = ((*in_buf) >> 4) % (1u32 << 2);
    out = out.offset(1);
    *out = ((*in_buf) >> 6) % (1u32 << 2);
    out = out.offset(1);
    *out = ((*in_buf) >> 8) % (1u32 << 2);
    out = out.offset(1);
    *out = ((*in_buf) >> 10) % (1u32 << 2);
    out = out.offset(1);
    *out = ((*in_buf) >> 12) % (1u32 << 2);
    out = out.offset(1);
    *out = ((*in_buf) >> 14) % (1u32 << 2);
    out = out.offset(1);
    *out = ((*in_buf) >> 16) % (1u32 << 2);
    out = out.offset(1);
    *out = ((*in_buf) >> 18) % (1u32 << 2);
    out = out.offset(1);
    *out = ((*in_buf) >> 20) % (1u32 << 2);
    out = out.offset(1);
    *out = ((*in_buf) >> 22) % (1u32 << 2);
    out = out.offset(1);
    *out = ((*in_buf) >> 24) % (1u32 << 2);
    out = out.offset(1);
    *out = ((*in_buf) >> 26) % (1u32 << 2);
    out = out.offset(1);
    *out = ((*in_buf) >> 28) % (1u32 << 2);
    out = out.offset(1);
    *out = (*in_buf) >> 30;

    in_buf.offset(1)
}

unsafe fn unpack3_32(mut in_buf: *const u32, mut out: *mut u32) -> *const u32 {
    *out = ((*in_buf) >> 0) % (1u32 << 3);
    out = out.offset(1);
    *out = ((*in_buf) >> 3) % (1u32 << 3);
    out = out.offset(1);
    *out = ((*in_buf) >> 6) % (1u32 << 3);
    out = out.offset(1);
    *out = ((*in_buf) >> 9) % (1u32 << 3);
    out = out.offset(1);
    *out = ((*in_buf) >> 12) % (1u32 << 3);
    out = out.offset(1);
    *out = ((*in_buf) >> 15) % (1u32 << 3);
    out = out.offset(1);
    *out = ((*in_buf) >> 18) % (1u32 << 3);
    out = out.offset(1);
    *out = ((*in_buf) >> 21) % (1u32 << 3);
    out = out.offset(1);
    *out = ((*in_buf) >> 24) % (1u32 << 3);
    out = out.offset(1);
    *out = ((*in_buf) >> 27) % (1u32 << 3);
    out = out.offset(1);
    *out = (*in_buf) >> 30;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 1)) << (3 - 1);
    out = out.offset(1);

    *out = ((*in_buf) >> 1) % (1u32 << 3);
    out = out.offset(1);
    *out = ((*in_buf) >> 4) % (1u32 << 3);
    out = out.offset(1);
    *out = ((*in_buf) >> 7) % (1u32 << 3);
    out = out.offset(1);
    *out = ((*in_buf) >> 10) % (1u32 << 3);
    out = out.offset(1);
    *out = ((*in_buf) >> 13) % (1u32 << 3);
    out = out.offset(1);
    *out = ((*in_buf) >> 16) % (1u32 << 3);
    out = out.offset(1);
    *out = ((*in_buf) >> 19) % (1u32 << 3);
    out = out.offset(1);
    *out = ((*in_buf) >> 22) % (1u32 << 3);
    out = out.offset(1);
    *out = ((*in_buf) >> 25) % (1u32 << 3);
    out = out.offset(1);
    *out = ((*in_buf) >> 28) % (1u32 << 3);
    out = out.offset(1);
    *out = (*in_buf) >> 31;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 2)) << (3 - 2);
    out = out.offset(1);

    *out = ((*in_buf) >> 2) % (1u32 << 3);
    out = out.offset(1);
    *out = ((*in_buf) >> 5) % (1u32 << 3);
    out = out.offset(1);
    *out = ((*in_buf) >> 8) % (1u32 << 3);
    out = out.offset(1);
    *out = ((*in_buf) >> 11) % (1u32 << 3);
    out = out.offset(1);
    *out = ((*in_buf) >> 14) % (1u32 << 3);
    out = out.offset(1);
    *out = ((*in_buf) >> 17) % (1u32 << 3);
    out = out.offset(1);
    *out = ((*in_buf) >> 20) % (1u32 << 3);
    out = out.offset(1);
    *out = ((*in_buf) >> 23) % (1u32 << 3);
    out = out.offset(1);
    *out = ((*in_buf) >> 26) % (1u32 << 3);
    out = out.offset(1);
    *out = (*in_buf) >> 29;

    in_buf.offset(1)
}

unsafe fn unpack4_32(mut in_buf: *const u32, mut out: *mut u32) -> *const u32 {
    *out = ((*in_buf) >> 0) % (1u32 << 4);
    out = out.offset(1);
    *out = ((*in_buf) >> 4) % (1u32 << 4);
    out = out.offset(1);
    *out = ((*in_buf) >> 8) % (1u32 << 4);
    out = out.offset(1);
    *out = ((*in_buf) >> 12) % (1u32 << 4);
    out = out.offset(1);
    *out = ((*in_buf) >> 16) % (1u32 << 4);
    out = out.offset(1);
    *out = ((*in_buf) >> 20) % (1u32 << 4);
    out = out.offset(1);
    *out = ((*in_buf) >> 24) % (1u32 << 4);
    out = out.offset(1);
    *out = ((*in_buf) >> 28) % (1u32 << 4);
    out = out.offset(1);
    in_buf = in_buf.offset(1);

    *out = ((*in_buf) >> 0) % (1u32 << 4);
    out = out.offset(1);
    *out = ((*in_buf) >> 4) % (1u32 << 4);
    out = out.offset(1);
    *out = ((*in_buf) >> 8) % (1u32 << 4);
    out = out.offset(1);
    *out = ((*in_buf) >> 12) % (1u32 << 4);
    out = out.offset(1);
    *out = ((*in_buf) >> 16) % (1u32 << 4);
    out = out.offset(1);
    *out = ((*in_buf) >> 20) % (1u32 << 4);
    out = out.offset(1);
    *out = ((*in_buf) >> 24) % (1u32 << 4);
    out = out.offset(1);
    *out = ((*in_buf) >> 28) % (1u32 << 4);
    out = out.offset(1);
    in_buf = in_buf.offset(1);

    *out = ((*in_buf) >> 0) % (1u32 << 4);
    out = out.offset(1);
    *out = ((*in_buf) >> 4) % (1u32 << 4);
    out = out.offset(1);
    *out = ((*in_buf) >> 8) % (1u32 << 4);
    out = out.offset(1);
    *out = ((*in_buf) >> 12) % (1u32 << 4);
    out = out.offset(1);
    *out = ((*in_buf) >> 16) % (1u32 << 4);
    out = out.offset(1);
    *out = ((*in_buf) >> 20) % (1u32 << 4);
    out = out.offset(1);
    *out = ((*in_buf) >> 24) % (1u32 << 4);
    out = out.offset(1);
    *out = ((*in_buf) >> 28) % (1u32 << 4);
    out = out.offset(1);
    in_buf = in_buf.offset(1);

    *out = ((*in_buf) >> 0) % (1u32 << 4);
    out = out.offset(1);
    *out = ((*in_buf) >> 4) % (1u32 << 4);
    out = out.offset(1);
    *out = ((*in_buf) >> 8) % (1u32 << 4);
    out = out.offset(1);
    *out = ((*in_buf) >> 12) % (1u32 << 4);
    out = out.offset(1);
    *out = ((*in_buf) >> 16) % (1u32 << 4);
    out = out.offset(1);
    *out = ((*in_buf) >> 20) % (1u32 << 4);
    out = out.offset(1);
    *out = ((*in_buf) >> 24) % (1u32 << 4);
    out = out.offset(1);
    *out = ((*in_buf) >> 28) % (1u32 << 4);

    in_buf.offset(1)
}

unsafe fn unpack5_32(mut in_buf: *const u32, mut out: *mut u32) -> *const u32 {
    *out = ((*in_buf) >> 0) % (1u32 << 5);
    out = out.offset(1);
    *out = ((*in_buf) >> 5) % (1u32 << 5);
    out = out.offset(1);
    *out = ((*in_buf) >> 10) % (1u32 << 5);
    out = out.offset(1);
    *out = ((*in_buf) >> 15) % (1u32 << 5);
    out = out.offset(1);
    *out = ((*in_buf) >> 20) % (1u32 << 5);
    out = out.offset(1);
    *out = ((*in_buf) >> 25) % (1u32 << 5);
    out = out.offset(1);
    *out = (*in_buf) >> 30;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 3)) << (5 - 3);
    out = out.offset(1);

    *out = ((*in_buf) >> 3) % (1u32 << 5);
    out = out.offset(1);
    *out = ((*in_buf) >> 8) % (1u32 << 5);
    out = out.offset(1);
    *out = ((*in_buf) >> 13) % (1u32 << 5);
    out = out.offset(1);
    *out = ((*in_buf) >> 18) % (1u32 << 5);
    out = out.offset(1);
    *out = ((*in_buf) >> 23) % (1u32 << 5);
    out = out.offset(1);
    *out = ((*in_buf) >> 28) % (1u32 << 5);
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 1)) << (5 - 1);
    out = out.offset(1);

    *out = ((*in_buf) >> 1) % (1u32 << 5);
    out = out.offset(1);
    *out = ((*in_buf) >> 6) % (1u32 << 5);
    out = out.offset(1);
    *out = ((*in_buf) >> 11) % (1u32 << 5);
    out = out.offset(1);
    *out = ((*in_buf) >> 16) % (1u32 << 5);
    out = out.offset(1);
    *out = ((*in_buf) >> 21) % (1u32 << 5);
    out = out.offset(1);
    *out = ((*in_buf) >> 26) % (1u32 << 5);
    out = out.offset(1);
    *out = (*in_buf) >> 31;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 4)) << (5 - 4);
    out = out.offset(1);

    *out = ((*in_buf) >> 4) % (1u32 << 5);
    out = out.offset(1);
    *out = ((*in_buf) >> 9) % (1u32 << 5);
    out = out.offset(1);
    *out = ((*in_buf) >> 14) % (1u32 << 5);
    out = out.offset(1);
    *out = ((*in_buf) >> 19) % (1u32 << 5);
    out = out.offset(1);
    *out = ((*in_buf) >> 24) % (1u32 << 5);
    out = out.offset(1);
    *out = (*in_buf) >> 29;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 2)) << (5 - 2);
    out = out.offset(1);

    *out = ((*in_buf) >> 2) % (1u32 << 5);
    out = out.offset(1);
    *out = ((*in_buf) >> 7) % (1u32 << 5);
    out = out.offset(1);
    *out = ((*in_buf) >> 12) % (1u32 << 5);
    out = out.offset(1);
    *out = ((*in_buf) >> 17) % (1u32 << 5);
    out = out.offset(1);
    *out = ((*in_buf) >> 22) % (1u32 << 5);
    out = out.offset(1);
    *out = (*in_buf) >> 27;

    in_buf.offset(1)
}

unsafe fn unpack6_32(mut in_buf: *const u32, mut out: *mut u32) -> *const u32 {
    *out = ((*in_buf) >> 0) % (1u32 << 6);
    out = out.offset(1);
    *out = ((*in_buf) >> 6) % (1u32 << 6);
    out = out.offset(1);
    *out = ((*in_buf) >> 12) % (1u32 << 6);
    out = out.offset(1);
    *out = ((*in_buf) >> 18) % (1u32 << 6);
    out = out.offset(1);
    *out = ((*in_buf) >> 24) % (1u32 << 6);
    out = out.offset(1);
    *out = (*in_buf) >> 30;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 4)) << (6 - 4);
    out = out.offset(1);

    *out = ((*in_buf) >> 4) % (1u32 << 6);
    out = out.offset(1);
    *out = ((*in_buf) >> 10) % (1u32 << 6);
    out = out.offset(1);
    *out = ((*in_buf) >> 16) % (1u32 << 6);
    out = out.offset(1);
    *out = ((*in_buf) >> 22) % (1u32 << 6);
    out = out.offset(1);
    *out = (*in_buf) >> 28;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 2)) << (6 - 2);
    out = out.offset(1);

    *out = ((*in_buf) >> 2) % (1u32 << 6);
    out = out.offset(1);
    *out = ((*in_buf) >> 8) % (1u32 << 6);
    out = out.offset(1);
    *out = ((*in_buf) >> 14) % (1u32 << 6);
    out = out.offset(1);
    *out = ((*in_buf) >> 20) % (1u32 << 6);
    out = out.offset(1);
    *out = (*in_buf) >> 26;
    in_buf = in_buf.offset(1);
    out = out.offset(1);

    *out = ((*in_buf) >> 0) % (1u32 << 6);
    out = out.offset(1);
    *out = ((*in_buf) >> 6) % (1u32 << 6);
    out = out.offset(1);
    *out = ((*in_buf) >> 12) % (1u32 << 6);
    out = out.offset(1);
    *out = ((*in_buf) >> 18) % (1u32 << 6);
    out = out.offset(1);
    *out = ((*in_buf) >> 24) % (1u32 << 6);
    out = out.offset(1);
    *out = (*in_buf) >> 30;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 4)) << (6 - 4);
    out = out.offset(1);

    *out = ((*in_buf) >> 4) % (1u32 << 6);
    out = out.offset(1);
    *out = ((*in_buf) >> 10) % (1u32 << 6);
    out = out.offset(1);
    *out = ((*in_buf) >> 16) % (1u32 << 6);
    out = out.offset(1);
    *out = ((*in_buf) >> 22) % (1u32 << 6);
    out = out.offset(1);
    *out = (*in_buf) >> 28;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 2)) << (6 - 2);
    out = out.offset(1);

    *out = ((*in_buf) >> 2) % (1u32 << 6);
    out = out.offset(1);
    *out = ((*in_buf) >> 8) % (1u32 << 6);
    out = out.offset(1);
    *out = ((*in_buf) >> 14) % (1u32 << 6);
    out = out.offset(1);
    *out = ((*in_buf) >> 20) % (1u32 << 6);
    out = out.offset(1);
    *out = (*in_buf) >> 26;

    in_buf.offset(1)
}

unsafe fn unpack7_32(mut in_buf: *const u32, mut out: *mut u32) -> *const u32 {
    *out = ((*in_buf) >> 0) % (1u32 << 7);
    out = out.offset(1);
    *out = ((*in_buf) >> 7) % (1u32 << 7);
    out = out.offset(1);
    *out = ((*in_buf) >> 14) % (1u32 << 7);
    out = out.offset(1);
    *out = ((*in_buf) >> 21) % (1u32 << 7);
    out = out.offset(1);
    *out = (*in_buf) >> 28;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 3)) << (7 - 3);
    out = out.offset(1);

    *out = ((*in_buf) >> 3) % (1u32 << 7);
    out = out.offset(1);
    *out = ((*in_buf) >> 10) % (1u32 << 7);
    out = out.offset(1);
    *out = ((*in_buf) >> 17) % (1u32 << 7);
    out = out.offset(1);
    *out = ((*in_buf) >> 24) % (1u32 << 7);
    out = out.offset(1);
    *out = (*in_buf) >> 31;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 6)) << (7 - 6);
    out = out.offset(1);

    *out = ((*in_buf) >> 6) % (1u32 << 7);
    out = out.offset(1);
    *out = ((*in_buf) >> 13) % (1u32 << 7);
    out = out.offset(1);
    *out = ((*in_buf) >> 20) % (1u32 << 7);
    out = out.offset(1);
    *out = (*in_buf) >> 27;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 2)) << (7 - 2);
    out = out.offset(1);

    *out = ((*in_buf) >> 2) % (1u32 << 7);
    out = out.offset(1);
    *out = ((*in_buf) >> 9) % (1u32 << 7);
    out = out.offset(1);
    *out = ((*in_buf) >> 16) % (1u32 << 7);
    out = out.offset(1);
    *out = ((*in_buf) >> 23) % (1u32 << 7);
    out = out.offset(1);
    *out = (*in_buf) >> 30;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 5)) << (7 - 5);
    out = out.offset(1);

    *out = ((*in_buf) >> 5) % (1u32 << 7);
    out = out.offset(1);
    *out = ((*in_buf) >> 12) % (1u32 << 7);
    out = out.offset(1);
    *out = ((*in_buf) >> 19) % (1u32 << 7);
    out = out.offset(1);
    *out = (*in_buf) >> 26;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 1)) << (7 - 1);
    out = out.offset(1);

    *out = ((*in_buf) >> 1) % (1u32 << 7);
    out = out.offset(1);
    *out = ((*in_buf) >> 8) % (1u32 << 7);
    out = out.offset(1);
    *out = ((*in_buf) >> 15) % (1u32 << 7);
    out = out.offset(1);
    *out = ((*in_buf) >> 22) % (1u32 << 7);
    out = out.offset(1);
    *out = (*in_buf) >> 29;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 4)) << (7 - 4);
    out = out.offset(1);

    *out = ((*in_buf) >> 4) % (1u32 << 7);
    out = out.offset(1);
    *out = ((*in_buf) >> 11) % (1u32 << 7);
    out = out.offset(1);
    *out = ((*in_buf) >> 18) % (1u32 << 7);
    out = out.offset(1);
    *out = (*in_buf) >> 25;

    in_buf.offset(1)
}

unsafe fn unpack8_32(mut in_buf: *const u32, mut out: *mut u32) -> *const u32 {
    *out = ((*in_buf) >> 0) % (1u32 << 8);
    out = out.offset(1);
    *out = ((*in_buf) >> 8) % (1u32 << 8);
    out = out.offset(1);
    *out = ((*in_buf) >> 16) % (1u32 << 8);
    out = out.offset(1);
    *out = (*in_buf) >> 24;
    out = out.offset(1);
    in_buf = in_buf.offset(1);

    *out = ((*in_buf) >> 0) % (1u32 << 8);
    out = out.offset(1);
    *out = ((*in_buf) >> 8) % (1u32 << 8);
    out = out.offset(1);
    *out = ((*in_buf) >> 16) % (1u32 << 8);
    out = out.offset(1);
    *out = (*in_buf) >> 24;
    out = out.offset(1);
    in_buf = in_buf.offset(1);

    *out = ((*in_buf) >> 0) % (1u32 << 8);
    out = out.offset(1);
    *out = ((*in_buf) >> 8) % (1u32 << 8);
    out = out.offset(1);
    *out = ((*in_buf) >> 16) % (1u32 << 8);
    out = out.offset(1);
    *out = (*in_buf) >> 24;
    out = out.offset(1);
    in_buf = in_buf.offset(1);

    *out = ((*in_buf) >> 0) % (1u32 << 8);
    out = out.offset(1);
    *out = ((*in_buf) >> 8) % (1u32 << 8);
    out = out.offset(1);
    *out = ((*in_buf) >> 16) % (1u32 << 8);
    out = out.offset(1);
    *out = (*in_buf) >> 24;
    out = out.offset(1);
    in_buf = in_buf.offset(1);

    *out = ((*in_buf) >> 0) % (1u32 << 8);
    out = out.offset(1);
    *out = ((*in_buf) >> 8) % (1u32 << 8);
    out = out.offset(1);
    *out = ((*in_buf) >> 16) % (1u32 << 8);
    out = out.offset(1);
    *out = (*in_buf) >> 24;
    out = out.offset(1);
    in_buf = in_buf.offset(1);

    *out = ((*in_buf) >> 0) % (1u32 << 8);
    out = out.offset(1);
    *out = ((*in_buf) >> 8) % (1u32 << 8);
    out = out.offset(1);
    *out = ((*in_buf) >> 16) % (1u32 << 8);
    out = out.offset(1);
    *out = (*in_buf) >> 24;
    out = out.offset(1);
    in_buf = in_buf.offset(1);

    *out = ((*in_buf) >> 0) % (1u32 << 8);
    out = out.offset(1);
    *out = ((*in_buf) >> 8) % (1u32 << 8);
    out = out.offset(1);
    *out = ((*in_buf) >> 16) % (1u32 << 8);
    out = out.offset(1);
    *out = (*in_buf) >> 24;
    out = out.offset(1);
    in_buf = in_buf.offset(1);

    *out = ((*in_buf) >> 0) % (1u32 << 8);
    out = out.offset(1);
    *out = ((*in_buf) >> 8) % (1u32 << 8);
    out = out.offset(1);
    *out = ((*in_buf) >> 16) % (1u32 << 8);
    out = out.offset(1);
    *out = (*in_buf) >> 24;

    in_buf.offset(1)
}

unsafe fn unpack9_32(mut in_buf: *const u32, mut out: *mut u32) -> *const u32 {
    *out = ((*in_buf) >> 0) % (1u32 << 9);
    out = out.offset(1);
    *out = ((*in_buf) >> 9) % (1u32 << 9);
    out = out.offset(1);
    *out = ((*in_buf) >> 18) % (1u32 << 9);
    out = out.offset(1);
    *out = (*in_buf) >> 27;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 4)) << (9 - 4);
    out = out.offset(1);

    *out = ((*in_buf) >> 4) % (1u32 << 9);
    out = out.offset(1);
    *out = ((*in_buf) >> 13) % (1u32 << 9);
    out = out.offset(1);
    *out = ((*in_buf) >> 22) % (1u32 << 9);
    out = out.offset(1);
    *out = (*in_buf) >> 31;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 8)) << (9 - 8);
    out = out.offset(1);

    *out = ((*in_buf) >> 8) % (1u32 << 9);
    out = out.offset(1);
    *out = ((*in_buf) >> 17) % (1u32 << 9);
    out = out.offset(1);
    *out = (*in_buf) >> 26;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 3)) << (9 - 3);
    out = out.offset(1);

    *out = ((*in_buf) >> 3) % (1u32 << 9);
    out = out.offset(1);
    *out = ((*in_buf) >> 12) % (1u32 << 9);
    out = out.offset(1);
    *out = ((*in_buf) >> 21) % (1u32 << 9);
    out = out.offset(1);
    *out = (*in_buf) >> 30;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 7)) << (9 - 7);
    out = out.offset(1);

    *out = ((*in_buf) >> 7) % (1u32 << 9);
    out = out.offset(1);
    *out = ((*in_buf) >> 16) % (1u32 << 9);
    out = out.offset(1);
    *out = (*in_buf) >> 25;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 2)) << (9 - 2);
    out = out.offset(1);

    *out = ((*in_buf) >> 2) % (1u32 << 9);
    out = out.offset(1);
    *out = ((*in_buf) >> 11) % (1u32 << 9);
    out = out.offset(1);
    *out = ((*in_buf) >> 20) % (1u32 << 9);
    out = out.offset(1);
    *out = (*in_buf) >> 29;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 6)) << (9 - 6);
    out = out.offset(1);

    *out = ((*in_buf) >> 6) % (1u32 << 9);
    out = out.offset(1);
    *out = ((*in_buf) >> 15) % (1u32 << 9);
    out = out.offset(1);
    *out = (*in_buf) >> 24;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 1)) << (9 - 1);
    out = out.offset(1);

    *out = ((*in_buf) >> 1) % (1u32 << 9);
    out = out.offset(1);
    *out = ((*in_buf) >> 10) % (1u32 << 9);
    out = out.offset(1);
    *out = ((*in_buf) >> 19) % (1u32 << 9);
    out = out.offset(1);
    *out = (*in_buf) >> 28;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 5)) << (9 - 5);
    out = out.offset(1);

    *out = ((*in_buf) >> 5) % (1u32 << 9);
    out = out.offset(1);
    *out = ((*in_buf) >> 14) % (1u32 << 9);
    out = out.offset(1);
    *out = (*in_buf) >> 23;

    in_buf.offset(1)
}

unsafe fn unpack10_32(mut in_buf: *const u32, mut out: *mut u32) -> *const u32 {
    *out = ((*in_buf) >> 0) % (1u32 << 10);
    out = out.offset(1);
    *out = ((*in_buf) >> 10) % (1u32 << 10);
    out = out.offset(1);
    *out = ((*in_buf) >> 20) % (1u32 << 10);
    out = out.offset(1);
    *out = (*in_buf) >> 30;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 8)) << (10 - 8);
    out = out.offset(1);

    *out = ((*in_buf) >> 8) % (1u32 << 10);
    out = out.offset(1);
    *out = ((*in_buf) >> 18) % (1u32 << 10);
    out = out.offset(1);
    *out = (*in_buf) >> 28;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 6)) << (10 - 6);
    out = out.offset(1);

    *out = ((*in_buf) >> 6) % (1u32 << 10);
    out = out.offset(1);
    *out = ((*in_buf) >> 16) % (1u32 << 10);
    out = out.offset(1);
    *out = (*in_buf) >> 26;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 4)) << (10 - 4);
    out = out.offset(1);

    *out = ((*in_buf) >> 4) % (1u32 << 10);
    out = out.offset(1);
    *out = ((*in_buf) >> 14) % (1u32 << 10);
    out = out.offset(1);
    *out = (*in_buf) >> 24;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 2)) << (10 - 2);
    out = out.offset(1);

    *out = ((*in_buf) >> 2) % (1u32 << 10);
    out = out.offset(1);
    *out = ((*in_buf) >> 12) % (1u32 << 10);
    out = out.offset(1);
    *out = (*in_buf) >> 22;
    in_buf = in_buf.offset(1);
    out = out.offset(1);

    *out = ((*in_buf) >> 0) % (1u32 << 10);
    out = out.offset(1);
    *out = ((*in_buf) >> 10) % (1u32 << 10);
    out = out.offset(1);
    *out = ((*in_buf) >> 20) % (1u32 << 10);
    out = out.offset(1);
    *out = (*in_buf) >> 30;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 8)) << (10 - 8);
    out = out.offset(1);

    *out = ((*in_buf) >> 8) % (1u32 << 10);
    out = out.offset(1);
    *out = ((*in_buf) >> 18) % (1u32 << 10);
    out = out.offset(1);
    *out = (*in_buf) >> 28;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 6)) << (10 - 6);
    out = out.offset(1);

    *out = ((*in_buf) >> 6) % (1u32 << 10);
    out = out.offset(1);
    *out = ((*in_buf) >> 16) % (1u32 << 10);
    out = out.offset(1);
    *out = (*in_buf) >> 26;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 4)) << (10 - 4);
    out = out.offset(1);

    *out = ((*in_buf) >> 4) % (1u32 << 10);
    out = out.offset(1);
    *out = ((*in_buf) >> 14) % (1u32 << 10);
    out = out.offset(1);
    *out = (*in_buf) >> 24;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 2)) << (10 - 2);
    out = out.offset(1);

    *out = ((*in_buf) >> 2) % (1u32 << 10);
    out = out.offset(1);
    *out = ((*in_buf) >> 12) % (1u32 << 10);
    out = out.offset(1);
    *out = (*in_buf) >> 22;

    in_buf.offset(1)
}

unsafe fn unpack11_32(mut in_buf: *const u32, mut out: *mut u32) -> *const u32 {
    *out = ((*in_buf) >> 0) % (1u32 << 11);
    out = out.offset(1);
    *out = ((*in_buf) >> 11) % (1u32 << 11);
    out = out.offset(1);
    *out = (*in_buf) >> 22;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 1)) << (11 - 1);
    out = out.offset(1);

    *out = ((*in_buf) >> 1) % (1u32 << 11);
    out = out.offset(1);
    *out = ((*in_buf) >> 12) % (1u32 << 11);
    out = out.offset(1);
    *out = (*in_buf) >> 23;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 2)) << (11 - 2);
    out = out.offset(1);

    *out = ((*in_buf) >> 2) % (1u32 << 11);
    out = out.offset(1);
    *out = ((*in_buf) >> 13) % (1u32 << 11);
    out = out.offset(1);
    *out = (*in_buf) >> 24;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 3)) << (11 - 3);
    out = out.offset(1);

    *out = ((*in_buf) >> 3) % (1u32 << 11);
    out = out.offset(1);
    *out = ((*in_buf) >> 14) % (1u32 << 11);
    out = out.offset(1);
    *out = (*in_buf) >> 25;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 4)) << (11 - 4);
    out = out.offset(1);

    *out = ((*in_buf) >> 4) % (1u32 << 11);
    out = out.offset(1);
    *out = ((*in_buf) >> 15) % (1u32 << 11);
    out = out.offset(1);
    *out = (*in_buf) >> 26;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 5)) << (11 - 5);
    out = out.offset(1);

    *out = ((*in_buf) >> 5) % (1u32 << 11);
    out = out.offset(1);
    *out = ((*in_buf) >> 16) % (1u32 << 11);
    out = out.offset(1);
    *out = (*in_buf) >> 27;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 6)) << (11 - 6);
    out = out.offset(1);

    *out = ((*in_buf) >> 6) % (1u32 << 11);
    out = out.offset(1);
    *out = ((*in_buf) >> 17) % (1u32 << 11);
    out = out.offset(1);
    *out = (*in_buf) >> 28;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 7)) << (11 - 7);
    out = out.offset(1);

    *out = ((*in_buf) >> 7) % (1u32 << 11);
    out = out.offset(1);
    *out = ((*in_buf) >> 18) % (1u32 << 11);
    out = out.offset(1);
    *out = (*in_buf) >> 29;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 8)) << (11 - 8);
    out = out.offset(1);

    *out = ((*in_buf) >> 8) % (1u32 << 11);
    out = out.offset(1);
    *out = ((*in_buf) >> 19) % (1u32 << 11);
    out = out.offset(1);
    *out = (*in_buf) >> 30;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 9)) << (11 - 9);
    out = out.offset(1);

    *out = ((*in_buf) >> 9) % (1u32 << 11);
    out = out.offset(1);
    *out = ((*in_buf) >> 20) % (1u32 << 11);
    out = out.offset(1);
    *out = (*in_buf) >> 31;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 10)) << (11 - 10);
    out = out.offset(1);

    *out = ((*in_buf) >> 10) % (1u32 << 11);
    out = out.offset(1);
    *out = (*in_buf) >> 21;

    in_buf.offset(1)
}

unsafe fn unpack12_32(mut in_buf: *const u32, mut out: *mut u32) -> *const u32 {
    *out = ((*in_buf) >> 0) % (1u32 << 12);
    out = out.offset(1);
    *out = ((*in_buf) >> 12) % (1u32 << 12);
    out = out.offset(1);
    *out = (*in_buf) >> 24;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 4)) << (12 - 4);
    out = out.offset(1);

    *out = ((*in_buf) >> 4) % (1u32 << 12);
    out = out.offset(1);
    *out = ((*in_buf) >> 16) % (1u32 << 12);
    out = out.offset(1);
    *out = (*in_buf) >> 28;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 8)) << (12 - 8);
    out = out.offset(1);

    *out = ((*in_buf) >> 8) % (1u32 << 12);
    out = out.offset(1);
    *out = (*in_buf) >> 20;
    in_buf = in_buf.offset(1);
    out = out.offset(1);

    *out = ((*in_buf) >> 0) % (1u32 << 12);
    out = out.offset(1);
    *out = ((*in_buf) >> 12) % (1u32 << 12);
    out = out.offset(1);
    *out = (*in_buf) >> 24;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 4)) << (12 - 4);
    out = out.offset(1);

    *out = ((*in_buf) >> 4) % (1u32 << 12);
    out = out.offset(1);
    *out = ((*in_buf) >> 16) % (1u32 << 12);
    out = out.offset(1);
    *out = (*in_buf) >> 28;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 8)) << (12 - 8);
    out = out.offset(1);

    *out = ((*in_buf) >> 8) % (1u32 << 12);
    out = out.offset(1);
    *out = (*in_buf) >> 20;
    in_buf = in_buf.offset(1);
    out = out.offset(1);

    *out = ((*in_buf) >> 0) % (1u32 << 12);
    out = out.offset(1);
    *out = ((*in_buf) >> 12) % (1u32 << 12);
    out = out.offset(1);
    *out = (*in_buf) >> 24;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 4)) << (12 - 4);
    out = out.offset(1);

    *out = ((*in_buf) >> 4) % (1u32 << 12);
    out = out.offset(1);
    *out = ((*in_buf) >> 16) % (1u32 << 12);
    out = out.offset(1);
    *out = (*in_buf) >> 28;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 8)) << (12 - 8);
    out = out.offset(1);

    *out = ((*in_buf) >> 8) % (1u32 << 12);
    out = out.offset(1);
    *out = (*in_buf) >> 20;
    in_buf = in_buf.offset(1);
    out = out.offset(1);

    *out = ((*in_buf) >> 0) % (1u32 << 12);
    out = out.offset(1);
    *out = ((*in_buf) >> 12) % (1u32 << 12);
    out = out.offset(1);
    *out = (*in_buf) >> 24;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 4)) << (12 - 4);
    out = out.offset(1);

    *out = ((*in_buf) >> 4) % (1u32 << 12);
    out = out.offset(1);
    *out = ((*in_buf) >> 16) % (1u32 << 12);
    out = out.offset(1);
    *out = (*in_buf) >> 28;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 8)) << (12 - 8);
    out = out.offset(1);

    *out = ((*in_buf) >> 8) % (1u32 << 12);
    out = out.offset(1);
    *out = (*in_buf) >> 20;

    in_buf.offset(1)
}

unsafe fn unpack13_32(mut in_buf: *const u32, mut out: *mut u32) -> *const u32 {
    *out = ((*in_buf) >> 0) % (1u32 << 13);
    out = out.offset(1);
    *out = ((*in_buf) >> 13) % (1u32 << 13);
    out = out.offset(1);
    *out = (*in_buf) >> 26;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 7)) << (13 - 7);
    out = out.offset(1);

    *out = ((*in_buf) >> 7) % (1u32 << 13);
    out = out.offset(1);
    *out = (*in_buf) >> 20;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 1)) << (13 - 1);
    out = out.offset(1);

    *out = ((*in_buf) >> 1) % (1u32 << 13);
    out = out.offset(1);
    *out = ((*in_buf) >> 14) % (1u32 << 13);
    out = out.offset(1);
    *out = (*in_buf) >> 27;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 8)) << (13 - 8);
    out = out.offset(1);

    *out = ((*in_buf) >> 8) % (1u32 << 13);
    out = out.offset(1);
    *out = (*in_buf) >> 21;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 2)) << (13 - 2);
    out = out.offset(1);

    *out = ((*in_buf) >> 2) % (1u32 << 13);
    out = out.offset(1);
    *out = ((*in_buf) >> 15) % (1u32 << 13);
    out = out.offset(1);
    *out = (*in_buf) >> 28;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 9)) << (13 - 9);
    out = out.offset(1);

    *out = ((*in_buf) >> 9) % (1u32 << 13);
    out = out.offset(1);
    *out = (*in_buf) >> 22;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 3)) << (13 - 3);
    out = out.offset(1);

    *out = ((*in_buf) >> 3) % (1u32 << 13);
    out = out.offset(1);
    *out = ((*in_buf) >> 16) % (1u32 << 13);
    out = out.offset(1);
    *out = (*in_buf) >> 29;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 10)) << (13 - 10);
    out = out.offset(1);

    *out = ((*in_buf) >> 10) % (1u32 << 13);
    out = out.offset(1);
    *out = (*in_buf) >> 23;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 4)) << (13 - 4);
    out = out.offset(1);

    *out = ((*in_buf) >> 4) % (1u32 << 13);
    out = out.offset(1);
    *out = ((*in_buf) >> 17) % (1u32 << 13);
    out = out.offset(1);
    *out = (*in_buf) >> 30;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 11)) << (13 - 11);
    out = out.offset(1);

    *out = ((*in_buf) >> 11) % (1u32 << 13);
    out = out.offset(1);
    *out = (*in_buf) >> 24;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 5)) << (13 - 5);
    out = out.offset(1);

    *out = ((*in_buf) >> 5) % (1u32 << 13);
    out = out.offset(1);
    *out = ((*in_buf) >> 18) % (1u32 << 13);
    out = out.offset(1);
    *out = (*in_buf) >> 31;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 12)) << (13 - 12);
    out = out.offset(1);

    *out = ((*in_buf) >> 12) % (1u32 << 13);
    out = out.offset(1);
    *out = (*in_buf) >> 25;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 6)) << (13 - 6);
    out = out.offset(1);

    *out = ((*in_buf) >> 6) % (1u32 << 13);
    out = out.offset(1);
    *out = (*in_buf) >> 19;

    in_buf.offset(1)
}

unsafe fn unpack14_32(mut in_buf: *const u32, mut out: *mut u32) -> *const u32 {
    *out = ((*in_buf) >> 0) % (1u32 << 14);
    out = out.offset(1);
    *out = ((*in_buf) >> 14) % (1u32 << 14);
    out = out.offset(1);
    *out = (*in_buf) >> 28;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 10)) << (14 - 10);
    out = out.offset(1);

    *out = ((*in_buf) >> 10) % (1u32 << 14);
    out = out.offset(1);
    *out = (*in_buf) >> 24;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 6)) << (14 - 6);
    out = out.offset(1);

    *out = ((*in_buf) >> 6) % (1u32 << 14);
    out = out.offset(1);
    *out = (*in_buf) >> 20;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 2)) << (14 - 2);
    out = out.offset(1);

    *out = ((*in_buf) >> 2) % (1u32 << 14);
    out = out.offset(1);
    *out = ((*in_buf) >> 16) % (1u32 << 14);
    out = out.offset(1);
    *out = (*in_buf) >> 30;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 12)) << (14 - 12);
    out = out.offset(1);

    *out = ((*in_buf) >> 12) % (1u32 << 14);
    out = out.offset(1);
    *out = (*in_buf) >> 26;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 8)) << (14 - 8);
    out = out.offset(1);

    *out = ((*in_buf) >> 8) % (1u32 << 14);
    out = out.offset(1);
    *out = (*in_buf) >> 22;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 4)) << (14 - 4);
    out = out.offset(1);

    *out = ((*in_buf) >> 4) % (1u32 << 14);
    out = out.offset(1);
    *out = (*in_buf) >> 18;
    in_buf = in_buf.offset(1);
    out = out.offset(1);

    *out = ((*in_buf) >> 0) % (1u32 << 14);
    out = out.offset(1);
    *out = ((*in_buf) >> 14) % (1u32 << 14);
    out = out.offset(1);
    *out = (*in_buf) >> 28;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 10)) << (14 - 10);
    out = out.offset(1);

    *out = ((*in_buf) >> 10) % (1u32 << 14);
    out = out.offset(1);
    *out = (*in_buf) >> 24;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 6)) << (14 - 6);
    out = out.offset(1);

    *out = ((*in_buf) >> 6) % (1u32 << 14);
    out = out.offset(1);
    *out = (*in_buf) >> 20;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 2)) << (14 - 2);
    out = out.offset(1);

    *out = ((*in_buf) >> 2) % (1u32 << 14);
    out = out.offset(1);
    *out = ((*in_buf) >> 16) % (1u32 << 14);
    out = out.offset(1);
    *out = (*in_buf) >> 30;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 12)) << (14 - 12);
    out = out.offset(1);

    *out = ((*in_buf) >> 12) % (1u32 << 14);
    out = out.offset(1);
    *out = (*in_buf) >> 26;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 8)) << (14 - 8);
    out = out.offset(1);

    *out = ((*in_buf) >> 8) % (1u32 << 14);
    out = out.offset(1);
    *out = (*in_buf) >> 22;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 4)) << (14 - 4);
    out = out.offset(1);

    *out = ((*in_buf) >> 4) % (1u32 << 14);
    out = out.offset(1);
    *out = (*in_buf) >> 18;

    in_buf.offset(1)
}

unsafe fn unpack15_32(mut in_buf: *const u32, mut out: *mut u32) -> *const u32 {
    *out = ((*in_buf) >> 0) % (1u32 << 15);
    out = out.offset(1);
    *out = ((*in_buf) >> 15) % (1u32 << 15);
    out = out.offset(1);
    *out = (*in_buf) >> 30;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 13)) << (15 - 13);
    out = out.offset(1);

    *out = ((*in_buf) >> 13) % (1u32 << 15);
    out = out.offset(1);
    *out = (*in_buf) >> 28;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 11)) << (15 - 11);
    out = out.offset(1);

    *out = ((*in_buf) >> 11) % (1u32 << 15);
    out = out.offset(1);
    *out = (*in_buf) >> 26;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 9)) << (15 - 9);
    out = out.offset(1);

    *out = ((*in_buf) >> 9) % (1u32 << 15);
    out = out.offset(1);
    *out = (*in_buf) >> 24;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 7)) << (15 - 7);
    out = out.offset(1);

    *out = ((*in_buf) >> 7) % (1u32 << 15);
    out = out.offset(1);
    *out = (*in_buf) >> 22;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 5)) << (15 - 5);
    out = out.offset(1);

    *out = ((*in_buf) >> 5) % (1u32 << 15);
    out = out.offset(1);
    *out = (*in_buf) >> 20;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 3)) << (15 - 3);
    out = out.offset(1);

    *out = ((*in_buf) >> 3) % (1u32 << 15);
    out = out.offset(1);
    *out = (*in_buf) >> 18;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 1)) << (15 - 1);
    out = out.offset(1);

    *out = ((*in_buf) >> 1) % (1u32 << 15);
    out = out.offset(1);
    *out = ((*in_buf) >> 16) % (1u32 << 15);
    out = out.offset(1);
    *out = (*in_buf) >> 31;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 14)) << (15 - 14);
    out = out.offset(1);

    *out = ((*in_buf) >> 14) % (1u32 << 15);
    out = out.offset(1);
    *out = (*in_buf) >> 29;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 12)) << (15 - 12);
    out = out.offset(1);

    *out = ((*in_buf) >> 12) % (1u32 << 15);
    out = out.offset(1);
    *out = (*in_buf) >> 27;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 10)) << (15 - 10);
    out = out.offset(1);

    *out = ((*in_buf) >> 10) % (1u32 << 15);
    out = out.offset(1);
    *out = (*in_buf) >> 25;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 8)) << (15 - 8);
    out = out.offset(1);

    *out = ((*in_buf) >> 8) % (1u32 << 15);
    out = out.offset(1);
    *out = (*in_buf) >> 23;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 6)) << (15 - 6);
    out = out.offset(1);

    *out = ((*in_buf) >> 6) % (1u32 << 15);
    out = out.offset(1);
    *out = (*in_buf) >> 21;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 4)) << (15 - 4);
    out = out.offset(1);

    *out = ((*in_buf) >> 4) % (1u32 << 15);
    out = out.offset(1);
    *out = (*in_buf) >> 19;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 2)) << (15 - 2);
    out = out.offset(1);

    *out = ((*in_buf) >> 2) % (1u32 << 15);
    out = out.offset(1);
    *out = (*in_buf) >> 17;

    in_buf.offset(1)
}

unsafe fn unpack16_32(mut in_buf: *const u32, mut out: *mut u32) -> *const u32 {
    *out = ((*in_buf) >> 0) % (1u32 << 16);
    out = out.offset(1);
    *out = (*in_buf) >> 16;
    out = out.offset(1);
    in_buf = in_buf.offset(1);

    *out = ((*in_buf) >> 0) % (1u32 << 16);
    out = out.offset(1);
    *out = (*in_buf) >> 16;
    out = out.offset(1);
    in_buf = in_buf.offset(1);

    *out = ((*in_buf) >> 0) % (1u32 << 16);
    out = out.offset(1);
    *out = (*in_buf) >> 16;
    out = out.offset(1);
    in_buf = in_buf.offset(1);

    *out = ((*in_buf) >> 0) % (1u32 << 16);
    out = out.offset(1);
    *out = (*in_buf) >> 16;
    out = out.offset(1);
    in_buf = in_buf.offset(1);

    *out = ((*in_buf) >> 0) % (1u32 << 16);
    out = out.offset(1);
    *out = (*in_buf) >> 16;
    out = out.offset(1);
    in_buf = in_buf.offset(1);

    *out = ((*in_buf) >> 0) % (1u32 << 16);
    out = out.offset(1);
    *out = (*in_buf) >> 16;
    out = out.offset(1);
    in_buf = in_buf.offset(1);

    *out = ((*in_buf) >> 0) % (1u32 << 16);
    out = out.offset(1);
    *out = (*in_buf) >> 16;
    out = out.offset(1);
    in_buf = in_buf.offset(1);

    *out = ((*in_buf) >> 0) % (1u32 << 16);
    out = out.offset(1);
    *out = (*in_buf) >> 16;
    out = out.offset(1);
    in_buf = in_buf.offset(1);

    *out = ((*in_buf) >> 0) % (1u32 << 16);
    out = out.offset(1);
    *out = (*in_buf) >> 16;
    out = out.offset(1);
    in_buf = in_buf.offset(1);

    *out = ((*in_buf) >> 0) % (1u32 << 16);
    out = out.offset(1);
    *out = (*in_buf) >> 16;
    out = out.offset(1);
    in_buf = in_buf.offset(1);

    *out = ((*in_buf) >> 0) % (1u32 << 16);
    out = out.offset(1);
    *out = (*in_buf) >> 16;
    out = out.offset(1);
    in_buf = in_buf.offset(1);

    *out = ((*in_buf) >> 0) % (1u32 << 16);
    out = out.offset(1);
    *out = (*in_buf) >> 16;
    out = out.offset(1);
    in_buf = in_buf.offset(1);

    *out = ((*in_buf) >> 0) % (1u32 << 16);
    out = out.offset(1);
    *out = (*in_buf) >> 16;
    out = out.offset(1);
    in_buf = in_buf.offset(1);

    *out = ((*in_buf) >> 0) % (1u32 << 16);
    out = out.offset(1);
    *out = (*in_buf) >> 16;
    out = out.offset(1);
    in_buf = in_buf.offset(1);

    *out = ((*in_buf) >> 0) % (1u32 << 16);
    out = out.offset(1);
    *out = (*in_buf) >> 16;
    out = out.offset(1);
    in_buf = in_buf.offset(1);

    *out = ((*in_buf) >> 0) % (1u32 << 16);
    out = out.offset(1);
    *out = (*in_buf) >> 16;

    in_buf.offset(1)
}

unsafe fn unpack17_32(mut in_buf: *const u32, mut out: *mut u32) -> *const u32 {
    *out = ((*in_buf) >> 0) % (1u32 << 17);
    out = out.offset(1);
    *out = (*in_buf) >> 17;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 2)) << (17 - 2);
    out = out.offset(1);

    *out = ((*in_buf) >> 2) % (1u32 << 17);
    out = out.offset(1);
    *out = (*in_buf) >> 19;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 4)) << (17 - 4);
    out = out.offset(1);

    *out = ((*in_buf) >> 4) % (1u32 << 17);
    out = out.offset(1);
    *out = (*in_buf) >> 21;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 6)) << (17 - 6);
    out = out.offset(1);

    *out = ((*in_buf) >> 6) % (1u32 << 17);
    out = out.offset(1);
    *out = (*in_buf) >> 23;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 8)) << (17 - 8);
    out = out.offset(1);

    *out = ((*in_buf) >> 8) % (1u32 << 17);
    out = out.offset(1);
    *out = (*in_buf) >> 25;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 10)) << (17 - 10);
    out = out.offset(1);

    *out = ((*in_buf) >> 10) % (1u32 << 17);
    out = out.offset(1);
    *out = (*in_buf) >> 27;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 12)) << (17 - 12);
    out = out.offset(1);

    *out = ((*in_buf) >> 12) % (1u32 << 17);
    out = out.offset(1);
    *out = (*in_buf) >> 29;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 14)) << (17 - 14);
    out = out.offset(1);

    *out = ((*in_buf) >> 14) % (1u32 << 17);
    out = out.offset(1);
    *out = (*in_buf) >> 31;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 16)) << (17 - 16);
    out = out.offset(1);

    *out = (*in_buf) >> 16;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 1)) << (17 - 1);
    out = out.offset(1);

    *out = ((*in_buf) >> 1) % (1u32 << 17);
    out = out.offset(1);
    *out = (*in_buf) >> 18;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 3)) << (17 - 3);
    out = out.offset(1);

    *out = ((*in_buf) >> 3) % (1u32 << 17);
    out = out.offset(1);
    *out = (*in_buf) >> 20;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 5)) << (17 - 5);
    out = out.offset(1);

    *out = ((*in_buf) >> 5) % (1u32 << 17);
    out = out.offset(1);
    *out = (*in_buf) >> 22;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 7)) << (17 - 7);
    out = out.offset(1);

    *out = ((*in_buf) >> 7) % (1u32 << 17);
    out = out.offset(1);
    *out = (*in_buf) >> 24;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 9)) << (17 - 9);
    out = out.offset(1);

    *out = ((*in_buf) >> 9) % (1u32 << 17);
    out = out.offset(1);
    *out = (*in_buf) >> 26;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 11)) << (17 - 11);
    out = out.offset(1);

    *out = ((*in_buf) >> 11) % (1u32 << 17);
    out = out.offset(1);
    *out = (*in_buf) >> 28;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 13)) << (17 - 13);
    out = out.offset(1);

    *out = ((*in_buf) >> 13) % (1u32 << 17);
    out = out.offset(1);
    *out = (*in_buf) >> 30;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 15)) << (17 - 15);
    out = out.offset(1);

    *out = (*in_buf) >> 15;

    in_buf.offset(1)
}

unsafe fn unpack18_32(mut in_buf: *const u32, mut out: *mut u32) -> *const u32 {
    *out = ((*in_buf) >> 0) % (1u32 << 18);
    out = out.offset(1);
    *out = (*in_buf) >> 18;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 4)) << (18 - 4);
    out = out.offset(1);

    *out = ((*in_buf) >> 4) % (1u32 << 18);
    out = out.offset(1);
    *out = (*in_buf) >> 22;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 8)) << (18 - 8);
    out = out.offset(1);

    *out = ((*in_buf) >> 8) % (1u32 << 18);
    out = out.offset(1);
    *out = (*in_buf) >> 26;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 12)) << (18 - 12);
    out = out.offset(1);

    *out = ((*in_buf) >> 12) % (1u32 << 18);
    out = out.offset(1);
    *out = (*in_buf) >> 30;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 16)) << (18 - 16);
    out = out.offset(1);

    *out = (*in_buf) >> 16;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 2)) << (18 - 2);
    out = out.offset(1);

    *out = ((*in_buf) >> 2) % (1u32 << 18);
    out = out.offset(1);
    *out = (*in_buf) >> 20;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 6)) << (18 - 6);
    out = out.offset(1);

    *out = ((*in_buf) >> 6) % (1u32 << 18);
    out = out.offset(1);
    *out = (*in_buf) >> 24;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 10)) << (18 - 10);
    out = out.offset(1);

    *out = ((*in_buf) >> 10) % (1u32 << 18);
    out = out.offset(1);
    *out = (*in_buf) >> 28;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 14)) << (18 - 14);
    out = out.offset(1);

    *out = (*in_buf) >> 14;
    in_buf = in_buf.offset(1);
    out = out.offset(1);

    *out = ((*in_buf) >> 0) % (1u32 << 18);
    out = out.offset(1);
    *out = (*in_buf) >> 18;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 4)) << (18 - 4);
    out = out.offset(1);

    *out = ((*in_buf) >> 4) % (1u32 << 18);
    out = out.offset(1);
    *out = (*in_buf) >> 22;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 8)) << (18 - 8);
    out = out.offset(1);

    *out = ((*in_buf) >> 8) % (1u32 << 18);
    out = out.offset(1);
    *out = (*in_buf) >> 26;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 12)) << (18 - 12);
    out = out.offset(1);

    *out = ((*in_buf) >> 12) % (1u32 << 18);
    out = out.offset(1);
    *out = (*in_buf) >> 30;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 16)) << (18 - 16);
    out = out.offset(1);

    *out = (*in_buf) >> 16;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 2)) << (18 - 2);
    out = out.offset(1);

    *out = ((*in_buf) >> 2) % (1u32 << 18);
    out = out.offset(1);
    *out = (*in_buf) >> 20;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 6)) << (18 - 6);
    out = out.offset(1);

    *out = ((*in_buf) >> 6) % (1u32 << 18);
    out = out.offset(1);
    *out = (*in_buf) >> 24;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 10)) << (18 - 10);
    out = out.offset(1);

    *out = ((*in_buf) >> 10) % (1u32 << 18);
    out = out.offset(1);
    *out = (*in_buf) >> 28;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 14)) << (18 - 14);
    out = out.offset(1);

    *out = (*in_buf) >> 14;

    in_buf.offset(1)
}

unsafe fn unpack19_32(mut in_buf: *const u32, mut out: *mut u32) -> *const u32 {
    *out = ((*in_buf) >> 0) % (1u32 << 19);
    out = out.offset(1);
    *out = (*in_buf) >> 19;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 6)) << (19 - 6);
    out = out.offset(1);

    *out = ((*in_buf) >> 6) % (1u32 << 19);
    out = out.offset(1);
    *out = (*in_buf) >> 25;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 12)) << (19 - 12);
    out = out.offset(1);

    *out = ((*in_buf) >> 12) % (1u32 << 19);
    out = out.offset(1);
    *out = (*in_buf) >> 31;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 18)) << (19 - 18);
    out = out.offset(1);

    *out = (*in_buf) >> 18;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 5)) << (19 - 5);
    out = out.offset(1);

    *out = ((*in_buf) >> 5) % (1u32 << 19);
    out = out.offset(1);
    *out = (*in_buf) >> 24;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 11)) << (19 - 11);
    out = out.offset(1);

    *out = ((*in_buf) >> 11) % (1u32 << 19);
    out = out.offset(1);
    *out = (*in_buf) >> 30;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 17)) << (19 - 17);
    out = out.offset(1);

    *out = (*in_buf) >> 17;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 4)) << (19 - 4);
    out = out.offset(1);

    *out = ((*in_buf) >> 4) % (1u32 << 19);
    out = out.offset(1);
    *out = (*in_buf) >> 23;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 10)) << (19 - 10);
    out = out.offset(1);

    *out = ((*in_buf) >> 10) % (1u32 << 19);
    out = out.offset(1);
    *out = (*in_buf) >> 29;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 16)) << (19 - 16);
    out = out.offset(1);

    *out = (*in_buf) >> 16;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 3)) << (19 - 3);
    out = out.offset(1);

    *out = ((*in_buf) >> 3) % (1u32 << 19);
    out = out.offset(1);
    *out = (*in_buf) >> 22;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 9)) << (19 - 9);
    out = out.offset(1);

    *out = ((*in_buf) >> 9) % (1u32 << 19);
    out = out.offset(1);
    *out = (*in_buf) >> 28;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 15)) << (19 - 15);
    out = out.offset(1);

    *out = (*in_buf) >> 15;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 2)) << (19 - 2);
    out = out.offset(1);

    *out = ((*in_buf) >> 2) % (1u32 << 19);
    out = out.offset(1);
    *out = (*in_buf) >> 21;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 8)) << (19 - 8);
    out = out.offset(1);

    *out = ((*in_buf) >> 8) % (1u32 << 19);
    out = out.offset(1);
    *out = (*in_buf) >> 27;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 14)) << (19 - 14);
    out = out.offset(1);

    *out = (*in_buf) >> 14;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 1)) << (19 - 1);
    out = out.offset(1);

    *out = ((*in_buf) >> 1) % (1u32 << 19);
    out = out.offset(1);
    *out = (*in_buf) >> 20;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 7)) << (19 - 7);
    out = out.offset(1);

    *out = ((*in_buf) >> 7) % (1u32 << 19);
    out = out.offset(1);
    *out = (*in_buf) >> 26;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 13)) << (19 - 13);
    out = out.offset(1);

    *out = (*in_buf) >> 13;

    in_buf.offset(1)
}

unsafe fn unpack20_32(mut in_buf: *const u32, mut out: *mut u32) -> *const u32 {
    *out = ((*in_buf) >> 0) % (1u32 << 20);
    out = out.offset(1);
    *out = (*in_buf) >> 20;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 8)) << (20 - 8);
    out = out.offset(1);

    *out = ((*in_buf) >> 8) % (1u32 << 20);
    out = out.offset(1);
    *out = (*in_buf) >> 28;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 16)) << (20 - 16);
    out = out.offset(1);

    *out = (*in_buf) >> 16;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 4)) << (20 - 4);
    out = out.offset(1);

    *out = ((*in_buf) >> 4) % (1u32 << 20);
    out = out.offset(1);
    *out = (*in_buf) >> 24;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 12)) << (20 - 12);
    out = out.offset(1);

    *out = (*in_buf) >> 12;
    in_buf = in_buf.offset(1);
    out = out.offset(1);

    *out = ((*in_buf) >> 0) % (1u32 << 20);
    out = out.offset(1);
    *out = (*in_buf) >> 20;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 8)) << (20 - 8);
    out = out.offset(1);

    *out = ((*in_buf) >> 8) % (1u32 << 20);
    out = out.offset(1);
    *out = (*in_buf) >> 28;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 16)) << (20 - 16);
    out = out.offset(1);

    *out = (*in_buf) >> 16;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 4)) << (20 - 4);
    out = out.offset(1);

    *out = ((*in_buf) >> 4) % (1u32 << 20);
    out = out.offset(1);
    *out = (*in_buf) >> 24;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 12)) << (20 - 12);
    out = out.offset(1);

    *out = (*in_buf) >> 12;
    in_buf = in_buf.offset(1);
    out = out.offset(1);

    *out = ((*in_buf) >> 0) % (1u32 << 20);
    out = out.offset(1);
    *out = (*in_buf) >> 20;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 8)) << (20 - 8);
    out = out.offset(1);

    *out = ((*in_buf) >> 8) % (1u32 << 20);
    out = out.offset(1);
    *out = (*in_buf) >> 28;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 16)) << (20 - 16);
    out = out.offset(1);

    *out = (*in_buf) >> 16;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 4)) << (20 - 4);
    out = out.offset(1);

    *out = ((*in_buf) >> 4) % (1u32 << 20);
    out = out.offset(1);
    *out = (*in_buf) >> 24;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 12)) << (20 - 12);
    out = out.offset(1);

    *out = (*in_buf) >> 12;
    in_buf = in_buf.offset(1);
    out = out.offset(1);

    *out = ((*in_buf) >> 0) % (1u32 << 20);
    out = out.offset(1);
    *out = (*in_buf) >> 20;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 8)) << (20 - 8);
    out = out.offset(1);

    *out = ((*in_buf) >> 8) % (1u32 << 20);
    out = out.offset(1);
    *out = (*in_buf) >> 28;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 16)) << (20 - 16);
    out = out.offset(1);

    *out = (*in_buf) >> 16;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 4)) << (20 - 4);
    out = out.offset(1);

    *out = ((*in_buf) >> 4) % (1u32 << 20);
    out = out.offset(1);
    *out = (*in_buf) >> 24;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 12)) << (20 - 12);
    out = out.offset(1);

    *out = (*in_buf) >> 12;

    in_buf.offset(1)
}

unsafe fn unpack21_32(mut in_buf: *const u32, mut out: *mut u32) -> *const u32 {
    *out = ((*in_buf) >> 0) % (1u32 << 21);
    out = out.offset(1);
    *out = (*in_buf) >> 21;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 10)) << (21 - 10);
    out = out.offset(1);

    *out = ((*in_buf) >> 10) % (1u32 << 21);
    out = out.offset(1);
    *out = (*in_buf) >> 31;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 20)) << (21 - 20);
    out = out.offset(1);

    *out = (*in_buf) >> 20;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 9)) << (21 - 9);
    out = out.offset(1);

    *out = ((*in_buf) >> 9) % (1u32 << 21);
    out = out.offset(1);
    *out = (*in_buf) >> 30;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 19)) << (21 - 19);
    out = out.offset(1);

    *out = (*in_buf) >> 19;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 8)) << (21 - 8);
    out = out.offset(1);

    *out = ((*in_buf) >> 8) % (1u32 << 21);
    out = out.offset(1);
    *out = (*in_buf) >> 29;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 18)) << (21 - 18);
    out = out.offset(1);

    *out = (*in_buf) >> 18;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 7)) << (21 - 7);
    out = out.offset(1);

    *out = ((*in_buf) >> 7) % (1u32 << 21);
    out = out.offset(1);
    *out = (*in_buf) >> 28;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 17)) << (21 - 17);
    out = out.offset(1);

    *out = (*in_buf) >> 17;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 6)) << (21 - 6);
    out = out.offset(1);

    *out = ((*in_buf) >> 6) % (1u32 << 21);
    out = out.offset(1);
    *out = (*in_buf) >> 27;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 16)) << (21 - 16);
    out = out.offset(1);

    *out = (*in_buf) >> 16;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 5)) << (21 - 5);
    out = out.offset(1);

    *out = ((*in_buf) >> 5) % (1u32 << 21);
    out = out.offset(1);
    *out = (*in_buf) >> 26;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 15)) << (21 - 15);
    out = out.offset(1);

    *out = (*in_buf) >> 15;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 4)) << (21 - 4);
    out = out.offset(1);

    *out = ((*in_buf) >> 4) % (1u32 << 21);
    out = out.offset(1);
    *out = (*in_buf) >> 25;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 14)) << (21 - 14);
    out = out.offset(1);

    *out = (*in_buf) >> 14;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 3)) << (21 - 3);
    out = out.offset(1);

    *out = ((*in_buf) >> 3) % (1u32 << 21);
    out = out.offset(1);
    *out = (*in_buf) >> 24;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 13)) << (21 - 13);
    out = out.offset(1);

    *out = (*in_buf) >> 13;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 2)) << (21 - 2);
    out = out.offset(1);

    *out = ((*in_buf) >> 2) % (1u32 << 21);
    out = out.offset(1);
    *out = (*in_buf) >> 23;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 12)) << (21 - 12);
    out = out.offset(1);

    *out = (*in_buf) >> 12;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 1)) << (21 - 1);
    out = out.offset(1);

    *out = ((*in_buf) >> 1) % (1u32 << 21);
    out = out.offset(1);
    *out = (*in_buf) >> 22;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 11)) << (21 - 11);
    out = out.offset(1);

    *out = (*in_buf) >> 11;

    in_buf.offset(1)
}

unsafe fn unpack22_32(mut in_buf: *const u32, mut out: *mut u32) -> *const u32 {
    *out = ((*in_buf) >> 0) % (1u32 << 22);
    out = out.offset(1);
    *out = (*in_buf) >> 22;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 12)) << (22 - 12);
    out = out.offset(1);

    *out = (*in_buf) >> 12;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 2)) << (22 - 2);
    out = out.offset(1);

    *out = ((*in_buf) >> 2) % (1u32 << 22);
    out = out.offset(1);
    *out = (*in_buf) >> 24;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 14)) << (22 - 14);
    out = out.offset(1);

    *out = (*in_buf) >> 14;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 4)) << (22 - 4);
    out = out.offset(1);

    *out = ((*in_buf) >> 4) % (1u32 << 22);
    out = out.offset(1);
    *out = (*in_buf) >> 26;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 16)) << (22 - 16);
    out = out.offset(1);

    *out = (*in_buf) >> 16;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 6)) << (22 - 6);
    out = out.offset(1);

    *out = ((*in_buf) >> 6) % (1u32 << 22);
    out = out.offset(1);
    *out = (*in_buf) >> 28;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 18)) << (22 - 18);
    out = out.offset(1);

    *out = (*in_buf) >> 18;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 8)) << (22 - 8);
    out = out.offset(1);

    *out = ((*in_buf) >> 8) % (1u32 << 22);
    out = out.offset(1);
    *out = (*in_buf) >> 30;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 20)) << (22 - 20);
    out = out.offset(1);

    *out = (*in_buf) >> 20;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 10)) << (22 - 10);
    out = out.offset(1);

    *out = (*in_buf) >> 10;
    in_buf = in_buf.offset(1);
    out = out.offset(1);

    *out = ((*in_buf) >> 0) % (1u32 << 22);
    out = out.offset(1);
    *out = (*in_buf) >> 22;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 12)) << (22 - 12);
    out = out.offset(1);

    *out = (*in_buf) >> 12;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 2)) << (22 - 2);
    out = out.offset(1);

    *out = ((*in_buf) >> 2) % (1u32 << 22);
    out = out.offset(1);
    *out = (*in_buf) >> 24;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 14)) << (22 - 14);
    out = out.offset(1);

    *out = (*in_buf) >> 14;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 4)) << (22 - 4);
    out = out.offset(1);

    *out = ((*in_buf) >> 4) % (1u32 << 22);
    out = out.offset(1);
    *out = (*in_buf) >> 26;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 16)) << (22 - 16);
    out = out.offset(1);

    *out = (*in_buf) >> 16;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 6)) << (22 - 6);
    out = out.offset(1);

    *out = ((*in_buf) >> 6) % (1u32 << 22);
    out = out.offset(1);
    *out = (*in_buf) >> 28;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 18)) << (22 - 18);
    out = out.offset(1);

    *out = (*in_buf) >> 18;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 8)) << (22 - 8);
    out = out.offset(1);

    *out = ((*in_buf) >> 8) % (1u32 << 22);
    out = out.offset(1);
    *out = (*in_buf) >> 30;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 20)) << (22 - 20);
    out = out.offset(1);

    *out = (*in_buf) >> 20;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 10)) << (22 - 10);
    out = out.offset(1);

    *out = (*in_buf) >> 10;

    in_buf.offset(1)
}

unsafe fn unpack23_32(mut in_buf: *const u32, mut out: *mut u32) -> *const u32 {
    *out = ((*in_buf) >> 0) % (1u32 << 23);
    out = out.offset(1);
    *out = (*in_buf) >> 23;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 14)) << (23 - 14);
    out = out.offset(1);

    *out = (*in_buf) >> 14;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 5)) << (23 - 5);
    out = out.offset(1);

    *out = ((*in_buf) >> 5) % (1u32 << 23);
    out = out.offset(1);
    *out = (*in_buf) >> 28;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 19)) << (23 - 19);
    out = out.offset(1);

    *out = (*in_buf) >> 19;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 10)) << (23 - 10);
    out = out.offset(1);

    *out = (*in_buf) >> 10;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 1)) << (23 - 1);
    out = out.offset(1);

    *out = ((*in_buf) >> 1) % (1u32 << 23);
    out = out.offset(1);
    *out = (*in_buf) >> 24;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 15)) << (23 - 15);
    out = out.offset(1);

    *out = (*in_buf) >> 15;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 6)) << (23 - 6);
    out = out.offset(1);

    *out = ((*in_buf) >> 6) % (1u32 << 23);
    out = out.offset(1);
    *out = (*in_buf) >> 29;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 20)) << (23 - 20);
    out = out.offset(1);

    *out = (*in_buf) >> 20;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 11)) << (23 - 11);
    out = out.offset(1);

    *out = (*in_buf) >> 11;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 2)) << (23 - 2);
    out = out.offset(1);

    *out = ((*in_buf) >> 2) % (1u32 << 23);
    out = out.offset(1);
    *out = (*in_buf) >> 25;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 16)) << (23 - 16);
    out = out.offset(1);

    *out = (*in_buf) >> 16;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 7)) << (23 - 7);
    out = out.offset(1);

    *out = ((*in_buf) >> 7) % (1u32 << 23);
    out = out.offset(1);
    *out = (*in_buf) >> 30;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 21)) << (23 - 21);
    out = out.offset(1);

    *out = (*in_buf) >> 21;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 12)) << (23 - 12);
    out = out.offset(1);

    *out = (*in_buf) >> 12;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 3)) << (23 - 3);
    out = out.offset(1);

    *out = ((*in_buf) >> 3) % (1u32 << 23);
    out = out.offset(1);
    *out = (*in_buf) >> 26;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 17)) << (23 - 17);
    out = out.offset(1);

    *out = (*in_buf) >> 17;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 8)) << (23 - 8);
    out = out.offset(1);

    *out = ((*in_buf) >> 8) % (1u32 << 23);
    out = out.offset(1);
    *out = (*in_buf) >> 31;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 22)) << (23 - 22);
    out = out.offset(1);

    *out = (*in_buf) >> 22;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 13)) << (23 - 13);
    out = out.offset(1);

    *out = (*in_buf) >> 13;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 4)) << (23 - 4);
    out = out.offset(1);

    *out = ((*in_buf) >> 4) % (1u32 << 23);
    out = out.offset(1);
    *out = (*in_buf) >> 27;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 18)) << (23 - 18);
    out = out.offset(1);

    *out = (*in_buf) >> 18;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 9)) << (23 - 9);
    out = out.offset(1);

    *out = (*in_buf) >> 9;

    in_buf.offset(1)
}

unsafe fn unpack24_32(mut in_buf: *const u32, mut out: *mut u32) -> *const u32 {
    *out = ((*in_buf) >> 0) % (1u32 << 24);
    out = out.offset(1);
    *out = (*in_buf) >> 24;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 16)) << (24 - 16);
    out = out.offset(1);

    *out = (*in_buf) >> 16;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 8)) << (24 - 8);
    out = out.offset(1);

    *out = (*in_buf) >> 8;
    in_buf = in_buf.offset(1);
    out = out.offset(1);

    *out = ((*in_buf) >> 0) % (1u32 << 24);
    out = out.offset(1);
    *out = (*in_buf) >> 24;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 16)) << (24 - 16);
    out = out.offset(1);

    *out = (*in_buf) >> 16;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 8)) << (24 - 8);
    out = out.offset(1);

    *out = (*in_buf) >> 8;
    in_buf = in_buf.offset(1);
    out = out.offset(1);

    *out = ((*in_buf) >> 0) % (1u32 << 24);
    out = out.offset(1);
    *out = (*in_buf) >> 24;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 16)) << (24 - 16);
    out = out.offset(1);

    *out = (*in_buf) >> 16;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 8)) << (24 - 8);
    out = out.offset(1);

    *out = (*in_buf) >> 8;
    in_buf = in_buf.offset(1);
    out = out.offset(1);

    *out = ((*in_buf) >> 0) % (1u32 << 24);
    out = out.offset(1);
    *out = (*in_buf) >> 24;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 16)) << (24 - 16);
    out = out.offset(1);

    *out = (*in_buf) >> 16;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 8)) << (24 - 8);
    out = out.offset(1);

    *out = (*in_buf) >> 8;
    in_buf = in_buf.offset(1);
    out = out.offset(1);

    *out = ((*in_buf) >> 0) % (1u32 << 24);
    out = out.offset(1);
    *out = (*in_buf) >> 24;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 16)) << (24 - 16);
    out = out.offset(1);

    *out = (*in_buf) >> 16;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 8)) << (24 - 8);
    out = out.offset(1);

    *out = (*in_buf) >> 8;
    in_buf = in_buf.offset(1);
    out = out.offset(1);

    *out = ((*in_buf) >> 0) % (1u32 << 24);
    out = out.offset(1);
    *out = (*in_buf) >> 24;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 16)) << (24 - 16);
    out = out.offset(1);

    *out = (*in_buf) >> 16;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 8)) << (24 - 8);
    out = out.offset(1);

    *out = (*in_buf) >> 8;
    in_buf = in_buf.offset(1);
    out = out.offset(1);

    *out = ((*in_buf) >> 0) % (1u32 << 24);
    out = out.offset(1);
    *out = (*in_buf) >> 24;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 16)) << (24 - 16);
    out = out.offset(1);

    *out = (*in_buf) >> 16;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 8)) << (24 - 8);
    out = out.offset(1);

    *out = (*in_buf) >> 8;
    in_buf = in_buf.offset(1);
    out = out.offset(1);

    *out = ((*in_buf) >> 0) % (1u32 << 24);
    out = out.offset(1);
    *out = (*in_buf) >> 24;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 16)) << (24 - 16);
    out = out.offset(1);

    *out = (*in_buf) >> 16;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 8)) << (24 - 8);
    out = out.offset(1);

    *out = (*in_buf) >> 8;

    in_buf.offset(1)
}

unsafe fn unpack25_32(mut in_buf: *const u32, mut out: *mut u32) -> *const u32 {
    *out = ((*in_buf) >> 0) % (1u32 << 25);
    out = out.offset(1);
    *out = (*in_buf) >> 25;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 18)) << (25 - 18);
    out = out.offset(1);

    *out = (*in_buf) >> 18;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 11)) << (25 - 11);
    out = out.offset(1);

    *out = (*in_buf) >> 11;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 4)) << (25 - 4);
    out = out.offset(1);

    *out = ((*in_buf) >> 4) % (1u32 << 25);
    out = out.offset(1);
    *out = (*in_buf) >> 29;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 22)) << (25 - 22);
    out = out.offset(1);

    *out = (*in_buf) >> 22;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 15)) << (25 - 15);
    out = out.offset(1);

    *out = (*in_buf) >> 15;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 8)) << (25 - 8);
    out = out.offset(1);

    *out = (*in_buf) >> 8;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 1)) << (25 - 1);
    out = out.offset(1);

    *out = ((*in_buf) >> 1) % (1u32 << 25);
    out = out.offset(1);
    *out = (*in_buf) >> 26;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 19)) << (25 - 19);
    out = out.offset(1);

    *out = (*in_buf) >> 19;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 12)) << (25 - 12);
    out = out.offset(1);

    *out = (*in_buf) >> 12;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 5)) << (25 - 5);
    out = out.offset(1);

    *out = ((*in_buf) >> 5) % (1u32 << 25);
    out = out.offset(1);
    *out = (*in_buf) >> 30;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 23)) << (25 - 23);
    out = out.offset(1);

    *out = (*in_buf) >> 23;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 16)) << (25 - 16);
    out = out.offset(1);

    *out = (*in_buf) >> 16;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 9)) << (25 - 9);
    out = out.offset(1);

    *out = (*in_buf) >> 9;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 2)) << (25 - 2);
    out = out.offset(1);

    *out = ((*in_buf) >> 2) % (1u32 << 25);
    out = out.offset(1);
    *out = (*in_buf) >> 27;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 20)) << (25 - 20);
    out = out.offset(1);

    *out = (*in_buf) >> 20;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 13)) << (25 - 13);
    out = out.offset(1);

    *out = (*in_buf) >> 13;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 6)) << (25 - 6);
    out = out.offset(1);

    *out = ((*in_buf) >> 6) % (1u32 << 25);
    out = out.offset(1);
    *out = (*in_buf) >> 31;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 24)) << (25 - 24);
    out = out.offset(1);

    *out = (*in_buf) >> 24;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 17)) << (25 - 17);
    out = out.offset(1);

    *out = (*in_buf) >> 17;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 10)) << (25 - 10);
    out = out.offset(1);

    *out = (*in_buf) >> 10;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 3)) << (25 - 3);
    out = out.offset(1);

    *out = ((*in_buf) >> 3) % (1u32 << 25);
    out = out.offset(1);
    *out = (*in_buf) >> 28;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 21)) << (25 - 21);
    out = out.offset(1);

    *out = (*in_buf) >> 21;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 14)) << (25 - 14);
    out = out.offset(1);

    *out = (*in_buf) >> 14;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 7)) << (25 - 7);
    out = out.offset(1);

    *out = (*in_buf) >> 7;

    in_buf.offset(1)
}

unsafe fn unpack26_32(mut in_buf: *const u32, mut out: *mut u32) -> *const u32 {
    *out = ((*in_buf) >> 0) % (1u32 << 26);
    out = out.offset(1);
    *out = (*in_buf) >> 26;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 20)) << (26 - 20);
    out = out.offset(1);

    *out = (*in_buf) >> 20;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 14)) << (26 - 14);
    out = out.offset(1);

    *out = (*in_buf) >> 14;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 8)) << (26 - 8);
    out = out.offset(1);

    *out = (*in_buf) >> 8;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 2)) << (26 - 2);
    out = out.offset(1);

    *out = ((*in_buf) >> 2) % (1u32 << 26);
    out = out.offset(1);
    *out = (*in_buf) >> 28;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 22)) << (26 - 22);
    out = out.offset(1);

    *out = (*in_buf) >> 22;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 16)) << (26 - 16);
    out = out.offset(1);

    *out = (*in_buf) >> 16;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 10)) << (26 - 10);
    out = out.offset(1);

    *out = (*in_buf) >> 10;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 4)) << (26 - 4);
    out = out.offset(1);

    *out = ((*in_buf) >> 4) % (1u32 << 26);
    out = out.offset(1);
    *out = (*in_buf) >> 30;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 24)) << (26 - 24);
    out = out.offset(1);

    *out = (*in_buf) >> 24;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 18)) << (26 - 18);
    out = out.offset(1);

    *out = (*in_buf) >> 18;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 12)) << (26 - 12);
    out = out.offset(1);

    *out = (*in_buf) >> 12;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 6)) << (26 - 6);
    out = out.offset(1);

    *out = (*in_buf) >> 6;
    in_buf = in_buf.offset(1);
    out = out.offset(1);

    *out = ((*in_buf) >> 0) % (1u32 << 26);
    out = out.offset(1);
    *out = (*in_buf) >> 26;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 20)) << (26 - 20);
    out = out.offset(1);

    *out = (*in_buf) >> 20;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 14)) << (26 - 14);
    out = out.offset(1);

    *out = (*in_buf) >> 14;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 8)) << (26 - 8);
    out = out.offset(1);

    *out = (*in_buf) >> 8;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 2)) << (26 - 2);
    out = out.offset(1);

    *out = ((*in_buf) >> 2) % (1u32 << 26);
    out = out.offset(1);
    *out = (*in_buf) >> 28;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 22)) << (26 - 22);
    out = out.offset(1);

    *out = (*in_buf) >> 22;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 16)) << (26 - 16);
    out = out.offset(1);

    *out = (*in_buf) >> 16;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 10)) << (26 - 10);
    out = out.offset(1);

    *out = (*in_buf) >> 10;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 4)) << (26 - 4);
    out = out.offset(1);

    *out = ((*in_buf) >> 4) % (1u32 << 26);
    out = out.offset(1);
    *out = (*in_buf) >> 30;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 24)) << (26 - 24);
    out = out.offset(1);

    *out = (*in_buf) >> 24;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 18)) << (26 - 18);
    out = out.offset(1);

    *out = (*in_buf) >> 18;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 12)) << (26 - 12);
    out = out.offset(1);

    *out = (*in_buf) >> 12;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 6)) << (26 - 6);
    out = out.offset(1);

    *out = (*in_buf) >> 6;

    in_buf.offset(1)
}

unsafe fn unpack27_32(mut in_buf: *const u32, mut out: *mut u32) -> *const u32 {
    *out = ((*in_buf) >> 0) % (1u32 << 27);
    out = out.offset(1);
    *out = (*in_buf) >> 27;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 22)) << (27 - 22);
    out = out.offset(1);

    *out = (*in_buf) >> 22;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 17)) << (27 - 17);
    out = out.offset(1);

    *out = (*in_buf) >> 17;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 12)) << (27 - 12);
    out = out.offset(1);

    *out = (*in_buf) >> 12;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 7)) << (27 - 7);
    out = out.offset(1);

    *out = (*in_buf) >> 7;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 2)) << (27 - 2);
    out = out.offset(1);

    *out = ((*in_buf) >> 2) % (1u32 << 27);
    out = out.offset(1);
    *out = (*in_buf) >> 29;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 24)) << (27 - 24);
    out = out.offset(1);

    *out = (*in_buf) >> 24;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 19)) << (27 - 19);
    out = out.offset(1);

    *out = (*in_buf) >> 19;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 14)) << (27 - 14);
    out = out.offset(1);

    *out = (*in_buf) >> 14;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 9)) << (27 - 9);
    out = out.offset(1);

    *out = (*in_buf) >> 9;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 4)) << (27 - 4);
    out = out.offset(1);

    *out = ((*in_buf) >> 4) % (1u32 << 27);
    out = out.offset(1);
    *out = (*in_buf) >> 31;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 26)) << (27 - 26);
    out = out.offset(1);

    *out = (*in_buf) >> 26;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 21)) << (27 - 21);
    out = out.offset(1);

    *out = (*in_buf) >> 21;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 16)) << (27 - 16);
    out = out.offset(1);

    *out = (*in_buf) >> 16;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 11)) << (27 - 11);
    out = out.offset(1);

    *out = (*in_buf) >> 11;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 6)) << (27 - 6);
    out = out.offset(1);

    *out = (*in_buf) >> 6;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 1)) << (27 - 1);
    out = out.offset(1);

    *out = ((*in_buf) >> 1) % (1u32 << 27);
    out = out.offset(1);
    *out = (*in_buf) >> 28;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 23)) << (27 - 23);
    out = out.offset(1);

    *out = (*in_buf) >> 23;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 18)) << (27 - 18);
    out = out.offset(1);

    *out = (*in_buf) >> 18;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 13)) << (27 - 13);
    out = out.offset(1);

    *out = (*in_buf) >> 13;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 8)) << (27 - 8);
    out = out.offset(1);

    *out = (*in_buf) >> 8;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 3)) << (27 - 3);
    out = out.offset(1);

    *out = ((*in_buf) >> 3) % (1u32 << 27);
    out = out.offset(1);
    *out = (*in_buf) >> 30;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 25)) << (27 - 25);
    out = out.offset(1);

    *out = (*in_buf) >> 25;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 20)) << (27 - 20);
    out = out.offset(1);

    *out = (*in_buf) >> 20;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 15)) << (27 - 15);
    out = out.offset(1);

    *out = (*in_buf) >> 15;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 10)) << (27 - 10);
    out = out.offset(1);

    *out = (*in_buf) >> 10;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 5)) << (27 - 5);
    out = out.offset(1);

    *out = (*in_buf) >> 5;

    in_buf.offset(1)
}

unsafe fn unpack28_32(mut in_buf: *const u32, mut out: *mut u32) -> *const u32 {
    *out = ((*in_buf) >> 0) % (1u32 << 28);
    out = out.offset(1);
    *out = (*in_buf) >> 28;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 24)) << (28 - 24);
    out = out.offset(1);

    *out = (*in_buf) >> 24;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 20)) << (28 - 20);
    out = out.offset(1);

    *out = (*in_buf) >> 20;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 16)) << (28 - 16);
    out = out.offset(1);

    *out = (*in_buf) >> 16;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 12)) << (28 - 12);
    out = out.offset(1);

    *out = (*in_buf) >> 12;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 8)) << (28 - 8);
    out = out.offset(1);

    *out = (*in_buf) >> 8;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 4)) << (28 - 4);
    out = out.offset(1);

    *out = (*in_buf) >> 4;
    in_buf = in_buf.offset(1);
    out = out.offset(1);

    *out = ((*in_buf) >> 0) % (1u32 << 28);
    out = out.offset(1);
    *out = (*in_buf) >> 28;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 24)) << (28 - 24);
    out = out.offset(1);

    *out = (*in_buf) >> 24;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 20)) << (28 - 20);
    out = out.offset(1);

    *out = (*in_buf) >> 20;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 16)) << (28 - 16);
    out = out.offset(1);

    *out = (*in_buf) >> 16;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 12)) << (28 - 12);
    out = out.offset(1);

    *out = (*in_buf) >> 12;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 8)) << (28 - 8);
    out = out.offset(1);

    *out = (*in_buf) >> 8;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 4)) << (28 - 4);
    out = out.offset(1);

    *out = (*in_buf) >> 4;
    in_buf = in_buf.offset(1);
    out = out.offset(1);

    *out = ((*in_buf) >> 0) % (1u32 << 28);
    out = out.offset(1);
    *out = (*in_buf) >> 28;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 24)) << (28 - 24);
    out = out.offset(1);

    *out = (*in_buf) >> 24;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 20)) << (28 - 20);
    out = out.offset(1);

    *out = (*in_buf) >> 20;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 16)) << (28 - 16);
    out = out.offset(1);

    *out = (*in_buf) >> 16;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 12)) << (28 - 12);
    out = out.offset(1);

    *out = (*in_buf) >> 12;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 8)) << (28 - 8);
    out = out.offset(1);

    *out = (*in_buf) >> 8;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 4)) << (28 - 4);
    out = out.offset(1);

    *out = (*in_buf) >> 4;
    in_buf = in_buf.offset(1);
    out = out.offset(1);

    *out = ((*in_buf) >> 0) % (1u32 << 28);
    out = out.offset(1);
    *out = (*in_buf) >> 28;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 24)) << (28 - 24);
    out = out.offset(1);

    *out = (*in_buf) >> 24;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 20)) << (28 - 20);
    out = out.offset(1);

    *out = (*in_buf) >> 20;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 16)) << (28 - 16);
    out = out.offset(1);

    *out = (*in_buf) >> 16;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 12)) << (28 - 12);
    out = out.offset(1);

    *out = (*in_buf) >> 12;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 8)) << (28 - 8);
    out = out.offset(1);

    *out = (*in_buf) >> 8;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 4)) << (28 - 4);
    out = out.offset(1);

    *out = (*in_buf) >> 4;

    in_buf.offset(1)
}

unsafe fn unpack29_32(mut in_buf: *const u32, mut out: *mut u32) -> *const u32 {
    *out = ((*in_buf) >> 0) % (1u32 << 29);
    out = out.offset(1);
    *out = (*in_buf) >> 29;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 26)) << (29 - 26);
    out = out.offset(1);

    *out = (*in_buf) >> 26;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 23)) << (29 - 23);
    out = out.offset(1);

    *out = (*in_buf) >> 23;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 20)) << (29 - 20);
    out = out.offset(1);

    *out = (*in_buf) >> 20;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 17)) << (29 - 17);
    out = out.offset(1);

    *out = (*in_buf) >> 17;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 14)) << (29 - 14);
    out = out.offset(1);

    *out = (*in_buf) >> 14;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 11)) << (29 - 11);
    out = out.offset(1);

    *out = (*in_buf) >> 11;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 8)) << (29 - 8);
    out = out.offset(1);

    *out = (*in_buf) >> 8;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 5)) << (29 - 5);
    out = out.offset(1);

    *out = (*in_buf) >> 5;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 2)) << (29 - 2);
    out = out.offset(1);

    *out = ((*in_buf) >> 2) % (1u32 << 29);
    out = out.offset(1);
    *out = (*in_buf) >> 31;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 28)) << (29 - 28);
    out = out.offset(1);

    *out = (*in_buf) >> 28;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 25)) << (29 - 25);
    out = out.offset(1);

    *out = (*in_buf) >> 25;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 22)) << (29 - 22);
    out = out.offset(1);

    *out = (*in_buf) >> 22;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 19)) << (29 - 19);
    out = out.offset(1);

    *out = (*in_buf) >> 19;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 16)) << (29 - 16);
    out = out.offset(1);

    *out = (*in_buf) >> 16;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 13)) << (29 - 13);
    out = out.offset(1);

    *out = (*in_buf) >> 13;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 10)) << (29 - 10);
    out = out.offset(1);

    *out = (*in_buf) >> 10;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 7)) << (29 - 7);
    out = out.offset(1);

    *out = (*in_buf) >> 7;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 4)) << (29 - 4);
    out = out.offset(1);

    *out = (*in_buf) >> 4;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 1)) << (29 - 1);
    out = out.offset(1);

    *out = ((*in_buf) >> 1) % (1u32 << 29);
    out = out.offset(1);
    *out = (*in_buf) >> 30;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 27)) << (29 - 27);
    out = out.offset(1);

    *out = (*in_buf) >> 27;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 24)) << (29 - 24);
    out = out.offset(1);

    *out = (*in_buf) >> 24;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 21)) << (29 - 21);
    out = out.offset(1);

    *out = (*in_buf) >> 21;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 18)) << (29 - 18);
    out = out.offset(1);

    *out = (*in_buf) >> 18;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 15)) << (29 - 15);
    out = out.offset(1);

    *out = (*in_buf) >> 15;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 12)) << (29 - 12);
    out = out.offset(1);

    *out = (*in_buf) >> 12;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 9)) << (29 - 9);
    out = out.offset(1);

    *out = (*in_buf) >> 9;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 6)) << (29 - 6);
    out = out.offset(1);

    *out = (*in_buf) >> 6;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 3)) << (29 - 3);
    out = out.offset(1);

    *out = (*in_buf) >> 3;

    in_buf.offset(1)
}

unsafe fn unpack30_32(mut in_buf: *const u32, mut out: *mut u32) -> *const u32 {
    *out = ((*in_buf) >> 0) % (1u32 << 30);
    out = out.offset(1);
    *out = (*in_buf) >> 30;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 28)) << (30 - 28);
    out = out.offset(1);

    *out = (*in_buf) >> 28;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 26)) << (30 - 26);
    out = out.offset(1);

    *out = (*in_buf) >> 26;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 24)) << (30 - 24);
    out = out.offset(1);

    *out = (*in_buf) >> 24;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 22)) << (30 - 22);
    out = out.offset(1);

    *out = (*in_buf) >> 22;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 20)) << (30 - 20);
    out = out.offset(1);

    *out = (*in_buf) >> 20;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 18)) << (30 - 18);
    out = out.offset(1);

    *out = (*in_buf) >> 18;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 16)) << (30 - 16);
    out = out.offset(1);

    *out = (*in_buf) >> 16;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 14)) << (30 - 14);
    out = out.offset(1);

    *out = (*in_buf) >> 14;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 12)) << (30 - 12);
    out = out.offset(1);

    *out = (*in_buf) >> 12;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 10)) << (30 - 10);
    out = out.offset(1);

    *out = (*in_buf) >> 10;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 8)) << (30 - 8);
    out = out.offset(1);

    *out = (*in_buf) >> 8;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 6)) << (30 - 6);
    out = out.offset(1);

    *out = (*in_buf) >> 6;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 4)) << (30 - 4);
    out = out.offset(1);

    *out = (*in_buf) >> 4;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 2)) << (30 - 2);
    out = out.offset(1);

    *out = (*in_buf) >> 2;
    in_buf = in_buf.offset(1);
    out = out.offset(1);

    *out = ((*in_buf) >> 0) % (1u32 << 30);
    out = out.offset(1);
    *out = (*in_buf) >> 30;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 28)) << (30 - 28);
    out = out.offset(1);

    *out = (*in_buf) >> 28;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 26)) << (30 - 26);
    out = out.offset(1);

    *out = (*in_buf) >> 26;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 24)) << (30 - 24);
    out = out.offset(1);

    *out = (*in_buf) >> 24;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 22)) << (30 - 22);
    out = out.offset(1);

    *out = (*in_buf) >> 22;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 20)) << (30 - 20);
    out = out.offset(1);

    *out = (*in_buf) >> 20;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 18)) << (30 - 18);
    out = out.offset(1);

    *out = (*in_buf) >> 18;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 16)) << (30 - 16);
    out = out.offset(1);

    *out = (*in_buf) >> 16;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 14)) << (30 - 14);
    out = out.offset(1);

    *out = (*in_buf) >> 14;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 12)) << (30 - 12);
    out = out.offset(1);

    *out = (*in_buf) >> 12;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 10)) << (30 - 10);
    out = out.offset(1);

    *out = (*in_buf) >> 10;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 8)) << (30 - 8);
    out = out.offset(1);

    *out = (*in_buf) >> 8;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 6)) << (30 - 6);
    out = out.offset(1);

    *out = (*in_buf) >> 6;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 4)) << (30 - 4);
    out = out.offset(1);

    *out = (*in_buf) >> 4;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 2)) << (30 - 2);
    out = out.offset(1);

    *out = (*in_buf) >> 2;

    in_buf.offset(1)
}

unsafe fn unpack31_32(mut in_buf: *const u32, mut out: *mut u32) -> *const u32 {
    *out = ((*in_buf) >> 0) % (1u32 << 31);
    out = out.offset(1);
    *out = (*in_buf) >> 31;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 30)) << (31 - 30);
    out = out.offset(1);

    *out = (*in_buf) >> 30;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 29)) << (31 - 29);
    out = out.offset(1);

    *out = (*in_buf) >> 29;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 28)) << (31 - 28);
    out = out.offset(1);

    *out = (*in_buf) >> 28;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 27)) << (31 - 27);
    out = out.offset(1);

    *out = (*in_buf) >> 27;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 26)) << (31 - 26);
    out = out.offset(1);

    *out = (*in_buf) >> 26;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 25)) << (31 - 25);
    out = out.offset(1);

    *out = (*in_buf) >> 25;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 24)) << (31 - 24);
    out = out.offset(1);

    *out = (*in_buf) >> 24;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 23)) << (31 - 23);
    out = out.offset(1);

    *out = (*in_buf) >> 23;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 22)) << (31 - 22);
    out = out.offset(1);

    *out = (*in_buf) >> 22;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 21)) << (31 - 21);
    out = out.offset(1);

    *out = (*in_buf) >> 21;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 20)) << (31 - 20);
    out = out.offset(1);

    *out = (*in_buf) >> 20;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 19)) << (31 - 19);
    out = out.offset(1);

    *out = (*in_buf) >> 19;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 18)) << (31 - 18);
    out = out.offset(1);

    *out = (*in_buf) >> 18;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 17)) << (31 - 17);
    out = out.offset(1);

    *out = (*in_buf) >> 17;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 16)) << (31 - 16);
    out = out.offset(1);

    *out = (*in_buf) >> 16;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 15)) << (31 - 15);
    out = out.offset(1);

    *out = (*in_buf) >> 15;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 14)) << (31 - 14);
    out = out.offset(1);

    *out = (*in_buf) >> 14;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 13)) << (31 - 13);
    out = out.offset(1);

    *out = (*in_buf) >> 13;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 12)) << (31 - 12);
    out = out.offset(1);

    *out = (*in_buf) >> 12;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 11)) << (31 - 11);
    out = out.offset(1);

    *out = (*in_buf) >> 11;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 10)) << (31 - 10);
    out = out.offset(1);

    *out = (*in_buf) >> 10;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 9)) << (31 - 9);
    out = out.offset(1);

    *out = (*in_buf) >> 9;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 8)) << (31 - 8);
    out = out.offset(1);

    *out = (*in_buf) >> 8;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 7)) << (31 - 7);
    out = out.offset(1);

    *out = (*in_buf) >> 7;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 6)) << (31 - 6);
    out = out.offset(1);

    *out = (*in_buf) >> 6;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 5)) << (31 - 5);
    out = out.offset(1);

    *out = (*in_buf) >> 5;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 4)) << (31 - 4);
    out = out.offset(1);

    *out = (*in_buf) >> 4;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 3)) << (31 - 3);
    out = out.offset(1);

    *out = (*in_buf) >> 3;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 2)) << (31 - 2);
    out = out.offset(1);

    *out = (*in_buf) >> 2;
    in_buf = in_buf.offset(1);
    *out |= ((*in_buf) % (1u32 << 1)) << (31 - 1);
    out = out.offset(1);

    *out = (*in_buf) >> 1;

    in_buf.offset(1)
}

unsafe fn unpack32_32(mut in_buf: *const u32, mut out: *mut u32) -> *const u32 {
    *out = (*in_buf) >> 0;
    in_buf = in_buf.offset(1);
    out = out.offset(1);

    *out = (*in_buf) >> 0;
    in_buf = in_buf.offset(1);
    out = out.offset(1);

    *out = (*in_buf) >> 0;
    in_buf = in_buf.offset(1);
    out = out.offset(1);

    *out = (*in_buf) >> 0;
    in_buf = in_buf.offset(1);
    out = out.offset(1);

    *out = (*in_buf) >> 0;
    in_buf = in_buf.offset(1);
    out = out.offset(1);

    *out = (*in_buf) >> 0;
    in_buf = in_buf.offset(1);
    out = out.offset(1);

    *out = (*in_buf) >> 0;
    in_buf = in_buf.offset(1);
    out = out.offset(1);

    *out = (*in_buf) >> 0;
    in_buf = in_buf.offset(1);
    out = out.offset(1);

    *out = (*in_buf) >> 0;
    in_buf = in_buf.offset(1);
    out = out.offset(1);

    *out = (*in_buf) >> 0;
    in_buf = in_buf.offset(1);
    out = out.offset(1);

    *out = (*in_buf) >> 0;
    in_buf = in_buf.offset(1);
    out = out.offset(1);

    *out = (*in_buf) >> 0;
    in_buf = in_buf.offset(1);
    out = out.offset(1);

    *out = (*in_buf) >> 0;
    in_buf = in_buf.offset(1);
    out = out.offset(1);

    *out = (*in_buf) >> 0;
    in_buf = in_buf.offset(1);
    out = out.offset(1);

    *out = (*in_buf) >> 0;
    in_buf = in_buf.offset(1);
    out = out.offset(1);

    *out = (*in_buf) >> 0;
    in_buf = in_buf.offset(1);
    out = out.offset(1);

    *out = (*in_buf) >> 0;
    in_buf = in_buf.offset(1);
    out = out.offset(1);

    *out = (*in_buf) >> 0;
    in_buf = in_buf.offset(1);
    out = out.offset(1);

    *out = (*in_buf) >> 0;
    in_buf = in_buf.offset(1);
    out = out.offset(1);

    *out = (*in_buf) >> 0;
    in_buf = in_buf.offset(1);
    out = out.offset(1);

    *out = (*in_buf) >> 0;
    in_buf = in_buf.offset(1);
    out = out.offset(1);

    *out = (*in_buf) >> 0;
    in_buf = in_buf.offset(1);
    out = out.offset(1);

    *out = (*in_buf) >> 0;
    in_buf = in_buf.offset(1);
    out = out.offset(1);

    *out = (*in_buf) >> 0;
    in_buf = in_buf.offset(1);
    out = out.offset(1);

    *out = (*in_buf) >> 0;
    in_buf = in_buf.offset(1);
    out = out.offset(1);

    *out = (*in_buf) >> 0;
    in_buf = in_buf.offset(1);
    out = out.offset(1);

    *out = (*in_buf) >> 0;
    in_buf = in_buf.offset(1);
    out = out.offset(1);

    *out = (*in_buf) >> 0;
    in_buf = in_buf.offset(1);
    out = out.offset(1);

    *out = (*in_buf) >> 0;
    in_buf = in_buf.offset(1);
    out = out.offset(1);

    *out = (*in_buf) >> 0;
    in_buf = in_buf.offset(1);
    out = out.offset(1);

    *out = (*in_buf) >> 0;
    in_buf = in_buf.offset(1);
    out = out.offset(1);

    *out = (*in_buf) >> 0;

    in_buf.offset(1)
}
