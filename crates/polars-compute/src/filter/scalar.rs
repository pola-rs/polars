use std::mem::MaybeUninit;

use arrow::array::PrimitiveArray;
use arrow::bitmap::Bitmap;
use arrow::types::NativeType;
use bytemuck::Pod;
use polars_utils::slice::load_padded_le_u64;


// If the ith bit of m is set (m & (1 << i)), then v[i] must be in-bounds.
// out must be valid for at least m.count_ones() + 4 writes.
// m_popcnt must be m.count_ones()
unsafe fn scalar_sparse_unchecked<T: Pod>(v: &[T], mut m: u64, mut m_popcnt: u32, out: *mut T) {
    let mut written = 0;
    
    while m > 0 {
        let idx = m.trailing_zeros() as usize;
        *out.add(written as usize) = *v.get_unchecked(idx);
        m &= m.wrapping_sub(1); // Clear least significant bit.
        written += 1;

        // tz % 64 otherwise we could go out of bounds
        let idx = (m.trailing_zeros() % 64) as usize;
        *out.add(written as usize) = *v.get_unchecked(idx);
        m &= m.wrapping_sub(1); // Clear least significant bit.
        written += 1;
    }
}

// v.len() >= 64 must hold.
// out must be valid for at least m.count_ones() + 1 writes.
unsafe fn scalar_filter64_unchecked<T: Pod>(v: &[T], mut m: u64, out: *mut T) {
    // Rust generated significantly better code if we write the below loop
    // with v as a pointer, and out.add(written) instead of incrementing out
    // directly.
    let mut written = 0usize;
    let mut src = v.as_ptr();

    // while m > 0 {
    //     // Skip whole bytes that are empty.
    //     let empty_bits = m.trailing_zeros() % 64 / 8 * 8;
    //     m >>= empty_bits;
    //     src = src.add(empty_bits as usize);

    for _ in 0..16 {
        for i in 0..4 {
            *out.add(written as usize) = *src;
            written += ((m >> i) & 1) as usize;
            src = src.add(1);
        }
        m >>= 4;
    }
}

// v.len() >= 64 must hold.
// m_popcnt > 0
// out must be valid for at least m.count_ones() + 1 writes.
unsafe fn scalar_filter64_unchecked1<T: Pod>(v: &[T], mut m: u64, m_popcnt: u32, out: *mut T) {
    // Rust generated significantly better code if we write the below loop
    // with v as a pointer, and out.add(written) instead of incrementing out
    // directly.
    let mut src = v.as_ptr();
    
    // let hi_start_offset = m_lo.count_ones() as usize;
    // let first_in_hi = m_hi.trailing_zeros() as usize;

    let mut lo_m = m;
    let mut hi_m = m;
    let mut lo_offset = 0;
    let mut hi_offset = m_popcnt as usize - 1;
    let mut lo_src = src;
    let mut hi_src = src.add(64);
    for i in 0..4 {
        for j in 0..8 {
            hi_src = hi_src.sub(1);
            *out.add(lo_offset as usize) = *lo_src;
            *out.add(hi_offset as usize) = *hi_src;
            lo_offset += ((lo_m >> j) & 1) as usize;
            hi_offset += ((hi_m >> (63 - j)) & 1) as usize;
            lo_src = src.add(1);
        }
        
        lo_m >>= 8;
        hi_m <<= 8;
    }
}


fn make_lut() {
    for mut m in 0..u8::MAX {
        let popcnt = m.count_ones();
        let mut out = (popcnt as u64) << 56;
        let mut offset = 0;
        while m > 0 {
            let idx = 1 + m.trailing_zeros() as u64;
            out |= idx << offset;
            offset += 8;
            m &= m.wrapping_sub(1);
        }
        print!("{out:016x}, ");
    }
}

#[rustfmt::skip]
static LUT: &'static [u64; 256] = &[
    0x0000000000000000, 0x0100000000000001, 0x0100000000000002, 0x0200000000000201,
    0x0100000000000003, 0x0200000000000301, 0x0200000000000302, 0x0300000000030201,
    0x0100000000000004, 0x0200000000000401, 0x0200000000000402, 0x0300000000040201,
    0x0200000000000403, 0x0300000000040301, 0x0300000000040302, 0x0400000004030201,
    0x0100000000000005, 0x0200000000000501, 0x0200000000000502, 0x0300000000050201,
    0x0200000000000503, 0x0300000000050301, 0x0300000000050302, 0x0400000005030201,
    0x0200000000000504, 0x0300000000050401, 0x0300000000050402, 0x0400000005040201,
    0x0300000000050403, 0x0400000005040301, 0x0400000005040302, 0x0500000504030201,
    0x0100000000000006, 0x0200000000000601, 0x0200000000000602, 0x0300000000060201,
    0x0200000000000603, 0x0300000000060301, 0x0300000000060302, 0x0400000006030201,
    0x0200000000000604, 0x0300000000060401, 0x0300000000060402, 0x0400000006040201,
    0x0300000000060403, 0x0400000006040301, 0x0400000006040302, 0x0500000604030201,
    0x0200000000000605, 0x0300000000060501, 0x0300000000060502, 0x0400000006050201,
    0x0300000000060503, 0x0400000006050301, 0x0400000006050302, 0x0500000605030201,
    0x0300000000060504, 0x0400000006050401, 0x0400000006050402, 0x0500000605040201,
    0x0400000006050403, 0x0500000605040301, 0x0500000605040302, 0x0600060504030201,
    0x0100000000000007, 0x0200000000000701, 0x0200000000000702, 0x0300000000070201,
    0x0200000000000703, 0x0300000000070301, 0x0300000000070302, 0x0400000007030201,
    0x0200000000000704, 0x0300000000070401, 0x0300000000070402, 0x0400000007040201,
    0x0300000000070403, 0x0400000007040301, 0x0400000007040302, 0x0500000704030201,
    0x0200000000000705, 0x0300000000070501, 0x0300000000070502, 0x0400000007050201,
    0x0300000000070503, 0x0400000007050301, 0x0400000007050302, 0x0500000705030201,
    0x0300000000070504, 0x0400000007050401, 0x0400000007050402, 0x0500000705040201,
    0x0400000007050403, 0x0500000705040301, 0x0500000705040302, 0x0600070504030201,
    0x0200000000000706, 0x0300000000070601, 0x0300000000070602, 0x0400000007060201,
    0x0300000000070603, 0x0400000007060301, 0x0400000007060302, 0x0500000706030201,
    0x0300000000070604, 0x0400000007060401, 0x0400000007060402, 0x0500000706040201,
    0x0400000007060403, 0x0500000706040301, 0x0500000706040302, 0x0600070604030201,
    0x0300000000070605, 0x0400000007060501, 0x0400000007060502, 0x0500000706050201,
    0x0400000007060503, 0x0500000706050301, 0x0500000706050302, 0x0600070605030201,
    0x0400000007060504, 0x0500000706050401, 0x0500000706050402, 0x0600070605040201,
    0x0500000706050403, 0x0600070605040301, 0x0600070605040302, 0x0707060504030201,
    0x0100000000000008, 0x0200000000000801, 0x0200000000000802, 0x0300000000080201,
    0x0200000000000803, 0x0300000000080301, 0x0300000000080302, 0x0400000008030201,
    0x0200000000000804, 0x0300000000080401, 0x0300000000080402, 0x0400000008040201,
    0x0300000000080403, 0x0400000008040301, 0x0400000008040302, 0x0500000804030201,
    0x0200000000000805, 0x0300000000080501, 0x0300000000080502, 0x0400000008050201,
    0x0300000000080503, 0x0400000008050301, 0x0400000008050302, 0x0500000805030201,
    0x0300000000080504, 0x0400000008050401, 0x0400000008050402, 0x0500000805040201,
    0x0400000008050403, 0x0500000805040301, 0x0500000805040302, 0x0600080504030201,
    0x0200000000000806, 0x0300000000080601, 0x0300000000080602, 0x0400000008060201,
    0x0300000000080603, 0x0400000008060301, 0x0400000008060302, 0x0500000806030201,
    0x0300000000080604, 0x0400000008060401, 0x0400000008060402, 0x0500000806040201,
    0x0400000008060403, 0x0500000806040301, 0x0500000806040302, 0x0600080604030201,
    0x0300000000080605, 0x0400000008060501, 0x0400000008060502, 0x0500000806050201,
    0x0400000008060503, 0x0500000806050301, 0x0500000806050302, 0x0600080605030201,
    0x0400000008060504, 0x0500000806050401, 0x0500000806050402, 0x0600080605040201,
    0x0500000806050403, 0x0600080605040301, 0x0600080605040302, 0x0708060504030201,
    0x0200000000000807, 0x0300000000080701, 0x0300000000080702, 0x0400000008070201,
    0x0300000000080703, 0x0400000008070301, 0x0400000008070302, 0x0500000807030201,
    0x0300000000080704, 0x0400000008070401, 0x0400000008070402, 0x0500000807040201,
    0x0400000008070403, 0x0500000807040301, 0x0500000807040302, 0x0600080704030201,
    0x0300000000080705, 0x0400000008070501, 0x0400000008070502, 0x0500000807050201,
    0x0400000008070503, 0x0500000807050301, 0x0500000807050302, 0x0600080705030201,
    0x0400000008070504, 0x0500000807050401, 0x0500000807050402, 0x0600080705040201,
    0x0500000807050403, 0x0600080705040301, 0x0600080705040302, 0x0708070504030201,
    0x0300000000080706, 0x0400000008070601, 0x0400000008070602, 0x0500000807060201,
    0x0400000008070603, 0x0500000807060301, 0x0500000807060302, 0x0600080706030201,
    0x0400000008070604, 0x0500000807060401, 0x0500000807060402, 0x0600080706040201,
    0x0500000807060403, 0x0600080706040301, 0x0600080706040302, 0x0708070604030201,
    0x0400000008070605, 0x0500000807060501, 0x0500000807060502, 0x0600080706050201,
    0x0500000807060503, 0x0600080706050301, 0x0600080706050302, 0x0708070605030201,
    0x0500000807060504, 0x0600080706050401, 0x0600080706050402, 0x0708070605040201,
    0x0600080706050403, 0x0708070605040301, 0x0708070605040302, 0x0807060504030201,
];

// v.len() >= 64 must hold.
// out must be valid for at least m.count_ones() + 1 writes.
unsafe fn scalar_filter64_unchecked_v2<T: Pod>(v: &[T], mut m: u64, out: *mut T) {
    // Rust generated significantly better code if we write the below loop
    // with v as a pointer, and out.add(written) instead of incrementing out
    // directly.
    let mut written = 0usize;
    let mut src = v.as_ptr();
    
    let mut idx_buf: MaybeUninit<[u8; 70]> = MaybeUninit::uninit();
    let idx_buf_ptr = idx_buf.as_mut_ptr() as *mut u8;
    let mut idx_offset = 0;

    for i in 0..8 {
        let one_offsets = LUT[(m >> 8*i) as u8 as usize];
        let popcnt = one_offsets >> 56;
        let v_offsets = one_offsets + i * 0x0101010101010101 - 0x0101010101010101;
        (idx_buf_ptr.add(idx_offset) as *mut u64).write_unaligned(v_offsets);
        idx_offset += popcnt as usize;
    }
    (idx_buf_ptr.add(idx_offset) as *mut u64).write_unaligned(0); // Initialize tail.
                                              

    let mut i = 0;
    while i < idx_offset {
        let tmp: [T; 8] = std::array::from_fn(|j|*v.get_unchecked(*idx_buf_ptr.add(i + j) as usize));
        core::ptr::copy_nonoverlapping(tmp.as_ptr(), out.add(i), 8);
        i += 8;
    }
}


pub fn filter_primitive_scalar<T: NativeType>(
    array: &PrimitiveArray<T>,
    mask: &Bitmap,
) -> PrimitiveArray<T> {
    assert_eq!(array.len(), mask.len());

    let mask_bits_set = mask.set_bits();
    let mut out = Vec::with_capacity(mask_bits_set + 16);
    let mut out_ptr = out.as_mut_ptr();
    
    let values = &array.values()[..];
    let (mut mask_bytes, offset, len) = mask.as_slice();
    if len == 0 {
        return PrimitiveArray::new_empty(array.data_type().clone());
    }

    // Handle offset.
    let mut value_idx = 0;
    if offset > 0 {
        let first_byte = mask_bytes[0];
        mask_bytes = &mask_bytes[1..];

        for byte_idx in offset..8 {
            let mask_bit = first_byte & (1 << byte_idx) != 0;
            if mask_bit && value_idx < len {
                unsafe {
                    *out_ptr = *values.get_unchecked(value_idx);
                    out_ptr = out_ptr.add(1);
                }
            }
            value_idx += 1;
        }
    }
    
    // Handle bulk.
    while value_idx + 64 <= len {
        let (mask_chunk, value_chunk);
        unsafe {
            mask_chunk = mask_bytes.get_unchecked(0..8);
            mask_bytes = mask_bytes.get_unchecked(8..);
            value_chunk = values.get_unchecked(value_idx..value_idx + 64);
            value_idx += 64;
        };
        let m = u64::from_le_bytes(mask_chunk.try_into().unwrap());

        if m == 0 {
            continue;
        }

        unsafe {
            if m == u64::MAX {
                core::ptr::copy_nonoverlapping(value_chunk.as_ptr(), out_ptr, 64);
                out_ptr = out_ptr.add(64);
                continue;
            }

            let m_popcnt = m.count_ones();
            if m_popcnt <= 16 {
                scalar_sparse_unchecked(value_chunk, m, m_popcnt, out_ptr)
            } else {
                scalar_filter64_unchecked(value_chunk, m, out_ptr)
            };
            out_ptr = out_ptr.add(m_popcnt as usize);
        }
    }
    
    if value_idx < len {
        let rest_len = len - value_idx;
        assert!(rest_len < 64);
        let m = load_padded_le_u64(mask_bytes) & ((1 << rest_len) - 1);
        unsafe {
            let value_chunk = values.get_unchecked(value_idx..);
            scalar_sparse_unchecked(value_chunk, m, m.count_ones(), out_ptr);
        }
    }
    
    unsafe {
        out.set_len(mask_bits_set);
    }

    
    let mut arr = PrimitiveArray::from_vec(out);
    if let Some(validity) = array.validity() {
        let filt_validity = super::boolean::filter_boolean_kernel(validity, mask);
        arr.set_validity(Some(filt_validity));
    }
    arr
}
