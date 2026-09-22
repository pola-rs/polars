pub mod binary;
pub mod no_order;
pub mod utf8;

/// Bytes handled at a time by the string encoders and decoders.
pub(crate) const BLOCK_SIZE: usize = 16;

/// Position of the first `needle` byte in `block`, or [`BLOCK_SIZE`] if there is none.
#[inline(always)]
pub(crate) fn find_byte(block: [u8; BLOCK_SIZE], needle: u8) -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;
        // SAFETY: SSE2 is always available on x86_64.
        unsafe {
            let v = _mm_loadu_si128(block.as_ptr() as *const __m128i);
            let eq = _mm_cmpeq_epi8(v, _mm_set1_epi8(needle as i8));
            let mask = _mm_movemask_epi8(eq) as u32;
            (mask | (1 << BLOCK_SIZE)).trailing_zeros() as usize
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        const LO: u64 = 0x0101_0101_0101_0101;
        const HI: u64 = 0x8080_8080_8080_8080;
        let pattern = LO * needle as u64;
        let lo = u64::from_le_bytes(block[..8].try_into().unwrap()) ^ pattern;
        let hi = u64::from_le_bytes(block[8..].try_into().unwrap()) ^ pattern;
        let lo = lo.wrapping_sub(LO) & !lo & HI;
        let hi = hi.wrapping_sub(LO) & !hi & HI;
        if lo != 0 {
            lo.trailing_zeros() as usize / 8
        } else if hi != 0 {
            8 + hi.trailing_zeros() as usize / 8
        } else {
            BLOCK_SIZE
        }
    }
}
