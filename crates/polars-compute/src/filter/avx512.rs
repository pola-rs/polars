use core::arch::x86_64::*;

// It's not possible to inline target_feature(enable = ...) functions into other
// functions without that enabled, so we use a macro for these very-similarly
// structured functions.
macro_rules! simd_filter {
    ($values: ident, $mask_bytes: ident, $out: ident, |$subchunk: ident, $m: ident: $MaskT: ty| $body:block) => {{
        const MASK_BITS: usize = std::mem::size_of::<$MaskT>() * 8;

        // Do a 64-element loop for sparse fast path.
        let chunks = $values.chunks_exact(64);
        $values = chunks.remainder();
        for chunk in chunks {
            let mask_chunk = $mask_bytes.get_unchecked(..8);
            $mask_bytes = $mask_bytes.get_unchecked(8..);
            let mut m64 = u64::from_le_bytes(mask_chunk.try_into().unwrap());

            // Fast-path: skip entire 64-element chunk.
            if m64 == 0 {
                continue;
            }

            for $subchunk in chunk.chunks_exact(MASK_BITS) {
                let $m = m64 as $MaskT;
                $body;
                m64 >>= MASK_BITS % 64;
            }
        }

        // Handle the SIMD-block-sized remainder.
        let subchunks = $values.chunks_exact(MASK_BITS);
        $values = subchunks.remainder();
        for $subchunk in subchunks {
            let mask_chunk = $mask_bytes.get_unchecked(..MASK_BITS / 8);
            $mask_bytes = $mask_bytes.get_unchecked(MASK_BITS / 8..);
            let $m = <$MaskT>::from_le_bytes(mask_chunk.try_into().unwrap());
            $body;
        }

        ($values, $mask_bytes, $out)
    }};
}

/// # Safety
/// out must be valid for 64 + bitslice(mask_bytes, 0..values.len()).count_ones() writes.
/// AVX512_VBMI2 must be enabled.
#[target_feature(enable = "avx512f")]
#[target_feature(enable = "avx512vbmi2")]
pub unsafe fn filter_u8_avx512vbmi2<'a>(
    mut values: &'a [u8],
    mut mask_bytes: &'a [u8],
    mut out: *mut u8,
) -> (&'a [u8], &'a [u8], *mut u8) {
    simd_filter!(values, mask_bytes, out, |vchunk, m: u64| {
        // We don't use compress-store instructions because they are very slow
        // on Zen. We are allowed to overshoot anyway.
        let v = _mm512_loadu_si512(vchunk.as_ptr().cast());
        let filtered = _mm512_maskz_compress_epi8(m, v);
        _mm512_storeu_si512(out.cast(), filtered);
        out = out.add(m.count_ones() as usize);
    })
}

/// # Safety
/// out must be valid for 32 + bitslice(mask_bytes, 0..values.len()).count_ones() writes.
/// AVX512_VBMI2 must be enabled.
#[target_feature(enable = "avx512f")]
#[target_feature(enable = "avx512vbmi2")]
pub unsafe fn filter_u16_avx512vbmi2<'a>(
    mut values: &'a [u16],
    mut mask_bytes: &'a [u8],
    mut out: *mut u16,
) -> (&'a [u16], &'a [u8], *mut u16) {
    simd_filter!(values, mask_bytes, out, |vchunk, m: u32| {
        let v = _mm512_loadu_si512(vchunk.as_ptr().cast());
        let filtered = _mm512_maskz_compress_epi16(m, v);
        _mm512_storeu_si512(out.cast(), filtered);
        out = out.add(m.count_ones() as usize);
    })
}

/// # Safety
/// out must be valid for 16 + bitslice(mask_bytes, 0..values.len()).count_ones() writes.
/// AVX512F must be enabled.
#[target_feature(enable = "avx512f")]
pub unsafe fn filter_u32_avx512f<'a>(
    mut values: &'a [u32],
    mut mask_bytes: &'a [u8],
    mut out: *mut u32,
) -> (&'a [u32], &'a [u8], *mut u32) {
    simd_filter!(values, mask_bytes, out, |vchunk, m: u16| {
        let v = _mm512_loadu_si512(vchunk.as_ptr().cast());
        let filtered = _mm512_maskz_compress_epi32(m, v);
        _mm512_storeu_si512(out.cast(), filtered);
        out = out.add(m.count_ones() as usize);
    })
}

/// # Safety
/// out must be valid for 8 + bitslice(mask_bytes, 0..values.len()).count_ones() writes.
/// AVX512F must be enabled.
#[target_feature(enable = "avx512f")]
pub unsafe fn filter_u64_avx512f<'a>(
    mut values: &'a [u64],
    mut mask_bytes: &'a [u8],
    mut out: *mut u64,
) -> (&'a [u64], &'a [u8], *mut u64) {
    simd_filter!(values, mask_bytes, out, |vchunk, m: u8| {
        let v = _mm512_loadu_si512(vchunk.as_ptr().cast());
        let filtered = _mm512_maskz_compress_epi64(m, v);
        _mm512_storeu_si512(out.cast(), filtered);
        out = out.add(m.count_ones() as usize);
    })
}
