use polars_arrow::legacy::kernels::string::num_continuation_bytes;

#[inline(always)]
pub(crate) fn is_utf8_codepoint_start(b: u8) -> bool {
    // The top two bits of a continuation byte are 10. Any other value is a
    // starting byte. We can use signed comparison to test for this in one
    // instruction, as the top bits 11, 00 and 01 are all more positive and thus
    // larger in signed comparison.
    (b as i8) >= (0b1100_0000_u8 as i8)
}

/// The number of codepoints in UTF-8 `s` that start before `byte_idx`.
#[inline]
pub(crate) fn byte_to_char_idx(s: &[u8], byte_idx: usize) -> usize {
    byte_idx - num_continuation_bytes(&s[..byte_idx]) as usize
}
