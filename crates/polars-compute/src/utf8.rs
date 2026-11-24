use polars_arrow::array::{Array, UInt32Array, Utf8ViewArray, View};
use polars_arrow::datatypes::ArrowDataType;
use polars_buffer::Buffer;

/// Number of characters (code points) in the UTF-8 encoded `bytes`.
#[inline]
pub fn utf8_len(bytes: &[u8]) -> usize {
    // Sum in u8 lanes per chunk so that this vectorizes well; a chunk has fewer than 256 bytes.
    let num_continuation_bytes: usize = bytes
        .chunks(128)
        .map(|chunk| chunk.iter().map(|&b| ((b as i8) < -0x40) as u8).sum::<u8>() as usize)
        .sum();
    bytes.len() - num_continuation_bytes
}

#[inline(always)]
pub fn is_utf8_codepoint_start(b: u8) -> bool {
    // The top two bits of a continuation byte are 10. Any other value is a
    // starting byte. We can use signed comparison to test for this in one
    // instruction, as the top bits 11, 00 and 01 are all more positive and thus
    // larger in signed comparison.
    (b as i8) >= (0b1100_0000_u8 as i8)
}

/// The number of codepoints in UTF-8 `s` that start before `byte_idx`.
#[inline]
pub fn byte_to_char_idx(s: &[u8], byte_idx: usize) -> usize {
    utf8_len(&s[..byte_idx])
}

/// Similar to char_to_byte_idx but if `char_idx` would be out-of-bounds the
/// number of codepoints in s is returned as an error.
pub fn char_to_byte_idx_or_cp_count(s: &str, char_idx: usize) -> Result<usize, usize> {
    let bytes = s.as_bytes();
    if char_idx == 0 {
        return Ok(0);
    }

    let mut offset = 0;
    let mut num_chars_seen = 0;

    // Auto-vectorized bulk processing, but skip if index is small.
    if char_idx >= 16 {
        while let Some(chunk) = bytes.get(offset..offset + 16) {
            let chunk_seen: usize = chunk
                .iter()
                .map(|b| is_utf8_codepoint_start(*b) as usize)
                .sum();
            if num_chars_seen + chunk_seen > char_idx {
                break;
            }
            offset += 16;
            num_chars_seen += chunk_seen;
        }
    }

    while let Some(b) = bytes.get(offset) {
        num_chars_seen += is_utf8_codepoint_start(*b) as usize;
        if num_chars_seen > char_idx {
            return Ok(offset);
        }
        offset += 1;
    }

    debug_assert!(offset == bytes.len());
    Err(num_chars_seen)
}

/// Given an offset to the start of the `char_idx`th codepoint, returns the
/// equivalent offset in bytes.
///
/// If `char_idx` would be out-of-bounds s.len() is returned.
#[inline]
pub fn char_to_byte_idx(s: &str, char_idx: usize) -> usize {
    if char_idx >= s.len() {
        // No need to even count.
        s.len()
    } else {
        char_to_byte_idx_or_cp_count(s, char_idx).unwrap_or(s.len())
    }
}

/// Similar to rev_char_to_byte_idx but if `char_idx` would be out-of-bounds the
/// number of codepoints in s is returned as an error.
pub fn rev_char_to_byte_idx_or_cp_count(s: &str, rev_char_idx: usize) -> Result<usize, usize> {
    let bytes = s.as_bytes();
    if rev_char_idx == 0 {
        return Ok(bytes.len());
    }

    let mut offset = s.len();
    let mut num_chars_seen = 0;

    // Auto-vectorized bulk processing, but skip if index is small.
    if rev_char_idx >= 16 {
        while offset >= 16 {
            let chunk = unsafe { bytes.get_unchecked(offset - 16..offset) };
            let chunk_seen: usize = chunk
                .iter()
                .map(|b| is_utf8_codepoint_start(*b) as usize)
                .sum();
            if num_chars_seen + chunk_seen >= rev_char_idx {
                break;
            }
            offset -= 16;
            num_chars_seen += chunk_seen;
        }
    }

    while offset > 0 {
        offset -= 1;
        let byte = unsafe { bytes.get_unchecked(offset) };
        num_chars_seen += is_utf8_codepoint_start(*byte) as usize;
        if num_chars_seen >= rev_char_idx {
            return Ok(offset);
        }
    }

    debug_assert!(offset == 0);
    Err(num_chars_seen)
}

/// Counts rev_char_idx code points from *the end* of the string, returning an
/// offset in bytes where this codepoint ends.
///
/// For example, rev_char_to_byte_idx(0, s) returns s.len(), and
/// rev_char_to_byte_idx(1, s) returns s.len() - width(last_codepoint_in_s).
///
/// If rev_char_idx is large enough that we would go out of bounds, 0 is returned.
pub fn rev_char_to_byte_idx(s: &str, rev_char_idx: usize) -> usize {
    if rev_char_idx >= s.len() {
        // No need to even count.
        0
    } else {
        rev_char_to_byte_idx_or_cp_count(s, rev_char_idx).unwrap_or(0)
    }
}

/// Number of characters (code points) in each string of `array`.
pub fn utf8view_len_chars(array: &Utf8ViewArray) -> UInt32Array {
    const INLINE_CONTINUATION_BITS: u128 = 0x8080_8080_8080_8080_8080_8080_0000_0000;

    let buffers = array.data_buffers();
    let values: Buffer<u32> = array
        .views()
        .iter()
        .map(|view| {
            if view.is_inline() {
                let data = view.as_u128().to_le();
                // Padding of null views need not be zero, so only keep the first `length` bytes.
                let used = u128::MAX >> (8 * (View::MAX_INLINE_SIZE - view.length));
                let continuation = data & !(data << 1) & INLINE_CONTINUATION_BITS & used;
                view.length - continuation.count_ones()
            } else {
                // SAFETY: views of a valid array point into its buffers.
                let bytes = unsafe { view.get_external_slice_unchecked(buffers) };
                utf8_len(bytes) as u32
            }
        })
        .collect();
    UInt32Array::new(ArrowDataType::UInt32, values, array.validity().cloned())
}
