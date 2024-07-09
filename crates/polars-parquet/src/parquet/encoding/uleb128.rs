// Reads an uleb128 encoded integer with at most 56 bits (8 bytes with 7 bits worth of payload each).
/// Returns the integer and the number of bytes that made up this integer.
/// If the returned length is bigger than 8 this means the integer required more than 8 bytes and the remaining bytes need to be read sequentially and combined with the return value.
///
/// # Safety
/// `data` needs to contain at least 8 bytes.
#[target_feature(enable = "bmi2")]
#[cfg(target_feature = "bmi2")]
pub unsafe fn decode_uleb_bmi2(data: &[u8]) -> (u64, usize) {
    const CONT_MARKER: u64 = 0x80808080_80808080;
    debug_assert!(data.len() >= 8);

    unsafe {
        let word = data.as_ptr().cast::<u64>().read_unaligned();
        // mask indicating continuation bytes
        let mask = std::arch::x86_64::_pext_u64(word, CONT_MARKER);
        let len = (!mask).trailing_zeros() + 1;
        // which payload bits to extract
        let ext = std::arch::x86_64::_bzhi_u64(!CONT_MARKER, 8 * len);
        let payload = std::arch::x86_64::_pext_u64(word, ext);

        (payload, len as _)
    }
}

pub fn decode(values: &[u8]) -> (u64, usize) {
    #[cfg(target_feature = "bmi2")]
    {
        if polars_utils::cpuid::has_fast_bmi2() && values.len() >= 8 {
            let (result, consumed) = unsafe { decode_uleb_bmi2(values) };
            if consumed <= 8 {
                return (result, consumed);
            }
        }
    }

    let mut result = 0;
    let mut shift = 0;

    let mut consumed = 0;
    for byte in values {
        consumed += 1;

        #[cfg(debug_assertions)]
        debug_assert!(!(shift == 63 && *byte > 1));

        result |= u64::from(byte & 0b01111111) << shift;

        if byte & 0b10000000 == 0 {
            break;
        }

        shift += 7;
    }
    (result, consumed)
}

/// Encodes `value` in ULEB128 into `container`. The exact number of bytes written
/// depends on `value`, and cannot be determined upfront. The maximum number of bytes
/// required are 10.
/// # Panic
/// This function may panic if `container.len() < 10` and `value` requires more bytes.
pub fn encode(mut value: u64, container: &mut [u8]) -> usize {
    let mut consumed = 0;
    let mut iter = container.iter_mut();
    loop {
        let mut byte = (value as u8) & !128;
        value >>= 7;
        if value != 0 {
            byte |= 128;
        }
        *iter.next().unwrap() = byte;
        consumed += 1;
        if value == 0 {
            break;
        }
    }
    consumed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_1() {
        let data = vec![0xe5, 0x8e, 0x26, 0xDE, 0xAD, 0xBE, 0xEF];
        let (value, len) = decode(&data);
        assert_eq!(value, 624_485);
        assert_eq!(len, 3);
    }

    #[test]
    fn decode_2() {
        let data = vec![0b00010000, 0b00000001, 0b00000011, 0b00000011];
        let (value, len) = decode(&data);
        assert_eq!(value, 16);
        assert_eq!(len, 1);
    }

    #[test]
    fn round_trip() {
        let original = 123124234u64;
        let mut container = [0u8; 10];
        let encoded_len = encode(original, &mut container);
        let (value, len) = decode(&container);
        assert_eq!(value, original);
        assert_eq!(len, encoded_len);
    }

    #[test]
    fn min_value() {
        let original = u64::MIN;
        let mut container = [0u8; 10];
        let encoded_len = encode(original, &mut container);
        let (value, len) = decode(&container);
        assert_eq!(value, original);
        assert_eq!(len, encoded_len);
    }

    #[test]
    fn max_value() {
        let original = u64::MAX;
        let mut container = [0u8; 10];
        let encoded_len = encode(original, &mut container);
        let (value, len) = decode(&container);
        assert_eq!(value, original);
        assert_eq!(len, encoded_len);
    }
}
