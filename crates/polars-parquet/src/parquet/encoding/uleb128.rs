use crate::error::Error;

pub fn decode(values: &[u8]) -> Result<(u64, usize), Error> {
    let mut result = 0;
    let mut shift = 0;

    let mut consumed = 0;
    for byte in values {
        consumed += 1;
        if shift == 63 && *byte > 1 {
            panic!()
        };

        result |= u64::from(byte & 0b01111111) << shift;

        if byte & 0b10000000 == 0 {
            break;
        }

        shift += 7;
    }
    Ok((result, consumed))
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
        let (value, len) = decode(&data).unwrap();
        assert_eq!(value, 624_485);
        assert_eq!(len, 3);
    }

    #[test]
    fn decode_2() {
        let data = vec![0b00010000, 0b00000001, 0b00000011, 0b00000011];
        let (value, len) = decode(&data).unwrap();
        assert_eq!(value, 16);
        assert_eq!(len, 1);
    }

    #[test]
    fn round_trip() {
        let original = 123124234u64;
        let mut container = [0u8; 10];
        let encoded_len = encode(original, &mut container);
        let (value, len) = decode(&container).unwrap();
        assert_eq!(value, original);
        assert_eq!(len, encoded_len);
    }

    #[test]
    fn min_value() {
        let original = u64::MIN;
        let mut container = [0u8; 10];
        let encoded_len = encode(original, &mut container);
        let (value, len) = decode(&container).unwrap();
        assert_eq!(value, original);
        assert_eq!(len, encoded_len);
    }

    #[test]
    fn max_value() {
        let original = u64::MAX;
        let mut container = [0u8; 10];
        let encoded_len = encode(original, &mut container);
        let (value, len) = decode(&container).unwrap();
        assert_eq!(value, original);
        assert_eq!(len, encoded_len);
    }
}
