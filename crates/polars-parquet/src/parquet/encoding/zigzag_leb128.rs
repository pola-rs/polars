use super::uleb128;

pub fn decode(values: &[u8]) -> (i64, usize) {
    let (u, consumed) = uleb128::decode(values);
    ((u >> 1) as i64 ^ -((u & 1) as i64), consumed)
}

pub fn encode(value: i64) -> ([u8; 10], usize) {
    let value = ((value << 1) ^ (value >> (64 - 1))) as u64;
    let mut a = [0u8; 10];
    let produced = uleb128::encode(value, &mut a);
    (a, produced)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode() {
        // see e.g. https://stackoverflow.com/a/2211086/931303
        let cases = vec![
            (0u8, 0i64),
            (1, -1),
            (2, 1),
            (3, -2),
            (4, 2),
            (5, -3),
            (6, 3),
            (7, -4),
            (8, 4),
            (9, -5),
        ];
        for (data, expected) in cases {
            let (result, _) = decode(&[data]);
            assert_eq!(result, expected)
        }
    }

    #[test]
    fn test_encode() {
        let cases = vec![
            (0u8, 0i64),
            (1, -1),
            (2, 1),
            (3, -2),
            (4, 2),
            (5, -3),
            (6, 3),
            (7, -4),
            (8, 4),
            (9, -5),
        ];
        for (expected, data) in cases {
            let (result, size) = encode(data);
            assert_eq!(size, 1);
            assert_eq!(result[0], expected)
        }
    }

    #[test]
    fn test_roundtrip() {
        let value = -1001212312;
        let (data, size) = encode(value);
        let (result, _) = decode(&data[..size]);
        assert_eq!(value, result);
    }
}
