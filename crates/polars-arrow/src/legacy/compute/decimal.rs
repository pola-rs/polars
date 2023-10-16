use atoi::FromRadix10SignedChecked;

fn significant_digits(bytes: &[u8]) -> u8 {
    (bytes.len() as u8) - leading_zeros(bytes)
}

fn leading_zeros(bytes: &[u8]) -> u8 {
    bytes.iter().take_while(|byte| **byte == b'0').count() as u8
}

fn split_decimal_bytes(bytes: &[u8]) -> (Option<&[u8]>, Option<&[u8]>) {
    let mut a = bytes.splitn(2, |x| *x == b'.');
    let lhs = a.next();
    let rhs = a.next();
    (lhs, rhs)
}

fn parse_integer_checked(bytes: &[u8]) -> Option<i128> {
    let (n, len) = i128::from_radix_10_signed_checked(bytes);
    n.filter(|_| len == bytes.len())
}

pub fn infer_scale(bytes: &[u8]) -> Option<u8> {
    let (_lhs, rhs) = split_decimal_bytes(bytes);
    rhs.map(significant_digits)
}

/// Deserializes bytes to a single i128 representing a decimal
/// The decimal precision and scale are not checked.
#[inline]
pub(super) fn deserialize_decimal(
    mut bytes: &[u8],
    precision: Option<u8>,
    scale: u8,
) -> Option<i128> {
    let negative = bytes.first() == Some(&b'-');
    if negative {
        bytes = &bytes[1..];
    };
    let (lhs, rhs) = split_decimal_bytes(bytes);
    let precision = precision.unwrap_or(u8::MAX);

    let lhs_b = lhs?;
    let abs = parse_integer_checked(lhs_b).and_then(|x| {
        match rhs {
            Some(rhs) => {
                parse_integer_checked(rhs)
                    .map(|y| (x, lhs_b, y, rhs))
                    .and_then(|(lhs, lhs_b, rhs, rhs_b)| {
                        let lhs_s = significant_digits(lhs_b);
                        let leading_zeros_rhs = leading_zeros(rhs_b);
                        let rhs_s = rhs_b.len() as u8 - leading_zeros_rhs;

                        // parameters don't match bytes
                        if lhs_s + rhs_s > precision || rhs_s > scale {
                            None
                        }
                        // significant digits don't fit scale
                        else if rhs_s < scale {
                            // scale: 2
                            // number: x.09
                            // significant digits: 1
                            // leading_zeros: 1
                            // parsed: 9
                            // so this is correct
                            if leading_zeros_rhs + rhs_s == scale {
                                Some((lhs, rhs))
                            }
                            // scale: 2
                            // number: x.9
                            // significant digits: 1
                            // parsed: 9
                            // so we must multiply by 10 to get 90
                            else {
                                let diff = scale as u32 - (rhs_s + leading_zeros_rhs) as u32;
                                Some((lhs, rhs * 10i128.pow(diff)))
                            }
                        }
                        // scale: 2
                        // number: x.90
                        // significant digits: 2
                        // parsed: 90
                        // so this is correct
                        else {
                            Some((lhs, rhs))
                        }
                    })
                    .map(|(lhs, rhs)| lhs * 10i128.pow(scale as u32) + rhs)
            },
            None => {
                if lhs_b.len() > precision as usize || scale != 0 {
                    return None;
                }
                parse_integer_checked(lhs_b)
            },
        }
    });
    if negative {
        Some(-abs?)
    } else {
        abs
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_decimal() {
        let precision = Some(8);
        let scale = 2;

        let val = "12.09";
        assert_eq!(
            deserialize_decimal(val.as_bytes(), precision, scale),
            Some(1209)
        );

        let val = "1200.90";
        assert_eq!(
            deserialize_decimal(val.as_bytes(), precision, scale),
            Some(120090)
        );

        let val = "143.9";
        assert_eq!(
            deserialize_decimal(val.as_bytes(), precision, scale),
            Some(14390)
        );

        let val = "-0.5";
        assert_eq!(
            deserialize_decimal(val.as_bytes(), precision, scale),
            Some(-50)
        );

        let val = "-1.5";
        assert_eq!(
            deserialize_decimal(val.as_bytes(), precision, scale),
            Some(-150)
        );

        let scale = 20;
        let val = "0.01";
        assert_eq!(
            deserialize_decimal(val.as_bytes(), precision, scale),
            Some(1000000000000000000)
        );
        let scale = 5;
        let val = "12ABC.34";
        assert_eq!(deserialize_decimal(val.as_bytes(), precision, scale), None);

        let val = "1ABC2.34";
        assert_eq!(deserialize_decimal(val.as_bytes(), precision, scale), None);

        let val = "12.3ABC4";
        assert_eq!(deserialize_decimal(val.as_bytes(), precision, scale), None);

        let val = "12.3.ABC4";
        assert_eq!(deserialize_decimal(val.as_bytes(), precision, scale), None);

        let val = "";
        assert_eq!(deserialize_decimal(val.as_bytes(), precision, scale), None);

        let val = "5.";
        assert_eq!(
            deserialize_decimal(val.as_bytes(), precision, scale),
            Some(500000i128)
        );

        let val = ".5";
        assert_eq!(
            deserialize_decimal(val.as_bytes(), precision, scale),
            Some(50000i128)
        );
    }
}
