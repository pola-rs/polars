use atoi::FromRadix10SignedChecked;

/// Count the number of b'0's at the beginning of a slice.
fn leading_zeros(bytes: &[u8]) -> u8 {
    bytes.iter().take_while(|byte| **byte == b'0').count() as u8
}

fn split_decimal_bytes(bytes: &[u8]) -> (Option<&[u8]>, Option<&[u8]>) {
    let mut a = bytes.splitn(2, |x| *x == b'.');
    let lhs = a.next();
    let rhs = a.next();
    (lhs, rhs)
}

/// Parse a single i128 from bytes, ensuring the entire slice is read.
fn parse_integer_checked(bytes: &[u8]) -> Option<i128> {
    let (n, len) = i128::from_radix_10_signed_checked(bytes);
    n.filter(|_| len == bytes.len())
}

/// Assuming bytes are a well-formed decimal number (with or without a separator),
/// infer the scale of the number.  If no separator is present, the scale is 0.
pub fn infer_scale(bytes: &[u8]) -> u8 {
    let (_lhs, rhs) = split_decimal_bytes(bytes);
    rhs.map_or(0, |x| x.len() as u8)
}

/// Deserialize bytes to a single i128 representing a decimal, at a specified precision
/// (optional) and scale (required).  If precision is not specified, it is assumed to be
/// 38 (the max precision allowed by the i128 representation).  The number is checked to
/// ensure it fits within the specified precision and scale.  Consistent with float parsing,
/// no decimal separator is required (eg "500", "500.", and "500.0" are all accepted); this allows
/// mixed integer/decimal sequences to be parsed as decimals.  All trailing zeros are assumed to
/// be significant, whether or not a separator is present: 1200 requires precision >= 4, while 1200.200
/// requires precision >= 7 and scale >= 3.  Returns None if the number is not well-formed, or does not
/// fit. Only b'.' is allowed as a decimal separator (issue #6698).
#[inline]
pub(crate) fn deserialize_decimal(
    mut bytes: &[u8],
    precision: Option<u8>,
    scale: u8,
) -> Option<i128> {
    // While parse_integer_checked will parse negative numbers, we want to handle
    // the negative sign ourselves, and so check for it initially, then handle it
    // at the end.
    let negative = bytes.first() == Some(&b'-');
    if negative {
        bytes = &bytes[1..];
    };
    let (lhs, rhs) = split_decimal_bytes(bytes);
    let precision = precision.unwrap_or(38);

    let lhs_b = lhs?;

    // For the purposes of decimal parsing, we assume that all digits other than leading zeros
    // are significant, eg, 001200 has 4 significant digits, not 2.  The Decimal type does
    // not allow negative scales, so all trailing zeros on the LHS of any decimal separator
    // will still take up space in the representation (eg, 1200 requires, at minimum, precision 4
    // at scale 0; there is no scale -2 where it would only need precision 2).
    let lhs_s = lhs_b.len() as u8 - leading_zeros(lhs_b);

    let abs = parse_integer_checked(lhs_b).and_then(|x| match rhs {
        // A decimal separator was found, so LHS and RHS need to be combined.
        Some(rhs) => parse_integer_checked(rhs)
            .map(|y| (x, y, rhs))
            .and_then(|(lhs, rhs, rhs_b)| {
                // We include all digits on the RHS, including both leading and trailing zeros,
                // as significant.  This is consistent with standard scientific practice for writing
                // numbers.  However, an alternative for parsing could truncate trailing zeros that extend
                // beyond the scale: we choose not to do this here.
                let scale_adjust = scale as i8 - rhs_b.len() as i8;

                if (lhs_s + scale > precision)
                    || (scale_adjust < 0)
                    || (rhs_b.first() == Some(&b'-'))
                {
                    // LHS significant figures and scale exceed precision,
                    // RHS significant figures (all digits in RHS) exceed scale, or
                    // RHS starts with a '-' and the number is not well-formed.
                    None
                } else if (rhs_b.len() as u8) == scale {
                    // RHS has exactly scale significant digits, so no adjustment
                    // is needed to RHS.
                    Some((lhs, rhs))
                } else {
                    // RHS needs adjustment to scale. scale_adjust is known to be
                    // positive.
                    Some((lhs, rhs * 10i128.pow(scale_adjust as u32)))
                }
            })
            .map(|(lhs, rhs)| lhs * 10i128.pow(scale as u32) + rhs),
        // No decimal separator was found; we have an integer / LHS only.
        None => {
            if (lhs_s + scale > precision) || lhs_b.is_empty() {
                // Either the integer itself exceeds the precision, or we simply have
                // no number at all / an empty string.
                return None;
            }
            Some(x * 10i128.pow(scale as u32))
        },
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
        assert_eq!(deserialize_decimal(val.as_bytes(), precision, scale), None);
        assert_eq!(
            deserialize_decimal(val.as_bytes(), None, scale),
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

        let val = "12.-3";
        assert_eq!(deserialize_decimal(val.as_bytes(), precision, scale), None);

        let val = "";
        assert_eq!(deserialize_decimal(val.as_bytes(), precision, scale), None);

        let val = "5.";
        assert_eq!(
            deserialize_decimal(val.as_bytes(), precision, scale),
            Some(500000i128)
        );

        let val = "5";
        assert_eq!(
            deserialize_decimal(val.as_bytes(), precision, scale),
            Some(500000i128)
        );

        let val = ".5";
        assert_eq!(
            deserialize_decimal(val.as_bytes(), precision, scale),
            Some(50000i128)
        );

        // Precision and scale fitting:
        let val = b"1200";
        assert_eq!(deserialize_decimal(val, None, 0), Some(1200));
        assert_eq!(deserialize_decimal(val, Some(4), 0), Some(1200));
        assert_eq!(deserialize_decimal(val, Some(3), 0), None);
        assert_eq!(deserialize_decimal(val, Some(4), 1), None);

        let val = b"1200.010";
        assert_eq!(deserialize_decimal(val, None, 0), None); // insufficient scale
        assert_eq!(deserialize_decimal(val, None, 3), Some(1200010)); // exact scale
        assert_eq!(deserialize_decimal(val, None, 6), Some(1200010000)); // excess scale
        assert_eq!(deserialize_decimal(val, Some(7), 0), None); // insufficient precision and scale
        assert_eq!(deserialize_decimal(val, Some(7), 3), Some(1200010)); // exact precision and scale
        assert_eq!(deserialize_decimal(val, Some(10), 6), Some(1200010000)); // exact precision, excess scale
        assert_eq!(deserialize_decimal(val, Some(5), 6), None); // insufficient precision, excess scale
        assert_eq!(deserialize_decimal(val, Some(5), 3), None); // insufficient precision, exact scale
        assert_eq!(deserialize_decimal(val, Some(12), 5), Some(120001000)); // excess precision, excess scale
        assert_eq!(deserialize_decimal(val, None, 35), None); // scale causes insufficient precision
    }
}
