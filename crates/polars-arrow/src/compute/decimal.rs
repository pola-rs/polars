use std::sync::atomic::{AtomicBool, Ordering};

use atoi::FromRadix10SignedChecked;
use num_traits::Euclid;

static TRIM_DECIMAL_ZEROS: AtomicBool = AtomicBool::new(false);

pub fn get_trim_decimal_zeros() -> bool {
    TRIM_DECIMAL_ZEROS.load(Ordering::Relaxed)
}
pub fn set_trim_decimal_zeros(trim: Option<bool>) {
    TRIM_DECIMAL_ZEROS.store(trim.unwrap_or(false), Ordering::Relaxed)
}

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
    // While parse_integer_checked will parse positive/negative numbers, we want to
    // handle the sign ourselves, and so check for it initially, then handle it
    // at the end.
    let negative = match bytes.first() {
        Some(s @ (b'+' | b'-')) => {
            bytes = &bytes[1..];
            *s == b'-'
        },
        _ => false,
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

    if lhs_s + scale > precision {
        // the integer already exceeds the precision
        return None;
    }

    let abs = parse_integer_checked(lhs_b).and_then(|x| match rhs {
        // A decimal separator was found, so LHS and RHS need to be combined.
        Some(mut rhs) => {
            if matches!(rhs.first(), Some(b'+' | b'-')) {
                // RHS starts with a '+'/'-' sign and the number is not well-formed.
                return None;
            }
            let scale_adjust = if (scale as usize) <= rhs.len() {
                // Truncate trailing digits that extend beyond the scale
                rhs = &rhs[..scale as usize];
                None
            } else {
                Some(scale as u32 - rhs.len() as u32)
            };

            parse_integer_checked(rhs).map(|y| {
                let lhs = x * 10i128.pow(scale as u32);
                let rhs = scale_adjust.map_or(y, |s| y * 10i128.pow(s));
                lhs + rhs
            })
        },
        // No decimal separator was found; we have an integer / LHS only.
        None => {
            if lhs_b.is_empty() {
                // we simply have no number at all / an empty string.
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

const BUF_LEN: usize = 48;

#[derive(Clone, Copy)]
pub struct FormatBuffer {
    data: [u8; BUF_LEN],
    len: usize,
}

impl Default for FormatBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl FormatBuffer {
    #[inline]
    pub const fn new() -> Self {
        Self {
            data: [0; BUF_LEN],
            len: 0,
        }
    }

    #[inline]
    pub fn as_str(&self) -> &str {
        unsafe { std::str::from_utf8_unchecked(&self.data[..self.len]) }
    }
}

const POW10: [i128; 39] = [
    1,
    10,
    100,
    1000,
    10000,
    100000,
    1000000,
    10000000,
    100000000,
    1000000000,
    10000000000,
    100000000000,
    1000000000000,
    10000000000000,
    100000000000000,
    1000000000000000,
    10000000000000000,
    100000000000000000,
    1000000000000000000,
    10000000000000000000,
    100000000000000000000,
    1000000000000000000000,
    10000000000000000000000,
    100000000000000000000000,
    1000000000000000000000000,
    10000000000000000000000000,
    100000000000000000000000000,
    1000000000000000000000000000,
    10000000000000000000000000000,
    100000000000000000000000000000,
    1000000000000000000000000000000,
    10000000000000000000000000000000,
    100000000000000000000000000000000,
    1000000000000000000000000000000000,
    10000000000000000000000000000000000,
    100000000000000000000000000000000000,
    1000000000000000000000000000000000000,
    10000000000000000000000000000000000000,
    100000000000000000000000000000000000000,
];

pub fn format_decimal(v: i128, scale: usize, trim_zeros: bool) -> FormatBuffer {
    const ZEROS: [u8; BUF_LEN] = [b'0'; BUF_LEN];

    let mut buf = FormatBuffer::new();
    let factor = POW10[scale];
    let (div, rem) = v.abs().div_rem_euclid(&factor);

    unsafe {
        let mut ptr = buf.data.as_mut_ptr();
        if v < 0 {
            *ptr = b'-';
            buf.len = 1;
            ptr = ptr.add(1);
        }
        let n_whole = itoap::write_to_ptr(ptr, div);
        buf.len += n_whole;
        ptr = ptr.add(n_whole);

        if scale == 0 {
            return buf;
        }

        *ptr = b'.';
        ptr = ptr.add(1);

        if rem != 0 {
            let mut frac_buf = [0_u8; BUF_LEN];
            let n_frac = itoap::write_to_ptr(frac_buf.as_mut_ptr(), rem);
            std::ptr::copy_nonoverlapping(ZEROS.as_ptr(), ptr, scale - n_frac);
            ptr = ptr.add(scale - n_frac);
            std::ptr::copy_nonoverlapping(frac_buf.as_mut_ptr(), ptr, n_frac);
            ptr = ptr.add(n_frac);
        } else {
            std::ptr::copy_nonoverlapping(ZEROS.as_ptr(), ptr, scale);
            ptr = ptr.add(scale);
        }
        buf.len += 1 + scale;

        if trim_zeros {
            ptr = ptr.sub(1);
            while *ptr == b'0' {
                ptr = ptr.sub(1);
                buf.len -= 1;
            }
            if *ptr == b'.' {
                buf.len -= 1;
            }
        }
    }

    buf
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

        let val = "+000000.5";
        assert_eq!(
            deserialize_decimal(val.as_bytes(), precision, scale),
            Some(50)
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
        assert_eq!(deserialize_decimal(val, None, 0), Some(1200)); // truncate scale
        assert_eq!(deserialize_decimal(val, None, 3), Some(1200010)); // exact scale
        assert_eq!(deserialize_decimal(val, None, 6), Some(1200010000)); // excess scale
        assert_eq!(deserialize_decimal(val, Some(7), 0), Some(1200)); // sufficient precision and truncate scale
        assert_eq!(deserialize_decimal(val, Some(7), 3), Some(1200010)); // exact precision and scale
        assert_eq!(deserialize_decimal(val, Some(10), 6), Some(1200010000)); // exact precision, excess scale
        assert_eq!(deserialize_decimal(val, Some(5), 6), None); // insufficient precision, excess scale
        assert_eq!(deserialize_decimal(val, Some(5), 3), None); // insufficient precision, exact scale
        assert_eq!(deserialize_decimal(val, Some(12), 5), Some(120001000)); // excess precision, excess scale
        assert_eq!(deserialize_decimal(val, None, 35), None); // scale causes insufficient precision
    }
}
