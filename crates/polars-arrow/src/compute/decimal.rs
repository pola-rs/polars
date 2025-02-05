use std::sync::atomic::{AtomicBool, Ordering};

use num_traits::Euclid;

static TRIM_DECIMAL_ZEROS: AtomicBool = AtomicBool::new(false);

pub fn get_trim_decimal_zeros() -> bool {
    TRIM_DECIMAL_ZEROS.load(Ordering::Relaxed)
}
pub fn set_trim_decimal_zeros(trim: Option<bool>) {
    TRIM_DECIMAL_ZEROS.store(trim.unwrap_or(false), Ordering::Relaxed)
}

/// Assuming bytes are a well-formed decimal number (with or without a separator),
/// infer the scale of the number.  If no separator is present, the scale is 0.
pub fn infer_scale(bytes: &[u8]) -> u8 {
    let Some(separator) = bytes.iter().position(|b| *b == b'.') else {
        return 0;
    };
    (bytes.len() - (1 + separator)) as u8
}

/// Deserialize bytes to a single i128 representing a decimal, at a specified
/// precision (optional) and scale (required). The number is checked to ensure
/// it fits within the specified precision and scale.  Consistent with float
/// parsing, no decimal separator is required (eg "500", "500.", and "500.0" are
/// all accepted); this allows mixed integer/decimal sequences to be parsed as
/// decimals.  All trailing zeros are assumed to be significant, whether or not
/// a separator is present: 1200 requires precision >= 4, while 1200.200
/// requires precision >= 7 and scale >= 3. Returns None if the number is not
/// well-formed, or does not fit. Only b'.' is allowed as a decimal separator
/// (issue #6698).
#[inline]
pub fn deserialize_decimal(bytes: &[u8], precision: Option<u8>, scale: u8) -> Option<i128> {
    let precision_digits = precision.unwrap_or(38).min(38) as usize;
    if scale as usize > precision_digits {
        return None;
    }

    let separator = bytes.iter().position(|b| *b == b'.').unwrap_or(bytes.len());
    let (mut int, mut frac) = bytes.split_at(separator);
    if frac.len() <= 1 || scale == 0 {
        // Only integer fast path.
        let n: i128 = atoi_simd::parse(int).ok()?;
        let ret = n.checked_mul(POW10[scale as usize] as i128)?;
        if precision.is_some() && ret >= POW10[precision_digits] as i128 {
            return None;
        }
        return Some(ret);
    }

    // Skip period.
    frac = &frac[1..];

    // Skip sign.
    let negative = match bytes.first() {
        Some(s @ (b'+' | b'-')) => {
            int = &int[1..];
            *s == b'-'
        },
        _ => false,
    };

    // Truncate trailing digits that extend beyond the scale.
    let frac_scale = if scale as usize <= frac.len() {
        frac = &frac[..scale as usize];
        0
    } else {
        scale as usize - frac.len()
    };

    // Parse and combine parts.
    let pint: u128 = if int.is_empty() {
        0
    } else {
        atoi_simd::parse_pos(int).ok()?
    };
    let pfrac: u128 = atoi_simd::parse_pos(frac).ok()?;

    let ret = pint
        .checked_mul(POW10[scale as usize])?
        .checked_add(pfrac.checked_mul(POW10[frac_scale])?)?;
    if precision.is_some() && ret >= POW10[precision_digits] {
        return None;
    }
    if negative {
        if ret > (1 << 127) {
            None
        } else {
            Some(ret.wrapping_neg() as i128)
        }
    } else {
        ret.try_into().ok()
    }
}

const MAX_DECIMAL_LEN: usize = 48;

#[derive(Clone, Copy)]
pub struct DecimalFmtBuffer {
    data: [u8; MAX_DECIMAL_LEN],
    len: usize,
}

impl Default for DecimalFmtBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl DecimalFmtBuffer {
    #[inline]
    pub const fn new() -> Self {
        Self {
            data: [0; MAX_DECIMAL_LEN],
            len: 0,
        }
    }

    pub fn format(&mut self, x: i128, scale: usize, trim_zeros: bool) -> &str {
        let factor = POW10[scale];
        let mut itoa_buf = itoa::Buffer::new();

        self.len = 0;
        let (div, rem) = x.unsigned_abs().div_rem_euclid(&factor);
        if x < 0 {
            self.data[0] = b'-';
            self.len += 1;
        }

        let div_fmt = itoa_buf.format(div);
        self.data[self.len..self.len + div_fmt.len()].copy_from_slice(div_fmt.as_bytes());
        self.len += div_fmt.len();

        if scale == 0 {
            return unsafe { std::str::from_utf8_unchecked(&self.data[..self.len]) };
        }

        self.data[self.len] = b'.';
        self.len += 1;

        let rem_fmt = itoa_buf.format(rem + factor); // + factor adds leading 1 where period would be.
        self.data[self.len..self.len + rem_fmt.len() - 1].copy_from_slice(rem_fmt[1..].as_bytes());
        self.len += rem_fmt.len() - 1;

        if trim_zeros {
            while self.data.get(self.len - 1) == Some(&b'0') {
                self.len -= 1;
            }
            if self.data.get(self.len - 1) == Some(&b'.') {
                self.len -= 1;
            }
        }

        unsafe { std::str::from_utf8_unchecked(&self.data[..self.len]) }
    }
}

const POW10: [u128; 39] = [
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
        assert_eq!(
            deserialize_decimal(val, None, 35),
            Some(120001000000000000000000000000000000000)
        );
        assert_eq!(deserialize_decimal(val, None, 36), None);
        assert_eq!(deserialize_decimal(val, Some(38), 35), None); // scale causes insufficient precision
    }
}
