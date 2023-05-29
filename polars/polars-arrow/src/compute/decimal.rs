use atoi::atoi;

// 90.xx has 2 sd
// 00900.xx has 3 sd
fn significant_digits_precision(bytes: &[u8]) -> u8 {
    (bytes.len() - bytes.iter().take_while(|byte| **byte == b'0').count()) as u8
}

// 0.009 has 3 sd
// 0.900 has 1 sd
fn significant_digits_scale(bytes: &[u8]) -> u8 {
    (bytes.len() - bytes.iter().rev().take_while(|byte| **byte == b'0').count()) as u8
}

fn split_decimal_bytes(bytes: &[u8]) -> (Option<&[u8]>, Option<&[u8]>) {
    let mut a = bytes.split(|x| *x == b'.');
    let lhs = a.next();
    let rhs = a.next();
    (lhs, rhs)
}

pub fn infer_params(bytes: &[u8]) -> Option<(u8, u8)> {
    let (lhs, rhs) = split_decimal_bytes(bytes);
    match (lhs, rhs) {
        (Some(lhs), Some(rhs)) => {
            let lhs_s = significant_digits_precision(lhs);
            let rhs_s = significant_digits_scale(rhs);

            let precision = lhs_s + rhs_s;
            let scale = rhs_s;
            Some((precision, scale))
        }
        (None, Some(rhs)) => {
            let precision = rhs.len() as u8;
            Some((precision, precision))
        }
        (Some(lhs), None) => {
            let precision = lhs.len() as u8;
            Some((precision, 0))
        }
        (None, None) => None,
    }
}

/// Deserializes bytes to a single i128 representing a decimal
/// The decimal precision and scale are not checked.
#[inline]
pub(super) fn deserialize_decimal(bytes: &[u8], precision: u8, scale: u8) -> Option<i128> {
    let (lhs, rhs) = split_decimal_bytes(bytes);
    match (lhs, rhs) {
        (Some(lhs), Some(rhs)) => atoi::<i128>(lhs).and_then(|x| {
            atoi::<i128>(rhs)
                .map(|y| (x, lhs, y, rhs))
                .and_then(|(lhs, lhs_b, rhs, rhs_b)| {
                    let lhs_s = significant_digits_precision(lhs_b);
                    let rhs_s = significant_digits_scale(rhs_b);

                    if lhs_s + rhs_s > precision || rhs_s > scale {
                        None
                    } else if rhs_s < scale {
                        let diff = scale - rhs_s;

                        Some((lhs, rhs * 10i128.pow(diff as u32), rhs_s))
                    } else {
                        Some((lhs, rhs, rhs_s))
                    }
                })
                .map(|(lhs, rhs, rhs_s)| lhs * 10i128.pow(scale as u32) + rhs)
        }),
        (None, Some(rhs)) => {
            if rhs.len() != precision as usize || rhs.len() != scale as usize {
                return None;
            }
            atoi::<i128>(rhs)
        }
        (Some(lhs), None) => {
            if lhs.len() != precision as usize || scale != 0 {
                return None;
            }
            atoi::<i128>(lhs)
        }
        (None, None) => None,
    }
}
