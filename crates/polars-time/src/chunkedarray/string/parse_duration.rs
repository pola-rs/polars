use polars_core::datatypes::TimeUnit;

const NS_PER_SECOND: i128 = 1_000_000_000;
const NS_PER_MINUTE: i128 = 60 * NS_PER_SECOND;
const NS_PER_HOUR: i128 = 60 * NS_PER_MINUTE;
const NS_PER_DAY: i128 = 24 * NS_PER_HOUR;

/// Read a run of ASCII digits starting at `pos`, returning
/// `(value, new_pos, n_digits)`. Returns `None` if no digit is found or the
/// value overflows an `i128`.
fn read_uint(bytes: &[u8], mut pos: usize) -> Option<(i128, usize, usize)> {
    let start = pos;
    let mut val: i128 = 0;
    while pos < bytes.len() && bytes[pos].is_ascii_digit() {
        val = val.checked_mul(10)?;
        val = val.checked_add((bytes[pos] - b'0') as i128)?;
        pos += 1;
    }
    if pos == start {
        return None;
    }
    Some((val, pos, pos - start))
}

/// Parse a "stopwatch"/colon-style duration string into an integer number of
/// `time_unit`s, according to `fmt`.
///
/// Unlike chrono's time parsing, fields are **not** range-checked, so elapsed
/// time values such as `80` minutes or `104` hours parse successfully. This
/// makes it suitable for parsing stopwatch / duration strings such as
/// `"80:00"`, `"-04:00:00"` or `"104:00:00.250"`.
///
/// Supported format specifiers:
/// * `%d` - days
/// * `%H` - hours
/// * `%M` - minutes
/// * `%S` - whole seconds
/// * `%f` - fractional seconds digits (the value is right-padded to nanoseconds;
///   extra digits beyond nanosecond resolution are truncated)
/// * `%.f` - an optional `.` followed by fractional seconds (like chrono)
/// * `%%` - a literal `%`
///
/// A leading `-` or `+` in the input sets the sign of the whole duration. Any
/// other character in `fmt` must match the input literally. The entire input
/// must be consumed, otherwise `None` is returned.
pub(super) fn parse_duration_string(input: &str, fmt: &str, time_unit: TimeUnit) -> Option<i64> {
    let bytes = input.as_bytes();
    let fbytes = fmt.as_bytes();
    let mut ipos = 0usize;
    let mut fpos = 0usize;

    // Optional leading sign applies to the whole duration.
    let negative = match bytes.first() {
        Some(b'-') => {
            ipos += 1;
            true
        },
        Some(b'+') => {
            ipos += 1;
            false
        },
        _ => false,
    };

    let mut total_ns: i128 = 0;

    while fpos < fbytes.len() {
        if fbytes[fpos] == b'%' {
            // A specifier must follow the '%'.
            let spec = *fbytes.get(fpos + 1)?;
            match spec {
                b'd' | b'H' | b'M' | b'S' => {
                    let (val, npos, _) = read_uint(bytes, ipos)?;
                    let scale = match spec {
                        b'd' => NS_PER_DAY,
                        b'H' => NS_PER_HOUR,
                        b'M' => NS_PER_MINUTE,
                        b'S' => NS_PER_SECOND,
                        _ => unreachable!(),
                    };
                    total_ns = total_ns.checked_add(val.checked_mul(scale)?)?;
                    ipos = npos;
                    fpos += 2;
                },
                b'f' => {
                    let (frac_ns, npos) = read_fraction(bytes, ipos)?;
                    total_ns = total_ns.checked_add(frac_ns)?;
                    ipos = npos;
                    fpos += 2;
                },
                b'.' => {
                    // `%.f`: an optional '.' followed by fractional seconds.
                    if *fbytes.get(fpos + 2)? != b'f' {
                        return None;
                    }
                    if ipos < bytes.len() && bytes[ipos] == b'.' {
                        ipos += 1;
                        let (frac_ns, npos) = read_fraction(bytes, ipos)?;
                        total_ns = total_ns.checked_add(frac_ns)?;
                        ipos = npos;
                    }
                    fpos += 3;
                },
                b'%' => {
                    if ipos >= bytes.len() || bytes[ipos] != b'%' {
                        return None;
                    }
                    ipos += 1;
                    fpos += 2;
                },
                _ => return None,
            }
        } else {
            // Literal character: must match the input exactly.
            if ipos >= bytes.len() || bytes[ipos] != fbytes[fpos] {
                return None;
            }
            ipos += 1;
            fpos += 1;
        }
    }

    // The whole input must have been consumed.
    if ipos != bytes.len() {
        return None;
    }

    let total = match time_unit {
        TimeUnit::Nanoseconds => total_ns,
        TimeUnit::Microseconds => total_ns / 1_000,
        TimeUnit::Milliseconds => total_ns / 1_000_000,
    };
    let total = if negative { -total } else { total };
    i64::try_from(total).ok()
}

/// Read fractional-second digits and return their value in nanoseconds along
/// with the new input position. Digits beyond nanosecond resolution (9 digits)
/// are consumed but truncated.
fn read_fraction(bytes: &[u8], pos: usize) -> Option<(i128, usize)> {
    let (_, npos, n_digits) = read_uint(bytes, pos)?;
    let mut frac_ns: i128 = 0;
    let mut scale: i128 = 100_000_000; // value of the first fractional digit in ns
    for &b in &bytes[pos..npos] {
        if scale == 0 {
            break;
        }
        frac_ns += (b - b'0') as i128 * scale;
        scale /= 10;
    }
    debug_assert!(n_digits >= 1);
    Some((frac_ns, npos))
}

#[cfg(test)]
mod test {
    use super::*;

    const NS: TimeUnit = TimeUnit::Nanoseconds;
    const US: TimeUnit = TimeUnit::Microseconds;
    const MS: TimeUnit = TimeUnit::Milliseconds;

    #[test]
    fn parses_issue_examples() {
        // "80:00" meaning 80 minutes (format %M:%S).
        assert_eq!(
            parse_duration_string("80:00", "%M:%S", NS),
            Some(80 * 60 * 1_000_000_000)
        );
        // "-04:00:00" minus 4 hours.
        assert_eq!(
            parse_duration_string("-04:00:00", "%H:%M:%S", NS),
            Some(-4 * 3_600 * 1_000_000_000)
        );
        // "104:00:00" -> 104 hours (note: not range-checked).
        assert_eq!(
            parse_duration_string("104:00:00", "%H:%M:%S", NS),
            Some(104 * 3_600 * 1_000_000_000)
        );
    }

    #[test]
    fn fractional_seconds() {
        // "80:00.000" should parse with an optional fraction.
        assert_eq!(
            parse_duration_string("80:00.000", "%M:%S%.f", NS),
            Some(80 * 60 * 1_000_000_000)
        );
        assert_eq!(
            parse_duration_string("00:00.250", "%M:%S%.f", MS),
            Some(250)
        );
        // %f without a leading dot (standalone fractional seconds).
        assert_eq!(parse_duration_string("250", "%f", MS), Some(250));
        // Sub-nanosecond digits are truncated, not rejected.
        assert_eq!(
            parse_duration_string("1.1234567899", "%S%.f", NS),
            Some(1_123_456_789)
        );
    }

    #[test]
    fn time_units() {
        assert_eq!(parse_duration_string("1", "%S", NS), Some(1_000_000_000));
        assert_eq!(parse_duration_string("1", "%S", US), Some(1_000_000));
        assert_eq!(parse_duration_string("1", "%S", MS), Some(1_000));
    }

    #[test]
    fn days_and_signs() {
        assert_eq!(
            parse_duration_string("2:03:04:05", "%d:%H:%M:%S", NS),
            Some(((2 * 86_400 + 3 * 3_600 + 4 * 60 + 5) as i64) * 1_000_000_000)
        );
        assert_eq!(parse_duration_string("+30", "%S", NS), Some(30_000_000_000));
    }

    #[test]
    fn rejects_invalid() {
        // Trailing characters not covered by the format.
        assert_eq!(parse_duration_string("01:02:03x", "%H:%M:%S", NS), None);
        // Missing separator.
        assert_eq!(parse_duration_string("0102", "%H:%M", NS), None);
        // Non-numeric where a number is expected.
        assert_eq!(parse_duration_string("ab:00", "%M:%S", NS), None);
        // Empty input where digits are required.
        assert_eq!(parse_duration_string("", "%S", NS), None);
    }
}
