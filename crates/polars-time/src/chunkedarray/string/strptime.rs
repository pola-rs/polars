//! Much more opinionated, but also much faster strptrime than the one given in Chrono.
//!
use atoi::FromRadix10;
use chrono::{NaiveDate, NaiveDateTime};
use once_cell::sync::Lazy;
use polars_utils::slice::GetSaferUnchecked;
use regex::Regex;

use crate::chunkedarray::{polars_bail, PolarsResult};

static HOUR_PATTERN: Lazy<Regex> = Lazy::new(|| Regex::new(r"%[_-]?[HkIl]").unwrap());
static MINUTE_PATTERN: Lazy<Regex> = Lazy::new(|| Regex::new(r"%[_-]?M").unwrap());
static SECOND_PATTERN: Lazy<Regex> = Lazy::new(|| Regex::new(r"%[_-]?S").unwrap());
static TWELVE_HOUR_PATTERN: Lazy<Regex> = Lazy::new(|| Regex::new(r"%[_-]?[Il]").unwrap());
static MERIDIEM_PATTERN: Lazy<Regex> = Lazy::new(|| Regex::new(r"%[_-]?[pP]").unwrap());

#[inline]
fn update_and_parse<T: atoi::FromRadix10>(
    incr: usize,
    offset: usize,
    vals: &[u8],
) -> Option<(T, usize)> {
    // this maybe oob because we cannot entirely sure about fmt lengths
    let new_offset = offset + incr;
    let bytes = vals.get(offset..new_offset)?;
    let (val, parsed) = T::from_radix_10(bytes);
    if parsed == 0 {
        None
    } else {
        Some((val, new_offset))
    }
}

#[inline]
fn parse_month_abbrev(val: &[u8], offset: usize) -> Option<(u32, usize)> {
    let new_offset = offset + 3;
    match &val[offset..new_offset] {
        b"Jan" => Some((1, new_offset)),
        b"Feb" => Some((2, new_offset)),
        b"Mar" => Some((3, new_offset)),
        b"Apr" => Some((4, new_offset)),
        b"May" => Some((5, new_offset)),
        b"Jun" => Some((6, new_offset)),
        b"Jul" => Some((7, new_offset)),
        b"Aug" => Some((8, new_offset)),
        b"Sep" => Some((9, new_offset)),
        b"Oct" => Some((10, new_offset)),
        b"Nov" => Some((11, new_offset)),
        b"Dec" => Some((12, new_offset)),
        _ => None,
    }
}

/// Tries to convert a chrono `fmt` to a `fmt` that the polars parser consumes.
/// E.g. chrono supports single letter date identifiers like %F, whereas polars only consumes
/// year, day, month distinctively with %Y, %d, %m.
pub(super) fn compile_fmt(fmt: &str) -> PolarsResult<String> {
    // (hopefully) temporary hacks. Ideally, chrono would return a ParseKindError indicating
    // if `fmt` is too long for NaiveDate. If that's implemented, then this check could
    // be removed, and that error could be matched against in `transform_datetime_*s`
    // See https://github.com/chronotope/chrono/issues/1075.
    if HOUR_PATTERN.is_match(fmt) ^ MINUTE_PATTERN.is_match(fmt) {
        polars_bail!(ComputeError: "Invalid format string: \
            Please either specify both hour and minute, or neither.");
    }
    if SECOND_PATTERN.is_match(fmt) && !HOUR_PATTERN.is_match(fmt) {
        polars_bail!(ComputeError: "Invalid format string: \
            Found seconds directive, but no hours directive.");
    }
    if TWELVE_HOUR_PATTERN.is_match(fmt) ^ MERIDIEM_PATTERN.is_match(fmt) {
        polars_bail!(ComputeError: "Invalid format string: \
            Please either specify both 12-hour directive and meridiem directive, or neither.");
    }

    Ok(fmt
        .replace("%D", "%m/%d/%y")
        .replace("%R", "%H:%M")
        .replace("%T", "%H:%M:%S")
        .replace("%X", "%H:%M:%S")
        .replace("%F", "%Y-%m-%d"))
}

#[derive(Default, Clone)]
pub(super) struct StrpTimeState {}

impl StrpTimeState {
    #[inline]
    // # Safety
    // Caller must ensure that fmt adheres to the fmt rules of chrono and `fmt_len` is correct.
    pub(super) unsafe fn parse(
        &mut self,
        val: &[u8],
        fmt: &[u8],
        fmt_len: u16,
    ) -> Option<NaiveDateTime> {
        let mut offset = 0;
        let mut negative = false;
        if val.starts_with(b"-") && fmt.starts_with(b"%Y") {
            offset = 1;
            negative = true;
        }
        if val.len() - offset != (fmt_len as usize) {
            return None;
        }

        const ESCAPE: u8 = b'%';
        let mut year: i32 = 1;
        // minimal day/month is always 1
        // otherwise chrono may panic.
        let mut month: u32 = 1;
        let mut day: u32 = 1;
        let mut hour: u32 = 0;
        let mut min: u32 = 0;
        let mut sec: u32 = 0;
        let mut nano: u32 = 0;

        let mut fmt_iter = fmt.iter();

        while let Some(fmt_b) = fmt_iter.next() {
            debug_assert!(offset < val.len());
            let b = *val.get_unchecked(offset);
            if *fmt_b == ESCAPE {
                // SAFETY: we must ensure we provide valid patterns
                let next = fmt_iter.next();
                debug_assert!(next.is_some());
                match next.unwrap_unchecked() {
                    b'Y' => {
                        (year, offset) = update_and_parse(4, offset, val)?;
                        if negative {
                            year *= -1
                        }
                    },
                    b'm' => {
                        (month, offset) = update_and_parse(2, offset, val)?;
                        if month > 12 {
                            return None;
                        }
                    },
                    b'b' => {
                        (month, offset) = parse_month_abbrev(val, offset)?;
                    },
                    b'd' => {
                        (day, offset) = update_and_parse(2, offset, val)?;
                    },
                    b'H' => {
                        (hour, offset) = update_and_parse(2, offset, val)?;
                    },
                    b'M' => {
                        (min, offset) = update_and_parse(2, offset, val)?;
                    },
                    b'S' => {
                        (sec, offset) = update_and_parse(2, offset, val)?;
                    },
                    b'y' => {
                        let new_offset = offset + 2;
                        let bytes = val.get_unchecked_release(offset..new_offset);

                        let (decade, parsed) = i32::from_radix_10(bytes);
                        if parsed == 0 {
                            return None;
                        }

                        if decade < 70 {
                            year = 2000 + decade;
                        } else {
                            year = 1900 + decade;
                        }
                        offset = new_offset;
                    },
                    b'9' => {
                        (nano, offset) = update_and_parse(9, offset, val)?;
                        break;
                    },
                    b'6' => {
                        (nano, offset) = update_and_parse(6, offset, val)?;
                        nano *= 1000;
                        break;
                    },
                    b'3' => {
                        (nano, offset) = update_and_parse(3, offset, val)?;
                        nano *= 1_000_000;
                        break;
                    },
                    _ => return None,
                }
            }
            // consume
            else if b == *fmt_b {
                offset += 1;
            } else {
                return None;
            }
        }
        // all values processed
        if offset == val.len() {
            NaiveDate::from_ymd_opt(year, month, day)
                .and_then(|nd| nd.and_hms_nano_opt(hour, min, sec, nano))
        }
        // remaining values did not match pattern
        else {
            None
        }
    }
}

pub(super) fn fmt_len(fmt: &[u8]) -> Option<u16> {
    let mut iter = fmt.iter();
    let mut cnt = 0;

    while let Some(&val) = iter.next() {
        match val {
            b'%' => match iter.next().expect("invalid pattern") {
                b'Y' => cnt += 4,
                b'y' => cnt += 2,
                b'd' => cnt += 2,
                b'm' => cnt += 2,
                b'b' => cnt += 3,
                b'H' => cnt += 2,
                b'M' => cnt += 2,
                b'S' => cnt += 2,
                b'9' => {
                    cnt += 9;
                    debug_assert_eq!(iter.next(), Some(&b'f'));
                    return Some(cnt);
                },
                b'6' => {
                    cnt += 6;
                    debug_assert_eq!(iter.next(), Some(&b'f'));
                    return Some(cnt);
                },
                b'3' => {
                    cnt += 3;
                    debug_assert_eq!(iter.next(), Some(&b'f'));
                    return Some(cnt);
                },
                _ => return None,
            },
            _ => {
                cnt += 1;
            },
        }
    }
    Some(cnt)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_parsing() {
        let patterns = [
            (
                "2021-01-01",
                "%Y-%m-%d",
                10,
                Some(
                    NaiveDate::from_ymd_opt(2021, 1, 1)
                        .unwrap()
                        .and_hms_nano_opt(0, 0, 0, 0)
                        .unwrap(),
                ),
            ),
            (
                "2021-01-01 07:45:12",
                "%Y-%m-%d %H:%M:%S",
                19,
                Some(
                    NaiveDate::from_ymd_opt(2021, 1, 1)
                        .unwrap()
                        .and_hms_nano_opt(7, 45, 12, 0)
                        .unwrap(),
                ),
            ),
            (
                "2021-01-01 07:45:12",
                "%Y-%m-%d %H:%M:%S",
                19,
                Some(
                    NaiveDate::from_ymd_opt(2021, 1, 1)
                        .unwrap()
                        .and_hms_nano_opt(7, 45, 12, 0)
                        .unwrap(),
                ),
            ),
            (
                "2019-04-18T02:45:55.555000000",
                "%Y-%m-%dT%H:%M:%S.%9f",
                29,
                Some(
                    NaiveDate::from_ymd_opt(2019, 4, 18)
                        .unwrap()
                        .and_hms_nano_opt(2, 45, 55, 555000000)
                        .unwrap(),
                ),
            ),
            (
                "2019-04-18T02:45:55.555000",
                "%Y-%m-%dT%H:%M:%S.%6f",
                26,
                Some(
                    NaiveDate::from_ymd_opt(2019, 4, 18)
                        .unwrap()
                        .and_hms_nano_opt(2, 45, 55, 555000000)
                        .unwrap(),
                ),
            ),
            (
                "2019-04-18T02:45:55.555",
                "%Y-%m-%dT%H:%M:%S.%3f",
                23,
                Some(
                    NaiveDate::from_ymd_opt(2019, 4, 18)
                        .unwrap()
                        .and_hms_nano_opt(2, 45, 55, 555000000)
                        .unwrap(),
                ),
            ),
        ];

        for (val, fmt, len, expected) in patterns {
            assert_eq!(fmt_len(fmt.as_bytes()).unwrap(), len);
            unsafe {
                assert_eq!(
                    StrpTimeState::default().parse(val.as_bytes(), fmt.as_bytes(), len),
                    expected
                )
            };
        }
    }
}
