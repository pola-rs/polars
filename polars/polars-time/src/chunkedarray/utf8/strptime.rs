//! Much more opinionated, but also much faster strptrime than the one given in Chrono.
//!
use chrono::{NaiveDate, NaiveDateTime};

#[inline]
unsafe fn update_and_parse<T: lexical::FromLexical>(
    incr: usize,
    offset: usize,
    vals: &[u8],
) -> Option<(T, usize)> {
    let new_offset = offset + incr;
    debug_assert!(new_offset <= vals.len());

    lexical::parse(vals.get_unchecked(offset..new_offset))
        .ok()
        .map(|v| (v, new_offset))
}

/// Tries to convert a chrono `fmt` to a `fmt` that the polars parser consumes.
/// E.g. chrono supports single letter date identifiers like %F, whereas polars only consumes
/// year, day, month distinctively with %Y, %d, %m.
pub(super) fn compile_fmt(fmt: &str) -> String {
    fmt.replace("%D", "%m/%d/%y")
        .replace("%R", "%H:%M")
        .replace("%T", "%H:%M:%S")
        .replace("%X", "%H:%M:%S")
        .replace("%F", "%Y-%m-%d")
}

#[inline]
// # Safety
// Caller must ensure that fmt adheres to the fmt rules of chrono and `fmt_len` is correct.
pub(super) unsafe fn parse(val: &[u8], fmt: &[u8], fmt_len: u16) -> Option<NaiveDateTime> {
    const ESCAPE: u8 = b'%';
    if val.len() < fmt_len as usize {
        return None;
    }
    let mut year: i32 = 0;
    let mut month: u32 = 0;
    let mut day: u32 = 0;
    let mut hour: u32 = 0;
    let mut min: u32 = 0;
    let mut sec: u32 = 0;
    let mut nano: u32 = 0;

    let mut fmt_iter = fmt.iter();

    let mut offset = 0;

    while let Some(fmt_b) = fmt_iter.next() {
        debug_assert!(offset < val.len());
        let b = *val.get_unchecked(offset);
        if *fmt_b == ESCAPE {
            match fmt_iter.next().expect("invalid fmt") {
                b'Y' => {
                    (year, offset) = update_and_parse(4, offset, val)?;
                }
                b'm' => {
                    (month, offset) = update_and_parse(2, offset, val)?;
                }
                b'd' => {
                    (day, offset) = update_and_parse(2, offset, val)?;
                }
                b'H' => {
                    (hour, offset) = update_and_parse(2, offset, val)?;
                }
                b'M' => {
                    (min, offset) = update_and_parse(2, offset, val)?;
                }
                b'S' => {
                    (sec, offset) = update_and_parse(2, offset, val)?;
                }
                b'y' => {
                    let new_offset = offset + 2;
                    let decade = lexical::parse::<i32, _>(&val[offset..new_offset]).ok()?;
                    if decade < 50 {
                        year = 2000 + decade;
                    } else {
                        year = 1900 + decade;
                    }
                    offset = new_offset;
                }
                b'9' => {
                    (nano, _) = update_and_parse(9, offset, val)?;
                    break;
                }
                b'6' => {
                    (nano, _) = update_and_parse(6, offset, val)?;
                    nano *= 1000;
                    break;
                }
                b'3' => {
                    (nano, _) = update_and_parse(3, offset, val)?;
                    nano *= 1_000_000;
                    break;
                }
                // utc can be ignored
                b'Z' => {}
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

    Some(NaiveDate::from_ymd(year, month, day).and_hms_nano(hour, min, sec, nano))
}

pub(super) fn fmt_len(fmt: &[u8]) -> Option<u16> {
    let mut iter = fmt.iter();
    let mut cnt = 0;

    while let Some(&val) = iter.next() {
        match val {
            b'%' => match iter.next().expect("invalid patter") {
                b'Y' => cnt += 4,
                b'y' => cnt += 2,
                b'd' => cnt += 2,
                b'm' => cnt += 2,
                b'H' => cnt += 2,
                b'M' => cnt += 2,
                b'S' => cnt += 2,
                b'Z' => cnt += 1,
                b'9' => {
                    cnt += 9;
                    debug_assert_eq!(iter.next(), Some(&b'f'));
                    return Some(cnt);
                }
                b'6' => {
                    cnt += 6;
                    debug_assert_eq!(iter.next(), Some(&b'f'));
                    return Some(cnt);
                }
                b'3' => {
                    cnt += 3;
                    debug_assert_eq!(iter.next(), Some(&b'f'));
                    return Some(cnt);
                }
                _ => return None,
            },
            _ => {
                cnt += 1;
            }
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
                Some(NaiveDate::from_ymd(2021, 1, 1).and_hms_nano(0, 0, 0, 0)),
            ),
            (
                "2021-01-01 07:45:12",
                "%Y-%m-%d %H:%M:%S",
                19,
                Some(NaiveDate::from_ymd(2021, 1, 1).and_hms_nano(7, 45, 12, 0)),
            ),
            (
                "2021-01-01 07:45:12",
                "%Y-%m-%d %H:%M:%S",
                19,
                Some(NaiveDate::from_ymd(2021, 1, 1).and_hms_nano(7, 45, 12, 0)),
            ),
            (
                "2019-04-18T02:45:55.555000000",
                "%Y-%m-%dT%H:%M:%S.%9f",
                29,
                Some(NaiveDate::from_ymd(2019, 4, 18).and_hms_nano(2, 45, 55, 555000000)),
            ),
            (
                "2019-04-18T02:45:55.555000",
                "%Y-%m-%dT%H:%M:%S.%6f",
                26,
                Some(NaiveDate::from_ymd(2019, 4, 18).and_hms_nano(2, 45, 55, 555000000)),
            ),
            (
                "2019-04-18T02:45:55.555",
                "%Y-%m-%dT%H:%M:%S.%3f",
                23,
                Some(NaiveDate::from_ymd(2019, 4, 18).and_hms_nano(2, 45, 55, 555000000)),
            ),
        ];

        for (val, fmt, len, expected) in patterns {
            assert_eq!(fmt_len(fmt.as_bytes()).unwrap(), len);
            unsafe { assert_eq!(parse(val.as_bytes(), fmt.as_bytes(), len), expected) };
        }
    }
}
