use chrono::{NaiveDate, NaiveDateTime};

/// The following specifiers are available both to formatting and parsing.
///
/// | Spec. | Example  | Description                                                                |
/// |-------|----------|----------------------------------------------------------------------------|
/// |       |          | **DATE SPECIFIERS:**                                                       |
/// | `%Y`  | `2001`   | The full proleptic Gregorian year, zero-padded to 4 digits. [^1]           |
/// | `%C`  | `20`     | The proleptic Gregorian year divided by 100, zero-padded to 2 digits. [^2] |
/// | `%y`  | `01`     | The proleptic Gregorian year modulo 100, zero-padded to 2 digits. [^2]     |
/// |       |          |                                                                            |
/// | `%m`  | `07`     | Month number (01--12), zero-padded to 2 digits.                            |
/// | `%b`  | `Jul`    | Abbreviated month name. Always 3 letters.                                  |
/// | `%B`  | `July`   | Full month name. Also accepts corresponding abbreviation in parsing.       |
/// | `%h`  | `Jul`    | Same as `%b`.                                                              |
/// |       |          |                                                                            |
/// | `%d`  | `08`     | Day number (01--31), zero-padded to 2 digits.                              |
/// | `%e`  | ` 8`     | Same as `%d` but space-padded. Same as `%_d`.                              |
/// |       |          |                                                                            |
/// | `%a`  | `Sun`    | Abbreviated weekday name. Always 3 letters.                                |
/// | `%A`  | `Sunday` | Full weekday name. Also accepts corresponding abbreviation in parsing.     |
/// | `%w`  | `0`      | Sunday = 0, Monday = 1, ..., Saturday = 6.                                 |
/// | `%u`  | `7`      | Monday = 1, Tuesday = 2, ..., Sunday = 7. (ISO 8601)                       |
/// |       |          |                                                                            |
/// | `%U`  | `28`     | Week number starting with Sunday (00--53), zero-padded to 2 digits. [^3]   |
/// | `%W`  | `27`     | Same as `%U`, but week 1 starts with the first Monday in that year instead.|
/// |       |          |                                                                            |
/// | `%G`  | `2001`   | Same as `%Y` but uses the year number in ISO 8601 week date. [^4]          |
/// | `%g`  | `01`     | Same as `%y` but uses the year number in ISO 8601 week date. [^4]          |
/// | `%V`  | `27`     | Same as `%U` but uses the week number in ISO 8601 week date (01--53). [^4] |
/// |       |          |                                                                            |
/// | `%j`  | `189`    | Day of the year (001--366), zero-padded to 3 digits.                       |
/// |       |          |                                                                            |
/// | `%D`  | `07/08/01`    | Month-day-year format. Same as `%m/%d/%y`.                            |
/// | `%x`  | `07/08/01`    | Locale's date representation (e.g., 12/31/99).                        |
/// | `%F`  | `2001-07-08`  | Year-month-day format (ISO 8601). Same as `%Y-%m-%d`.                 |
/// | `%v`  | ` 8-Jul-2001` | Day-month-year format. Same as `%e-%b-%Y`.                            |
/// |       |          |                                                                            |
/// |       |          | **TIME SPECIFIERS:**                                                       |
/// | `%H`  | `00`     | Hour number (00--23), zero-padded to 2 digits.                             |
/// | `%k`  | ` 0`     | Same as `%H` but space-padded. Same as `%_H`.                              |
/// | `%I`  | `12`     | Hour number in 12-hour clocks (01--12), zero-padded to 2 digits.           |
/// | `%l`  | `12`     | Same as `%I` but space-padded. Same as `%_I`.                              |
/// |       |          |                                                                            |
/// | `%P`  | `am`     | `am` or `pm` in 12-hour clocks.                                            |
/// | `%p`  | `AM`     | `AM` or `PM` in 12-hour clocks.                                            |
/// |       |          |                                                                            |
/// | `%M`  | `34`     | Minute number (00--59), zero-padded to 2 digits.                           |
/// | `%S`  | `60`     | Second number (00--60), zero-padded to 2 digits. [^5]                      |
/// | `%f`  | `026490000`   | The fractional seconds (in nanoseconds) since last whole second. [^8] |
/// | `%.f` | `.026490`| Similar to `.%f` but left-aligned. These all consume the leading dot. [^8] |
/// | `%.3f`| `.026`        | Similar to `.%f` but left-aligned but fixed to a length of 3. [^8]    |
/// | `%.6f`| `.026490`     | Similar to `.%f` but left-aligned but fixed to a length of 6. [^8]    |
/// | `%.9f`| `.026490000`  | Similar to `.%f` but left-aligned but fixed to a length of 9. [^8]    |
/// | `%3f` | `026`         | Similar to `%.3f` but without the leading dot. [^8]                   |
/// | `%6f` | `026490`      | Similar to `%.6f` but without the leading dot. [^8]                   |
/// | `%9f` | `026490000`   | Similar to `%.9f` but without the leading dot. [^8]                   |
/// |       |               |                                                                       |
/// | `%R`  | `00:34`       | Hour-minute format. Same as `%H:%M`.                                  |
/// | `%T`  | `00:34:60`    | Hour-minute-second format. Same as `%H:%M:%S`.                        |
/// | `%X`  | `00:34:60`    | Locale's time representation (e.g., 23:13:48).                        |
/// | `%r`  | `12:34:60 AM` | Hour-minute-second format in 12-hour clocks. Same as `%I:%M:%S %p`.   |
/// |       |          |                                                                            |
/// |       |          | **TIME ZONE SPECIFIERS:**                                                  |
/// | `%Z`  | `ACST`   | Local time zone name. Skips all non-whitespace characters during parsing. [^9] |
/// | `%z`  | `+0930`  | Offset from the local time to UTC (with UTC being `+0000`).                |
/// | `%:z` | `+09:30` | Same as `%z` but with a colon.                                             |
/// | `%#z` | `+09`    | *Parsing only:* Same as `%z` but allows minutes to be missing or present.  |
/// |       |          |                                                                            |
/// |       |          | **DATE & TIME SPECIFIERS:**                                                |
/// |`%c`|`Sun Jul  8 00:34:60 2001`|Locale's date and time (e.g., Thu Mar  3 23:05:25 2005).       |
/// | `%+`  | `2001-07-08T00:34:60.026490+09:30` | ISO 8601 / RFC 3339 date & time format. [^6]     |
/// |       |               |                                                                       |
/// | `%s`  | `994518299`   | UNIX timestamp, the number of seconds since 1970-01-01 00:00 UTC. [^7]|
/// |       |          |                                                                            |
/// |       |          | **SPECIAL SPECIFIERS:**                                                    |
/// | `%t`  |          | Literal tab (`\t`).                                                        |
/// | `%n`  |          | Literal newline (`\n`).                                                    |
/// | `%%`  |          | Literal percent sign.                                                      |


#[inline]
fn update_and_parse<T: lexical::FromLexical>(incr: usize, offset: usize, vals: &[u8]) -> Option<(T, usize)> {
    let new_offset = offset + incr;
    lexical::parse(&vals[offset..new_offset]).ok().map(|v| (v, new_offset))
}


#[inline]
pub fn parse(val: &[u8], fmt: &[u8], fmt_len: u16) -> Option<NaiveDateTime> {
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

    while let  Some(fmt_b) = fmt_iter.next() {
        let b = val[offset];
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
                    year = 2000 + lexical::parse::<i32, _>(&val[offset..new_offset]).ok()?;
                    offset = new_offset;
                }
                b => panic!("char: {} not implemented", *b as char)
            }
        }
        // consume
        else if b == *fmt_b {
            offset += 1;
        } else {
            return None
        }
    }

    Some(NaiveDate::from_ymd(year, month, day).and_hms_nano(hour, min, sec, nano))
}

fn fmt_len(fmt: &[u8]) -> u16 {
    let mut iter = fmt.iter();
    let mut cnt = 0;

    while let Some(&val) = iter.next() {
        match val {
            b'%' => {
                match iter.next().expect("invalid patter") {
                    b'Y' => cnt += 4,
                    b'y' => cnt += 2,
                    b'd' => cnt += 2,
                    b'm' => cnt += 2,
                    b'H' => cnt += 2,
                    b'M' => cnt += 2,
                    b'S' => cnt += 2,
                    b => panic!("char: {} not implemented", *b as char)
                }
            },
            _ => {
                cnt += 1;
            }
        }
    }
    cnt
}


#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_parsing() {
        let patterns = [
            ("2021-01-01", "%Y-%m-%d", 10, Some(NaiveDate::from_ymd(2021, 1, 1).and_hms_nano(0, 0, 0, 0))),
            ("2021-01-01 07:45:12", "%Y-%m-%d %H:%M:%S", 19, Some(NaiveDate::from_ymd(2021, 1, 1).and_hms_nano(7, 45, 12, 0)))
        ];

        for (val, fmt, len, expected) in patterns {
            assert_eq!(fmt_len(fmt.as_bytes()), len);
            assert_eq!(parse(val.as_bytes(), fmt.as_bytes(), len), expected);
        }



    }
}