use polars_core::prelude::TimeUnit;
use polars_error::{PolarsResult, polars_bail, polars_err};

#[derive(Clone, Copy)]
enum Field {
    Hour,
    Minute,
    Second,
    Fraction { digits: Option<usize>, dot: bool },
}

enum Token {
    Literal(Vec<u8>),
    Field(Field),
}

pub(super) struct DurationFormat {
    tokens: Vec<Token>,
}

impl DurationFormat {
    pub(super) fn compile(fmt: &str) -> PolarsResult<Self> {
        let bytes = fmt.as_bytes();
        let mut tokens = Vec::new();
        let mut literal = Vec::new();
        let mut pos = 0;
        let mut seen = [false; 4];

        while pos < bytes.len() {
            if bytes[pos] != b'%' {
                literal.push(bytes[pos]);
                pos += 1;
                continue;
            }
            if !literal.is_empty() {
                tokens.push(Token::Literal(std::mem::take(&mut literal)));
            }
            pos += 1;
            if pos >= bytes.len() {
                polars_bail!(InvalidOperation: "invalid duration format: trailing '%'" );
            }
            if bytes[pos] == b'%' {
                literal.push(b'%');
                pos += 1;
                continue;
            }

            let mut dot = false;
            if bytes[pos] == b'.' {
                dot = true;
                pos += 1;
            }
            let start = pos;
            while pos < bytes.len() && bytes[pos].is_ascii_digit() {
                pos += 1;
            }
            let digits = if start == pos {
                None
            } else {
                Some(
                    std::str::from_utf8(&bytes[start..pos])
                        .unwrap()
                        .parse::<usize>()
                        .map_err(|_| polars_err!(InvalidOperation: "invalid duration precision"))?,
                )
            };
            if pos >= bytes.len() {
                polars_bail!(InvalidOperation: "invalid duration format directive");
            }

            let (field, seen_idx) = match bytes[pos] {
                b'H' if !dot && digits.is_none() => (Field::Hour, 0),
                b'M' if !dot && digits.is_none() => (Field::Minute, 1),
                b'S' if !dot && digits.is_none() => (Field::Second, 2),
                b'f' if digits.is_none_or(|n| (1..=9).contains(&n)) => {
                    (Field::Fraction { digits, dot }, 3)
                },
                directive => polars_bail!(
                    InvalidOperation:
                    "unsupported duration format directive '%{}{}{}'",
                    if dot { "." } else { "" },
                    digits.map_or(String::new(), |n| n.to_string()),
                    directive as char
                ),
            };
            if seen[seen_idx] {
                polars_bail!(InvalidOperation: "duration format contains a duplicate field");
            }
            seen[seen_idx] = true;
            tokens.push(Token::Field(field));
            pos += 1;
        }
        if !literal.is_empty() {
            tokens.push(Token::Literal(literal));
        }
        if !seen[..3].iter().any(|v| *v) {
            polars_bail!(InvalidOperation: "duration format must contain %H, %M, or %S");
        }
        Ok(Self { tokens })
    }

    pub(super) fn parse(&self, value: &str, time_unit: TimeUnit) -> Option<i64> {
        let bytes = value.as_bytes();
        let (negative, mut pos) = match bytes.first() {
            Some(b'-') => (true, 1),
            Some(b'+') => (false, 1),
            _ => (false, 0),
        };
        let mut values = [0i64; 3];
        let mut fraction_ns = 0i64;
        let highest = self.tokens.iter().find_map(|token| match token {
            Token::Field(Field::Hour) => Some(0),
            Token::Field(Field::Minute) => Some(1),
            Token::Field(Field::Second) => Some(2),
            _ => None,
        })?;

        for token in &self.tokens {
            match token {
                Token::Literal(literal) => {
                    if !bytes.get(pos..)?.starts_with(literal) {
                        return None;
                    }
                    pos += literal.len();
                },
                Token::Field(Field::Fraction { digits, dot }) => {
                    if *dot {
                        if bytes.get(pos) != Some(&b'.') {
                            return None;
                        }
                        pos += 1;
                    }
                    let start = pos;
                    let max = digits.unwrap_or(9);
                    while pos < bytes.len() && bytes[pos].is_ascii_digit() && pos - start < max {
                        pos += 1;
                    }
                    let count = pos - start;
                    if count == 0 || digits.is_some_and(|n| n != count) {
                        return None;
                    }
                    let parsed = std::str::from_utf8(&bytes[start..pos])
                        .ok()?
                        .parse::<i64>()
                        .ok()?;
                    fraction_ns = parsed.checked_mul(10i64.pow((9 - count) as u32))?;
                },
                Token::Field(field) => {
                    let idx = match field {
                        Field::Hour => 0,
                        Field::Minute => 1,
                        Field::Second => 2,
                        Field::Fraction { .. } => unreachable!(),
                    };
                    let start = pos;
                    while pos < bytes.len() && bytes[pos].is_ascii_digit() {
                        pos += 1;
                    }
                    if start == pos {
                        return None;
                    }
                    let parsed = std::str::from_utf8(&bytes[start..pos])
                        .ok()?
                        .parse::<i64>()
                        .ok()?;
                    if idx > highest && parsed >= 60 {
                        return None;
                    }
                    values[idx] = parsed;
                },
            }
        }
        if pos != bytes.len() {
            return None;
        }

        let seconds = values[0]
            .checked_mul(3600)?
            .checked_add(values[1].checked_mul(60)?)?
            .checked_add(values[2])?;
        let ns = seconds
            .checked_mul(1_000_000_000)?
            .checked_add(fraction_ns)?;
        let value = match time_unit {
            TimeUnit::Nanoseconds => ns,
            TimeUnit::Microseconds => ns / 1_000,
            TimeUnit::Milliseconds => ns / 1_000_000,
        };
        if negative {
            value.checked_neg()
        } else {
            Some(value)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn compile_error(format: &str) -> String {
        DurationFormat::compile(format).err().unwrap().to_string()
    }

    #[test]
    fn parses_unbounded_highest_component_and_sign() {
        let hours = DurationFormat::compile("%H:%M:%S").unwrap();
        assert_eq!(
            hours.parse("104:00:00", TimeUnit::Milliseconds),
            Some(374_400_000)
        );
        assert_eq!(
            hours.parse("-04:00:00", TimeUnit::Milliseconds),
            Some(-14_400_000)
        );

        let minutes = DurationFormat::compile("%M:%S").unwrap();
        assert_eq!(
            minutes.parse("80:00", TimeUnit::Milliseconds),
            Some(4_800_000)
        );
        assert_eq!(minutes.parse("23:483", TimeUnit::Milliseconds), None);
    }

    #[test]
    fn parses_fractional_seconds() {
        let fmt = DurationFormat::compile("%M:%S%.3f").unwrap();
        assert_eq!(
            fmt.parse("80:00.125", TimeUnit::Microseconds),
            Some(4_800_125_000)
        );

        let variable = DurationFormat::compile("%S.%f").unwrap();
        assert_eq!(
            variable.parse("1.123456789", TimeUnit::Nanoseconds),
            Some(1_123_456_789)
        );
        assert_eq!(variable.parse("+1.5", TimeUnit::Milliseconds), Some(1_500));
        assert_eq!(variable.parse("1.", TimeUnit::Nanoseconds), None);

        let exact = DurationFormat::compile("%S-%3f").unwrap();
        assert_eq!(
            exact.parse("1-123", TimeUnit::Nanoseconds),
            Some(1_123_000_000)
        );
        assert_eq!(exact.parse("1-12", TimeUnit::Nanoseconds), None);
    }

    #[test]
    fn parses_literals_and_rejects_invalid_values() {
        let fmt = DurationFormat::compile("elapsed %% %Hh %Mm %Ss").unwrap();
        assert_eq!(
            fmt.parse("elapsed % 2h 03m 04s", TimeUnit::Milliseconds),
            Some(7_384_000)
        );
        assert_eq!(
            fmt.parse("remaining % 2h 03m 04s", TimeUnit::Milliseconds),
            None
        );
        assert_eq!(
            fmt.parse("elapsed % 2h 60m 04s", TimeUnit::Milliseconds),
            None
        );
        assert_eq!(
            fmt.parse("elapsed % 2h xm 04s", TimeUnit::Milliseconds),
            None
        );
        assert_eq!(
            fmt.parse("elapsed % 2h 03m 04s trailing", TimeUnit::Milliseconds),
            None
        );
        assert_eq!(fmt.parse("+", TimeUnit::Milliseconds), None);
    }

    #[test]
    fn rejects_invalid_formats() {
        assert!(compile_error("duration").contains("must contain"));
        assert!(compile_error("%H:%H").contains("duplicate field"));
        assert!(compile_error("%Q").contains("unsupported"));
        assert!(compile_error("%.").contains("invalid duration format directive"));
        assert!(compile_error("%0f").contains("unsupported"));
        assert!(compile_error("%10f").contains("unsupported"));
        assert!(compile_error("%").contains("trailing"));
        assert!(
            compile_error("%999999999999999999999999999999999999999f")
                .contains("invalid duration precision")
        );
    }

    #[test]
    fn rejects_overflow() {
        let fmt = DurationFormat::compile("%H:%M:%S").unwrap();
        assert_eq!(
            fmt.parse("999999999999999999999:00:00", TimeUnit::Nanoseconds),
            None
        );
        assert_eq!(fmt.parse("2562048:00:00", TimeUnit::Nanoseconds), None);
    }
}
