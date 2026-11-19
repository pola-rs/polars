use polars_core::prelude::DataType;
use polars_error::{PolarsResult, polars_bail, polars_ensure, polars_err};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Align {
    Left,
    Right,
    Center,
    /// Padding goes between the sign and the digits.
    AfterSign,
}

/// How values of a data type are formatted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FormatKind {
    Str,
    Int,
    Float,
}

/// A Python format specification:
/// `[[fill]align][sign][#][0][width][grouping][.precision][type]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FormatSpec {
    pub fill: Option<char>,
    pub align: Option<Align>,
    pub sign: Option<u8>,
    pub alternate: bool,
    pub zero: bool,
    pub width: usize,
    pub grouping: Option<u8>,
    pub precision: Option<usize>,
    pub ty: Option<u8>,
}

impl FormatSpec {
    pub fn parse(spec: &str) -> PolarsResult<Self> {
        let align_of = |c: char| match c {
            '<' => Some(Align::Left),
            '>' => Some(Align::Right),
            '^' => Some(Align::Center),
            '=' => Some(Align::AfterSign),
            _ => None,
        };
        let mut fill = None;
        let mut align = None;
        let mut rest = spec;
        let mut chars = spec.chars();
        if let Some(c) = chars.next() {
            if let Some(a) = chars.next().and_then(align_of) {
                fill = Some(c);
                align = Some(a);
                rest = chars.as_str();
            } else if let Some(a) = align_of(c) {
                align = Some(a);
                rest = &spec[1..];
            }
        }

        let bytes = rest.as_bytes();
        let mut i = 0;
        let sign = next_if(bytes, &mut i, |b| matches!(b, b'+' | b'-' | b' '));
        let alternate = next_if(bytes, &mut i, |b| b == b'#').is_some();
        let zero = next_if(bytes, &mut i, |b| b == b'0').is_some();
        let width = parse_number(spec, bytes, &mut i)?.unwrap_or(0);
        let grouping = next_if(bytes, &mut i, |b| matches!(b, b',' | b'_'));
        let precision = if next_if(bytes, &mut i, |b| b == b'.').is_some() {
            Some(parse_number(spec, bytes, &mut i)?.ok_or_else(
                || polars_err!(InvalidOperation: "missing precision in format specifier '{spec}'"),
            )?)
        } else {
            None
        };
        let ty = next_if(bytes, &mut i, |b| b"sbcdoxXeEfFgG%".contains(&b));
        polars_ensure!(
            i == bytes.len(),
            InvalidOperation: "invalid format specifier '{spec}'"
        );

        Ok(Self {
            fill,
            align,
            sign,
            alternate,
            zero,
            width,
            grouping,
            precision,
            ty,
        })
    }

    /// Checks that values of `dtype` can be formatted with this specification.
    pub fn check(&self, dtype: &DataType) -> PolarsResult<()> {
        if !matches!(dtype, DataType::Null | DataType::Unknown(_)) {
            self.kind(dtype)?;
        }
        Ok(())
    }

    pub fn kind(&self, dtype: &DataType) -> PolarsResult<FormatKind> {
        let ty = self.ty.map(char::from);
        // Unlike in Python, `s` writes any value as its string.
        let kind = match dtype {
            _ if ty == Some('s') => FormatKind::Str,
            dt if dt.is_integer() => FormatKind::Int,
            DataType::Boolean if ty.is_some() => FormatKind::Int,
            dt if dt.is_float() || dt.is_decimal() => FormatKind::Float,
            _ => FormatKind::Str,
        };
        match kind {
            FormatKind::Str => {
                if let Some(t) = ty.filter(|t| *t != 's') {
                    polars_bail!(InvalidOperation: "format code '{t}' is not supported for {dtype}");
                }
                let options = [
                    (self.sign.is_some(), "a sign"),
                    (self.alternate, "'#'"),
                    (self.grouping.is_some(), "a thousands separator"),
                    (self.align == Some(Align::AfterSign), "'=' alignment"),
                ];
                if let Some((_, option)) = options.iter().find(|(used, _)| *used) {
                    polars_bail!(InvalidOperation: "{option} is not allowed in a format specifier for {dtype}");
                }
            },
            FormatKind::Int if ty.is_none_or(|t| "dbcoxX".contains(t)) => {
                polars_ensure!(
                    self.precision.is_none(),
                    InvalidOperation: "a precision is not allowed in a format specifier for {dtype}"
                );
                if self.grouping == Some(b',') && ty.is_some_and(|t| t != 'd') {
                    polars_bail!(InvalidOperation: "',' is not allowed with format code '{}'", ty.unwrap());
                }
                if ty == Some('c') {
                    polars_ensure!(
                        self.sign.is_none() && !self.alternate && self.grouping.is_none(),
                        InvalidOperation: "format code 'c' does not allow a sign, '#' or a thousands separator"
                    );
                }
            },
            FormatKind::Int | FormatKind::Float => {
                if let Some(t) = ty.filter(|t| !"eEfFgG%".contains(*t)) {
                    polars_bail!(InvalidOperation: "format code '{t}' is not supported for {dtype}");
                }
            },
        }
        Ok(kind)
    }

    /// The fill character used for padding.
    pub fn fill(&self) -> char {
        self.fill.unwrap_or(if self.zero { '0' } else { ' ' })
    }
}

fn parse_number(spec: &str, bytes: &[u8], i: &mut usize) -> PolarsResult<Option<usize>> {
    let start = *i;
    while bytes.get(*i).is_some_and(u8::is_ascii_digit) {
        *i += 1;
    }
    if start == *i {
        return Ok(None);
    }
    let digits = std::str::from_utf8(&bytes[start..*i]).unwrap();
    // Longer strings can't be stored.
    let n = digits.parse::<i32>().map_err(
        |_| polars_err!(InvalidOperation: "number too large in format specifier '{spec}'"),
    )?;
    Ok(Some(n as usize))
}

fn next_if(bytes: &[u8], i: &mut usize, f: fn(u8) -> bool) -> Option<u8> {
    let b = bytes.get(*i).copied().filter(|b| f(*b))?;
    *i += 1;
    Some(b)
}
