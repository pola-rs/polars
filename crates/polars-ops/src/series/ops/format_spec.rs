use std::fmt::Write;

use polars_core::prelude::*;
use polars_defs::format_spec::{Align, FormatKind, FormatSpec};
use polars_error::{PolarsResult, polars_bail};

fn write_str(spec: &FormatSpec, out: &mut String, s: &str) {
    let s = match spec.precision {
        Some(p) => s.char_indices().nth(p).map_or(s, |(i, _)| &s[..i]),
        None => s,
    };
    let pad = if spec.width > 0 {
        spec.width.saturating_sub(s.chars().count())
    } else {
        0
    };
    let (left, right) = match spec.align.unwrap_or(Align::Left) {
        Align::Right => (pad, 0),
        Align::Center => (pad / 2, pad - pad / 2),
        Align::Left | Align::AfterSign => (0, pad),
    };
    push_fill(out, spec.fill(), left);
    out.push_str(s);
    push_fill(out, spec.fill(), right);
}

/// Writes a number given as its sign, a prefix such as `0x`, the integer digits and the rest.
fn write_number(
    spec: &FormatSpec,
    out: &mut String,
    negative: bool,
    prefix: &str,
    (digits, rest): (&str, &str),
    group: usize,
) {
    let sign = match (negative, spec.sign) {
        (true, _) => "-",
        (false, Some(b'+')) => "+",
        (false, Some(b' ')) => " ",
        _ => "",
    };
    let int_len = digits.len();
    let rest_len = rest.chars().count();
    let fill = spec.fill();
    let align = spec.align.unwrap_or(if spec.zero {
        Align::AfterSign
    } else {
        Align::Right
    });

    let separator = spec.grouping.filter(|_| int_len > 0).map(char::from);
    let mut n_digits = int_len;
    let mut digits_len = int_len;
    if separator.is_some() {
        // Zero padding is grouped as well.
        let min_len = if fill == '0' && align == Align::AfterSign {
            spec.width
                .saturating_sub(sign.len() + prefix.len() + rest_len)
        } else {
            0
        };
        n_digits = n_digits.max(min_len - min_len / (group + 1));
        digits_len = n_digits + (n_digits - 1) / group;
        while digits_len < min_len {
            n_digits += 1;
            digits_len = n_digits + (n_digits - 1) / group;
        }
    }

    let pad = spec
        .width
        .saturating_sub(sign.len() + prefix.len() + digits_len + rest_len);
    let (left, right) = match align {
        Align::Left => (0, pad),
        Align::Right => (pad, 0),
        Align::Center => (pad / 2, pad - pad / 2),
        Align::AfterSign => (0, 0),
    };
    push_fill(out, fill, left);
    out.push_str(sign);
    out.push_str(prefix);
    if align == Align::AfterSign {
        push_fill(out, fill, pad);
    }
    match separator {
        Some(separator) => {
            let n_zeros = n_digits - int_len;
            let padded = std::iter::repeat_n('0', n_zeros).chain(digits.chars());
            for (i, digit) in padded.enumerate() {
                if i > 0 && (n_digits - i).is_multiple_of(group) {
                    out.push(separator);
                }
                out.push(digit);
            }
        },
        None => out.push_str(digits),
    }
    out.push_str(rest);
    push_fill(out, fill, right);
}

fn write_int(
    spec: &FormatSpec,
    out: &mut String,
    num: &mut String,
    negative: bool,
    magnitude: u128,
) -> PolarsResult<()> {
    num.clear();
    let prefix = |p| if spec.alternate { p } else { "" };
    // Formatting a `u64` is faster than a `u128`.
    macro_rules! write_magnitude {
        ($fmt:literal) => {
            match u64::try_from(magnitude) {
                Ok(m) => write!(num, $fmt, m),
                Err(_) => write!(num, $fmt, magnitude),
            }
            .unwrap()
        };
    }
    let (prefix, group) = match spec.ty {
        None | Some(b'd') => {
            write_magnitude!("{}");
            ("", 3)
        },
        Some(b'b') => {
            write_magnitude!("{:b}");
            (prefix("0b"), 4)
        },
        Some(b'o') => {
            write_magnitude!("{:o}");
            (prefix("0o"), 4)
        },
        Some(b'x') => {
            write_magnitude!("{:x}");
            (prefix("0x"), 4)
        },
        Some(b'X') => {
            write_magnitude!("{:X}");
            (prefix("0X"), 4)
        },
        Some(b'c') => {
            let c = u32::try_from(magnitude)
                .ok()
                .filter(|_| !negative)
                .and_then(char::from_u32);
            let Some(c) = c else {
                let sign = if negative { "-" } else { "" };
                polars_bail!(ComputeError: "invalid Unicode code point: {sign}{magnitude}");
            };
            num.push(c);
            write_number(spec, out, false, "", ("", num), 3);
            return Ok(());
        },
        _ => {
            let v = magnitude as f64;
            write_float(spec, out, num, if negative { -v } else { v });
            return Ok(());
        },
    };
    write_number(spec, out, negative, prefix, (num, ""), group);
    Ok(())
}

fn write_float(spec: &FormatSpec, out: &mut String, num: &mut String, v: f64) {
    num.clear();
    let negative = v.is_sign_negative() && !v.is_nan();
    let v = if spec.ty == Some(b'%') {
        v.abs() * 100.0
    } else {
        v.abs()
    };
    if !v.is_finite() {
        num.push_str(if v.is_nan() { "nan" } else { "inf" });
        if spec.ty == Some(b'%') {
            num.push('%');
        }
    } else {
        let alternate = spec.alternate;
        match spec.ty {
            Some(b'f' | b'F') => write_fixed(num, v, spec.precision.unwrap_or(6), alternate, false),
            Some(b'%') => {
                write_fixed(num, v, spec.precision.unwrap_or(6), alternate, false);
                num.push('%');
            },
            Some(b'e' | b'E') => write_exp(num, v, spec.precision.unwrap_or(6), alternate, false),
            Some(b'g' | b'G') => {
                write_general(num, v, spec.precision.unwrap_or(6), alternate, false)
            },
            _ => write_general(num, v, spec.precision.unwrap_or(6), alternate, true),
        }
    }
    if matches!(spec.ty, Some(b'E' | b'F' | b'G')) {
        num.make_ascii_uppercase();
    }
    write_number(spec, out, negative, "", split_digits(num), 3);
}

/// Fixed point formatting of a decimal, rounding half to even.
#[cfg(feature = "dtype-decimal")]
fn write_decimal(
    spec: &FormatSpec,
    out: &mut String,
    num: &mut String,
    mantissa: i128,
    scale: usize,
) {
    num.clear();
    let percent = spec.ty == Some(b'%');
    let scale = scale as i64 - if percent { 2 } else { 0 };
    let precision = spec.precision.unwrap_or(scale.max(0) as usize);
    let magnitude = mantissa.unsigned_abs();
    let shift = scale - precision as i64;
    let (value, n_trailing_zeros) = if shift > 0 {
        let divisor = 10u128.pow(shift as u32);
        let (q, r) = (magnitude / divisor, magnitude % divisor);
        let half = divisor / 2;
        (q + u128::from(r > half || (r == half && q % 2 == 1)), 0)
    } else if magnitude != 0 {
        (magnitude, shift.unsigned_abs() as usize)
    } else {
        (0, 0)
    };
    if precision > 0 {
        // At least one digit before the decimal point.
        let n_digits = value.checked_ilog10().map_or(1, |d| d as usize + 1) + n_trailing_zeros;
        num.extend(std::iter::repeat_n(
            '0',
            (precision + 1).saturating_sub(n_digits),
        ));
    }
    write!(num, "{value}").unwrap();
    num.extend(std::iter::repeat_n('0', n_trailing_zeros));
    if precision > 0 {
        num.insert(num.len() - precision, '.');
    } else if spec.alternate {
        num.push('.');
    }
    if percent {
        num.push('%');
    }
    write_number(spec, out, mantissa < 0, "", split_digits(num), 3);
}

/// Splits a number without a sign into its integer digits and the rest.
fn split_digits(num: &str) -> (&str, &str) {
    num.split_at(num.bytes().take_while(u8::is_ascii_digit).count())
}

fn push_fill(out: &mut String, fill: char, n: usize) {
    out.extend(std::iter::repeat_n(fill, n));
}

fn strip_zeros(num: &mut String) {
    if num.contains('.') {
        let len = num.trim_end_matches('0').trim_end_matches('.').len();
        num.truncate(len);
    }
}

// A float has at most 1074 digits after the decimal point and 767 significant digits, so
// digits past these limits are zeros. This also stays within the precision Rust can format.
const MAX_FIXED_DIGITS: usize = 1100;
const MAX_EXP_DIGITS: usize = 800;

fn write_fixed(num: &mut String, v: f64, precision: usize, alternate: bool, strip: bool) {
    let exact = precision.min(MAX_FIXED_DIGITS);
    write!(num, "{v:.exact$}").unwrap();
    if strip {
        strip_zeros(num);
    } else {
        push_fill(num, '0', precision - exact);
        if alternate && precision == 0 {
            num.push('.');
        }
    }
}

fn write_exp(num: &mut String, v: f64, precision: usize, alternate: bool, strip: bool) {
    let exact = precision.min(MAX_EXP_DIGITS);
    write!(num, "{v:.exact$e}").unwrap();
    let e = num.find('e').unwrap();
    let exp: i32 = num[e + 1..].parse().unwrap();
    num.truncate(e);
    if strip {
        strip_zeros(num);
    } else {
        push_fill(num, '0', precision - exact);
        if alternate && precision == 0 {
            num.push('.');
        }
    }
    let sign = if exp < 0 { '-' } else { '+' };
    write!(num, "e{sign}{:02}", exp.unsigned_abs()).unwrap();
}

/// The `g` format code, or no format code with a precision when `dot_zero` is set.
fn write_general(num: &mut String, v: f64, precision: usize, alternate: bool, dot_zero: bool) {
    let p = precision.max(1);
    // The decimal exponent after rounding to `p` significant digits.
    let exp = if v == 0.0 {
        0
    } else {
        write!(num, "{v:.*e}", (p - 1).min(MAX_EXP_DIGITS)).unwrap();
        let exp: i64 = num[num.find('e').unwrap() + 1..].parse().unwrap();
        num.clear();
        exp
    };
    let threshold = p as i64 - i64::from(dot_zero);
    if exp < -4 || exp >= threshold {
        write_exp(num, v, p - 1, alternate, !alternate);
    } else {
        write_fixed(num, v, (p as i64 - 1 - exp) as usize, alternate, !alternate);
        if dot_zero && !num.contains('.') {
            num.push_str(".0");
        }
    }
}

/// Formats each value with a Python format specification, such as `>10` or `,.2f`.
pub(crate) fn format_value(c: &Column, spec: &str) -> PolarsResult<Column> {
    let spec = &FormatSpec::parse(spec)?;
    let dtype = c.dtype();
    if dtype.is_null() {
        return Ok(Column::full_null(
            c.name().clone(),
            c.len(),
            &DataType::String,
        ));
    }
    let mut num = String::new();
    let out = match spec.kind(dtype)? {
        FormatKind::Str => {
            let c = c.cast(&DataType::String)?;
            c.str()?
                .apply_into_string_amortized(|v, buf| write_str(spec, buf, v))
        },
        FormatKind::Int => {
            let c = if dtype.is_bool() {
                c.cast(&DataType::UInt64)?
            } else {
                c.clone()
            };
            match c.dtype() {
                #[cfg(feature = "dtype-i128")]
                DataType::Int128 => c.i128()?.try_apply_into_string_amortized(|v, buf| {
                    write_int(spec, buf, &mut num, v < 0, v.unsigned_abs())
                })?,
                #[cfg(feature = "dtype-u128")]
                DataType::UInt128 => c.u128()?.try_apply_into_string_amortized(|v, buf| {
                    write_int(spec, buf, &mut num, false, v)
                })?,
                dt if dt.is_signed_integer() => {
                    let c = c.cast(&DataType::Int64)?;
                    c.i64()?.try_apply_into_string_amortized(|v, buf| {
                        write_int(spec, buf, &mut num, v < 0, u128::from(v.unsigned_abs()))
                    })?
                },
                _ => {
                    let c = c.cast(&DataType::UInt64)?;
                    c.u64()?.try_apply_into_string_amortized(|v, buf| {
                        write_int(spec, buf, &mut num, false, u128::from(v))
                    })?
                },
            }
        },
        // Without a format code or precision, floats and decimals are written as in a cast to
        // a string.
        FormatKind::Float if spec.ty.is_none() && spec.precision.is_none() => {
            let c = c.cast(&DataType::String)?;
            if spec.width == 0 && spec.sign.is_none() && spec.grouping.is_none() && !spec.alternate
            {
                return Ok(c);
            }
            c.str()?.apply_into_string_amortized(|v, buf| {
                let (negative, v) = v.strip_prefix('-').map_or((false, v), |v| (true, v));
                let (digits, rest) = split_digits(v);
                if spec.alternate && !digits.is_empty() && !rest.contains('.') {
                    // The alternate form always has a decimal point.
                    num.clear();
                    num.push('.');
                    num.push_str(rest);
                    write_number(spec, buf, negative, "", (digits, &num), 3)
                } else {
                    write_number(spec, buf, negative, "", (digits, rest), 3)
                }
            })
        },
        #[cfg(feature = "dtype-decimal")]
        FormatKind::Float if dtype.is_decimal() && matches!(spec.ty, Some(b'f' | b'F' | b'%')) => {
            let ca = c.decimal()?;
            let scale = ca.scale();
            ca.physical()
                .apply_into_string_amortized(|v, buf| write_decimal(spec, buf, &mut num, v, scale))
        },
        FormatKind::Float => {
            let c = c.cast(&DataType::Float64)?;
            c.f64()?
                .apply_into_string_amortized(|v, buf| write_float(spec, buf, &mut num, v))
        },
    };
    Ok(out.with_name(c.name().clone()).into_column())
}
