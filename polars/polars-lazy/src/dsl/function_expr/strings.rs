use std::borrow::Cow;

use polars_arrow::utils::CustomIterTools;
#[cfg(feature = "regex")]
use regex::{escape, Regex};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::*;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, PartialEq, Debug, Eq, Hash)]
pub enum StringFunction {
    Contains {
        pat: String,
        literal: bool,
    },
    StartsWith(String),
    EndsWith(String),
    Extract {
        pat: String,
        group_index: usize,
    },
    #[cfg(feature = "string_justify")]
    Zfill(usize),
    #[cfg(feature = "string_justify")]
    LJust {
        width: usize,
        fillchar: char,
    },
    #[cfg(feature = "string_justify")]
    RJust {
        width: usize,
        fillchar: char,
    },
    ExtractAll(String),
    CountMatch(String),
    #[cfg(feature = "temporal")]
    Strptime(StrpTimeOptions),
    #[cfg(feature = "concat_str")]
    Concat(String),
    #[cfg(feature = "regex")]
    Replace {
        // replace_single or replace_all
        all: bool,
        literal: bool,
    },
    Uppercase,
    Lowercase,
}

impl Display for StringFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use self::*;
        match self {
            StringFunction::Contains { .. } => write!(f, "str.contains"),
            StringFunction::StartsWith(_) => write!(f, "str.starts_with"),
            StringFunction::EndsWith(_) => write!(f, "str.ends_with"),
            StringFunction::Extract { .. } => write!(f, "str.extract"),
            #[cfg(feature = "string_justify")]
            StringFunction::Zfill(_) => write!(f, "str.zfill"),
            #[cfg(feature = "string_justify")]
            StringFunction::LJust { .. } => write!(f, "str.ljust"),
            #[cfg(feature = "string_justify")]
            StringFunction::RJust { .. } => write!(f, "str.rjust"),
            StringFunction::ExtractAll(_) => write!(f, "str.extract_all"),
            StringFunction::CountMatch(_) => write!(f, "str.count_match"),
            #[cfg(feature = "temporal")]
            StringFunction::Strptime(_) => write!(f, "str.strptime"),
            #[cfg(feature = "concat_str")]
            StringFunction::Concat(_) => write!(f, "str.concat"),
            #[cfg(feature = "regex")]
            StringFunction::Replace { .. } => write!(f, "str.replace"),
            StringFunction::Uppercase => write!(f, "str.uppercase"),
            StringFunction::Lowercase => write!(f, "str.lowercase"),
        }
    }
}

pub(super) fn uppercase(s: &Series) -> Result<Series> {
    let ca = s.utf8()?;
    Ok(ca.to_uppercase().into_series())
}

pub(super) fn lowercase(s: &Series) -> Result<Series> {
    let ca = s.utf8()?;
    Ok(ca.to_lowercase().into_series())
}

pub(super) fn contains(s: &Series, pat: &str, literal: bool) -> Result<Series> {
    let ca = s.utf8()?;
    if literal {
        ca.contains_literal(pat).map(|ca| ca.into_series())
    } else {
        ca.contains(pat).map(|ca| ca.into_series())
    }
}

pub(super) fn ends_with(s: &Series, sub: &str) -> Result<Series> {
    let ca = s.utf8()?;
    Ok(ca.ends_with(sub).into_series())
}
pub(super) fn starts_with(s: &Series, sub: &str) -> Result<Series> {
    let ca = s.utf8()?;
    Ok(ca.starts_with(sub).into_series())
}

/// Extract a regex pattern from the a string value.
pub(super) fn extract(s: &Series, pat: &str, group_index: usize) -> Result<Series> {
    let pat = pat.to_string();

    let ca = s.utf8()?;
    ca.extract(&pat, group_index).map(|ca| ca.into_series())
}

#[cfg(feature = "string_justify")]
pub(super) fn zfill(s: &Series, alignment: usize) -> Result<Series> {
    let ca = s.utf8()?;
    Ok(ca.zfill(alignment).into_series())
}

#[cfg(feature = "string_justify")]
pub(super) fn ljust(s: &Series, width: usize, fillchar: char) -> Result<Series> {
    let ca = s.utf8()?;
    Ok(ca.ljust(width, fillchar).into_series())
}
#[cfg(feature = "string_justify")]
pub(super) fn rjust(s: &Series, width: usize, fillchar: char) -> Result<Series> {
    let ca = s.utf8()?;
    Ok(ca.rjust(width, fillchar).into_series())
}

pub(super) fn extract_all(s: &Series, pat: &str) -> Result<Series> {
    let pat = pat.to_string();

    let ca = s.utf8()?;
    ca.extract_all(&pat).map(|ca| ca.into_series())
}

pub(super) fn count_match(s: &Series, pat: &str) -> Result<Series> {
    let pat = pat.to_string();

    let ca = s.utf8()?;
    ca.count_match(&pat).map(|ca| ca.into_series())
}

#[cfg(feature = "temporal")]
pub(super) fn strptime(s: &Series, options: &StrpTimeOptions) -> Result<Series> {
    let ca = s.utf8()?;

    let out = match &options.date_dtype {
        DataType::Date => {
            if options.exact {
                ca.as_date(options.fmt.as_deref())?.into_series()
            } else {
                ca.as_date_not_exact(options.fmt.as_deref())?.into_series()
            }
        }
        DataType::Datetime(tu, _) => {
            if options.exact {
                ca.as_datetime(options.fmt.as_deref(), *tu)?.into_series()
            } else {
                ca.as_datetime_not_exact(options.fmt.as_deref(), *tu)?
                    .into_series()
            }
        }
        DataType::Time => {
            if options.exact {
                ca.as_time(options.fmt.as_deref())?.into_series()
            } else {
                return Err(PolarsError::ComputeError(
                    format!("non-exact not implemented for dtype {:?}", DataType::Time).into(),
                ));
            }
        }
        dt => {
            return Err(PolarsError::ComputeError(
                format!("not implemented for dtype {:?}", dt).into(),
            ))
        }
    };
    if options.strict {
        if out.null_count() != ca.null_count() {
            Err(PolarsError::ComputeError(
                "strict conversion to dates failed, maybe set strict=False".into(),
            ))
        } else {
            Ok(out.into_series())
        }
    } else {
        Ok(out.into_series())
    }
}

#[cfg(feature = "concat_str")]
pub(super) fn concat(s: &Series, delimiter: &str) -> Result<Series> {
    Ok(s.str_concat(delimiter).into_series())
}

impl From<StringFunction> for FunctionExpr {
    fn from(str: StringFunction) -> Self {
        FunctionExpr::StringExpr(str)
    }
}

#[cfg(feature = "regex")]
fn get_pat(pat: &Utf8Chunked) -> Result<&str> {
    pat.get(0).ok_or_else(|| {
        PolarsError::ComputeError("pattern may not be 'null' in 'replace' expression".into())
    })
}

fn iter_and_replace<'a, F>(ca: &'a Utf8Chunked, val: &'a Utf8Chunked, f: F) -> Utf8Chunked
where
    F: Fn(&'a str, &'a str) -> Cow<'a, str>,
{
    let mut out: Utf8Chunked = ca
        .into_iter()
        .zip(val.into_iter())
        .map(|(opt_src, opt_val)| match (opt_src, opt_val) {
            (Some(src), Some(val)) => Some(f(src, val)),
            _ => None,
        })
        .collect_trusted();

    out.rename(ca.name());
    out
}

#[cfg(feature = "regex")]
fn replace_single<'a>(
    ca: &'a Utf8Chunked,
    pat: &'a Utf8Chunked,
    val: &'a Utf8Chunked,
    literal: bool,
) -> Result<Utf8Chunked> {
    match (pat.len(), val.len()) {
        (1, 1) => {
            let pat = get_pat(pat)?;
            let val = val.get(0).ok_or_else(|| PolarsError::ComputeError("value may not be 'null' in 'replace' expression".into()))?;

            match literal {
                true => ca.replace_literal(pat, val),
                false => ca.replace(pat, val),
            }
        }
        (1, len_val) => {
            let mut pat = get_pat(pat)?.to_string();
            if len_val != ca.len() {
                return Err(PolarsError::ComputeError(format!("The replacement value expression in 'str.replace' should be equal to the length of the string column.\
                Got column length: {} and replacement value length: {}", ca.len(), len_val).into()))
            }

            if literal {
                pat = escape(&pat)
            }


            let reg = Regex::new(&pat)?;
            let lit = pat.chars().all(|c| !c.is_ascii_punctuation());

            let f = |s: &'a str, val: &'a str| {
                if lit && (s.len() <= 32) {
                    Cow::Owned(s.replacen(&pat, val, 1))
                } else {
                    reg.replace(s, val)
                }
            };
            Ok(iter_and_replace(ca, val, f))
        }
        _ => Err(PolarsError::ComputeError("A dynamic pattern length in the 'str.replace' expressions are not yet supported. Consider open a feature request for this.".into()))
    }
}

#[cfg(feature = "regex")]
fn replace_all<'a>(
    ca: &'a Utf8Chunked,
    pat: &'a Utf8Chunked,
    val: &'a Utf8Chunked,
    literal: bool,
) -> Result<Utf8Chunked> {
    match (pat.len(), val.len()) {
        (1, 1) => {
            let pat = get_pat(pat)?;
            let val = val.get(0).ok_or_else(|| PolarsError::ComputeError("value may not be 'null' in 'replace' expression".into()))?;

            match literal {
                true => ca.replace_literal_all(pat, val),
                false => ca.replace_all(pat, val),
            }
        }
        (1, len_val) => {
            let mut pat = get_pat(pat)?.to_string();
            if len_val != ca.len() {
                return Err(PolarsError::ComputeError(format!("The replacement value expression in 'str.replace' should be equal to the length of the string column.\
                Got column length: {} and replacement value length: {}", ca.len(), len_val).into()))
            }

            if literal {
                pat = escape(&pat)
            }

            let reg = Regex::new(&pat)?;

            let f = |s: &'a str, val: &'a str| {
                reg.replace_all(s, val)
            };
            Ok(iter_and_replace(ca, val, f))
        }
        _ => Err(PolarsError::ComputeError("A dynamic pattern length in the 'str.replace' expressions are not yet supported. Consider open a feature request for this.".into()))
    }
}

#[cfg(feature = "regex")]
pub(super) fn replace(s: &[Series], literal: bool, all: bool) -> Result<Series> {
    let column = &s[0];
    let pat = &s[1];
    let val = &s[2];

    let column = column.utf8()?;
    let pat = pat.utf8()?;
    let val = val.utf8()?;

    if all {
        replace_all(column, pat, val, literal)
    } else {
        replace_single(column, pat, val, literal)
    }
    .map(|ca| ca.into_series())
}
