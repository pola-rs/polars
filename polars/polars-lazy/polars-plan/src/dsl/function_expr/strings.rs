use std::borrow::Cow;

#[cfg(feature = "timezones")]
use once_cell::sync::Lazy;
use polars_arrow::utils::CustomIterTools;
#[cfg(feature = "regex")]
use regex::{escape, Regex};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "timezones")]
static TZ_AWARE_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(%z)|(%:z)|(%#z)|(^%\+$)").unwrap());

use super::*;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, PartialEq, Debug, Eq, Hash)]
pub enum StringFunction {
    #[cfg(feature = "regex")]
    Contains {
        literal: bool,
        strict: bool,
    },
    StartsWith,
    EndsWith,
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
    ExtractAll,
    CountMatch(String),
    #[cfg(feature = "temporal")]
    Strptime(StrpTimeOptions),
    #[cfg(feature = "concat_str")]
    ConcatVertical(String),
    #[cfg(feature = "concat_str")]
    ConcatHorizontal(String),
    #[cfg(feature = "regex")]
    Replace {
        // replace_single or replace_all
        all: bool,
        literal: bool,
    },
    Uppercase,
    Lowercase,
    Strip(Option<String>),
    RStrip(Option<String>),
    LStrip(Option<String>),
    #[cfg(feature = "string_from_radix")]
    FromRadix(Option<u32>),
}

impl Display for StringFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use self::*;
        let s = match self {
            #[cfg(feature = "regex")]
            StringFunction::Contains { .. } => "contains",
            StringFunction::StartsWith { .. } => "starts_with",
            StringFunction::EndsWith { .. } => "ends_with",
            StringFunction::Extract { .. } => "extract",
            #[cfg(feature = "string_justify")]
            StringFunction::Zfill(_) => "zfill",
            #[cfg(feature = "string_justify")]
            StringFunction::LJust { .. } => "str.ljust",
            #[cfg(feature = "string_justify")]
            StringFunction::RJust { .. } => "rjust",
            StringFunction::ExtractAll => "extract_all",
            StringFunction::CountMatch(_) => "count_match",
            #[cfg(feature = "temporal")]
            StringFunction::Strptime(_) => "strptime",
            #[cfg(feature = "concat_str")]
            StringFunction::ConcatVertical(_) => "concat_vertical",
            #[cfg(feature = "concat_str")]
            StringFunction::ConcatHorizontal(_) => "concat_horizontal",
            #[cfg(feature = "regex")]
            StringFunction::Replace { .. } => "replace",
            StringFunction::Uppercase => "uppercase",
            StringFunction::Lowercase => "lowercase",
            StringFunction::Strip(_) => "strip",
            StringFunction::LStrip(_) => "lstrip",
            StringFunction::RStrip(_) => "rstrip",
            #[cfg(feature = "string_from_radix")]
            StringFunction::FromRadix { .. } => "from_radix",
        };

        write!(f, "str.{s}")
    }
}

pub(super) fn uppercase(s: &Series) -> PolarsResult<Series> {
    let ca = s.utf8()?;
    Ok(ca.to_uppercase().into_series())
}

pub(super) fn lowercase(s: &Series) -> PolarsResult<Series> {
    let ca = s.utf8()?;
    Ok(ca.to_lowercase().into_series())
}

#[cfg(feature = "regex")]
pub(super) fn contains(s: &[Series], literal: bool, strict: bool) -> PolarsResult<Series> {
    let ca = &s[0].utf8()?;
    let pat = &s[1].utf8()?;

    let mut out: BooleanChunked = match pat.len() {
        1 => match pat.get(0) {
            Some(pat) => {
                if literal {
                    ca.contains_literal(pat)?
                } else {
                    ca.contains(pat)?
                }
            }
            None => BooleanChunked::full(ca.name(), false, ca.len()),
        },
        _ => {
            if literal {
                ca.into_iter()
                    .zip(pat.into_iter())
                    .map(|(opt_src, opt_val)| match (opt_src, opt_val) {
                        (Some(src), Some(pat)) => src.contains(pat),
                        _ => false,
                    })
                    .collect_trusted()
            } else if strict {
                ca.into_iter()
                    .zip(pat.into_iter())
                    .map(|(opt_src, opt_val)| match (opt_src, opt_val) {
                        (Some(src), Some(pat)) => {
                            let re = Regex::new(pat)?;
                            Ok(re.is_match(src))
                        }
                        _ => Ok(false),
                    })
                    .collect::<PolarsResult<_>>()?
            } else {
                ca.into_iter()
                    .zip(pat.into_iter())
                    .map(|(opt_src, opt_val)| match (opt_src, opt_val) {
                        (Some(src), Some(pat)) => {
                            let re = Regex::new(pat).ok()?;
                            Some(re.is_match(src))
                        }
                        _ => Some(false),
                    })
                    .collect_trusted()
            }
        }
    };

    out.rename(ca.name());
    Ok(out.into_series())
}

pub(super) fn ends_with(s: &[Series]) -> PolarsResult<Series> {
    let ca = &s[0].utf8()?;
    let sub = &s[1].utf8()?;

    let mut out: BooleanChunked = match sub.len() {
        1 => match sub.get(0) {
            Some(s) => ca.ends_with(s),
            None => BooleanChunked::full(ca.name(), false, ca.len()),
        },
        _ => ca
            .into_iter()
            .zip(sub.into_iter())
            .map(|(opt_src, opt_val)| match (opt_src, opt_val) {
                (Some(src), Some(val)) => src.ends_with(val),
                _ => false,
            })
            .collect_trusted(),
    };

    out.rename(ca.name());
    Ok(out.into_series())
}

pub(super) fn starts_with(s: &[Series]) -> PolarsResult<Series> {
    let ca = &s[0].utf8()?;
    let sub = &s[1].utf8()?;

    let mut out: BooleanChunked = match sub.len() {
        1 => match sub.get(0) {
            Some(s) => ca.starts_with(s),
            None => BooleanChunked::full(ca.name(), false, ca.len()),
        },
        _ => ca
            .into_iter()
            .zip(sub.into_iter())
            .map(|(opt_src, opt_val)| match (opt_src, opt_val) {
                (Some(src), Some(val)) => src.starts_with(val),
                _ => false,
            })
            .collect_trusted(),
    };

    out.rename(ca.name());
    Ok(out.into_series())
}

/// Extract a regex pattern from the a string value.
pub(super) fn extract(s: &Series, pat: &str, group_index: usize) -> PolarsResult<Series> {
    let pat = pat.to_string();

    let ca = s.utf8()?;
    ca.extract(&pat, group_index).map(|ca| ca.into_series())
}

#[cfg(feature = "string_justify")]
pub(super) fn zfill(s: &Series, alignment: usize) -> PolarsResult<Series> {
    let ca = s.utf8()?;
    Ok(ca.zfill(alignment).into_series())
}

#[cfg(feature = "string_justify")]
pub(super) fn ljust(s: &Series, width: usize, fillchar: char) -> PolarsResult<Series> {
    let ca = s.utf8()?;
    Ok(ca.ljust(width, fillchar).into_series())
}
#[cfg(feature = "string_justify")]
pub(super) fn rjust(s: &Series, width: usize, fillchar: char) -> PolarsResult<Series> {
    let ca = s.utf8()?;
    Ok(ca.rjust(width, fillchar).into_series())
}

pub(super) fn strip(s: &Series, matches: Option<&str>) -> PolarsResult<Series> {
    let ca = s.utf8()?;
    if let Some(matches) = matches {
        if matches.chars().count() == 1 {
            // Fast path for when a single character is passed
            Ok(ca
                .apply(|s| Cow::Borrowed(s.trim_matches(matches.chars().next().unwrap())))
                .into_series())
        } else {
            Ok(ca
                .apply(|s| Cow::Borrowed(s.trim_matches(|c| matches.contains(c))))
                .into_series())
        }
    } else {
        Ok(ca.apply(|s| Cow::Borrowed(s.trim())).into_series())
    }
}

pub(super) fn lstrip(s: &Series, matches: Option<&str>) -> PolarsResult<Series> {
    let ca = s.utf8()?;

    if let Some(matches) = matches {
        if matches.chars().count() == 1 {
            // Fast path for when a single character is passed
            Ok(ca
                .apply(|s| Cow::Borrowed(s.trim_start_matches(matches.chars().next().unwrap())))
                .into_series())
        } else {
            Ok(ca
                .apply(|s| Cow::Borrowed(s.trim_start_matches(|c| matches.contains(c))))
                .into_series())
        }
    } else {
        Ok(ca.apply(|s| Cow::Borrowed(s.trim_start())).into_series())
    }
}

pub(super) fn rstrip(s: &Series, matches: Option<&str>) -> PolarsResult<Series> {
    let ca = s.utf8()?;
    if let Some(matches) = matches {
        if matches.chars().count() == 1 {
            // Fast path for when a single character is passed
            Ok(ca
                .apply(|s| Cow::Borrowed(s.trim_end_matches(matches.chars().next().unwrap())))
                .into_series())
        } else {
            Ok(ca
                .apply(|s| Cow::Borrowed(s.trim_end_matches(|c| matches.contains(c))))
                .into_series())
        }
    } else {
        Ok(ca.apply(|s| Cow::Borrowed(s.trim_end())).into_series())
    }
}

pub(super) fn extract_all(args: &[Series]) -> PolarsResult<Series> {
    let s = &args[0];
    let pat = &args[1];

    let ca = s.utf8()?;
    let pat = pat.utf8()?;

    if pat.len() == 1 {
        let pat = pat
            .get(0)
            .ok_or_else(|| PolarsError::ComputeError("Expected a pattern got null".into()))?;
        ca.extract_all(pat).map(|ca| ca.into_series())
    } else {
        ca.extract_all_many(pat).map(|ca| ca.into_series())
    }
}

pub(super) fn count_match(s: &Series, pat: &str) -> PolarsResult<Series> {
    let pat = pat.to_string();

    let ca = s.utf8()?;
    ca.count_match(&pat).map(|ca| ca.into_series())
}

#[cfg(feature = "temporal")]
pub(super) fn strptime(s: &Series, options: &StrpTimeOptions) -> PolarsResult<Series> {
    let tz_aware = match (options.tz_aware, &options.fmt) {
        (true, Some(_)) => true,
        (true, None) => {
            return Err(PolarsError::ComputeError(
                "Passing 'tz_aware=True' without 'fmt' is not yet supported. Please specify 'fmt'."
                    .into(),
            ));
        }
        #[cfg(feature = "timezones")]
        (false, Some(fmt)) => TZ_AWARE_RE.is_match(fmt),
        (false, _) => false,
    };
    #[cfg(feature = "timezones")]
    if !tz_aware && options.utc {
        return Err(PolarsError::ComputeError(
            "Cannot use 'utc=True' with tz-naive data. Parse the data as naive, and then use `.dt.convert_time_zone('UTC')`.".into(),
        ));
    }
    let ca = s.utf8()?;

    let out = match &options.date_dtype {
        DataType::Date => {
            if options.exact {
                ca.as_date(options.fmt.as_deref(), options.cache)?
                    .into_series()
            } else {
                ca.as_date_not_exact(options.fmt.as_deref())?.into_series()
            }
        }
        DataType::Datetime(tu, tz) => {
            if tz.is_some() && tz_aware {
                return Err(PolarsError::ComputeError(
                    "Cannot use strptime with both 'tz_aware=True' and tz-aware Datetime.".into(),
                ));
            }
            if options.exact {
                ca.as_datetime(
                    options.fmt.as_deref(),
                    *tu,
                    options.cache,
                    tz_aware,
                    options.utc,
                    tz.as_ref(),
                )?
                .into_series()
            } else {
                ca.as_datetime_not_exact(options.fmt.as_deref(), *tu, tz.as_ref())?
                    .into_series()
            }
        }
        DataType::Time => {
            if options.exact {
                ca.as_time(options.fmt.as_deref(), options.cache)?
                    .into_series()
            } else {
                return Err(PolarsError::ComputeError(
                    format!("non-exact not implemented for dtype {:?}", DataType::Time).into(),
                ));
            }
        }
        dt => {
            return Err(PolarsError::ComputeError(
                format!("not implemented for dtype {dt:?}").into(),
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
pub(super) fn concat(s: &Series, delimiter: &str) -> PolarsResult<Series> {
    Ok(s.str_concat(delimiter).into_series())
}

#[cfg(feature = "concat_str")]
pub(super) fn concat_hor(s: &[Series], delimiter: &str) -> PolarsResult<Series> {
    polars_core::functions::concat_str(s, delimiter).map(|ca| ca.into_series())
}

impl From<StringFunction> for FunctionExpr {
    fn from(str: StringFunction) -> Self {
        FunctionExpr::StringExpr(str)
    }
}

#[cfg(feature = "regex")]
fn get_pat(pat: &Utf8Chunked) -> PolarsResult<&str> {
    pat.get(0).ok_or_else(|| {
        PolarsError::ComputeError("pattern may not be 'null' in 'replace' expression".into())
    })
}

// used only if feature="regex"
#[allow(dead_code)]
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
) -> PolarsResult<Utf8Chunked> {
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
) -> PolarsResult<Utf8Chunked> {
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
pub(super) fn replace(s: &[Series], literal: bool, all: bool) -> PolarsResult<Series> {
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

#[cfg(feature = "string_from_radix")]
pub(super) fn from_radix(s: &Series, radix: Option<u32>) -> PolarsResult<Series> {
    let ca = s.utf8()?;
    Ok(ca.parse_int(radix).into_series())
}
