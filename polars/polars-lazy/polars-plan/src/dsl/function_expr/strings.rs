use std::borrow::Cow;

#[cfg(feature = "timezones")]
use once_cell::sync::Lazy;
use polars_arrow::utils::CustomIterTools;
#[cfg(feature = "regex")]
use regex::{escape, Regex};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "timezones")]
static TZ_AWARE_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(%z)|(%:z)|(%::z)|(%:::z)|(%#z)|(^%\+$)").unwrap());

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
    Strptime(DataType, StrptimeOptions),
    #[cfg(feature = "concat_str")]
    ConcatVertical(String),
    #[cfg(feature = "concat_str")]
    ConcatHorizontal(String),
    #[cfg(feature = "regex")]
    Replace {
        // negative is replace all
        // how many matches to replace
        n: i64,
        literal: bool,
    },
    Uppercase,
    Lowercase,
    #[cfg(feature = "nightly")]
    Titlecase,
    Strip(Option<String>),
    RStrip(Option<String>),
    LStrip(Option<String>),
    #[cfg(feature = "string_from_radix")]
    FromRadix(u32, bool),
    Slice(i64, Option<u64>),
    Explode,
    #[cfg(feature = "dtype-decimal")]
    ToDecimal(usize),
    #[cfg(feature = "extract_jsonpath")]
    JsonExtract {
        dtype: Option<DataType>,
        infer_schema_len: Option<usize>,
    },
}

impl StringFunction {
    pub(super) fn get_field(&self, mapper: FieldsMapper) -> PolarsResult<Field> {
        use StringFunction::*;
        match self {
            #[cfg(feature = "regex")]
            Contains { .. } => mapper.with_dtype(DataType::Boolean),
            EndsWith | StartsWith => mapper.with_dtype(DataType::Boolean),
            Extract { .. } => mapper.with_same_dtype(),
            ExtractAll => mapper.with_dtype(DataType::List(Box::new(DataType::Utf8))),
            CountMatch(_) => mapper.with_dtype(DataType::UInt32),
            #[cfg(feature = "string_justify")]
            Zfill { .. } | LJust { .. } | RJust { .. } => mapper.with_same_dtype(),
            #[cfg(feature = "temporal")]
            Strptime(dtype, _) => mapper.with_dtype(dtype.clone()),
            #[cfg(feature = "concat_str")]
            ConcatVertical(_) | ConcatHorizontal(_) => mapper.with_same_dtype(),
            #[cfg(feature = "regex")]
            Replace { .. } => mapper.with_same_dtype(),
            Uppercase | Lowercase | Strip(_) | LStrip(_) | RStrip(_) | Slice(_, _) => {
                mapper.with_same_dtype()
            }
            #[cfg(feature = "nightly")]
            Titlecase => mapper.with_same_dtype(),
            #[cfg(feature = "string_from_radix")]
            FromRadix { .. } => mapper.with_dtype(DataType::Int32),
            Explode => mapper.with_same_dtype(),
            #[cfg(feature = "dtype-decimal")]
            ToDecimal(_) => mapper.with_dtype(DataType::Decimal(None, None)),
            #[cfg(feature = "extract_jsonpath")]
            JsonExtract { dtype, .. } => mapper.with_opt_dtype(dtype.clone()),
        }
    }
}

impl Display for StringFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
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
            StringFunction::Strptime(_, _) => "strptime",
            #[cfg(feature = "concat_str")]
            StringFunction::ConcatVertical(_) => "concat_vertical",
            #[cfg(feature = "concat_str")]
            StringFunction::ConcatHorizontal(_) => "concat_horizontal",
            #[cfg(feature = "regex")]
            StringFunction::Replace { .. } => "replace",
            StringFunction::Uppercase => "uppercase",
            StringFunction::Lowercase => "lowercase",
            #[cfg(feature = "nightly")]
            StringFunction::Titlecase => "titlecase",
            StringFunction::Strip(_) => "strip",
            StringFunction::LStrip(_) => "lstrip",
            StringFunction::RStrip(_) => "rstrip",
            #[cfg(feature = "string_from_radix")]
            StringFunction::FromRadix { .. } => "from_radix",
            StringFunction::Slice(_, _) => "str_slice",
            StringFunction::Explode => "explode",
            #[cfg(feature = "dtype-decimal")]
            StringFunction::ToDecimal(_) => "to_decimal",
            #[cfg(feature = "extract_jsonpath")]
            StringFunction::JsonExtract { .. } => "json_extract",
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

#[cfg(feature = "nightly")]
pub(super) fn titlecase(s: &Series) -> PolarsResult<Series> {
    let ca = s.utf8()?;
    Ok(ca.to_titlecase().into_series())
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
                    ca.contains(pat, strict)?
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
                        (Some(src), Some(pat)) => Regex::new(pat).ok().map(|re| re.is_match(src)),
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
            .ok_or_else(|| polars_err!(ComputeError: "expected a pattern, got null"))?;
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
pub(super) fn strptime(
    s: &Series,
    dtype: DataType,
    options: &StrptimeOptions,
) -> PolarsResult<Series> {
    match dtype {
        DataType::Date => to_date(s, options),
        DataType::Datetime(time_unit, time_zone) => {
            to_datetime(s, &time_unit, time_zone.as_ref(), options)
        }
        DataType::Time => to_time(s, options),
        dt => polars_bail!(ComputeError: "not implemented for dtype {}", dt),
    }
}

#[cfg(feature = "dtype-date")]
fn to_date(s: &Series, options: &StrptimeOptions) -> PolarsResult<Series> {
    let ca = s.utf8()?;
    let out = {
        if options.exact {
            ca.as_date(options.format.as_deref(), options.cache)?
                .into_series()
        } else {
            ca.as_date_not_exact(options.format.as_deref())?
                .into_series()
        }
    };

    if options.strict {
        polars_ensure!(
            out.null_count() == ca.null_count(),
            ComputeError:
            "strict conversion to dates failed.\n\
            \n\
            You might want to try:\n\
            - setting `strict=False`\n\
            - explicitly specifying a `format`"
        );
    }
    Ok(out.into_series())
}

#[cfg(feature = "dtype-datetime")]
fn to_datetime(
    s: &Series,
    time_unit: &TimeUnit,
    time_zone: Option<&TimeZone>,
    options: &StrptimeOptions,
) -> PolarsResult<Series> {
    let tz_aware = match &options.format {
        #[cfg(feature = "timezones")]
        Some(format) => TZ_AWARE_RE.is_match(format),
        _ => false,
    };
    if let (Some(tz), true) = (time_zone, tz_aware) {
        if tz != "UTC" {
            polars_bail!(
                ComputeError:
                "if using strftime/to_datetime with a time-zone-aware format, the output will be in UTC. Please either drop the time zone from the function call, or set it to UTC. \
                If you are trying to convert the output to a different time zone, please use `convert_time_zone`."
            )
        }
    };

    let ca = s.utf8()?;
    let out = if options.exact {
        ca.as_datetime(
            options.format.as_deref(),
            *time_unit,
            options.cache,
            tz_aware,
            time_zone,
        )?
        .into_series()
    } else {
        ca.as_datetime_not_exact(options.format.as_deref(), *time_unit, tz_aware, time_zone)?
            .into_series()
    };

    if options.strict {
        polars_ensure!(
            out.null_count() == ca.null_count(),
            ComputeError:
            "strict conversion to datetimes failed.\n\
            \n\
            You might want to try:\n\
            - setting `strict=False`\n\
            - explicitly specifying a `format`"
        );
    }
    Ok(out.into_series())
}

#[cfg(feature = "dtype-time")]
fn to_time(s: &Series, options: &StrptimeOptions) -> PolarsResult<Series> {
    polars_ensure!(
        options.exact, ComputeError: "non-exact not implemented for Time data type"
    );

    let ca = s.utf8()?;
    let out = ca
        .as_time(options.format.as_deref(), options.cache)?
        .into_series();

    if options.strict {
        polars_ensure!(
            out.null_count() == ca.null_count(),
            ComputeError:
            "strict conversion to times failed.\n\
            \n\
            You might want to try:\n\
            - setting `strict=False`\n\
            - explicitly specifying a `format`"
        );
    }
    Ok(out.into_series())
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
    pat.get(0).ok_or_else(
        || polars_err!(ComputeError: "pattern cannot be 'null' in 'replace' expression"),
    )
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
fn is_literal_pat(pat: &str) -> bool {
    pat.chars().all(|c| !c.is_ascii_punctuation())
}

#[cfg(feature = "regex")]
fn replace_n<'a>(
    ca: &'a Utf8Chunked,
    pat: &'a Utf8Chunked,
    val: &'a Utf8Chunked,
    literal: bool,
    n: usize,
) -> PolarsResult<Utf8Chunked> {
    match (pat.len(), val.len()) {
        (1, 1) => {
            let pat = get_pat(pat)?;
            let val = val.get(0).ok_or_else(
                || polars_err!(ComputeError: "value cannot be 'null' in 'replace' expression"),
            )?;
            let literal = literal || is_literal_pat(pat);

            match literal {
                true => ca.replace_literal(pat, val, n),
                false => {
                    if n > 1 {
                        polars_bail!(ComputeError: "regex replacement with 'n > 1' not yet supported")
                    }
                    ca.replace(pat, val)
                }
            }
        }
        (1, len_val) => {
            if n > 1 {
                polars_bail!(ComputeError: "multivalue replacement with 'n > 1' not yet supported")
            }
            let mut pat = get_pat(pat)?.to_string();
            polars_ensure!(
                len_val == ca.len(),
                ComputeError:
                "replacement value length ({}) does not match string column length ({})",
                len_val, ca.len(),
            );
            let literal = literal || is_literal_pat(&pat);

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
        _ => polars_bail!(
            ComputeError: "dynamic pattern length in 'str.replace' expressions is not supported yet"
        ),
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
            let val = val.get(0).ok_or_else(
                || polars_err!(ComputeError: "value cannot be 'null' in 'replace' expression"),
            )?;
            let literal = literal || is_literal_pat(pat);

            match literal {
                true => ca.replace_literal_all(pat, val),
                false => ca.replace_all(pat, val),
            }
        }
        (1, len_val) => {
            let mut pat = get_pat(pat)?.to_string();
            polars_ensure!(
                len_val == ca.len(),
                ComputeError:
                "replacement value length ({}) does not match string column length ({})",
                len_val, ca.len(),
            );
            let literal = literal || is_literal_pat(&pat);

            if literal {
                pat = escape(&pat)
            }

            let reg = Regex::new(&pat)?;

            let f = |s: &'a str, val: &'a str| reg.replace_all(s, val);
            Ok(iter_and_replace(ca, val, f))
        }
        _ => polars_bail!(
            ComputeError: "dynamic pattern length in 'str.replace' expressions is not supported yet"
        ),
    }
}

#[cfg(feature = "regex")]
pub(super) fn replace(s: &[Series], literal: bool, n: i64) -> PolarsResult<Series> {
    let column = &s[0];
    let pat = &s[1];
    let val = &s[2];

    let all = n < 0;

    let column = column.utf8()?;
    let pat = pat.utf8()?;
    let val = val.utf8()?;

    if all {
        replace_all(column, pat, val, literal)
    } else {
        replace_n(column, pat, val, literal, n as usize)
    }
    .map(|ca| ca.into_series())
}

#[cfg(feature = "string_from_radix")]
pub(super) fn from_radix(s: &Series, radix: u32, strict: bool) -> PolarsResult<Series> {
    let ca = s.utf8()?;
    ca.parse_int(radix, strict).map(|ok| ok.into_series())
}
pub(super) fn str_slice(s: &Series, start: i64, length: Option<u64>) -> PolarsResult<Series> {
    let ca = s.utf8()?;
    ca.str_slice(start, length).map(|ca| ca.into_series())
}

pub(super) fn explode(s: &Series) -> PolarsResult<Series> {
    let ca = s.utf8()?;
    ca.explode()
}

#[cfg(feature = "dtype-decimal")]
pub(super) fn to_decimal(s: &Series, infer_len: usize) -> PolarsResult<Series> {
    let ca = s.utf8()?;
    ca.to_decimal(infer_len)
}

#[cfg(feature = "extract_jsonpath")]
pub(super) fn json_extract(
    s: &Series,
    dtype: Option<DataType>,
    infer_schema_len: Option<usize>,
) -> PolarsResult<Series> {
    let ca = s.utf8()?;
    ca.json_extract(dtype, infer_schema_len)
}
