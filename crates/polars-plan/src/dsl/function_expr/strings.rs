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
    #[cfg(feature = "concat_str")]
    ConcatHorizontal(String),
    #[cfg(feature = "concat_str")]
    ConcatVertical(String),
    #[cfg(feature = "regex")]
    Contains {
        literal: bool,
        strict: bool,
    },
    CountMatches(bool),
    EndsWith,
    Explode,
    Extract {
        pat: String,
        group_index: usize,
    },
    ExtractAll,
    #[cfg(feature = "extract_groups")]
    ExtractGroups {
        dtype: DataType,
        pat: String,
    },
    #[cfg(feature = "string_from_radix")]
    FromRadix(u32, bool),
    NChars,
    Length,
    #[cfg(feature = "string_justify")]
    LJust {
        width: usize,
        fillchar: char,
    },
    Lowercase,
    #[cfg(feature = "extract_jsonpath")]
    JsonExtract {
        dtype: Option<DataType>,
        infer_schema_len: Option<usize>,
    },
    #[cfg(feature = "regex")]
    Replace {
        // negative is replace all
        // how many matches to replace
        n: i64,
        literal: bool,
    },
    #[cfg(feature = "string_justify")]
    RJust {
        width: usize,
        fillchar: char,
    },
    Slice(i64, Option<u64>),
    StartsWith,
    StripChars(Option<String>),
    StripCharsStart(Option<String>),
    StripCharsEnd(Option<String>),
    StripPrefix(String),
    StripSuffix(String),
    #[cfg(feature = "temporal")]
    Strptime(DataType, StrptimeOptions),
    Split,
    SplitInclusive,
    #[cfg(feature = "dtype-decimal")]
    ToDecimal(usize),
    #[cfg(feature = "nightly")]
    Titlecase,
    Uppercase,
    #[cfg(feature = "string_justify")]
    Zfill(usize),
}

impl StringFunction {
    pub(super) fn get_field(&self, mapper: FieldsMapper) -> PolarsResult<Field> {
        use StringFunction::*;
        match self {
            #[cfg(feature = "concat_str")]
            ConcatVertical(_) | ConcatHorizontal(_) => mapper.with_dtype(DataType::Utf8),
            #[cfg(feature = "regex")]
            Contains { .. } => mapper.with_dtype(DataType::Boolean),
            CountMatches(_) => mapper.with_dtype(DataType::UInt32),
            EndsWith | StartsWith => mapper.with_dtype(DataType::Boolean),
            Explode => mapper.with_same_dtype(),
            Extract { .. } => mapper.with_same_dtype(),
            ExtractAll => mapper.with_dtype(DataType::List(Box::new(DataType::Utf8))),
            #[cfg(feature = "extract_groups")]
            ExtractGroups { dtype, .. } => mapper.with_dtype(dtype.clone()),
            #[cfg(feature = "string_from_radix")]
            FromRadix { .. } => mapper.with_dtype(DataType::Int32),
            #[cfg(feature = "extract_jsonpath")]
            JsonExtract { dtype, .. } => mapper.with_opt_dtype(dtype.clone()),
            Length => mapper.with_dtype(DataType::UInt32),
            NChars => mapper.with_dtype(DataType::UInt32),
            #[cfg(feature = "regex")]
            Replace { .. } => mapper.with_same_dtype(),
            #[cfg(feature = "temporal")]
            Strptime(dtype, _) => mapper.with_dtype(dtype.clone()),
            Split | SplitInclusive => mapper.with_dtype(DataType::List(Box::new(DataType::Utf8))),
            #[cfg(feature = "nightly")]
            Titlecase => mapper.with_same_dtype(),
            #[cfg(feature = "dtype-decimal")]
            ToDecimal(_) => mapper.with_dtype(DataType::Decimal(None, None)),
            Uppercase
            | Lowercase
            | StripChars(_)
            | StripCharsStart(_)
            | StripCharsEnd(_)
            | StripPrefix(_)
            | StripSuffix(_)
            | Slice(_, _) => mapper.with_same_dtype(),
            #[cfg(feature = "string_justify")]
            Zfill { .. } | LJust { .. } | RJust { .. } => mapper.with_same_dtype(),
        }
    }
}

impl Display for StringFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            #[cfg(feature = "regex")]
            StringFunction::Contains { .. } => "contains",
            StringFunction::CountMatches(_) => "count_matches",
            StringFunction::EndsWith { .. } => "ends_with",
            StringFunction::Extract { .. } => "extract",
            #[cfg(feature = "concat_str")]
            StringFunction::ConcatHorizontal(_) => "concat_horizontal",
            #[cfg(feature = "concat_str")]
            StringFunction::ConcatVertical(_) => "concat_vertical",
            StringFunction::Explode => "explode",
            StringFunction::ExtractAll => "extract_all",
            #[cfg(feature = "extract_groups")]
            StringFunction::ExtractGroups { .. } => "extract_groups",
            #[cfg(feature = "string_from_radix")]
            StringFunction::FromRadix { .. } => "from_radix",
            #[cfg(feature = "extract_jsonpath")]
            StringFunction::JsonExtract { .. } => "json_extract",
            #[cfg(feature = "string_justify")]
            StringFunction::LJust { .. } => "str.ljust",
            StringFunction::Length => "str_lengths",
            StringFunction::Lowercase => "lowercase",
            StringFunction::NChars => "n_chars",
            #[cfg(feature = "string_justify")]
            StringFunction::RJust { .. } => "rjust",
            #[cfg(feature = "regex")]
            StringFunction::Replace { .. } => "replace",
            StringFunction::Slice(_, _) => "str_slice",
            StringFunction::StartsWith { .. } => "starts_with",
            StringFunction::StripChars(_) => "strip_chars",
            StringFunction::StripCharsStart(_) => "strip_chars_start",
            StringFunction::StripCharsEnd(_) => "strip_chars_end",
            StringFunction::StripPrefix(_) => "strip_prefix",
            StringFunction::StripSuffix(_) => "strip_suffix",
            #[cfg(feature = "temporal")]
            StringFunction::Strptime(_, _) => "strptime",
            StringFunction::Split => "split",
            StringFunction::SplitInclusive => "split_inclusive",
            #[cfg(feature = "nightly")]
            StringFunction::Titlecase => "titlecase",
            #[cfg(feature = "dtype-decimal")]
            StringFunction::ToDecimal(_) => "to_decimal",
            StringFunction::Uppercase => "uppercase",
            #[cfg(feature = "string_justify")]
            StringFunction::Zfill(_) => "zfill",
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

pub(super) fn n_chars(s: &Series) -> PolarsResult<Series> {
    let ca = s.utf8()?;
    Ok(ca.str_n_chars().into_series())
}

pub(super) fn lengths(s: &Series) -> PolarsResult<Series> {
    let ca = s.utf8()?;
    Ok(ca.str_lengths().into_series())
}

#[cfg(feature = "regex")]
pub(super) fn contains(s: &[Series], literal: bool, strict: bool) -> PolarsResult<Series> {
    // TODO! move to polars-ops
    let ca = s[0].utf8()?;
    let pat = s[1].utf8()?;

    let mut out: BooleanChunked = match pat.len() {
        1 => match pat.get(0) {
            Some(pat) => {
                if literal {
                    ca.contains_literal(pat)?
                } else {
                    ca.contains(pat, strict)?
                }
            },
            None => BooleanChunked::full(ca.name(), false, ca.len()),
        },
        _ => {
            if literal {
                ca.into_iter()
                    .zip(pat)
                    .map(|(opt_src, opt_val)| match (opt_src, opt_val) {
                        (Some(src), Some(pat)) => src.contains(pat),
                        _ => false,
                    })
                    .collect_trusted()
            } else if strict {
                ca.into_iter()
                    .zip(pat)
                    .map(|(opt_src, opt_val)| match (opt_src, opt_val) {
                        (Some(src), Some(pat)) => {
                            let re = Regex::new(pat)?;
                            Ok(re.is_match(src))
                        },
                        _ => Ok(false),
                    })
                    .collect::<PolarsResult<_>>()?
            } else {
                ca.into_iter()
                    .zip(pat)
                    .map(|(opt_src, opt_val)| match (opt_src, opt_val) {
                        (Some(src), Some(pat)) => Regex::new(pat).ok().map(|re| re.is_match(src)),
                        _ => Some(false),
                    })
                    .collect_trusted()
            }
        },
    };

    out.rename(ca.name());
    Ok(out.into_series())
}

pub(super) fn ends_with(s: &[Series]) -> PolarsResult<Series> {
    let ca = &s[0].utf8()?.as_binary();
    let suffix = &s[1].utf8()?.as_binary();

    Ok(ca
        .ends_with_chunked(suffix)
        .with_name(ca.name())
        .into_series())
}

pub(super) fn starts_with(s: &[Series]) -> PolarsResult<Series> {
    let ca = &s[0].utf8()?.as_binary();
    let prefix = &s[1].utf8()?.as_binary();

    Ok(ca
        .starts_with_chunked(prefix)
        .with_name(ca.name())
        .into_series())
}

/// Extract a regex pattern from the a string value.
pub(super) fn extract(s: &Series, pat: &str, group_index: usize) -> PolarsResult<Series> {
    let pat = pat.to_string();

    let ca = s.utf8()?;
    ca.extract(&pat, group_index).map(|ca| ca.into_series())
}

#[cfg(feature = "extract_groups")]
/// Extract all capture groups from a regex pattern as a struct
pub(super) fn extract_groups(s: &Series, pat: &str, dtype: &DataType) -> PolarsResult<Series> {
    let ca = s.utf8()?;
    ca.extract_groups(pat, dtype)
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

pub(super) fn strip_chars(s: &Series, matches: Option<&str>) -> PolarsResult<Series> {
    let ca = s.utf8()?;
    if let Some(matches) = matches {
        if matches.chars().count() == 1 {
            // Fast path for when a single character is passed
            Ok(ca
                .apply_values(|s| Cow::Borrowed(s.trim_matches(matches.chars().next().unwrap())))
                .into_series())
        } else {
            Ok(ca
                .apply_values(|s| Cow::Borrowed(s.trim_matches(|c| matches.contains(c))))
                .into_series())
        }
    } else {
        Ok(ca.apply_values(|s| Cow::Borrowed(s.trim())).into_series())
    }
}

pub(super) fn strip_chars_start(s: &Series, matches: Option<&str>) -> PolarsResult<Series> {
    let ca = s.utf8()?;

    if let Some(matches) = matches {
        if matches.chars().count() == 1 {
            // Fast path for when a single character is passed
            Ok(ca
                .apply_values(|s| {
                    Cow::Borrowed(s.trim_start_matches(matches.chars().next().unwrap()))
                })
                .into_series())
        } else {
            Ok(ca
                .apply_values(|s| Cow::Borrowed(s.trim_start_matches(|c| matches.contains(c))))
                .into_series())
        }
    } else {
        Ok(ca
            .apply_values(|s| Cow::Borrowed(s.trim_start()))
            .into_series())
    }
}

pub(super) fn strip_chars_end(s: &Series, matches: Option<&str>) -> PolarsResult<Series> {
    let ca = s.utf8()?;
    if let Some(matches) = matches {
        if matches.chars().count() == 1 {
            // Fast path for when a single character is passed
            Ok(ca
                .apply_values(|s| {
                    Cow::Borrowed(s.trim_end_matches(matches.chars().next().unwrap()))
                })
                .into_series())
        } else {
            Ok(ca
                .apply_values(|s| Cow::Borrowed(s.trim_end_matches(|c| matches.contains(c))))
                .into_series())
        }
    } else {
        Ok(ca
            .apply_values(|s| Cow::Borrowed(s.trim_end()))
            .into_series())
    }
}

pub(super) fn strip_prefix(s: &Series, prefix: &str) -> PolarsResult<Series> {
    let ca = s.utf8()?;
    Ok(ca
        .apply_values(|s| Cow::Borrowed(s.strip_prefix(prefix).unwrap_or(s)))
        .into_series())
}

pub(super) fn strip_suffix(s: &Series, suffix: &str) -> PolarsResult<Series> {
    let ca = s.utf8()?;
    Ok(ca
        .apply_values(|s| Cow::Borrowed(s.strip_suffix(suffix).unwrap_or(s)))
        .into_series())
}

pub(super) fn extract_all(args: &[Series]) -> PolarsResult<Series> {
    let s = &args[0];
    let pat = &args[1];

    let ca = s.utf8()?;
    let pat = pat.utf8()?;

    if pat.len() == 1 {
        if let Some(pat) = pat.get(0) {
            ca.extract_all(pat).map(|ca| ca.into_series())
        } else {
            Ok(Series::full_null(
                ca.name(),
                ca.len(),
                &DataType::List(Box::new(DataType::Utf8)),
            ))
        }
    } else {
        ca.extract_all_many(pat).map(|ca| ca.into_series())
    }
}

pub(super) fn count_matches(args: &[Series], literal: bool) -> PolarsResult<Series> {
    let s = &args[0];
    let pat = &args[1];

    let ca = s.utf8()?;
    let pat = pat.utf8()?;
    if pat.len() == 1 {
        if let Some(pat) = pat.get(0) {
            ca.count_matches(pat, literal).map(|ca| ca.into_series())
        } else {
            Ok(Series::full_null(ca.name(), ca.len(), &DataType::UInt32))
        }
    } else {
        ca.count_matches_many(pat, literal)
            .map(|ca| ca.into_series())
    }
}

#[cfg(feature = "temporal")]
pub(super) fn strptime(
    s: &[Series],
    dtype: DataType,
    options: &StrptimeOptions,
) -> PolarsResult<Series> {
    match dtype {
        DataType::Date => to_date(&s[0], options),
        DataType::Datetime(time_unit, time_zone) => {
            to_datetime(s, &time_unit, time_zone.as_ref(), options)
        },
        DataType::Time => to_time(&s[0], options),
        dt => polars_bail!(ComputeError: "not implemented for dtype {}", dt),
    }
}

pub(super) fn split(s: &[Series]) -> PolarsResult<Series> {
    let ca = s[0].utf8()?;
    let by = s[1].utf8()?;

    if by.len() == 1 {
        if let Some(by) = by.get(0) {
            Ok(ca.split(by).into_series())
        } else {
            Ok(Series::full_null(
                ca.name(),
                ca.len(),
                &DataType::List(Box::new(DataType::Utf8)),
            ))
        }
    } else {
        Ok(ca.split_many(by).into_series())
    }
}

pub(super) fn split_inclusive(s: &[Series]) -> PolarsResult<Series> {
    let ca = s[0].utf8()?;
    let by = s[1].utf8()?;

    if by.len() == 1 {
        if let Some(by) = by.get(0) {
            Ok(ca.split_inclusive(by).into_series())
        } else {
            Ok(Series::full_null(
                ca.name(),
                ca.len(),
                &DataType::List(Box::new(DataType::Utf8)),
            ))
        }
    } else {
        Ok(ca.split_inclusive_many(by).into_series())
    }
}

fn handle_temporal_parsing_error(
    ca: &Utf8Chunked,
    out: &Series,
    format: Option<&str>,
    has_non_exact_option: bool,
) -> PolarsResult<()> {
    let failure_mask = !ca.is_null() & out.is_null();
    let all_failures = ca.filter(&failure_mask)?;
    let first_failures = all_failures.unique()?.slice(0, 10).sort(false);
    let n_failures = all_failures.len();
    let n_failures_unique = all_failures.n_unique()?;
    let exact_addendum = if has_non_exact_option {
        "- setting `exact=False` (note: this is much slower!)\n"
    } else {
        ""
    };
    let format_addendum;
    if let Some(format) = format {
        format_addendum = format!(
            "- checking whether the format provided ('{}') is correct",
            format
        );
    } else {
        format_addendum = String::from("- explicitly specifying `format`");
    }
    polars_bail!(
        ComputeError:
        "strict {} parsing failed for {} value(s) ({} unique): {}\n\
        \n\
        You might want to try:\n\
        - setting `strict=False`\n\
        {}\
        {}",
        out.dtype(),
        n_failures,
        n_failures_unique,
        first_failures.into_series().fmt_list(),
        exact_addendum,
        format_addendum,
    )
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

    if options.strict && ca.null_count() != out.null_count() {
        handle_temporal_parsing_error(ca, &out, options.format.as_deref(), true)?;
    }
    Ok(out.into_series())
}

#[cfg(feature = "dtype-datetime")]
fn to_datetime(
    s: &[Series],
    time_unit: &TimeUnit,
    time_zone: Option<&TimeZone>,
    options: &StrptimeOptions,
) -> PolarsResult<Series> {
    let datetime_strings = &s[0].utf8().unwrap();
    let ambiguous = &s[1].utf8().unwrap();
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

    let out = if options.exact {
        datetime_strings
            .as_datetime(
                options.format.as_deref(),
                *time_unit,
                options.cache,
                tz_aware,
                time_zone,
                ambiguous,
            )?
            .into_series()
    } else {
        datetime_strings
            .as_datetime_not_exact(
                options.format.as_deref(),
                *time_unit,
                tz_aware,
                time_zone,
                ambiguous,
            )?
            .into_series()
    };

    if options.strict && datetime_strings.null_count() != out.null_count() {
        handle_temporal_parsing_error(datetime_strings, &out, options.format.as_deref(), true)?;
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

    if options.strict && ca.null_count() != out.null_count() {
        handle_temporal_parsing_error(ca, &out, options.format.as_deref(), false)?;
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
        .zip(val)
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
                },
            }
        },
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
        },
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
        },
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
        },
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
