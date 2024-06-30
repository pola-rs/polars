use std::borrow::Cow;

use arrow::legacy::utils::CustomIterTools;
#[cfg(feature = "timezones")]
use once_cell::sync::Lazy;
#[cfg(feature = "timezones")]
use polars_core::chunked_array::temporal::validate_time_zone;
use polars_core::utils::handle_casting_failures;
#[cfg(feature = "dtype-struct")]
use polars_utils::format_smartstring;
#[cfg(feature = "regex")]
use regex::{escape, Regex};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::*;
use crate::{map, map_as_slice};

#[cfg(all(feature = "regex", feature = "timezones"))]
static TZ_AWARE_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(%z)|(%:z)|(%::z)|(%:::z)|(%#z)|(^%\+$)").unwrap());

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, PartialEq, Debug, Eq, Hash)]
pub enum StringFunction {
    #[cfg(feature = "concat_str")]
    ConcatHorizontal {
        delimiter: String,
        ignore_nulls: bool,
    },
    #[cfg(feature = "concat_str")]
    ConcatVertical {
        delimiter: String,
        ignore_nulls: bool,
    },
    #[cfg(feature = "regex")]
    Contains {
        literal: bool,
        strict: bool,
    },
    CountMatches(bool),
    EndsWith,
    Extract(usize),
    ExtractAll,
    #[cfg(feature = "extract_groups")]
    ExtractGroups {
        dtype: DataType,
        pat: String,
    },
    #[cfg(feature = "regex")]
    Find {
        literal: bool,
        strict: bool,
    },
    #[cfg(feature = "string_to_integer")]
    ToInteger(bool),
    LenBytes,
    LenChars,
    Lowercase,
    #[cfg(feature = "extract_jsonpath")]
    JsonDecode {
        dtype: Option<DataType>,
        infer_schema_len: Option<usize>,
    },
    #[cfg(feature = "extract_jsonpath")]
    JsonPathMatch,
    #[cfg(feature = "regex")]
    Replace {
        // negative is replace all
        // how many matches to replace
        n: i64,
        literal: bool,
    },
    #[cfg(feature = "string_reverse")]
    Reverse,
    #[cfg(feature = "string_pad")]
    PadStart {
        length: usize,
        fill_char: char,
    },
    #[cfg(feature = "string_pad")]
    PadEnd {
        length: usize,
        fill_char: char,
    },
    Slice,
    Head,
    Tail,
    #[cfg(feature = "string_encoding")]
    HexEncode,
    #[cfg(feature = "binary_encoding")]
    HexDecode(bool),
    #[cfg(feature = "string_encoding")]
    Base64Encode,
    #[cfg(feature = "binary_encoding")]
    Base64Decode(bool),
    StartsWith,
    StripChars,
    StripCharsStart,
    StripCharsEnd,
    StripPrefix,
    StripSuffix,
    #[cfg(feature = "dtype-struct")]
    SplitExact {
        n: usize,
        inclusive: bool,
    },
    #[cfg(feature = "dtype-struct")]
    SplitN(usize),
    #[cfg(feature = "temporal")]
    Strptime(DataType, StrptimeOptions),
    Split(bool),
    #[cfg(feature = "dtype-decimal")]
    ToDecimal(usize),
    #[cfg(feature = "nightly")]
    Titlecase,
    Uppercase,
    #[cfg(feature = "string_pad")]
    ZFill,
    #[cfg(feature = "find_many")]
    ContainsMany {
        ascii_case_insensitive: bool,
    },
    #[cfg(feature = "find_many")]
    ReplaceMany {
        ascii_case_insensitive: bool,
    },
    #[cfg(feature = "find_many")]
    ExtractMany {
        ascii_case_insensitive: bool,
        overlapping: bool,
    },
}

impl StringFunction {
    pub(super) fn get_field(&self, mapper: FieldsMapper) -> PolarsResult<Field> {
        use StringFunction::*;
        match self {
            #[cfg(feature = "concat_str")]
            ConcatVertical { .. } | ConcatHorizontal { .. } => mapper.with_dtype(DataType::String),
            #[cfg(feature = "regex")]
            Contains { .. } => mapper.with_dtype(DataType::Boolean),
            CountMatches(_) => mapper.with_dtype(DataType::UInt32),
            EndsWith | StartsWith => mapper.with_dtype(DataType::Boolean),
            Extract(_) => mapper.with_same_dtype(),
            ExtractAll => mapper.with_dtype(DataType::List(Box::new(DataType::String))),
            #[cfg(feature = "extract_groups")]
            ExtractGroups { dtype, .. } => mapper.with_dtype(dtype.clone()),
            #[cfg(feature = "string_to_integer")]
            ToInteger { .. } => mapper.with_dtype(DataType::Int64),
            #[cfg(feature = "regex")]
            Find { .. } => mapper.with_dtype(DataType::UInt32),
            #[cfg(feature = "extract_jsonpath")]
            JsonDecode { dtype, .. } => mapper.with_opt_dtype(dtype.clone()),
            #[cfg(feature = "extract_jsonpath")]
            JsonPathMatch => mapper.with_dtype(DataType::String),
            LenBytes => mapper.with_dtype(DataType::UInt32),
            LenChars => mapper.with_dtype(DataType::UInt32),
            #[cfg(feature = "regex")]
            Replace { .. } => mapper.with_same_dtype(),
            #[cfg(feature = "string_reverse")]
            Reverse => mapper.with_same_dtype(),
            #[cfg(feature = "temporal")]
            Strptime(dtype, _) => mapper.with_dtype(dtype.clone()),
            Split(_) => mapper.with_dtype(DataType::List(Box::new(DataType::String))),
            #[cfg(feature = "nightly")]
            Titlecase => mapper.with_same_dtype(),
            #[cfg(feature = "dtype-decimal")]
            ToDecimal(_) => mapper.with_dtype(DataType::Decimal(None, None)),
            #[cfg(feature = "string_encoding")]
            HexEncode => mapper.with_same_dtype(),
            #[cfg(feature = "binary_encoding")]
            HexDecode(_) => mapper.with_dtype(DataType::Binary),
            #[cfg(feature = "string_encoding")]
            Base64Encode => mapper.with_same_dtype(),
            #[cfg(feature = "binary_encoding")]
            Base64Decode(_) => mapper.with_dtype(DataType::Binary),
            Uppercase | Lowercase | StripChars | StripCharsStart | StripCharsEnd | StripPrefix
            | StripSuffix | Slice | Head | Tail => mapper.with_same_dtype(),
            #[cfg(feature = "string_pad")]
            PadStart { .. } | PadEnd { .. } | ZFill => mapper.with_same_dtype(),
            #[cfg(feature = "dtype-struct")]
            SplitExact { n, .. } => mapper.with_dtype(DataType::Struct(
                (0..n + 1)
                    .map(|i| Field::from_owned(format_smartstring!("field_{i}"), DataType::String))
                    .collect(),
            )),
            #[cfg(feature = "dtype-struct")]
            SplitN(n) => mapper.with_dtype(DataType::Struct(
                (0..*n)
                    .map(|i| Field::from_owned(format_smartstring!("field_{i}"), DataType::String))
                    .collect(),
            )),
            #[cfg(feature = "find_many")]
            ContainsMany { .. } => mapper.with_dtype(DataType::Boolean),
            #[cfg(feature = "find_many")]
            ReplaceMany { .. } => mapper.with_same_dtype(),
            #[cfg(feature = "find_many")]
            ExtractMany { .. } => mapper.with_dtype(DataType::List(Box::new(DataType::String))),
        }
    }
}

impl Display for StringFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use StringFunction::*;
        let s = match self {
            #[cfg(feature = "regex")]
            Contains { .. } => "contains",
            CountMatches(_) => "count_matches",
            EndsWith { .. } => "ends_with",
            Extract(_) => "extract",
            #[cfg(feature = "concat_str")]
            ConcatHorizontal { .. } => "concat_horizontal",
            #[cfg(feature = "concat_str")]
            ConcatVertical { .. } => "concat_vertical",
            ExtractAll => "extract_all",
            #[cfg(feature = "extract_groups")]
            ExtractGroups { .. } => "extract_groups",
            #[cfg(feature = "string_to_integer")]
            ToInteger { .. } => "to_integer",
            #[cfg(feature = "regex")]
            Find { .. } => "find",
            Head { .. } => "head",
            Tail { .. } => "tail",
            #[cfg(feature = "extract_jsonpath")]
            JsonDecode { .. } => "json_decode",
            #[cfg(feature = "extract_jsonpath")]
            JsonPathMatch => "json_path_match",
            LenBytes => "len_bytes",
            Lowercase => "lowercase",
            LenChars => "len_chars",
            #[cfg(feature = "string_pad")]
            PadEnd { .. } => "pad_end",
            #[cfg(feature = "string_pad")]
            PadStart { .. } => "pad_start",
            #[cfg(feature = "regex")]
            Replace { .. } => "replace",
            #[cfg(feature = "string_reverse")]
            Reverse => "reverse",
            #[cfg(feature = "string_encoding")]
            HexEncode => "hex_encode",
            #[cfg(feature = "binary_encoding")]
            HexDecode(_) => "hex_decode",
            #[cfg(feature = "string_encoding")]
            Base64Encode => "base64_encode",
            #[cfg(feature = "binary_encoding")]
            Base64Decode(_) => "base64_decode",
            Slice => "slice",
            StartsWith { .. } => "starts_with",
            StripChars => "strip_chars",
            StripCharsStart => "strip_chars_start",
            StripCharsEnd => "strip_chars_end",
            StripPrefix => "strip_prefix",
            StripSuffix => "strip_suffix",
            #[cfg(feature = "dtype-struct")]
            SplitExact { inclusive, .. } => {
                if *inclusive {
                    "split_exact_inclusive"
                } else {
                    "split_exact"
                }
            },
            #[cfg(feature = "dtype-struct")]
            SplitN(_) => "splitn",
            #[cfg(feature = "temporal")]
            Strptime(_, _) => "strptime",
            Split(inclusive) => {
                if *inclusive {
                    "split_inclusive"
                } else {
                    "split"
                }
            },
            #[cfg(feature = "nightly")]
            Titlecase => "titlecase",
            #[cfg(feature = "dtype-decimal")]
            ToDecimal(_) => "to_decimal",
            Uppercase => "uppercase",
            #[cfg(feature = "string_pad")]
            ZFill => "zfill",
            #[cfg(feature = "find_many")]
            ContainsMany { .. } => "contains_many",
            #[cfg(feature = "find_many")]
            ReplaceMany { .. } => "replace_many",
            #[cfg(feature = "find_many")]
            ExtractMany { .. } => "extract_many",
        };
        write!(f, "str.{s}")
    }
}

impl From<StringFunction> for SpecialEq<Arc<dyn SeriesUdf>> {
    fn from(func: StringFunction) -> Self {
        use StringFunction::*;
        match func {
            #[cfg(feature = "regex")]
            Contains { literal, strict } => map_as_slice!(strings::contains, literal, strict),
            CountMatches(literal) => {
                map_as_slice!(strings::count_matches, literal)
            },
            EndsWith { .. } => map_as_slice!(strings::ends_with),
            StartsWith { .. } => map_as_slice!(strings::starts_with),
            Extract(group_index) => map_as_slice!(strings::extract, group_index),
            ExtractAll => {
                map_as_slice!(strings::extract_all)
            },
            #[cfg(feature = "extract_groups")]
            ExtractGroups { pat, dtype } => {
                map!(strings::extract_groups, &pat, &dtype)
            },
            #[cfg(feature = "regex")]
            Find { literal, strict } => map_as_slice!(strings::find, literal, strict),
            LenBytes => map!(strings::len_bytes),
            LenChars => map!(strings::len_chars),
            #[cfg(feature = "string_pad")]
            PadEnd { length, fill_char } => {
                map!(strings::pad_end, length, fill_char)
            },
            #[cfg(feature = "string_pad")]
            PadStart { length, fill_char } => {
                map!(strings::pad_start, length, fill_char)
            },
            #[cfg(feature = "string_pad")]
            ZFill => {
                map_as_slice!(strings::zfill)
            },
            #[cfg(feature = "temporal")]
            Strptime(dtype, options) => {
                map_as_slice!(strings::strptime, dtype.clone(), &options)
            },
            Split(inclusive) => {
                map_as_slice!(strings::split, inclusive)
            },
            #[cfg(feature = "dtype-struct")]
            SplitExact { n, inclusive } => map_as_slice!(strings::split_exact, n, inclusive),
            #[cfg(feature = "dtype-struct")]
            SplitN(n) => map_as_slice!(strings::splitn, n),
            #[cfg(feature = "concat_str")]
            ConcatVertical {
                delimiter,
                ignore_nulls,
            } => map!(strings::join, &delimiter, ignore_nulls),
            #[cfg(feature = "concat_str")]
            ConcatHorizontal {
                delimiter,
                ignore_nulls,
            } => map_as_slice!(strings::concat_hor, &delimiter, ignore_nulls),
            #[cfg(feature = "regex")]
            Replace { n, literal } => map_as_slice!(strings::replace, literal, n),
            #[cfg(feature = "string_reverse")]
            Reverse => map!(strings::reverse),
            Uppercase => map!(uppercase),
            Lowercase => map!(lowercase),
            #[cfg(feature = "nightly")]
            Titlecase => map!(strings::titlecase),
            StripChars => map_as_slice!(strings::strip_chars),
            StripCharsStart => map_as_slice!(strings::strip_chars_start),
            StripCharsEnd => map_as_slice!(strings::strip_chars_end),
            StripPrefix => map_as_slice!(strings::strip_prefix),
            StripSuffix => map_as_slice!(strings::strip_suffix),
            #[cfg(feature = "string_to_integer")]
            ToInteger(strict) => map_as_slice!(strings::to_integer, strict),
            Slice => map_as_slice!(strings::str_slice),
            Head => map_as_slice!(strings::str_head),
            Tail => map_as_slice!(strings::str_tail),
            #[cfg(feature = "string_encoding")]
            HexEncode => map!(strings::hex_encode),
            #[cfg(feature = "binary_encoding")]
            HexDecode(strict) => map!(strings::hex_decode, strict),
            #[cfg(feature = "string_encoding")]
            Base64Encode => map!(strings::base64_encode),
            #[cfg(feature = "binary_encoding")]
            Base64Decode(strict) => map!(strings::base64_decode, strict),
            #[cfg(feature = "dtype-decimal")]
            ToDecimal(infer_len) => map!(strings::to_decimal, infer_len),
            #[cfg(feature = "extract_jsonpath")]
            JsonDecode {
                dtype,
                infer_schema_len,
            } => map!(strings::json_decode, dtype.clone(), infer_schema_len),
            #[cfg(feature = "extract_jsonpath")]
            JsonPathMatch => map_as_slice!(strings::json_path_match),
            #[cfg(feature = "find_many")]
            ContainsMany {
                ascii_case_insensitive,
            } => {
                map_as_slice!(contains_many, ascii_case_insensitive)
            },
            #[cfg(feature = "find_many")]
            ReplaceMany {
                ascii_case_insensitive,
            } => {
                map_as_slice!(replace_many, ascii_case_insensitive)
            },
            #[cfg(feature = "find_many")]
            ExtractMany {
                ascii_case_insensitive,
                overlapping,
            } => {
                map_as_slice!(extract_many, ascii_case_insensitive, overlapping)
            },
        }
    }
}

#[cfg(feature = "find_many")]
fn contains_many(s: &[Series], ascii_case_insensitive: bool) -> PolarsResult<Series> {
    let ca = s[0].str()?;
    let patterns = s[1].str()?;
    polars_ops::chunked_array::strings::contains_any(ca, patterns, ascii_case_insensitive)
        .map(|out| out.into_series())
}

#[cfg(feature = "find_many")]
fn replace_many(s: &[Series], ascii_case_insensitive: bool) -> PolarsResult<Series> {
    let ca = s[0].str()?;
    let patterns = s[1].str()?;
    let replace_with = s[2].str()?;
    polars_ops::chunked_array::strings::replace_all(
        ca,
        patterns,
        replace_with,
        ascii_case_insensitive,
    )
    .map(|out| out.into_series())
}

#[cfg(feature = "find_many")]
fn extract_many(
    s: &[Series],
    ascii_case_insensitive: bool,
    overlapping: bool,
) -> PolarsResult<Series> {
    let ca = s[0].str()?;
    let patterns = &s[1];

    polars_ops::chunked_array::strings::extract_many(
        ca,
        patterns,
        ascii_case_insensitive,
        overlapping,
    )
    .map(|out| out.into_series())
}

fn uppercase(s: &Series) -> PolarsResult<Series> {
    let ca = s.str()?;
    Ok(ca.to_uppercase().into_series())
}

fn lowercase(s: &Series) -> PolarsResult<Series> {
    let ca = s.str()?;
    Ok(ca.to_lowercase().into_series())
}

#[cfg(feature = "nightly")]
pub(super) fn titlecase(s: &Series) -> PolarsResult<Series> {
    let ca = s.str()?;
    Ok(ca.to_titlecase().into_series())
}

pub(super) fn len_chars(s: &Series) -> PolarsResult<Series> {
    let ca = s.str()?;
    Ok(ca.str_len_chars().into_series())
}

pub(super) fn len_bytes(s: &Series) -> PolarsResult<Series> {
    let ca = s.str()?;
    Ok(ca.str_len_bytes().into_series())
}

#[cfg(feature = "regex")]
pub(super) fn contains(s: &[Series], literal: bool, strict: bool) -> PolarsResult<Series> {
    let ca = s[0].str()?;
    let pat = s[1].str()?;
    ca.contains_chunked(pat, literal, strict)
        .map(|ok| ok.into_series())
}

#[cfg(feature = "regex")]
pub(super) fn find(s: &[Series], literal: bool, strict: bool) -> PolarsResult<Series> {
    let ca = s[0].str()?;
    let pat = s[1].str()?;
    ca.find_chunked(pat, literal, strict)
        .map(|ok| ok.into_series())
}

pub(super) fn ends_with(s: &[Series]) -> PolarsResult<Series> {
    let ca = &s[0].str()?.as_binary();
    let suffix = &s[1].str()?.as_binary();

    Ok(ca.ends_with_chunked(suffix).into_series())
}

pub(super) fn starts_with(s: &[Series]) -> PolarsResult<Series> {
    let ca = &s[0].str()?.as_binary();
    let prefix = &s[1].str()?.as_binary();

    Ok(ca.starts_with_chunked(prefix).into_series())
}

/// Extract a regex pattern from the a string value.
pub(super) fn extract(s: &[Series], group_index: usize) -> PolarsResult<Series> {
    let ca = s[0].str()?;
    let pat = s[1].str()?;
    ca.extract(pat, group_index).map(|ca| ca.into_series())
}

#[cfg(feature = "extract_groups")]
/// Extract all capture groups from a regex pattern as a struct
pub(super) fn extract_groups(s: &Series, pat: &str, dtype: &DataType) -> PolarsResult<Series> {
    let ca = s.str()?;
    ca.extract_groups(pat, dtype)
}

#[cfg(feature = "string_pad")]
pub(super) fn pad_start(s: &Series, length: usize, fill_char: char) -> PolarsResult<Series> {
    let ca = s.str()?;
    Ok(ca.pad_start(length, fill_char).into_series())
}

#[cfg(feature = "string_pad")]
pub(super) fn pad_end(s: &Series, length: usize, fill_char: char) -> PolarsResult<Series> {
    let ca = s.str()?;
    Ok(ca.pad_end(length, fill_char).into_series())
}

#[cfg(feature = "string_pad")]
pub(super) fn zfill(s: &[Series]) -> PolarsResult<Series> {
    let ca = s[0].str()?;
    let length_s = s[1].strict_cast(&DataType::UInt64)?;
    let length = length_s.u64()?;
    Ok(ca.zfill(length).into_series())
}

pub(super) fn strip_chars(s: &[Series]) -> PolarsResult<Series> {
    let ca = s[0].str()?;
    let pat_s = &s[1];
    ca.strip_chars(pat_s).map(|ok| ok.into_series())
}

pub(super) fn strip_chars_start(s: &[Series]) -> PolarsResult<Series> {
    let ca = s[0].str()?;
    let pat_s = &s[1];
    ca.strip_chars_start(pat_s).map(|ok| ok.into_series())
}

pub(super) fn strip_chars_end(s: &[Series]) -> PolarsResult<Series> {
    let ca = s[0].str()?;
    let pat_s = &s[1];
    ca.strip_chars_end(pat_s).map(|ok| ok.into_series())
}

pub(super) fn strip_prefix(s: &[Series]) -> PolarsResult<Series> {
    let ca = s[0].str()?;
    let prefix = s[1].str()?;
    Ok(ca.strip_prefix(prefix).into_series())
}

pub(super) fn strip_suffix(s: &[Series]) -> PolarsResult<Series> {
    let ca = s[0].str()?;
    let suffix = s[1].str()?;
    Ok(ca.strip_suffix(suffix).into_series())
}

pub(super) fn extract_all(args: &[Series]) -> PolarsResult<Series> {
    let s = &args[0];
    let pat = &args[1];

    let ca = s.str()?;
    let pat = pat.str()?;

    if pat.len() == 1 {
        if let Some(pat) = pat.get(0) {
            ca.extract_all(pat).map(|ca| ca.into_series())
        } else {
            Ok(Series::full_null(
                ca.name(),
                ca.len(),
                &DataType::List(Box::new(DataType::String)),
            ))
        }
    } else {
        ca.extract_all_many(pat).map(|ca| ca.into_series())
    }
}

pub(super) fn count_matches(args: &[Series], literal: bool) -> PolarsResult<Series> {
    let s = &args[0];
    let pat = &args[1];

    let ca = s.str()?;
    let pat = pat.str()?;
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
        #[cfg(feature = "dtype-date")]
        DataType::Date => to_date(&s[0], options),
        #[cfg(feature = "dtype-datetime")]
        DataType::Datetime(time_unit, time_zone) => {
            to_datetime(s, &time_unit, time_zone.as_ref(), options)
        },
        #[cfg(feature = "dtype-time")]
        DataType::Time => to_time(&s[0], options),
        dt => polars_bail!(ComputeError: "not implemented for dtype {}", dt),
    }
}

#[cfg(feature = "dtype-struct")]
pub(super) fn split_exact(s: &[Series], n: usize, inclusive: bool) -> PolarsResult<Series> {
    let ca = s[0].str()?;
    let by = s[1].str()?;

    if inclusive {
        ca.split_exact_inclusive(by, n).map(|ca| ca.into_series())
    } else {
        ca.split_exact(by, n).map(|ca| ca.into_series())
    }
}

#[cfg(feature = "dtype-struct")]
pub(super) fn splitn(s: &[Series], n: usize) -> PolarsResult<Series> {
    let ca = s[0].str()?;
    let by = s[1].str()?;

    ca.splitn(by, n).map(|ca| ca.into_series())
}

pub(super) fn split(s: &[Series], inclusive: bool) -> PolarsResult<Series> {
    let ca = s[0].str()?;
    let by = s[1].str()?;

    if inclusive {
        Ok(ca.split_inclusive(by).into_series())
    } else {
        Ok(ca.split(by).into_series())
    }
}

#[cfg(feature = "dtype-date")]
fn to_date(s: &Series, options: &StrptimeOptions) -> PolarsResult<Series> {
    let ca = s.str()?;
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
        handle_casting_failures(s, &out)?;
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
    let datetime_strings = &s[0].str()?;
    let ambiguous = &s[1].str()?;
    let tz_aware = match &options.format {
        #[cfg(all(feature = "regex", feature = "timezones"))]
        Some(format) => TZ_AWARE_RE.is_match(format),
        _ => false,
    };
    #[cfg(feature = "timezones")]
    if let Some(time_zone) = time_zone {
        validate_time_zone(time_zone)?;
    }
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
        handle_casting_failures(&s[0], &out)?;
    }
    Ok(out.into_series())
}

#[cfg(feature = "dtype-time")]
fn to_time(s: &Series, options: &StrptimeOptions) -> PolarsResult<Series> {
    polars_ensure!(
        options.exact, ComputeError: "non-exact not implemented for Time data type"
    );

    let ca = s.str()?;
    let out = ca
        .as_time(options.format.as_deref(), options.cache)?
        .into_series();

    if options.strict && ca.null_count() != out.null_count() {
        handle_casting_failures(s, &out)?;
    }
    Ok(out.into_series())
}

#[cfg(feature = "concat_str")]
pub(super) fn join(s: &Series, delimiter: &str, ignore_nulls: bool) -> PolarsResult<Series> {
    let str_s = s.cast(&DataType::String)?;
    let joined = polars_ops::chunked_array::str_join(str_s.str()?, delimiter, ignore_nulls);
    Ok(joined.into_series())
}

#[cfg(feature = "concat_str")]
pub(super) fn concat_hor(
    series: &[Series],
    delimiter: &str,
    ignore_nulls: bool,
) -> PolarsResult<Series> {
    let str_series: Vec<_> = series
        .iter()
        .map(|s| s.cast(&DataType::String))
        .collect::<PolarsResult<_>>()?;
    let cas: Vec<_> = str_series.iter().map(|s| s.str().unwrap()).collect();
    Ok(polars_ops::chunked_array::hor_str_concat(&cas, delimiter, ignore_nulls)?.into_series())
}

impl From<StringFunction> for FunctionExpr {
    fn from(str: StringFunction) -> Self {
        FunctionExpr::StringExpr(str)
    }
}

#[cfg(feature = "regex")]
fn get_pat(pat: &StringChunked) -> PolarsResult<&str> {
    pat.get(0).ok_or_else(
        || polars_err!(ComputeError: "pattern cannot be 'null' in 'replace' expression"),
    )
}

// used only if feature="regex"
#[allow(dead_code)]
fn iter_and_replace<'a, F>(ca: &'a StringChunked, val: &'a StringChunked, f: F) -> StringChunked
where
    F: Fn(&'a str, &'a str) -> Cow<'a, str>,
{
    let mut out: StringChunked = ca
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
    ca: &'a StringChunked,
    pat: &'a StringChunked,
    val: &'a StringChunked,
    literal: bool,
    n: usize,
) -> PolarsResult<StringChunked> {
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
    ca: &'a StringChunked,
    pat: &'a StringChunked,
    val: &'a StringChunked,
    literal: bool,
) -> PolarsResult<StringChunked> {
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

    let column = column.str()?;
    let pat = pat.str()?;
    let val = val.str()?;

    if all {
        replace_all(column, pat, val, literal)
    } else {
        replace_n(column, pat, val, literal, n as usize)
    }
    .map(|ca| ca.into_series())
}

#[cfg(feature = "string_reverse")]
pub(super) fn reverse(s: &Series) -> PolarsResult<Series> {
    let ca = s.str()?;
    Ok(ca.str_reverse().into_series())
}

#[cfg(feature = "string_to_integer")]
pub(super) fn to_integer(s: &[Series], strict: bool) -> PolarsResult<Series> {
    let ca = s[0].str()?;
    let base = s[1].strict_cast(&DataType::UInt32)?;
    ca.to_integer(base.u32()?, strict)
        .map(|ok| ok.into_series())
}

fn _ensure_lengths(s: &[Series]) -> bool {
    // Calculate the post-broadcast length and ensure everything is consistent.
    let len = s
        .iter()
        .map(|series| series.len())
        .filter(|l| *l != 1)
        .max()
        .unwrap_or(1);
    s.iter()
        .all(|series| series.len() == 1 || series.len() == len)
}

pub(super) fn str_slice(s: &[Series]) -> PolarsResult<Series> {
    polars_ensure!(
        _ensure_lengths(s),
        ComputeError: "all series in `str_slice` should have equal or unit length",
    );
    let ca = s[0].str()?;
    let offset = &s[1];
    let length = &s[2];
    Ok(ca.str_slice(offset, length)?.into_series())
}

pub(super) fn str_head(s: &[Series]) -> PolarsResult<Series> {
    polars_ensure!(
        _ensure_lengths(s),
        ComputeError: "all series in `str_head` should have equal or unit length",
    );
    let ca = s[0].str()?;
    let n = &s[1];
    Ok(ca.str_head(n)?.into_series())
}

pub(super) fn str_tail(s: &[Series]) -> PolarsResult<Series> {
    polars_ensure!(
        _ensure_lengths(s),
        ComputeError: "all series in `str_tail` should have equal or unit length",
    );
    let ca = s[0].str()?;
    let n = &s[1];
    Ok(ca.str_tail(n)?.into_series())
}

#[cfg(feature = "string_encoding")]
pub(super) fn hex_encode(s: &Series) -> PolarsResult<Series> {
    Ok(s.str()?.hex_encode().into_series())
}

#[cfg(feature = "binary_encoding")]
pub(super) fn hex_decode(s: &Series, strict: bool) -> PolarsResult<Series> {
    s.str()?.hex_decode(strict).map(|ca| ca.into_series())
}

#[cfg(feature = "string_encoding")]
pub(super) fn base64_encode(s: &Series) -> PolarsResult<Series> {
    Ok(s.str()?.base64_encode().into_series())
}

#[cfg(feature = "binary_encoding")]
pub(super) fn base64_decode(s: &Series, strict: bool) -> PolarsResult<Series> {
    s.str()?.base64_decode(strict).map(|ca| ca.into_series())
}

#[cfg(feature = "dtype-decimal")]
pub(super) fn to_decimal(s: &Series, infer_len: usize) -> PolarsResult<Series> {
    let ca = s.str()?;
    ca.to_decimal(infer_len)
}

#[cfg(feature = "extract_jsonpath")]
pub(super) fn json_decode(
    s: &Series,
    dtype: Option<DataType>,
    infer_schema_len: Option<usize>,
) -> PolarsResult<Series> {
    let ca = s.str()?;
    ca.json_decode(dtype, infer_schema_len)
}

#[cfg(feature = "extract_jsonpath")]
pub(super) fn json_path_match(s: &[Series]) -> PolarsResult<Series> {
    let ca = s[0].str()?;
    let pat = s[1].str()?;
    Ok(ca.json_path_match(pat)?.into_series())
}
