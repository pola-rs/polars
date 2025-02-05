use std::borrow::Cow;

use arrow::legacy::utils::CustomIterTools;
#[cfg(feature = "timezones")]
use once_cell::sync::Lazy;
#[cfg(feature = "timezones")]
use polars_core::chunked_array::temporal::validate_time_zone;
use polars_core::utils::handle_casting_failures;
#[cfg(feature = "dtype-struct")]
use polars_utils::format_pl_smallstr;
#[cfg(feature = "regex")]
use regex::{escape, NoExpand, Regex};
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
        delimiter: PlSmallStr,
        ignore_nulls: bool,
    },
    #[cfg(feature = "concat_str")]
    ConcatVertical {
        delimiter: PlSmallStr,
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
        pat: PlSmallStr,
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
    #[cfg(feature = "string_normalize")]
    Normalize {
        form: UnicodeForm,
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
    ContainsAny {
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
    #[cfg(feature = "find_many")]
    FindMany {
        ascii_case_insensitive: bool,
        overlapping: bool,
    },
    #[cfg(feature = "regex")]
    EscapeRegex,
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
            #[cfg(feature = "string_normalize")]
            Normalize { .. } => mapper.with_same_dtype(),
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
                    .map(|i| Field::new(format_pl_smallstr!("field_{i}"), DataType::String))
                    .collect(),
            )),
            #[cfg(feature = "dtype-struct")]
            SplitN(n) => mapper.with_dtype(DataType::Struct(
                (0..*n)
                    .map(|i| Field::new(format_pl_smallstr!("field_{i}"), DataType::String))
                    .collect(),
            )),
            #[cfg(feature = "find_many")]
            ContainsAny { .. } => mapper.with_dtype(DataType::Boolean),
            #[cfg(feature = "find_many")]
            ReplaceMany { .. } => mapper.with_same_dtype(),
            #[cfg(feature = "find_many")]
            ExtractMany { .. } => mapper.with_dtype(DataType::List(Box::new(DataType::String))),
            #[cfg(feature = "find_many")]
            FindMany { .. } => mapper.with_dtype(DataType::List(Box::new(DataType::UInt32))),
            #[cfg(feature = "regex")]
            EscapeRegex => mapper.with_same_dtype(),
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
            #[cfg(feature = "string_normalize")]
            Normalize { .. } => "normalize",
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
            ContainsAny { .. } => "contains_any",
            #[cfg(feature = "find_many")]
            ReplaceMany { .. } => "replace_many",
            #[cfg(feature = "find_many")]
            ExtractMany { .. } => "extract_many",
            #[cfg(feature = "find_many")]
            FindMany { .. } => "extract_many",
            #[cfg(feature = "regex")]
            EscapeRegex => "escape_regex",
        };
        write!(f, "str.{s}")
    }
}

impl From<StringFunction> for SpecialEq<Arc<dyn ColumnsUdf>> {
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
            #[cfg(feature = "string_normalize")]
            Normalize { form } => map!(strings::normalize, form.clone()),
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
            ContainsAny {
                ascii_case_insensitive,
            } => {
                map_as_slice!(contains_any, ascii_case_insensitive)
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
            #[cfg(feature = "find_many")]
            FindMany {
                ascii_case_insensitive,
                overlapping,
            } => {
                map_as_slice!(find_many, ascii_case_insensitive, overlapping)
            },
            #[cfg(feature = "regex")]
            EscapeRegex => map!(escape_regex),
        }
    }
}

#[cfg(feature = "find_many")]
fn contains_any(s: &[Column], ascii_case_insensitive: bool) -> PolarsResult<Column> {
    let ca = s[0].str()?;
    let patterns = s[1].str()?;
    polars_ops::chunked_array::strings::contains_any(ca, patterns, ascii_case_insensitive)
        .map(|out| out.into_column())
}

#[cfg(feature = "find_many")]
fn replace_many(s: &[Column], ascii_case_insensitive: bool) -> PolarsResult<Column> {
    let ca = s[0].str()?;
    let patterns = s[1].str()?;
    let replace_with = s[2].str()?;
    polars_ops::chunked_array::strings::replace_all(
        ca,
        patterns,
        replace_with,
        ascii_case_insensitive,
    )
    .map(|out| out.into_column())
}

#[cfg(feature = "find_many")]
fn extract_many(
    s: &[Column],
    ascii_case_insensitive: bool,
    overlapping: bool,
) -> PolarsResult<Column> {
    let ca = s[0].str()?;
    let patterns = &s[1];

    polars_ops::chunked_array::strings::extract_many(
        ca,
        patterns.as_materialized_series(),
        ascii_case_insensitive,
        overlapping,
    )
    .map(|out| out.into_column())
}

#[cfg(feature = "find_many")]
fn find_many(
    s: &[Column],
    ascii_case_insensitive: bool,
    overlapping: bool,
) -> PolarsResult<Column> {
    let ca = s[0].str()?;
    let patterns = &s[1];

    polars_ops::chunked_array::strings::find_many(
        ca,
        patterns.as_materialized_series(),
        ascii_case_insensitive,
        overlapping,
    )
    .map(|out| out.into_column())
}

fn uppercase(s: &Column) -> PolarsResult<Column> {
    let ca = s.str()?;
    Ok(ca.to_uppercase().into_column())
}

fn lowercase(s: &Column) -> PolarsResult<Column> {
    let ca = s.str()?;
    Ok(ca.to_lowercase().into_column())
}

#[cfg(feature = "nightly")]
pub(super) fn titlecase(s: &Column) -> PolarsResult<Column> {
    let ca = s.str()?;
    Ok(ca.to_titlecase().into_column())
}

pub(super) fn len_chars(s: &Column) -> PolarsResult<Column> {
    let ca = s.str()?;
    Ok(ca.str_len_chars().into_column())
}

pub(super) fn len_bytes(s: &Column) -> PolarsResult<Column> {
    let ca = s.str()?;
    Ok(ca.str_len_bytes().into_column())
}

#[cfg(feature = "regex")]
pub(super) fn contains(s: &[Column], literal: bool, strict: bool) -> PolarsResult<Column> {
    _check_same_length(s, "contains")?;
    let ca = s[0].str()?;
    let pat = s[1].str()?;
    ca.contains_chunked(pat, literal, strict)
        .map(|ok| ok.into_column())
}

#[cfg(feature = "regex")]
pub(super) fn find(s: &[Column], literal: bool, strict: bool) -> PolarsResult<Column> {
    _check_same_length(s, "find")?;
    let ca = s[0].str()?;
    let pat = s[1].str()?;
    ca.find_chunked(pat, literal, strict)
        .map(|ok| ok.into_column())
}

pub(super) fn ends_with(s: &[Column]) -> PolarsResult<Column> {
    _check_same_length(s, "ends_with")?;
    let ca = &s[0].str()?.as_binary();
    let suffix = &s[1].str()?.as_binary();

    Ok(ca.ends_with_chunked(suffix).into_column())
}

pub(super) fn starts_with(s: &[Column]) -> PolarsResult<Column> {
    _check_same_length(s, "starts_with")?;
    let ca = s[0].str()?;
    let prefix = s[1].str()?;
    Ok(ca.starts_with_chunked(prefix).into_column())
}

/// Extract a regex pattern from the a string value.
pub(super) fn extract(s: &[Column], group_index: usize) -> PolarsResult<Column> {
    let ca = s[0].str()?;
    let pat = s[1].str()?;
    ca.extract(pat, group_index).map(|ca| ca.into_column())
}

#[cfg(feature = "extract_groups")]
/// Extract all capture groups from a regex pattern as a struct
pub(super) fn extract_groups(s: &Column, pat: &str, dtype: &DataType) -> PolarsResult<Column> {
    let ca = s.str()?;
    ca.extract_groups(pat, dtype).map(Column::from)
}

#[cfg(feature = "string_pad")]
pub(super) fn pad_start(s: &Column, length: usize, fill_char: char) -> PolarsResult<Column> {
    let ca = s.str()?;
    Ok(ca.pad_start(length, fill_char).into_column())
}

#[cfg(feature = "string_pad")]
pub(super) fn pad_end(s: &Column, length: usize, fill_char: char) -> PolarsResult<Column> {
    let ca = s.str()?;
    Ok(ca.pad_end(length, fill_char).into_column())
}

#[cfg(feature = "string_pad")]
pub(super) fn zfill(s: &[Column]) -> PolarsResult<Column> {
    _check_same_length(s, "zfill")?;
    let ca = s[0].str()?;
    let length_s = s[1].strict_cast(&DataType::UInt64)?;
    let length = length_s.u64()?;
    Ok(ca.zfill(length).into_column())
}

pub(super) fn strip_chars(s: &[Column]) -> PolarsResult<Column> {
    _check_same_length(s, "strip_chars")?;
    let ca = s[0].str()?;
    let pat_s = &s[1];
    ca.strip_chars(pat_s).map(|ok| ok.into_column())
}

pub(super) fn strip_chars_start(s: &[Column]) -> PolarsResult<Column> {
    _check_same_length(s, "strip_chars_start")?;
    let ca = s[0].str()?;
    let pat_s = &s[1];
    ca.strip_chars_start(pat_s).map(|ok| ok.into_column())
}

pub(super) fn strip_chars_end(s: &[Column]) -> PolarsResult<Column> {
    _check_same_length(s, "strip_chars_end")?;
    let ca = s[0].str()?;
    let pat_s = &s[1];
    ca.strip_chars_end(pat_s).map(|ok| ok.into_column())
}

pub(super) fn strip_prefix(s: &[Column]) -> PolarsResult<Column> {
    _check_same_length(s, "strip_prefix")?;
    let ca = s[0].str()?;
    let prefix = s[1].str()?;
    Ok(ca.strip_prefix(prefix).into_column())
}

pub(super) fn strip_suffix(s: &[Column]) -> PolarsResult<Column> {
    _check_same_length(s, "strip_suffix")?;
    let ca = s[0].str()?;
    let suffix = s[1].str()?;
    Ok(ca.strip_suffix(suffix).into_column())
}

pub(super) fn extract_all(args: &[Column]) -> PolarsResult<Column> {
    let s = &args[0];
    let pat = &args[1];

    let ca = s.str()?;
    let pat = pat.str()?;

    if pat.len() == 1 {
        if let Some(pat) = pat.get(0) {
            ca.extract_all(pat).map(|ca| ca.into_column())
        } else {
            Ok(Column::full_null(
                ca.name().clone(),
                ca.len(),
                &DataType::List(Box::new(DataType::String)),
            ))
        }
    } else {
        ca.extract_all_many(pat).map(|ca| ca.into_column())
    }
}

pub(super) fn count_matches(args: &[Column], literal: bool) -> PolarsResult<Column> {
    let s = &args[0];
    let pat = &args[1];

    let ca = s.str()?;
    let pat = pat.str()?;
    if pat.len() == 1 {
        if let Some(pat) = pat.get(0) {
            ca.count_matches(pat, literal).map(|ca| ca.into_column())
        } else {
            Ok(Column::full_null(
                ca.name().clone(),
                ca.len(),
                &DataType::UInt32,
            ))
        }
    } else {
        ca.count_matches_many(pat, literal)
            .map(|ca| ca.into_column())
    }
}

#[cfg(feature = "temporal")]
pub(super) fn strptime(
    s: &[Column],
    dtype: DataType,
    options: &StrptimeOptions,
) -> PolarsResult<Column> {
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
pub(super) fn split_exact(s: &[Column], n: usize, inclusive: bool) -> PolarsResult<Column> {
    let ca = s[0].str()?;
    let by = s[1].str()?;

    if inclusive {
        ca.split_exact_inclusive(by, n).map(|ca| ca.into_column())
    } else {
        ca.split_exact(by, n).map(|ca| ca.into_column())
    }
}

#[cfg(feature = "dtype-struct")]
pub(super) fn splitn(s: &[Column], n: usize) -> PolarsResult<Column> {
    let ca = s[0].str()?;
    let by = s[1].str()?;

    ca.splitn(by, n).map(|ca| ca.into_column())
}

pub(super) fn split(s: &[Column], inclusive: bool) -> PolarsResult<Column> {
    let ca = s[0].str()?;
    let by = s[1].str()?;

    if inclusive {
        Ok(ca.split_inclusive(by).into_column())
    } else {
        Ok(ca.split(by).into_column())
    }
}

#[cfg(feature = "dtype-date")]
fn to_date(s: &Column, options: &StrptimeOptions) -> PolarsResult<Column> {
    let ca = s.str()?;
    let out = {
        if options.exact {
            ca.as_date(options.format.as_deref(), options.cache)?
                .into_column()
        } else {
            ca.as_date_not_exact(options.format.as_deref())?
                .into_column()
        }
    };

    if options.strict && ca.null_count() != out.null_count() {
        handle_casting_failures(s.as_materialized_series(), out.as_materialized_series())?;
    }
    Ok(out.into_column())
}

#[cfg(feature = "dtype-datetime")]
fn to_datetime(
    s: &[Column],
    time_unit: &TimeUnit,
    time_zone: Option<&TimeZone>,
    options: &StrptimeOptions,
) -> PolarsResult<Column> {
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
            .into_column()
    } else {
        datetime_strings
            .as_datetime_not_exact(
                options.format.as_deref(),
                *time_unit,
                tz_aware,
                time_zone,
                ambiguous,
            )?
            .into_column()
    };

    if options.strict && datetime_strings.null_count() != out.null_count() {
        handle_casting_failures(s[0].as_materialized_series(), out.as_materialized_series())?;
    }
    Ok(out.into_column())
}

#[cfg(feature = "dtype-time")]
fn to_time(s: &Column, options: &StrptimeOptions) -> PolarsResult<Column> {
    polars_ensure!(
        options.exact, ComputeError: "non-exact not implemented for Time data type"
    );

    let ca = s.str()?;
    let out = ca
        .as_time(options.format.as_deref(), options.cache)?
        .into_column();

    if options.strict && ca.null_count() != out.null_count() {
        handle_casting_failures(s.as_materialized_series(), out.as_materialized_series())?;
    }
    Ok(out.into_column())
}

#[cfg(feature = "concat_str")]
pub(super) fn join(s: &Column, delimiter: &str, ignore_nulls: bool) -> PolarsResult<Column> {
    let str_s = s.cast(&DataType::String)?;
    let joined = polars_ops::chunked_array::str_join(str_s.str()?, delimiter, ignore_nulls);
    Ok(joined.into_column())
}

#[cfg(feature = "concat_str")]
pub(super) fn concat_hor(
    series: &[Column],
    delimiter: &str,
    ignore_nulls: bool,
) -> PolarsResult<Column> {
    let str_series: Vec<_> = series
        .iter()
        .map(|s| s.cast(&DataType::String))
        .collect::<PolarsResult<_>>()?;
    let cas: Vec<_> = str_series.iter().map(|s| s.str().unwrap()).collect();
    Ok(polars_ops::chunked_array::hor_str_concat(&cas, delimiter, ignore_nulls)?.into_column())
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

    out.rename(ca.name().clone());
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
            let lit = is_literal_pat(&pat);
            let literal_pat = literal || lit;

            if literal_pat {
                pat = escape(&pat)
            }

            let reg = Regex::new(&pat)?;

            let f = |s: &'a str, val: &'a str| {
                if lit && (s.len() <= 32) {
                    Cow::Owned(s.replacen(&pat, val, 1))
                } else {
                    // According to the docs for replace
                    // when literal = True then capture groups are ignored.
                    if literal {
                        reg.replace(s, NoExpand(val))
                    } else {
                        reg.replace(s, val)
                    }
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

            let literal_pat = literal || is_literal_pat(&pat);

            if literal_pat {
                pat = escape(&pat)
            }

            let reg = Regex::new(&pat)?;

            let f = |s: &'a str, val: &'a str| {
                // According to the docs for replace_all
                // when literal = True then capture groups are ignored.
                if literal {
                    reg.replace_all(s, NoExpand(val))
                } else {
                    reg.replace_all(s, val)
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
pub(super) fn replace(s: &[Column], literal: bool, n: i64) -> PolarsResult<Column> {
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
    .map(|ca| ca.into_column())
}

#[cfg(feature = "string_normalize")]
pub(super) fn normalize(s: &Column, form: UnicodeForm) -> PolarsResult<Column> {
    let ca = s.str()?;
    Ok(ca.str_normalize(form).into_column())
}

#[cfg(feature = "string_reverse")]
pub(super) fn reverse(s: &Column) -> PolarsResult<Column> {
    let ca = s.str()?;
    Ok(ca.str_reverse().into_column())
}

#[cfg(feature = "string_to_integer")]
pub(super) fn to_integer(s: &[Column], strict: bool) -> PolarsResult<Column> {
    let ca = s[0].str()?;
    let base = s[1].strict_cast(&DataType::UInt32)?;
    ca.to_integer(base.u32()?, strict)
        .map(|ok| ok.into_column())
}

fn _ensure_lengths(s: &[Column]) -> bool {
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

fn _check_same_length(s: &[Column], fn_name: &str) -> Result<(), PolarsError> {
    polars_ensure!(
        _ensure_lengths(s),
        ComputeError: "all series in `str.{}()` should have equal or unit length",
        fn_name
    );
    Ok(())
}

pub(super) fn str_slice(s: &[Column]) -> PolarsResult<Column> {
    _check_same_length(s, "slice")?;
    let ca = s[0].str()?;
    let offset = &s[1];
    let length = &s[2];
    Ok(ca.str_slice(offset, length)?.into_column())
}

pub(super) fn str_head(s: &[Column]) -> PolarsResult<Column> {
    _check_same_length(s, "head")?;
    let ca = s[0].str()?;
    let n = &s[1];
    Ok(ca.str_head(n)?.into_column())
}

pub(super) fn str_tail(s: &[Column]) -> PolarsResult<Column> {
    _check_same_length(s, "tail")?;
    let ca = s[0].str()?;
    let n = &s[1];
    Ok(ca.str_tail(n)?.into_column())
}

#[cfg(feature = "string_encoding")]
pub(super) fn hex_encode(s: &Column) -> PolarsResult<Column> {
    Ok(s.str()?.hex_encode().into_column())
}

#[cfg(feature = "binary_encoding")]
pub(super) fn hex_decode(s: &Column, strict: bool) -> PolarsResult<Column> {
    s.str()?.hex_decode(strict).map(|ca| ca.into_column())
}

#[cfg(feature = "string_encoding")]
pub(super) fn base64_encode(s: &Column) -> PolarsResult<Column> {
    Ok(s.str()?.base64_encode().into_column())
}

#[cfg(feature = "binary_encoding")]
pub(super) fn base64_decode(s: &Column, strict: bool) -> PolarsResult<Column> {
    s.str()?.base64_decode(strict).map(|ca| ca.into_column())
}

#[cfg(feature = "dtype-decimal")]
pub(super) fn to_decimal(s: &Column, infer_len: usize) -> PolarsResult<Column> {
    let ca = s.str()?;
    ca.to_decimal(infer_len).map(Column::from)
}

#[cfg(feature = "extract_jsonpath")]
pub(super) fn json_decode(
    s: &Column,
    dtype: Option<DataType>,
    infer_schema_len: Option<usize>,
) -> PolarsResult<Column> {
    let ca = s.str()?;
    ca.json_decode(dtype, infer_schema_len).map(Column::from)
}

#[cfg(feature = "extract_jsonpath")]
pub(super) fn json_path_match(s: &[Column]) -> PolarsResult<Column> {
    _check_same_length(s, "json_path_match")?;
    let ca = s[0].str()?;
    let pat = s[1].str()?;
    Ok(ca.json_path_match(pat)?.into_column())
}

#[cfg(feature = "regex")]
pub(super) fn escape_regex(s: &Column) -> PolarsResult<Column> {
    let ca = s.str()?;
    Ok(ca.str_escape_regex().into_column())
}
