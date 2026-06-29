#[cfg(feature = "dtype-decimal")]
use polars_compute::decimal::DEC128_MAX_PREC;
#[cfg(feature = "dtype-struct")]
use polars_utils::format_pl_smallstr;

use super::*;

#[cfg(all(feature = "regex", feature = "timezones"))]
polars_utils::regex_cache::cached_regex! {
    pub static TZ_AWARE_RE = r"(%z)|(%:z)|(%::z)|(%:::z)|(%#z)|(^%\+$)";
}

#[cfg_attr(feature = "ir_serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, PartialEq, Debug, Eq, Hash)]
pub enum IRStringFunction {
    Format {
        format: PlSmallStr,
        insertions: Arc<[usize]>,
    },
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
    ToInteger {
        dtype: Option<DataType>,
        strict: bool,
    },
    LenBytes,
    LenChars,
    Lowercase,
    #[cfg(feature = "extract_jsonpath")]
    JsonDecode(DataType),
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
        fill_char: char,
    },
    #[cfg(feature = "string_pad")]
    PadEnd {
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
    // DataType can only be Date/Datetime/Time
    Strptime(DataType, StrptimeOptions),
    Split(bool),
    #[cfg(feature = "regex")]
    SplitRegex {
        inclusive: bool,
        strict: bool,
    },
    #[cfg(feature = "dtype-decimal")]
    ToDecimal {
        scale: usize,
    },
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
        leftmost: bool,
    },
    #[cfg(feature = "find_many")]
    ExtractMany {
        ascii_case_insensitive: bool,
        overlapping: bool,
        leftmost: bool,
    },
    #[cfg(feature = "find_many")]
    FindMany {
        ascii_case_insensitive: bool,
        overlapping: bool,
        leftmost: bool,
    },
    #[cfg(feature = "regex")]
    EscapeRegex,
}

impl IRStringFunction {
    pub(super) fn get_field(&self, mapper: FieldsMapper) -> PolarsResult<Field> {
        use IRStringFunction::*;
        match self {
            Format { .. } => mapper.with_dtype(DataType::String),
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
            ToInteger { dtype, .. } => mapper.with_dtype(dtype.clone().unwrap_or(DataType::Int64)),
            #[cfg(feature = "regex")]
            Find { .. } => mapper.with_dtype(DataType::UInt32),
            #[cfg(feature = "extract_jsonpath")]
            JsonDecode(dtype) => mapper.with_dtype(dtype.clone()),
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
            Strptime(dtype, options) => match dtype {
                #[cfg(feature = "dtype-datetime")]
                DataType::Datetime(time_unit, time_zone) => {
                    let mut time_zone = time_zone.clone();
                    #[cfg(all(feature = "regex", feature = "timezones"))]
                    if options
                        .format
                        .as_ref()
                        .is_some_and(|format| TZ_AWARE_RE.is_match(format.as_str()))
                        && time_zone.is_none()
                    {
                        time_zone = Some(time_zone.unwrap_or(TimeZone::UTC));
                    }
                    mapper.with_dtype(DataType::Datetime(*time_unit, time_zone))
                },
                _ => mapper.with_dtype(dtype.clone()),
            },
            Split(_) => mapper.with_dtype(DataType::List(DataType::String.into())),
            #[cfg(feature = "regex")]
            SplitRegex { .. } => mapper.with_dtype(DataType::List(DataType::String.into())),
            #[cfg(feature = "nightly")]
            Titlecase => mapper.with_same_dtype(),
            #[cfg(feature = "dtype-decimal")]
            ToDecimal { scale } => mapper.with_dtype(DataType::Decimal(DEC128_MAX_PREC, *scale)),
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

    pub fn function_options(&self) -> FunctionOptions {
        use IRStringFunction as S;
        match self {
            S::Format { .. } => FunctionOptions::elementwise(),
            #[cfg(feature = "concat_str")]
            S::ConcatHorizontal { .. } => FunctionOptions::elementwise()
                .with_flags(|f| f | FunctionFlags::INPUT_WILDCARD_EXPANSION),
            #[cfg(feature = "concat_str")]
            S::ConcatVertical { .. } => FunctionOptions::aggregation(),
            #[cfg(feature = "regex")]
            S::Contains { .. } => {
                FunctionOptions::elementwise().with_supertyping(Default::default())
            },
            S::CountMatches(_) => FunctionOptions::elementwise(),
            S::EndsWith | S::StartsWith | S::Extract(_) => {
                FunctionOptions::elementwise().with_supertyping(Default::default())
            },
            S::ExtractAll => FunctionOptions::elementwise(),
            #[cfg(feature = "extract_groups")]
            S::ExtractGroups { .. } => FunctionOptions::elementwise(),
            #[cfg(feature = "string_to_integer")]
            S::ToInteger { .. } => FunctionOptions::elementwise(),
            #[cfg(feature = "regex")]
            S::Find { .. } => FunctionOptions::elementwise().with_supertyping(Default::default()),
            #[cfg(feature = "extract_jsonpath")]
            S::JsonDecode { .. } => FunctionOptions::elementwise(),
            #[cfg(feature = "extract_jsonpath")]
            S::JsonPathMatch => FunctionOptions::elementwise(),
            S::LenBytes | S::LenChars => FunctionOptions::elementwise(),
            #[cfg(feature = "regex")]
            S::Replace { .. } => {
                FunctionOptions::elementwise().with_supertyping(Default::default())
            },
            #[cfg(feature = "string_normalize")]
            S::Normalize { .. } => FunctionOptions::elementwise(),
            #[cfg(feature = "string_reverse")]
            S::Reverse => FunctionOptions::elementwise(),
            #[cfg(feature = "temporal")]
            S::Strptime(_, options) if options.format.is_some() => FunctionOptions::elementwise(),
            #[cfg(feature = "temporal")]
            S::Strptime(_, _) => FunctionOptions::elementwise_with_infer(),
            S::Split(_) => FunctionOptions::elementwise(),
            #[cfg(feature = "nightly")]
            S::Titlecase => FunctionOptions::elementwise(),
            #[cfg(feature = "dtype-decimal")]
            S::ToDecimal { .. } => FunctionOptions::elementwise(),
            #[cfg(feature = "string_encoding")]
            S::HexEncode | S::Base64Encode => FunctionOptions::elementwise(),
            #[cfg(feature = "binary_encoding")]
            S::HexDecode(_) | S::Base64Decode(_) => FunctionOptions::elementwise(),
            S::Uppercase | S::Lowercase => FunctionOptions::elementwise(),
            S::StripChars
            | S::StripCharsStart
            | S::StripCharsEnd
            | S::StripPrefix
            | S::StripSuffix
            | S::Head
            | S::Tail => FunctionOptions::elementwise(),
            S::Slice => FunctionOptions::elementwise(),
            #[cfg(feature = "string_pad")]
            S::PadStart { .. } | S::PadEnd { .. } | S::ZFill => FunctionOptions::elementwise(),
            #[cfg(feature = "dtype-struct")]
            S::SplitExact { .. } => FunctionOptions::elementwise(),
            #[cfg(feature = "dtype-struct")]
            S::SplitN(_) => FunctionOptions::elementwise(),
            #[cfg(feature = "regex")]
            S::SplitRegex { .. } => FunctionOptions::elementwise(),
            #[cfg(feature = "find_many")]
            S::ContainsAny { .. } => FunctionOptions::elementwise(),
            #[cfg(feature = "find_many")]
            S::ReplaceMany { .. } => FunctionOptions::elementwise(),
            #[cfg(feature = "find_many")]
            S::ExtractMany { .. } => FunctionOptions::elementwise(),
            #[cfg(feature = "find_many")]
            S::FindMany { .. } => FunctionOptions::elementwise(),
            #[cfg(feature = "regex")]
            S::EscapeRegex => FunctionOptions::elementwise(),
        }
    }
}

impl Display for IRStringFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use IRStringFunction::*;
        let s = match self {
            Format { .. } => "format",
            #[cfg(feature = "regex")]
            Contains { .. } => "contains",
            CountMatches(_) => "count_matches",
            EndsWith => "ends_with",
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
            Head => "head",
            Tail => "tail",
            #[cfg(feature = "extract_jsonpath")]
            JsonDecode(..) => "json_decode",
            #[cfg(feature = "extract_jsonpath")]
            JsonPathMatch => "json_path_match",
            LenBytes => "len_bytes",
            Lowercase => "to_lowercase",
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
            StartsWith => "starts_with",
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
            #[cfg(feature = "regex")]
            SplitRegex { inclusive, .. } => {
                if *inclusive {
                    "split_regex_inclusive"
                } else {
                    "split_regex"
                }
            },
            #[cfg(feature = "nightly")]
            Titlecase => "to_titlecase",
            #[cfg(feature = "dtype-decimal")]
            ToDecimal { .. } => "to_decimal",
            Uppercase => "to_uppercase",
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

impl From<IRStringFunction> for IRFunctionExpr {
    fn from(str: IRStringFunction) -> Self {
        IRFunctionExpr::StringExpr(str)
    }
}
