#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::*;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Clone, PartialEq, Debug, Hash)]
pub enum StringFunction {
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
    JsonDecode(DataTypeExpr),
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
    Strptime(DataTypeExpr, StrptimeOptions),
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

impl Display for StringFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use StringFunction::*;
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
            JsonDecode { .. } => "json_decode",
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

impl From<StringFunction> for FunctionExpr {
    fn from(value: StringFunction) -> Self {
        FunctionExpr::StringExpr(value)
    }
}
