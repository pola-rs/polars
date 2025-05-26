use super::*;
/// Specialized expressions for [`Series`] of [`DataType::String`].
pub struct StringNameSpace(pub(crate) Expr);

impl StringNameSpace {
    /// Check if a string value contains a literal substring.
    #[cfg(feature = "regex")]
    pub fn contains_literal(self, pat: Expr) -> Expr {
        self.0.map_binary(
            StringFunction::Contains {
                literal: true,
                strict: false,
            },
            pat,
        )
    }

    /// Check if this column of strings contains a Regex. If `strict` is `true`, then it is an error if any `pat` is
    /// an invalid regex, whereas if `strict` is `false`, an invalid regex will simply evaluate to `false`.
    #[cfg(feature = "regex")]
    pub fn contains(self, pat: Expr, strict: bool) -> Expr {
        self.0.map_binary(
            StringFunction::Contains {
                literal: false,
                strict,
            },
            pat,
        )
    }

    /// Uses aho-corasick to find many patterns.
    ///
    /// # Arguments
    /// - `patterns`: an expression that evaluates to a String column
    /// - `ascii_case_insensitive`: Enable ASCII-aware case insensitive matching.
    ///   When this option is enabled, searching will be performed without respect to case for
    ///   ASCII letters (a-z and A-Z) only.
    #[cfg(feature = "find_many")]
    pub fn contains_any(self, patterns: Expr, ascii_case_insensitive: bool) -> Expr {
        self.0.map_binary(
            StringFunction::ContainsAny {
                ascii_case_insensitive,
            },
            patterns,
        )
    }

    /// Uses aho-corasick to replace many patterns.
    /// # Arguments
    /// - `patterns`: an expression that evaluates to a String column
    /// - `replace_with`: an expression that evaluates to a String column
    /// - `ascii_case_insensitive`: Enable ASCII-aware case-insensitive matching.
    ///   When this option is enabled, searching will be performed without respect to case for
    ///   ASCII letters (a-z and A-Z) only.
    #[cfg(feature = "find_many")]
    pub fn replace_many(
        self,
        patterns: Expr,
        replace_with: Expr,
        ascii_case_insensitive: bool,
    ) -> Expr {
        self.0.map_ternary(
            StringFunction::ReplaceMany {
                ascii_case_insensitive,
            },
            patterns,
            replace_with,
        )
    }

    /// Uses aho-corasick to replace many patterns.
    /// # Arguments
    /// - `patterns`: an expression that evaluates to a String column
    /// - `ascii_case_insensitive`: Enable ASCII-aware case-insensitive matching.
    ///   When this option is enabled, searching will be performed without respect to case for
    ///   ASCII letters (a-z and A-Z) only.
    /// - `overlapping`: Whether matches may overlap.
    #[cfg(feature = "find_many")]
    pub fn extract_many(
        self,
        patterns: Expr,
        ascii_case_insensitive: bool,
        overlapping: bool,
    ) -> Expr {
        self.0.map_binary(
            StringFunction::ExtractMany {
                ascii_case_insensitive,
                overlapping,
            },
            patterns,
        )
    }

    /// Uses aho-corasick to find many patterns.
    /// # Arguments
    /// - `patterns`: an expression that evaluates to a String column
    /// - `ascii_case_insensitive`: Enable ASCII-aware case-insensitive matching.
    ///   When this option is enabled, searching will be performed without respect to case for
    ///   ASCII letters (a-z and A-Z) only.
    /// - `overlapping`: Whether matches may overlap.
    #[cfg(feature = "find_many")]
    pub fn find_many(
        self,
        patterns: Expr,
        ascii_case_insensitive: bool,
        overlapping: bool,
    ) -> Expr {
        self.0.map_binary(
            StringFunction::FindMany {
                ascii_case_insensitive,
                overlapping,
            },
            patterns,
        )
    }

    /// Check if a string value ends with the `sub` string.
    pub fn ends_with(self, sub: Expr) -> Expr {
        self.0.map_binary(StringFunction::EndsWith, sub)
    }

    /// Check if a string value starts with the `sub` string.
    pub fn starts_with(self, sub: Expr) -> Expr {
        self.0.map_binary(StringFunction::StartsWith, sub)
    }

    #[cfg(feature = "string_encoding")]
    pub fn hex_encode(self) -> Expr {
        self.0.map_unary(StringFunction::HexEncode)
    }

    #[cfg(feature = "binary_encoding")]
    pub fn hex_decode(self, strict: bool) -> Expr {
        self.0.map_unary(StringFunction::HexDecode(strict))
    }

    #[cfg(feature = "string_encoding")]
    pub fn base64_encode(self) -> Expr {
        self.0.map_unary(StringFunction::Base64Encode)
    }

    #[cfg(feature = "binary_encoding")]
    pub fn base64_decode(self, strict: bool) -> Expr {
        self.0.map_unary(StringFunction::Base64Decode(strict))
    }

    /// Extract a regex pattern from the a string value. If `group_index` is out of bounds, null is returned.
    pub fn extract(self, pat: Expr, group_index: usize) -> Expr {
        self.0.map_binary(StringFunction::Extract(group_index), pat)
    }

    #[cfg(feature = "extract_groups")]
    // Extract all captures groups from a regex pattern as a struct
    pub fn extract_groups(self, pat: &str) -> PolarsResult<Expr> {
        // regex will be compiled twice, because it doesn't support serde
        // and we need to compile it here to determine the output datatype

        use polars_utils::format_pl_smallstr;
        let reg = polars_utils::regex_cache::compile_regex(pat)?;
        let names = reg
            .capture_names()
            .enumerate()
            .skip(1)
            .map(|(idx, opt_name)| {
                opt_name
                    .map(PlSmallStr::from_str)
                    .unwrap_or_else(|| format_pl_smallstr!("{idx}"))
            })
            .collect::<Vec<_>>();

        let dtype = DataType::Struct(
            names
                .iter()
                .map(|name| Field::new(name.clone(), DataType::String))
                .collect(),
        );

        Ok(self.0.map_unary(StringFunction::ExtractGroups {
            dtype,
            pat: pat.into(),
        }))
    }

    /// Pad the start of the string until it reaches the given length.
    ///
    /// Padding is done using the specified `fill_char`.
    /// Strings with length equal to or greater than the given length are
    /// returned as-is.
    #[cfg(feature = "string_pad")]
    pub fn pad_start(self, length: usize, fill_char: char) -> Expr {
        self.0
            .map_unary(StringFunction::PadStart { length, fill_char })
    }

    /// Pad the end of the string until it reaches the given length.
    ///
    /// Padding is done using the specified `fill_char`.
    /// Strings with length equal to or greater than the given length are
    /// returned as-is.
    #[cfg(feature = "string_pad")]
    pub fn pad_end(self, length: usize, fill_char: char) -> Expr {
        self.0
            .map_unary(StringFunction::PadEnd { length, fill_char })
    }

    /// Pad the start of the string with zeros until it reaches the given length.
    ///
    /// A sign prefix (`-`) is handled by inserting the padding after the sign
    /// character rather than before.
    /// Strings with length equal to or greater than the given length are
    /// returned as-is.
    #[cfg(feature = "string_pad")]
    pub fn zfill(self, length: Expr) -> Expr {
        self.0.map_binary(StringFunction::ZFill, length)
    }

    /// Find the index of a literal substring within another string value.
    #[cfg(feature = "regex")]
    pub fn find_literal(self, pat: Expr) -> Expr {
        self.0.map_binary(
            StringFunction::Find {
                literal: true,
                strict: false,
            },
            pat,
        )
    }

    /// Find the index of a substring defined by a regular expressions within another string value.
    #[cfg(feature = "regex")]
    pub fn find(self, pat: Expr, strict: bool) -> Expr {
        self.0.map_binary(
            StringFunction::Find {
                literal: false,
                strict,
            },
            pat,
        )
    }

    /// Extract each successive non-overlapping match in an individual string as an array
    pub fn extract_all(self, pat: Expr) -> Expr {
        self.0.map_binary(StringFunction::ExtractAll, pat)
    }

    /// Count all successive non-overlapping regex matches.
    pub fn count_matches(self, pat: Expr, literal: bool) -> Expr {
        self.0
            .map_binary(StringFunction::CountMatches(literal), pat)
    }

    /// Convert a String column into a Date/Datetime/Time column.
    #[cfg(feature = "temporal")]
    pub fn strptime(self, dtype: DataType, options: StrptimeOptions, ambiguous: Expr) -> Expr {
        let is_column_independent = is_column_independent(&self.0);
        // Only elementwise if the format is explicitly set, or we're constant.
        self.0
            .map_binary(StringFunction::Strptime(dtype, options), ambiguous)
            .with_function_options(|mut options| {
                // @HACK. This needs to be done because literals still block predicate pushdown,
                // but this should be an exception in the predicate pushdown.
                if is_column_independent {
                    options.set_elementwise();
                }
                options
            })
    }

    /// Convert a String column into a Date column.
    #[cfg(feature = "dtype-date")]
    pub fn to_date(self, options: StrptimeOptions) -> Expr {
        self.strptime(DataType::Date, options, lit("raise"))
    }

    /// Convert a String column into a Datetime column.
    #[cfg(feature = "dtype-datetime")]
    pub fn to_datetime(
        self,
        time_unit: Option<TimeUnit>,
        time_zone: Option<TimeZone>,
        options: StrptimeOptions,
        ambiguous: Expr,
    ) -> Expr {
        // If time_unit is None, try to infer it from the format or set a default
        let time_unit = match (&options.format, time_unit) {
            (_, Some(time_unit)) => time_unit,
            (Some(format), None) => {
                if format.contains("%.9f") || format.contains("%9f") {
                    TimeUnit::Nanoseconds
                } else if format.contains("%.3f") || format.contains("%3f") {
                    TimeUnit::Milliseconds
                } else {
                    TimeUnit::Microseconds
                }
            },
            (None, None) => TimeUnit::Microseconds,
        };

        self.strptime(DataType::Datetime(time_unit, time_zone), options, ambiguous)
    }

    /// Convert a String column into a Time column.
    #[cfg(feature = "dtype-time")]
    pub fn to_time(self, options: StrptimeOptions) -> Expr {
        self.strptime(DataType::Time, options, lit("raise"))
    }

    /// Convert a String column into a Decimal column.
    #[cfg(feature = "dtype-decimal")]
    pub fn to_decimal(self, infer_length: usize) -> Expr {
        self.0.map_unary(StringFunction::ToDecimal(infer_length))
    }

    /// Concat the values into a string array.
    /// # Arguments
    ///
    /// * `delimiter` - A string that will act as delimiter between values.
    #[cfg(feature = "concat_str")]
    pub fn join(self, delimiter: &str, ignore_nulls: bool) -> Expr {
        self.0.map_unary(StringFunction::ConcatVertical {
            delimiter: delimiter.into(),
            ignore_nulls,
        })
    }

    /// Split the string by a substring. The resulting dtype is `List<String>`.
    pub fn split(self, by: Expr) -> Expr {
        self.0.map_binary(StringFunction::Split(false), by)
    }

    /// Split the string by a substring and keep the substring. The resulting dtype is `List<String>`.
    pub fn split_inclusive(self, by: Expr) -> Expr {
        self.0.map_binary(StringFunction::Split(true), by)
    }

    #[cfg(feature = "dtype-struct")]
    /// Split exactly `n` times by a given substring. The resulting dtype is [`DataType::Struct`].
    pub fn split_exact(self, by: Expr, n: usize) -> Expr {
        self.0.map_binary(
            StringFunction::SplitExact {
                n,
                inclusive: false,
            },
            by,
        )
    }

    #[cfg(feature = "dtype-struct")]
    /// Split exactly `n` times by a given substring and keep the substring.
    /// The resulting dtype is [`DataType::Struct`].
    pub fn split_exact_inclusive(self, by: Expr, n: usize) -> Expr {
        self.0
            .map_binary(StringFunction::SplitExact { n, inclusive: true }, by)
    }

    #[cfg(feature = "dtype-struct")]
    /// Split by a given substring, returning exactly `n` items. If there are more possible splits,
    /// keeps the remainder of the string intact. The resulting dtype is [`DataType::Struct`].
    pub fn splitn(self, by: Expr, n: usize) -> Expr {
        self.0.map_binary(StringFunction::SplitN(n), by)
    }

    #[cfg(feature = "regex")]
    /// Replace values that match a regex `pat` with a `value`.
    pub fn replace(self, pat: Expr, value: Expr, literal: bool) -> Expr {
        self.0
            .map_ternary(StringFunction::Replace { n: 1, literal }, pat, value)
    }

    #[cfg(feature = "regex")]
    /// Replace values that match a regex `pat` with a `value`.
    pub fn replace_n(self, pat: Expr, value: Expr, literal: bool, n: i64) -> Expr {
        self.0
            .map_ternary(StringFunction::Replace { n, literal }, pat, value)
    }

    #[cfg(feature = "regex")]
    /// Replace all values that match a regex `pat` with a `value`.
    pub fn replace_all(self, pat: Expr, value: Expr, literal: bool) -> Expr {
        self.0
            .map_ternary(StringFunction::Replace { n: -1, literal }, pat, value)
    }

    #[cfg(feature = "string_normalize")]
    /// Normalize each string
    pub fn normalize(self, form: UnicodeForm) -> Expr {
        self.0.map_unary(StringFunction::Normalize { form })
    }

    #[cfg(feature = "string_reverse")]
    /// Reverse each string
    pub fn reverse(self) -> Expr {
        self.0.map_unary(StringFunction::Reverse)
    }

    /// Remove leading and trailing characters, or whitespace if matches is None.
    pub fn strip_chars(self, matches: Expr) -> Expr {
        self.0.map_binary(StringFunction::StripChars, matches)
    }

    /// Remove leading characters, or whitespace if matches is None.
    pub fn strip_chars_start(self, matches: Expr) -> Expr {
        self.0.map_binary(StringFunction::StripCharsStart, matches)
    }

    /// Remove trailing characters, or whitespace if matches is None.
    pub fn strip_chars_end(self, matches: Expr) -> Expr {
        self.0.map_binary(StringFunction::StripCharsEnd, matches)
    }

    /// Remove prefix.
    pub fn strip_prefix(self, prefix: Expr) -> Expr {
        self.0.map_binary(StringFunction::StripPrefix, prefix)
    }

    /// Remove suffix.
    pub fn strip_suffix(self, suffix: Expr) -> Expr {
        self.0.map_binary(StringFunction::StripSuffix, suffix)
    }

    /// Convert all characters to lowercase.
    pub fn to_lowercase(self) -> Expr {
        self.0.map_unary(StringFunction::Lowercase)
    }

    /// Convert all characters to uppercase.
    pub fn to_uppercase(self) -> Expr {
        self.0.map_unary(StringFunction::Uppercase)
    }

    /// Convert all characters to titlecase.
    #[cfg(feature = "nightly")]
    pub fn to_titlecase(self) -> Expr {
        self.0.map_unary(StringFunction::Titlecase)
    }

    #[cfg(feature = "string_to_integer")]
    /// Parse string in base radix into decimal.
    pub fn to_integer(self, base: Expr, strict: bool) -> Expr {
        self.0.map_binary(StringFunction::ToInteger(strict), base)
    }

    /// Return the length of each string as the number of bytes.
    ///
    /// When working with non-ASCII text, the length in bytes is not the same
    /// as the length in characters. You may want to use
    /// [`len_chars`] instead. Note that `len_bytes` is much more
    /// performant (_O(1)_) than [`len_chars`] (_O(n)_).
    ///
    /// [`len_chars`]: StringNameSpace::len_chars
    pub fn len_bytes(self) -> Expr {
        self.0.map_unary(StringFunction::LenBytes)
    }

    /// Return the length of each string as the number of characters.
    ///
    /// When working with ASCII text, use [`len_bytes`] instead to achieve
    /// equivalent output with much better performance:
    /// [`len_bytes`] runs in _O(1)_, while `len_chars` runs in _O(n)_.
    ///
    /// [`len_bytes`]: StringNameSpace::len_bytes
    pub fn len_chars(self) -> Expr {
        self.0.map_unary(StringFunction::LenChars)
    }

    /// Slice the string values.
    pub fn slice(self, offset: Expr, length: Expr) -> Expr {
        self.0.map_ternary(StringFunction::Slice, offset, length)
    }

    /// Take the first `n` characters of the string values.
    pub fn head(self, n: Expr) -> Expr {
        self.0.map_binary(StringFunction::Head, n)
    }

    /// Take the last `n` characters of the string values.
    pub fn tail(self, n: Expr) -> Expr {
        self.0.map_binary(StringFunction::Tail, n)
    }

    #[cfg(feature = "extract_jsonpath")]
    pub fn json_decode(self, dtype: Option<DataType>, infer_schema_len: Option<usize>) -> Expr {
        self.0.map_unary(StringFunction::JsonDecode {
            dtype,
            infer_schema_len,
        })
    }

    #[cfg(feature = "extract_jsonpath")]
    pub fn json_path_match(self, pat: Expr) -> Expr {
        self.0.map_binary(StringFunction::JsonPathMatch, pat)
    }

    #[cfg(feature = "regex")]
    pub fn escape_regex(self) -> Expr {
        self.0.map_unary(StringFunction::EscapeRegex)
    }
}
