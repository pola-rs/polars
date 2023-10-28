use super::function_expr::StringFunction;
use super::*;
/// Specialized expressions for [`Series`] of [`DataType::Utf8`].
pub struct StringNameSpace(pub(crate) Expr);

impl StringNameSpace {
    /// Check if a string value contains a literal substring.
    #[cfg(feature = "regex")]
    pub fn contains_literal(self, pat: Expr) -> Expr {
        self.0.map_many_private(
            FunctionExpr::StringExpr(StringFunction::Contains {
                literal: true,
                strict: false,
            }),
            &[pat],
            false,
            true,
        )
    }

    /// Check if this column of strings contains a Regex. If `strict` is `true`, then it is an error if any `pat` is
    /// an invalid regex, whereas if `strict` is `false`, an invalid regex will simply evaluate to `false`.
    #[cfg(feature = "regex")]
    pub fn contains(self, pat: Expr, strict: bool) -> Expr {
        self.0.map_many_private(
            FunctionExpr::StringExpr(StringFunction::Contains {
                literal: false,
                strict,
            }),
            &[pat],
            false,
            true,
        )
    }

    /// Check if a string value ends with the `sub` string.
    pub fn ends_with(self, sub: Expr) -> Expr {
        self.0.map_many_private(
            FunctionExpr::StringExpr(StringFunction::EndsWith),
            &[sub],
            false,
            true,
        )
    }

    /// Check if a string value starts with the `sub` string.
    pub fn starts_with(self, sub: Expr) -> Expr {
        self.0.map_many_private(
            FunctionExpr::StringExpr(StringFunction::StartsWith),
            &[sub],
            false,
            true,
        )
    }

    /// Extract a regex pattern from the a string value. If `group_index` is out of bounds, null is returned.
    pub fn extract(self, pat: &str, group_index: usize) -> Expr {
        let pat = pat.to_string();
        self.0
            .map_private(StringFunction::Extract { pat, group_index }.into())
    }

    #[cfg(feature = "extract_groups")]
    // Extract all captures groups from a regex pattern as a struct
    pub fn extract_groups(self, pat: &str) -> PolarsResult<Expr> {
        // regex will be compiled twice, because it doesn't support serde
        // and we need to compile it here to determine the output datatype
        let reg = regex::Regex::new(pat)?;
        let names = reg
            .capture_names()
            .enumerate()
            .skip(1)
            .map(|(idx, opt_name)| {
                opt_name
                    .map(|name| name.to_string())
                    .unwrap_or_else(|| format!("{idx}"))
            })
            .collect::<Vec<_>>();

        let dtype = DataType::Struct(
            names
                .iter()
                .map(|name| Field::new(name.as_str(), DataType::Utf8))
                .collect(),
        );

        Ok(self.0.map_private(
            StringFunction::ExtractGroups {
                dtype,
                pat: pat.to_string(),
            }
            .into(),
        ))
    }

    /// Pad the start of the string until it reaches the given length.
    ///
    /// Padding is done using the specified `fill_char`.
    /// Strings with length equal to or greater than the given length are
    /// returned as-is.
    #[cfg(feature = "string_pad")]
    pub fn pad_start(self, length: usize, fill_char: char) -> Expr {
        self.0
            .map_private(StringFunction::PadStart { length, fill_char }.into())
    }

    /// Pad the end of the string until it reaches the given length.
    ///
    /// Padding is done using the specified `fill_char`.
    /// Strings with length equal to or greater than the given length are
    /// returned as-is.
    #[cfg(feature = "string_pad")]
    pub fn pad_end(self, length: usize, fill_char: char) -> Expr {
        self.0
            .map_private(StringFunction::PadEnd { length, fill_char }.into())
    }

    /// Pad the start of the string with zeros until it reaches the given length.
    ///
    /// A sign prefix (`-`) is handled by inserting the padding after the sign
    /// character rather than before.
    /// Strings with length equal to or greater than the given length are
    /// returned as-is.
    #[cfg(feature = "string_pad")]
    pub fn zfill(self, length: usize) -> Expr {
        self.0.map_private(StringFunction::ZFill(length).into())
    }

    /// Extract each successive non-overlapping match in an individual string as an array
    pub fn extract_all(self, pat: Expr) -> Expr {
        self.0
            .map_many_private(StringFunction::ExtractAll.into(), &[pat], false, false)
    }

    /// Count all successive non-overlapping regex matches.
    pub fn count_matches(self, pat: Expr, literal: bool) -> Expr {
        self.0.map_many_private(
            StringFunction::CountMatches(literal).into(),
            &[pat],
            false,
            false,
        )
    }

    /// Convert a Utf8 column into a Date/Datetime/Time column.
    #[cfg(feature = "temporal")]
    pub fn strptime(self, dtype: DataType, options: StrptimeOptions, ambiguous: Expr) -> Expr {
        self.0.map_many_private(
            StringFunction::Strptime(dtype, options).into(),
            &[ambiguous],
            false,
            false,
        )
    }

    /// Convert a Utf8 column into a Date column.
    #[cfg(feature = "dtype-date")]
    pub fn to_date(self, options: StrptimeOptions) -> Expr {
        self.strptime(DataType::Date, options, lit("raise"))
    }

    /// Convert a Utf8 column into a Datetime column.
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
                if format.contains("%.9f")
                    || format.contains("%9f")
                    || format.contains("%f")
                    || format.contains("%.f")
                {
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

    /// Convert a Utf8 column into a Time column.
    #[cfg(feature = "dtype-time")]
    pub fn to_time(self, options: StrptimeOptions) -> Expr {
        self.strptime(DataType::Time, options, lit("raise"))
    }

    /// Convert a Utf8 column into a Decimal column.
    #[cfg(feature = "dtype-decimal")]
    pub fn to_decimal(self, infer_length: usize) -> Expr {
        self.0
            .map_private(StringFunction::ToDecimal(infer_length).into())
    }

    /// Concat the values into a string array.
    /// # Arguments
    ///
    /// * `delimiter` - A string that will act as delimiter between values.
    #[cfg(feature = "concat_str")]
    pub fn concat(self, delimiter: &str) -> Expr {
        self.0
            .apply_private(StringFunction::ConcatVertical(delimiter.to_owned()).into())
            .with_function_options(|mut options| {
                options.returns_scalar = true;
                options.collect_groups = ApplyOptions::GroupWise;
                options
            })
    }

    /// Split the string by a substring. The resulting dtype is `List<Utf8>`.
    pub fn split(self, by: Expr) -> Expr {
        self.0
            .map_many_private(StringFunction::Split(false).into(), &[by], false, false)
    }

    /// Split the string by a substring and keep the substring. The resulting dtype is `List<Utf8>`.
    pub fn split_inclusive(self, by: Expr) -> Expr {
        self.0
            .map_many_private(StringFunction::Split(true).into(), &[by], false, false)
    }

    #[cfg(feature = "dtype-struct")]
    /// Split exactly `n` times by a given substring. The resulting dtype is [`DataType::Struct`].
    pub fn split_exact(self, by: Expr, n: usize) -> Expr {
        self.0.map_many_private(
            StringFunction::SplitExact {
                n,
                inclusive: false,
            }
            .into(),
            &[by],
            false,
            false,
        )
    }

    #[cfg(feature = "dtype-struct")]
    /// Split exactly `n` times by a given substring and keep the substring.
    /// The resulting dtype is [`DataType::Struct`].
    pub fn split_exact_inclusive(self, by: Expr, n: usize) -> Expr {
        self.0.map_many_private(
            StringFunction::SplitExact { n, inclusive: true }.into(),
            &[by],
            false,
            false,
        )
    }

    #[cfg(feature = "dtype-struct")]
    /// Split by a given substring, returning exactly `n` items. If there are more possible splits,
    /// keeps the remainder of the string intact. The resulting dtype is [`DataType::Struct`].
    pub fn splitn(self, by: Expr, n: usize) -> Expr {
        self.0
            .map_many_private(StringFunction::SplitN(n).into(), &[by], false, false)
    }

    #[cfg(feature = "regex")]
    /// Replace values that match a regex `pat` with a `value`.
    pub fn replace(self, pat: Expr, value: Expr, literal: bool) -> Expr {
        self.0.map_many_private(
            FunctionExpr::StringExpr(StringFunction::Replace { n: 1, literal }),
            &[pat, value],
            false,
            true,
        )
    }

    #[cfg(feature = "regex")]
    /// Replace values that match a regex `pat` with a `value`.
    pub fn replace_n(self, pat: Expr, value: Expr, literal: bool, n: i64) -> Expr {
        self.0.map_many_private(
            FunctionExpr::StringExpr(StringFunction::Replace { n, literal }),
            &[pat, value],
            false,
            true,
        )
    }

    #[cfg(feature = "regex")]
    /// Replace all values that match a regex `pat` with a `value`.
    pub fn replace_all(self, pat: Expr, value: Expr, literal: bool) -> Expr {
        self.0.map_many_private(
            FunctionExpr::StringExpr(StringFunction::Replace { n: -1, literal }),
            &[pat, value],
            false,
            true,
        )
    }

    /// Remove leading and trailing characters, or whitespace if matches is None.
    pub fn strip_chars(self, matches: Expr) -> Expr {
        self.0.map_many_private(
            FunctionExpr::StringExpr(StringFunction::StripChars),
            &[matches],
            false,
            false,
        )
    }

    /// Remove leading characters, or whitespace if matches is None.
    pub fn strip_chars_start(self, matches: Expr) -> Expr {
        self.0.map_many_private(
            FunctionExpr::StringExpr(StringFunction::StripCharsStart),
            &[matches],
            false,
            false,
        )
    }

    /// Remove trailing characters, or whitespace if matches is None.
    pub fn strip_chars_end(self, matches: Expr) -> Expr {
        self.0.map_many_private(
            FunctionExpr::StringExpr(StringFunction::StripCharsEnd),
            &[matches],
            false,
            false,
        )
    }

    /// Remove prefix.
    pub fn strip_prefix(self, prefix: Expr) -> Expr {
        self.0.map_many_private(
            FunctionExpr::StringExpr(StringFunction::StripPrefix),
            &[prefix],
            false,
            false,
        )
    }

    /// Remove suffix.
    pub fn strip_suffix(self, suffix: Expr) -> Expr {
        self.0.map_many_private(
            FunctionExpr::StringExpr(StringFunction::StripSuffix),
            &[suffix],
            false,
            false,
        )
    }

    /// Convert all characters to lowercase.
    pub fn to_lowercase(self) -> Expr {
        self.0
            .map_private(FunctionExpr::StringExpr(StringFunction::Lowercase))
    }

    /// Convert all characters to uppercase.
    pub fn to_uppercase(self) -> Expr {
        self.0
            .map_private(FunctionExpr::StringExpr(StringFunction::Uppercase))
    }

    /// Convert all characters to titlecase.
    #[cfg(feature = "nightly")]
    pub fn to_titlecase(self) -> Expr {
        self.0
            .map_private(FunctionExpr::StringExpr(StringFunction::Titlecase))
    }

    #[cfg(feature = "string_from_radix")]
    /// Parse string in base radix into decimal.
    pub fn from_radix(self, radix: u32, strict: bool) -> Expr {
        self.0
            .map_private(FunctionExpr::StringExpr(StringFunction::FromRadix(
                radix, strict,
            )))
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
        self.0
            .map_private(FunctionExpr::StringExpr(StringFunction::LenBytes))
    }

    /// Return the length of each string as the number of characters.
    ///
    /// When working with ASCII text, use [`len_bytes`] instead to achieve
    /// equivalent output with much better performance:
    /// [`len_bytes`] runs in _O(1)_, while `len_chars` runs in _O(n)_.
    ///
    /// [`len_bytes`]: StringNameSpace::len_bytes
    pub fn len_chars(self) -> Expr {
        self.0
            .map_private(FunctionExpr::StringExpr(StringFunction::LenChars))
    }

    /// Slice the string values.
    pub fn slice(self, start: i64, length: Option<u64>) -> Expr {
        self.0
            .map_private(FunctionExpr::StringExpr(StringFunction::Slice(
                start, length,
            )))
    }

    pub fn explode(self) -> Expr {
        self.0
            .apply_private(FunctionExpr::StringExpr(StringFunction::Explode))
    }

    #[cfg(feature = "extract_jsonpath")]
    pub fn json_extract(self, dtype: Option<DataType>, infer_schema_len: Option<usize>) -> Expr {
        self.0
            .map_private(FunctionExpr::StringExpr(StringFunction::JsonExtract {
                dtype,
                infer_schema_len,
            }))
    }
}
