#[cfg(feature = "dtype-struct")]
use polars_arrow::export::arrow::array::{MutableArray, MutableUtf8Array};
#[cfg(feature = "dtype-struct")]
use polars_utils::format_smartstring;

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
            true,
        )
    }

    /// Check if a string value ends with the `sub` string.
    pub fn ends_with(self, sub: Expr) -> Expr {
        self.0.map_many_private(
            FunctionExpr::StringExpr(StringFunction::EndsWith),
            &[sub],
            true,
        )
    }

    /// Check if a string value starts with the `sub` string.
    pub fn starts_with(self, sub: Expr) -> Expr {
        self.0.map_many_private(
            FunctionExpr::StringExpr(StringFunction::StartsWith),
            &[sub],
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

    /// Return a copy of the string left filled with ASCII '0' digits to make a string of length width.
    /// A leading sign prefix ('+'/'-') is handled by inserting the padding after the sign character
    /// rather than before.
    /// The original string is returned if width is less than or equal to `s.len()`.
    #[cfg(feature = "string_justify")]
    pub fn zfill(self, alignment: usize) -> Expr {
        self.0.map_private(StringFunction::Zfill(alignment).into())
    }

    /// Return the string left justified in a string of length width.
    /// Padding is done using the specified `fillchar`,
    /// The original string is returned if width is less than or equal to `s.len()`.
    #[cfg(feature = "string_justify")]
    pub fn ljust(self, width: usize, fillchar: char) -> Expr {
        self.0
            .map_private(StringFunction::LJust { width, fillchar }.into())
    }

    /// Return the string right justified in a string of length width.
    /// Padding is done using the specified `fillchar`,
    /// The original string is returned if width is less than or equal to `s.len()`.
    #[cfg(feature = "string_justify")]
    pub fn rjust(self, width: usize, fillchar: char) -> Expr {
        self.0
            .map_private(StringFunction::RJust { width, fillchar }.into())
    }

    /// Extract each successive non-overlapping match in an individual string as an array
    pub fn extract_all(self, pat: Expr) -> Expr {
        self.0
            .map_many_private(StringFunction::ExtractAll.into(), &[pat], false)
    }

    /// Count all successive non-overlapping regex matches.
    pub fn count_matches(self, pat: Expr, literal: bool) -> Expr {
        self.0
            .map_many_private(StringFunction::CountMatches(literal).into(), &[pat], false)
    }

    /// Convert a Utf8 column into a Date/Datetime/Time column.
    #[cfg(feature = "temporal")]
    pub fn strptime(self, dtype: DataType, options: StrptimeOptions, ambiguous: Expr) -> Expr {
        self.0.map_many_private(
            StringFunction::Strptime(dtype, options).into(),
            &[ambiguous],
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
        let delimiter = delimiter.to_owned();

        Expr::Function {
            input: vec![self.0],
            function: StringFunction::ConcatVertical(delimiter).into(),
            options: FunctionOptions {
                collect_groups: ApplyOptions::ApplyGroups,
                input_wildcard_expansion: false,
                auto_explode: true,
                ..Default::default()
            },
        }
    }

    /// Split the string by a substring. The resulting dtype is `List<Utf8>`.
    pub fn split(self, by: Expr) -> Expr {
        self.0
            .map_many_private(StringFunction::Split.into(), &[by], false)
    }

    /// Split the string by a substring and keep the substring. The resulting dtype is `List<Utf8>`.
    pub fn split_inclusive(self, by: Expr) -> Expr {
        self.0
            .map_many_private(StringFunction::SplitInclusive.into(), &[by], false)
    }

    #[cfg(feature = "dtype-struct")]
    /// Split exactly `n` times by a given substring. The resulting dtype is [`DataType::Struct`].
    pub fn split_exact(self, by: &str, n: usize) -> Expr {
        let by = by.to_string();

        let function = move |s: Series| {
            let ca = s.utf8()?;

            let mut arrs = (0..n + 1)
                .map(|_| MutableUtf8Array::<i64>::with_capacity(ca.len()))
                .collect::<Vec<_>>();

            ca.into_iter().for_each(|opt_s| match opt_s {
                None => {
                    for arr in &mut arrs {
                        arr.push_null()
                    }
                },
                Some(s) => {
                    let mut arr_iter = arrs.iter_mut();
                    let split_iter = s.split(&by);
                    (split_iter)
                        .zip(&mut arr_iter)
                        .for_each(|(splitted, arr)| arr.push(Some(splitted)));
                    // fill the remaining with null
                    for arr in arr_iter {
                        arr.push_null()
                    }
                },
            });
            let fields = arrs
                .into_iter()
                .enumerate()
                .map(|(i, mut arr)| {
                    Series::try_from((format!("field_{i}").as_str(), arr.as_box())).unwrap()
                })
                .collect::<Vec<_>>();
            Ok(Some(StructChunked::new(ca.name(), &fields)?.into_series()))
        };
        self.0
            .map(
                function,
                GetOutput::from_type(DataType::Struct(
                    (0..n + 1)
                        .map(|i| {
                            Field::from_owned(format_smartstring!("field_{i}"), DataType::Utf8)
                        })
                        .collect(),
                )),
            )
            .with_fmt("str.split_exact")
    }

    #[cfg(feature = "dtype-struct")]
    /// Split exactly `n` times by a given substring and keep the substring.
    /// The resulting dtype is [`DataType::Struct`].
    pub fn split_exact_inclusive(self, by: &str, n: usize) -> Expr {
        let by = by.to_string();

        let function = move |s: Series| {
            let ca = s.utf8()?;

            let mut arrs = (0..n + 1)
                .map(|_| MutableUtf8Array::<i64>::with_capacity(ca.len()))
                .collect::<Vec<_>>();

            ca.into_iter().for_each(|opt_s| match opt_s {
                None => {
                    for arr in &mut arrs {
                        arr.push_null()
                    }
                },
                Some(s) => {
                    let mut arr_iter = arrs.iter_mut();
                    let split_iter = s.split_inclusive(&by);
                    (split_iter)
                        .zip(&mut arr_iter)
                        .for_each(|(splitted, arr)| arr.push(Some(splitted)));
                    // fill the remaining with null
                    for arr in arr_iter {
                        arr.push_null()
                    }
                },
            });
            let fields = arrs
                .into_iter()
                .enumerate()
                .map(|(i, mut arr)| {
                    Series::try_from((format!("field_{i}").as_str(), arr.as_box())).unwrap()
                })
                .collect::<Vec<_>>();
            Ok(Some(StructChunked::new(ca.name(), &fields)?.into_series()))
        };
        self.0
            .map(
                function,
                GetOutput::from_type(DataType::Struct(
                    (0..n + 1)
                        .map(|i| {
                            Field::from_owned(format_smartstring!("field_{i}"), DataType::Utf8)
                        })
                        .collect(),
                )),
            )
            .with_fmt("str.split_exact")
    }

    #[cfg(feature = "dtype-struct")]
    /// Split by a given substring, returning exactly `n` items. If there are more possible splits,
    /// keeps the remainder of the string intact. The resulting dtype is [`DataType::Struct`].
    pub fn splitn(self, by: &str, n: usize) -> Expr {
        let by = by.to_string();

        let function = move |s: Series| {
            let ca = s.utf8()?;

            let mut arrs = (0..n)
                .map(|_| MutableUtf8Array::<i64>::with_capacity(ca.len()))
                .collect::<Vec<_>>();

            ca.into_iter().for_each(|opt_s| match opt_s {
                None => {
                    for arr in &mut arrs {
                        arr.push_null()
                    }
                },
                Some(s) => {
                    let mut arr_iter = arrs.iter_mut();
                    let split_iter = s.splitn(n, &by);
                    (split_iter)
                        .zip(&mut arr_iter)
                        .for_each(|(splitted, arr)| arr.push(Some(splitted)));
                    // fill the remaining with null
                    for arr in arr_iter {
                        arr.push_null()
                    }
                },
            });
            let fields = arrs
                .into_iter()
                .enumerate()
                .map(|(i, mut arr)| {
                    Series::try_from((format!("field_{i}").as_str(), arr.as_box())).unwrap()
                })
                .collect::<Vec<_>>();
            Ok(Some(StructChunked::new(ca.name(), &fields)?.into_series()))
        };
        self.0
            .map(
                function,
                GetOutput::from_type(DataType::Struct(
                    (0..n)
                        .map(|i| {
                            Field::from_owned(format_smartstring!("field_{i}"), DataType::Utf8)
                        })
                        .collect(),
                )),
            )
            .with_fmt("str.splitn")
    }

    #[cfg(feature = "regex")]
    /// Replace values that match a regex `pat` with a `value`.
    pub fn replace(self, pat: Expr, value: Expr, literal: bool) -> Expr {
        self.0.map_many_private(
            FunctionExpr::StringExpr(StringFunction::Replace { n: 1, literal }),
            &[pat, value],
            true,
        )
    }

    #[cfg(feature = "regex")]
    /// Replace values that match a regex `pat` with a `value`.
    pub fn replace_n(self, pat: Expr, value: Expr, literal: bool, n: i64) -> Expr {
        self.0.map_many_private(
            FunctionExpr::StringExpr(StringFunction::Replace { n, literal }),
            &[pat, value],
            true,
        )
    }

    #[cfg(feature = "regex")]
    /// Replace all values that match a regex `pat` with a `value`.
    pub fn replace_all(self, pat: Expr, value: Expr, literal: bool) -> Expr {
        self.0.map_many_private(
            FunctionExpr::StringExpr(StringFunction::Replace { n: -1, literal }),
            &[pat, value],
            true,
        )
    }

    /// Remove leading and trailing characters, or whitespace if matches is None.
    pub fn strip_chars(self, matches: Option<String>) -> Expr {
        self.0
            .map_private(FunctionExpr::StringExpr(StringFunction::StripChars(
                matches,
            )))
    }

    /// Remove leading characters, or whitespace if matches is None.
    pub fn strip_chars_start(self, matches: Option<String>) -> Expr {
        self.0
            .map_private(FunctionExpr::StringExpr(StringFunction::StripCharsStart(
                matches,
            )))
    }

    /// Remove trailing characters, or whitespace if matches is None.
    pub fn strip_chars_end(self, matches: Option<String>) -> Expr {
        self.0
            .map_private(FunctionExpr::StringExpr(StringFunction::StripCharsEnd(
                matches,
            )))
    }

    /// Remove prefix.
    pub fn strip_prefix(self, prefix: String) -> Expr {
        self.0
            .map_private(FunctionExpr::StringExpr(StringFunction::StripPrefix(
                prefix,
            )))
    }

    /// Remove suffix.
    pub fn strip_suffix(self, suffix: String) -> Expr {
        self.0
            .map_private(FunctionExpr::StringExpr(StringFunction::StripSuffix(
                suffix,
            )))
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

    /// Return the number of characters in the string (not bytes).
    pub fn n_chars(self) -> Expr {
        self.0
            .map_private(FunctionExpr::StringExpr(StringFunction::NChars))
    }

    /// Return the number of bytes in the string (not characters).
    pub fn lengths(self) -> Expr {
        self.0
            .map_private(FunctionExpr::StringExpr(StringFunction::Length))
    }

    /// Slice the string values.
    pub fn str_slice(self, start: i64, length: Option<u64>) -> Expr {
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
