use polars_arrow::array::ValueSize;
#[cfg(feature = "dtype-struct")]
use polars_arrow::export::arrow::array::{MutableArray, MutableUtf8Array};

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

    /// Check if a string value contains a Regex substring.
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

    /// Extract a regex pattern from the a string value.
    pub fn extract(self, pat: &str, group_index: usize) -> Expr {
        let pat = pat.to_string();
        self.0
            .map_private(StringFunction::Extract { pat, group_index }.into())
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
    pub fn count_match(self, pat: &str) -> Expr {
        let pat = pat.to_string();
        self.0.map_private(StringFunction::CountMatch(pat).into())
    }

    #[cfg(feature = "temporal")]
    pub fn strptime(self, options: StrpTimeOptions) -> Expr {
        self.0.map_private(StringFunction::Strptime(options).into())
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
    pub fn split(self, by: &str) -> Expr {
        let by = by.to_string();

        let function = move |s: Series| {
            let ca = s.utf8()?;

            let mut builder = ListUtf8ChunkedBuilder::new(s.name(), s.len(), ca.get_values_size());
            ca.into_iter().for_each(|opt_s| match opt_s {
                None => builder.append_null(),
                Some(s) => {
                    let iter = s.split(&by);
                    builder.append_values_iter(iter);
                }
            });
            Ok(Some(builder.finish().into_series()))
        };
        self.0
            .map(
                function,
                GetOutput::from_type(DataType::List(Box::new(DataType::Utf8))),
            )
            .with_fmt("str.split")
    }

    /// Split the string by a substring and keep the substring. The resulting dtype is `List<Utf8>`.
    pub fn split_inclusive(self, by: &str) -> Expr {
        let by = by.to_string();

        let function = move |s: Series| {
            let ca = s.utf8()?;

            let mut builder = ListUtf8ChunkedBuilder::new(s.name(), s.len(), ca.get_values_size());
            ca.into_iter().for_each(|opt_s| match opt_s {
                None => builder.append_null(),
                Some(s) => {
                    let iter = s.split_inclusive(&by);
                    builder.append_values_iter(iter);
                }
            });
            Ok(Some(builder.finish().into_series()))
        };
        self.0
            .map(
                function,
                GetOutput::from_type(DataType::List(Box::new(DataType::Utf8))),
            )
            .with_fmt("str.split_inclusive")
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
                }
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
                }
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
                        .map(|i| Field::from_owned(format!("field_{i}"), DataType::Utf8))
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
                }
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
                }
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
                        .map(|i| Field::from_owned(format!("field_{i}"), DataType::Utf8))
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
                }
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
                }
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
                        .map(|i| Field::from_owned(format!("field_{i}"), DataType::Utf8))
                        .collect(),
                )),
            )
            .with_fmt("str.splitn")
    }

    #[cfg(feature = "regex")]
    /// Replace values that match a regex `pat` with a `value`.
    pub fn replace(self, pat: Expr, value: Expr, literal: bool) -> Expr {
        self.0.map_many_private(
            FunctionExpr::StringExpr(StringFunction::Replace {
                all: false,
                literal,
            }),
            &[pat, value],
            true,
        )
    }

    #[cfg(feature = "regex")]
    /// Replace all values that match a regex `pat` with a `value`.
    pub fn replace_all(self, pat: Expr, value: Expr, literal: bool) -> Expr {
        self.0.map_many_private(
            FunctionExpr::StringExpr(StringFunction::Replace { all: true, literal }),
            &[pat, value],
            true,
        )
    }

    /// Remove leading and trailing characters, or whitespace if matches is None.
    pub fn strip(self, matches: Option<String>) -> Expr {
        self.0
            .map_private(FunctionExpr::StringExpr(StringFunction::Strip(matches)))
    }

    /// Remove leading characters, or whitespace if matches is None.
    pub fn lstrip(self, matches: Option<String>) -> Expr {
        self.0
            .map_private(FunctionExpr::StringExpr(StringFunction::LStrip(matches)))
    }

    /// Remove trailing characters, or whitespace if matches is None..
    pub fn rstrip(self, matches: Option<String>) -> Expr {
        self.0
            .map_private(FunctionExpr::StringExpr(StringFunction::RStrip(matches)))
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

    #[cfg(feature = "string_from_radix")]
    /// Parse string in base radix into decimal
    pub fn from_radix(self, radix: Option<u32>) -> Expr {
        self.0
            .map_private(FunctionExpr::StringExpr(StringFunction::FromRadix(radix)))
    }
}
