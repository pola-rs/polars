use polars_arrow::array::ValueSize;
#[cfg(feature = "dtype-struct")]
use polars_arrow::export::arrow::array::{MutableArray, MutableUtf8Array};

use super::function_expr::StringFunction;
use super::*;
/// Specialized expressions for [`Series`] of [`DataType::Utf8`].
pub struct StringNameSpace(pub(crate) Expr);

impl StringNameSpace {
    /// Check if a string value contains a literal substring.
    pub fn contains_literal<S: AsRef<str>>(self, pat: S) -> Expr {
        let pat = pat.as_ref().into();
        self.0.map_private(
            StringFunction::Contains { pat, literal: true }.into(),
            "str.contains_literal",
        )
    }

    /// Check if a string value contains a Regex substring.
    pub fn contains<S: AsRef<str>>(self, pat: S) -> Expr {
        let pat = pat.as_ref().into();
        self.0.map_private(
            StringFunction::Contains {
                pat,
                literal: false,
            }
            .into(),
            "str.contains",
        )
    }

    /// Check if a string value ends with the `sub` string.
    pub fn ends_with<S: AsRef<str>>(self, sub: S) -> Expr {
        let sub = sub.as_ref().into();
        self.0
            .map_private(StringFunction::EndsWith(sub).into(), "str.ends_with")
    }

    /// Check if a string value starts with the `sub` string.
    pub fn starts_with<S: AsRef<str>>(self, sub: S) -> Expr {
        let sub = sub.as_ref().into();
        self.0
            .map_private(StringFunction::StartsWith(sub).into(), "str.starts_with")
    }

    /// Extract a regex pattern from the a string value.
    pub fn extract(self, pat: &str, group_index: usize) -> Expr {
        let pat = pat.to_string();
        self.0.map_private(
            StringFunction::Extract { pat, group_index }.into(),
            "str.extract",
        )
    }

    /// Return a copy of the string left filled with ASCII '0' digits to make a string of length width.
    /// A leading sign prefix ('+'/'-') is handled by inserting the padding after the sign character
    /// rather than before.
    /// The original string is returned if width is less than or equal to `s.len()`.
    #[cfg(feature = "string_justify")]
    #[cfg_attr(docsrs, doc(cfg(feature = "string_justify")))]
    pub fn zfill(self, alignment: usize) -> Expr {
        self.0
            .map_private(StringFunction::Zfill(alignment).into(), "str.zfill")
    }

    /// Return the string left justified in a string of length width.
    /// Padding is done using the specified `fillchar`,
    /// The original string is returned if width is less than or equal to `s.len()`.
    #[cfg(feature = "string_justify")]
    #[cfg_attr(docsrs, doc(cfg(feature = "string_justify")))]
    pub fn ljust(self, width: usize, fillchar: char) -> Expr {
        self.0.map_private(
            StringFunction::LJust { width, fillchar }.into(),
            "str.ljust",
        )
    }

    /// Return the string right justified in a string of length width.
    /// Padding is done using the specified `fillchar`,
    /// The original string is returned if width is less than or equal to `s.len()`.
    #[cfg(feature = "string_justify")]
    #[cfg_attr(docsrs, doc(cfg(feature = "string_justify")))]
    pub fn rjust(self, width: usize, fillchar: char) -> Expr {
        self.0.map_private(
            StringFunction::RJust { width, fillchar }.into(),
            "str.rjust",
        )
    }

    /// Extract each successive non-overlapping match in an individual string as an array
    pub fn extract_all(self, pat: &str) -> Expr {
        let pat = pat.to_string();
        self.0
            .map_private(StringFunction::ExtractAll(pat).into(), "str.extract_all")
    }

    /// Count all successive non-overlapping regex matches.
    pub fn count_match(self, pat: &str) -> Expr {
        let pat = pat.to_string();
        self.0
            .map_private(StringFunction::CountMatch(pat).into(), "str.count_match")
    }

    #[cfg(feature = "temporal")]
    pub fn strptime(self, options: StrpTimeOptions) -> Expr {
        self.0
            .map_private(StringFunction::Strptime(options).into(), "str.strptime")
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
            function: StringFunction::Concat(delimiter).into(),
            options: FunctionOptions {
                collect_groups: ApplyOptions::ApplyGroups,
                input_wildcard_expansion: false,
                auto_explode: true,
                fmt_str: "str.concat",
            },
        }
    }

    /// Split the string by a substring.
    // Split exactly `n` times by a given substring. The resulting dtype is `List<Utf8>`.
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
            Ok(builder.finish().into_series())
        };
        self.0
            .map(
                function,
                GetOutput::from_type(DataType::List(Box::new(DataType::Utf8))),
            )
            .with_fmt("str.split")
    }

    #[cfg(feature = "dtype-struct")]
    // Split exactly `n` times by a given substring. The resulting dtype is [`DataType::Struct`].
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
            Ok(StructChunked::new(ca.name(), &fields)?.into_series())
        };
        self.0
            .map(
                function,
                GetOutput::from_type(DataType::Struct(
                    (0..n + 1)
                        .map(|i| Field::new(&format!("field_{i}"), DataType::Utf8))
                        .collect(),
                )),
            )
            .with_fmt("str.split_exact")
    }

    #[cfg(feature = "dtype-struct")]
    // Split exactly `n` times by a given substring and keep the substring.
    // The resulting dtype is [`DataType::Struct`].
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
            Ok(StructChunked::new(ca.name(), &fields)?.into_series())
        };
        self.0
            .map(
                function,
                GetOutput::from_type(DataType::Struct(
                    (0..n + 1)
                        .map(|i| Field::new(&format!("field_{i}"), DataType::Utf8))
                        .collect(),
                )),
            )
            .with_fmt("str.split_exact")
    }

    /// Split the string by a substring and keep the substring.
    // Split exactly `n` times by a given substring. The resulting dtype is `List<Utf8>`.
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
            Ok(builder.finish().into_series())
        };
        self.0
            .map(
                function,
                GetOutput::from_type(DataType::List(Box::new(DataType::Utf8))),
            )
            .with_fmt("str.split_inclusive")
    }
}
