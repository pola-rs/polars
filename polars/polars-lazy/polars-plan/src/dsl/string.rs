use super::function_expr::StringFunction;
use super::*;
/// Specialized expressions for [`Series`] of [`DataType::Utf8`].
pub struct StringNameSpace(pub(crate) Expr);

impl StringNameSpace {
    /// Check if a string value contains a literal substring.
    pub fn contains_literal<S: AsRef<str>>(self, pat: S) -> Expr {
        let pat = pat.as_ref().into();
        self.0
            .map_private(StringFunction::Contains { pat, literal: true }.into())
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
        )
    }

    /// Check if a string value ends with the `sub` string.
    pub fn ends_with<S: AsRef<str>>(self, sub: S) -> Expr {
        let sub = sub.as_ref().into();
        self.0.map_private(StringFunction::EndsWith(sub).into())
    }

    /// Check if a string value starts with the `sub` string.
    pub fn starts_with<S: AsRef<str>>(self, sub: S) -> Expr {
        let sub = sub.as_ref().into();
        self.0.map_private(StringFunction::StartsWith(sub).into())
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
        self.0.map_private(
            StringFunction::Split {
                by: by.into(),
                inclusive: false,
            }
            .into(),
        )
    }

    /// Split the string by a substring and keep the substring. The resulting dtype is `List<Utf8>`.
    pub fn split_inclusive(self, by: &str) -> Expr {
        self.0.map_private(
            StringFunction::Split {
                by: by.into(),
                inclusive: true,
            }
            .into(),
        )
    }

    #[cfg(feature = "dtype-struct")]
    /// Split exactly `n` times by a given substring. The resulting dtype is [`DataType::Struct`].
    pub fn split_exact(self, by: &str, n: usize) -> Expr {
        self.0.map_private(
            StringFunction::SplitExact {
                by: by.into(),
                inclusive: false,
                n,
            }
            .into(),
        )
    }

    #[cfg(feature = "dtype-struct")]
    /// Split exactly `n` times by a given substring and keep the substring.
    /// The resulting dtype is [`DataType::Struct`].
    pub fn split_exact_inclusive(self, by: &str, n: usize) -> Expr {
        self.0.map_private(
            StringFunction::SplitExact {
                by: by.into(),
                inclusive: true,
                n,
            }
            .into(),
        )
    }

    #[cfg(feature = "dtype-struct")]
    /// Split by a given substring, returning exactly `n` items. If there are more possible splits,
    /// keeps the remainder of the string intact. The resulting dtype is [`DataType::Struct`].
    pub fn splitn(self, by: &str, n: usize) -> Expr {
        self.0
            .map_private(StringFunction::SplitN { by: by.into(), n }.into())
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
}
