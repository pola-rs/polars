use polars::prelude::*;
use pyo3::prelude::*;

use crate::conversion::Wrap;
use crate::error::PyPolarsErr;
use crate::PyExpr;

#[pymethods]
impl PyExpr {
    fn str_join(&self, delimiter: &str, ignore_nulls: bool) -> Self {
        self.inner
            .clone()
            .str()
            .join(delimiter, ignore_nulls)
            .into()
    }

    #[pyo3(signature = (format, strict, exact, cache))]
    fn str_to_date(&self, format: Option<String>, strict: bool, exact: bool, cache: bool) -> Self {
        let options = StrptimeOptions {
            format,
            strict,
            exact,
            cache,
        };
        self.inner.clone().str().to_date(options).into()
    }

    #[pyo3(signature = (format, time_unit, time_zone, strict, exact, cache, ambiguous))]
    fn str_to_datetime(
        &self,
        format: Option<String>,
        time_unit: Option<Wrap<TimeUnit>>,
        time_zone: Option<TimeZone>,
        strict: bool,
        exact: bool,
        cache: bool,
        ambiguous: Self,
    ) -> Self {
        let options = StrptimeOptions {
            format,
            strict,
            exact,
            cache,
        };
        self.inner
            .clone()
            .str()
            .to_datetime(
                time_unit.map(|tu| tu.0),
                time_zone,
                options,
                ambiguous.inner,
            )
            .into()
    }

    #[pyo3(signature = (format, strict, cache))]
    fn str_to_time(&self, format: Option<String>, strict: bool, cache: bool) -> Self {
        let options = StrptimeOptions {
            format,
            strict,
            cache,
            exact: true,
        };
        self.inner.clone().str().to_time(options).into()
    }

    fn str_strip_chars(&self, matches: Self) -> Self {
        self.inner.clone().str().strip_chars(matches.inner).into()
    }

    fn str_strip_chars_start(&self, matches: Self) -> Self {
        self.inner
            .clone()
            .str()
            .strip_chars_start(matches.inner)
            .into()
    }

    fn str_strip_chars_end(&self, matches: Self) -> Self {
        self.inner
            .clone()
            .str()
            .strip_chars_end(matches.inner)
            .into()
    }

    fn str_strip_prefix(&self, prefix: Self) -> Self {
        self.inner.clone().str().strip_prefix(prefix.inner).into()
    }

    fn str_strip_suffix(&self, suffix: Self) -> Self {
        self.inner.clone().str().strip_suffix(suffix.inner).into()
    }

    fn str_slice(&self, offset: Self, length: Self) -> Self {
        self.inner
            .clone()
            .str()
            .slice(offset.inner, length.inner)
            .into()
    }

    fn str_head(&self, n: Self) -> Self {
        self.inner.clone().str().head(n.inner).into()
    }

    fn str_tail(&self, n: Self) -> Self {
        self.inner.clone().str().tail(n.inner).into()
    }

    fn str_to_uppercase(&self) -> Self {
        self.inner.clone().str().to_uppercase().into()
    }

    fn str_to_lowercase(&self) -> Self {
        self.inner.clone().str().to_lowercase().into()
    }

    #[cfg(feature = "nightly")]
    fn str_to_titlecase(&self) -> Self {
        self.inner.clone().str().to_titlecase().into()
    }

    fn str_len_bytes(&self) -> Self {
        self.inner.clone().str().len_bytes().into()
    }

    fn str_len_chars(&self) -> Self {
        self.inner.clone().str().len_chars().into()
    }

    #[cfg(feature = "regex")]
    fn str_replace_n(&self, pat: Self, val: Self, literal: bool, n: i64) -> Self {
        self.inner
            .clone()
            .str()
            .replace_n(pat.inner, val.inner, literal, n)
            .into()
    }

    #[cfg(feature = "regex")]
    fn str_replace_all(&self, pat: Self, val: Self, literal: bool) -> Self {
        self.inner
            .clone()
            .str()
            .replace_all(pat.inner, val.inner, literal)
            .into()
    }

    fn str_reverse(&self) -> Self {
        self.inner.clone().str().reverse().into()
    }

    fn str_pad_start(&self, length: usize, fill_char: char) -> Self {
        self.inner.clone().str().pad_start(length, fill_char).into()
    }

    fn str_pad_end(&self, length: usize, fill_char: char) -> Self {
        self.inner.clone().str().pad_end(length, fill_char).into()
    }

    fn str_zfill(&self, length: Self) -> Self {
        self.inner.clone().str().zfill(length.inner).into()
    }

    #[pyo3(signature = (pat, literal, strict))]
    #[cfg(feature = "regex")]
    fn str_contains(&self, pat: Self, literal: Option<bool>, strict: bool) -> Self {
        match literal {
            Some(true) => self.inner.clone().str().contains_literal(pat.inner).into(),
            _ => self.inner.clone().str().contains(pat.inner, strict).into(),
        }
    }

    #[pyo3(signature = (pat, literal, strict))]
    #[cfg(feature = "regex")]
    fn str_find(&self, pat: Self, literal: Option<bool>, strict: bool) -> Self {
        match literal {
            Some(true) => self.inner.clone().str().find_literal(pat.inner).into(),
            _ => self.inner.clone().str().find(pat.inner, strict).into(),
        }
    }

    fn str_ends_with(&self, sub: Self) -> Self {
        self.inner.clone().str().ends_with(sub.inner).into()
    }

    fn str_starts_with(&self, sub: Self) -> Self {
        self.inner.clone().str().starts_with(sub.inner).into()
    }

    fn str_hex_encode(&self) -> Self {
        self.inner.clone().str().hex_encode().into()
    }

    #[cfg(feature = "binary_encoding")]
    fn str_hex_decode(&self, strict: bool) -> Self {
        self.inner.clone().str().hex_decode(strict).into()
    }

    fn str_base64_encode(&self) -> Self {
        self.inner.clone().str().base64_encode().into()
    }

    #[cfg(feature = "binary_encoding")]
    fn str_base64_decode(&self, strict: bool) -> Self {
        self.inner.clone().str().base64_decode(strict).into()
    }

    fn str_to_integer(&self, base: Self, strict: bool) -> Self {
        self.inner
            .clone()
            .str()
            .to_integer(base.inner, strict)
            .with_fmt("str.to_integer")
            .into()
    }

    #[cfg(feature = "extract_jsonpath")]
    fn str_json_decode(
        &self,
        dtype: Option<Wrap<DataType>>,
        infer_schema_len: Option<usize>,
    ) -> Self {
        let dtype = dtype.map(|wrap| wrap.0);
        self.inner
            .clone()
            .str()
            .json_decode(dtype, infer_schema_len)
            .into()
    }

    #[cfg(feature = "extract_jsonpath")]
    fn str_json_path_match(&self, pat: Self) -> Self {
        self.inner.clone().str().json_path_match(pat.inner).into()
    }

    fn str_extract(&self, pat: Self, group_index: usize) -> Self {
        self.inner
            .clone()
            .str()
            .extract(pat.inner, group_index)
            .into()
    }

    fn str_extract_all(&self, pat: Self) -> Self {
        self.inner.clone().str().extract_all(pat.inner).into()
    }

    #[cfg(feature = "extract_groups")]
    fn str_extract_groups(&self, pat: &str) -> PyResult<Self> {
        Ok(self
            .inner
            .clone()
            .str()
            .extract_groups(pat)
            .map_err(PyPolarsErr::from)?
            .into())
    }

    fn str_count_matches(&self, pat: Self, literal: bool) -> Self {
        self.inner
            .clone()
            .str()
            .count_matches(pat.inner, literal)
            .into()
    }

    fn str_split(&self, by: Self) -> Self {
        self.inner.clone().str().split(by.inner).into()
    }

    fn str_split_inclusive(&self, by: Self) -> Self {
        self.inner.clone().str().split_inclusive(by.inner).into()
    }

    fn str_split_exact(&self, by: Self, n: usize) -> Self {
        self.inner.clone().str().split_exact(by.inner, n).into()
    }

    fn str_split_exact_inclusive(&self, by: Self, n: usize) -> Self {
        self.inner
            .clone()
            .str()
            .split_exact_inclusive(by.inner, n)
            .into()
    }

    fn str_splitn(&self, by: Self, n: usize) -> Self {
        self.inner.clone().str().splitn(by.inner, n).into()
    }

    fn str_to_decimal(&self, infer_len: usize) -> Self {
        self.inner.clone().str().to_decimal(infer_len).into()
    }

    #[cfg(feature = "find_many")]
    fn str_contains_any(&self, patterns: PyExpr, ascii_case_insensitive: bool) -> Self {
        self.inner
            .clone()
            .str()
            .contains_any(patterns.inner, ascii_case_insensitive)
            .into()
    }
    #[cfg(feature = "find_many")]
    fn str_replace_many(
        &self,
        patterns: PyExpr,
        replace_with: PyExpr,
        ascii_case_insensitive: bool,
    ) -> Self {
        self.inner
            .clone()
            .str()
            .replace_many(patterns.inner, replace_with.inner, ascii_case_insensitive)
            .into()
    }

    #[cfg(feature = "find_many")]
    fn str_extract_many(
        &self,
        patterns: PyExpr,
        ascii_case_insensitive: bool,
        overlapping: bool,
    ) -> Self {
        self.inner
            .clone()
            .str()
            .extract_many(patterns.inner, ascii_case_insensitive, overlapping)
            .into()
    }
}
