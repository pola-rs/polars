use polars::prelude::*;
use pyo3::prelude::*;

use crate::conversion::Wrap;
use crate::error::PyPolarsErr;
use crate::PyExpr;

#[pymethods]
impl PyExpr {
    fn str_concat(&self, delimiter: &str, ignore_nulls: bool) -> Self {
        self.inner
            .clone()
            .str()
            .concat(delimiter, ignore_nulls)
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

    fn str_slice(&self, start: i64, length: Option<u64>) -> Self {
        self.inner.clone().str().slice(start, length).into()
    }

    fn str_explode(&self) -> Self {
        self.inner.clone().str().explode().into()
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

    #[cfg(feature = "lazy_regex")]
    fn str_replace_n(&self, pat: Self, val: Self, literal: bool, n: i64) -> Self {
        self.inner
            .clone()
            .str()
            .replace_n(pat.inner, val.inner, literal, n)
            .into()
    }

    #[cfg(feature = "lazy_regex")]
    fn str_replace_all(&self, pat: Self, val: Self, literal: bool) -> Self {
        self.inner
            .clone()
            .str()
            .replace_all(pat.inner, val.inner, literal)
            .into()
    }

    fn str_pad_start(&self, length: usize, fill_char: char) -> Self {
        self.inner.clone().str().pad_start(length, fill_char).into()
    }

    fn str_pad_end(&self, length: usize, fill_char: char) -> Self {
        self.inner.clone().str().pad_end(length, fill_char).into()
    }

    fn str_zfill(&self, length: usize) -> Self {
        self.inner.clone().str().zfill(length).into()
    }

    #[pyo3(signature = (pat, literal, strict))]
    #[cfg(feature = "lazy_regex")]
    fn str_contains(&self, pat: Self, literal: Option<bool>, strict: bool) -> Self {
        match literal {
            Some(true) => self.inner.clone().str().contains_literal(pat.inner).into(),
            _ => self.inner.clone().str().contains(pat.inner, strict).into(),
        }
    }

    fn str_ends_with(&self, sub: Self) -> Self {
        self.inner.clone().str().ends_with(sub.inner).into()
    }

    fn str_starts_with(&self, sub: Self) -> Self {
        self.inner.clone().str().starts_with(sub.inner).into()
    }

    fn str_hex_encode(&self) -> Self {
        self.inner
            .clone()
            .map(
                move |s| s.utf8().map(|s| Some(s.hex_encode().into_series())),
                GetOutput::same_type(),
            )
            .with_fmt("str.hex_encode")
            .into()
    }

    #[cfg(feature = "binary_encoding")]
    fn str_hex_decode(&self, strict: bool) -> Self {
        self.inner
            .clone()
            .map(
                move |s| s.utf8()?.hex_decode(strict).map(|s| Some(s.into_series())),
                GetOutput::from_type(DataType::Binary),
            )
            .with_fmt("str.hex_decode")
            .into()
    }

    fn str_base64_encode(&self) -> Self {
        self.inner
            .clone()
            .map(
                move |s| s.utf8().map(|s| Some(s.base64_encode().into_series())),
                GetOutput::same_type(),
            )
            .with_fmt("str.base64_encode")
            .into()
    }

    #[cfg(feature = "binary_encoding")]
    fn str_base64_decode(&self, strict: bool) -> Self {
        self.inner
            .clone()
            .map(
                move |s| {
                    s.utf8()?
                        .base64_decode(strict)
                        .map(|s| Some(s.into_series()))
                },
                GetOutput::from_type(DataType::Binary),
            )
            .with_fmt("str.base64_decode")
            .into()
    }

    fn str_parse_int(&self, radix: u32, strict: bool) -> Self {
        self.inner
            .clone()
            .str()
            .from_radix(radix, strict)
            .with_fmt("str.parse_int")
            .into()
    }

    #[cfg(feature = "extract_jsonpath")]
    fn str_json_extract(
        &self,
        dtype: Option<Wrap<DataType>>,
        infer_schema_len: Option<usize>,
    ) -> Self {
        let dtype = dtype.map(|wrap| wrap.0);
        self.inner
            .clone()
            .str()
            .json_extract(dtype, infer_schema_len)
            .into()
    }

    #[cfg(feature = "extract_jsonpath")]
    fn str_json_path_match(&self, pat: String) -> Self {
        let function = move |s: Series| {
            let ca = s.utf8()?;
            match ca.json_path_match(&pat) {
                Ok(ca) => Ok(Some(ca.into_series())),
                Err(e) => Err(PolarsError::ComputeError(format!("{e:?}").into())),
            }
        };
        self.inner
            .clone()
            .map(function, GetOutput::from_type(DataType::Utf8))
            .with_fmt("str.json_path_match")
            .into()
    }

    fn str_extract(&self, pat: &str, group_index: usize) -> Self {
        self.inner.clone().str().extract(pat, group_index).into()
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
}
