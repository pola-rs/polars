#[cfg(feature = "extract_jsonpath")]
mod json_path;

#[cfg(feature = "string_encoding")]
mod encoding;

use crate::prelude::*;
use arrow::compute::substring::substring;
use polars_arrow::kernels::string::*;
use regex::Regex;
use std::borrow::Cow;

fn f_regex_extract<'a>(reg: &Regex, input: &'a str, group_index: usize) -> Option<Cow<'a, str>> {
    reg.captures(input)
        .and_then(|cap| cap.get(group_index).map(|m| Cow::Borrowed(m.as_str())))
}

impl Utf8Chunked {
    /// Get the length of the string values.
    pub fn str_lengths(&self) -> UInt32Chunked {
        self.apply_kernel_cast(&string_lengths)
    }

    /// Check if strings contain a regex pattern
    pub fn contains(&self, pat: &str) -> Result<BooleanChunked> {
        let reg = Regex::new(pat)?;
        let f = |s| reg.is_match(s);
        let mut ca: BooleanChunked = if !self.has_validity() {
            self.into_no_null_iter().map(f).collect()
        } else {
            self.into_iter().map(|opt_s| opt_s.map(f)).collect()
        };
        ca.rename(self.name());
        Ok(ca)
    }

    /// Replace the leftmost (sub)string by a regex pattern
    pub fn replace(&self, pat: &str, val: &str) -> Result<Utf8Chunked> {
        let reg = Regex::new(pat)?;
        let f = |s| reg.replace(s, val);
        Ok(self.apply(f))
    }

    /// Replace all (sub)strings by a regex pattern
    pub fn replace_all(&self, pat: &str, val: &str) -> Result<Utf8Chunked> {
        let reg = Regex::new(pat)?;
        let f = |s| reg.replace_all(s, val);
        Ok(self.apply(f))
    }

    /// Extract the nth capture group from pattern
    pub fn extract(&self, pat: &str, group_index: usize) -> Result<Utf8Chunked> {
        let reg = Regex::new(pat)?;
        Ok(self.apply_on_opt(|e| e.and_then(|input| f_regex_extract(&reg, input, group_index))))
    }

    /// Modify the strings to their lowercase equivalent
    #[must_use]
    pub fn to_lowercase(&self) -> Utf8Chunked {
        self.apply(|s| str::to_lowercase(s).into())
    }

    /// Modify the strings to their uppercase equivalent
    #[must_use]
    pub fn to_uppercase(&self) -> Utf8Chunked {
        self.apply(|s| str::to_uppercase(s).into())
    }

    /// Concat with the values from a second Utf8Chunked
    #[must_use]
    pub fn concat(&self, other: &Utf8Chunked) -> Self {
        self + other
    }

    /// Slice the string values
    /// Determines a substring starting from `start` and with optional length `length` of each of the elements in `array`.
    /// `start` can be negative, in which case the start counts from the end of the string.
    pub fn str_slice(&self, start: i64, length: Option<u64>) -> Result<Self> {
        let chunks = self
            .downcast_iter()
            .map(|c| Ok(substring(c, start, &length)?.into()))
            .collect::<arrow::error::Result<_>>()?;

        Ok(Self::from_chunks(self.name(), chunks))
    }
}
