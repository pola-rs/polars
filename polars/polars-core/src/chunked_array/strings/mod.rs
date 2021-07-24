#[cfg(feature = "extract_jsonpath")]
mod json_path;

use crate::chunked_array::kernels::strings::string_lengths;
use crate::prelude::*;
use arrow::compute::kernels::substring::substring;
use regex::Regex;

impl Utf8Chunked {
    /// Get the length of the string values.
    pub fn str_lengths(&self) -> UInt32Chunked {
        self.apply_kernel_cast(string_lengths)
    }

    /// Check if strings contain a regex pattern
    pub fn contains(&self, pat: &str) -> Result<BooleanChunked> {
        let reg = Regex::new(pat)?;
        let f = |s| reg.is_match(s);
        let mut ca: BooleanChunked = if self.null_count() == 0 {
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

    /// Modify the strings to their lowercase equivalent
    pub fn to_lowercase(&self) -> Utf8Chunked {
        self.apply(|s| str::to_lowercase(s).into())
    }

    /// Modify the strings to their uppercase equivalent
    pub fn to_uppercase(&self) -> Utf8Chunked {
        self.apply(|s| str::to_uppercase(s).into())
    }

    /// Concat with the values from a second Utf8Chunked
    pub fn concat(&self, other: &Utf8Chunked) -> Self {
        self + other
    }

    /// Slice the string values
    /// Determines a substring starting from `start` and with optional length `length` of each of the elements in `array`.
    /// `start` can be negative, in which case the start counts from the end of the string.
    pub fn str_slice(&self, start: i64, length: Option<u64>) -> Result<Self> {
        let chunks = self
            .downcast_iter()
            .map(|c| substring(c, start, &length))
            .collect::<arrow::error::Result<_>>()?;

        Ok(Self::new_from_chunks(self.name(), chunks))
    }
}
