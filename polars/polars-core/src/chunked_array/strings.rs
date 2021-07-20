use crate::chunked_array::kernels::strings::string_lengths;
use crate::prelude::*;
use arrow::compute::kernels::substring::substring;
use regex::Regex;
#[cfg(feature = "extract_jsonpath")]
use {
    jsonpath_lib::Compiled,
    serde_json::{from_str, Value},
    std::borrow::Cow,
    std::string::String,
};

#[cfg(feature = "extract_jsonpath")]
fn extract_json<'a>(expr: &'a Compiled, json_str: Option<&'a str>) -> Option<Cow<'a, str>> {
    if let Some(value) = json_str {
        let json: Value = from_str(value).unwrap();
        let result = expr.select(&json).unwrap();
        if result.is_empty() {
            None
        } else if result[0].is_string() {
            Some(Cow::Owned(String::from(result[0].as_str().unwrap())))
        } else if result[0].is_i64() {
            Some(Cow::Owned(result[0].as_i64().unwrap().to_string()))
        } else if result[0].is_f64() {
            Some(Cow::Owned(result[0].as_f64().unwrap().to_string()))
        } else if result[0].is_boolean() {
            Some(Cow::Owned(result[0].as_bool().unwrap().to_string()))
        } else {
            None
        }
    } else {
        None
    }
}

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
    /// Extract json path, first match
    /// Refer to https://goessner.net/articles/JsonPath/
    #[cfg(feature = "extract_jsonpath")]
    pub fn extract_json_path_single(&self, json_path: &str) -> Result<Utf8Chunked> {
        let pat = Compiled::compile(json_path).unwrap();
        Ok(self.apply_on_opt(|str_val| extract_json(&pat, str_val)))
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
