use crate::prelude::*;
use crate::utils::Xob;
use regex::Regex;

macro_rules! apply_closure_to_primitive {
    ($self:expr, $f:expr) => {{
        if $self.null_count() == 0 {
            let ca: Xob<_> = $self.into_no_null_iter().map($f).collect();
            ca.into_inner()
        } else {
            let ca = $self.into_iter().map(|opt_s| opt_s.map($f)).collect();
            ca
        }
    }};
}

macro_rules! apply_closure {
    ($self:expr, $f:expr) => {{
        if $self.null_count() == 0 {
            let ca = $self.into_no_null_iter().map($f).collect();
            ca
        } else {
            let ca = $self.into_iter().map(|opt_s| opt_s.map($f)).collect();
            ca
        }
    }};
}

impl Utf8Chunked {
    /// Get the length of the string values.
    pub fn str_lengths(&self) -> UInt32Chunked {
        let f = |s: &str| s.len() as u32;
        apply_closure_to_primitive!(self, f)
    }

    /// Check if strings contain a regex pattern
    pub fn contains(&self, pat: &str) -> Result<BooleanChunked> {
        let reg = Regex::new(pat)?;
        let f = |s| reg.is_match(s);
        Ok(apply_closure_to_primitive!(self, f))
    }

    /// Replace the leftmost (sub)string by a regex pattern
    pub fn replace(&self, pat: &str, val: &str) -> Result<Utf8Chunked> {
        let reg = Regex::new(pat)?;
        let f = |s| reg.replace(s, val);
        Ok(apply_closure!(self, f))
    }

    /// Replace all (sub)strings by a regex pattern
    pub fn replace_all(&self, pat: &str, val: &str) -> Result<Utf8Chunked> {
        let reg = Regex::new(pat)?;
        let f = |s| reg.replace_all(s, val);
        Ok(apply_closure!(self, f))
    }

    /// Modify the strings to their lowercase equivalent
    pub fn to_lowercase(&self) -> Utf8Chunked {
        self.apply(str::to_lowercase)
    }

    /// Modify the strings to their uppercase equivalent
    pub fn to_uppercase(&self) -> Utf8Chunked {
        self.apply(str::to_uppercase)
    }
}
