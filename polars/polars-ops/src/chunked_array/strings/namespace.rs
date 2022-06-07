use super::*;

use polars_arrow::{
    export::arrow::{self, compute::substring::substring},
    kernels::string::*,
};
use polars_core::export::regex::Regex;
use std::borrow::Cow;

fn f_regex_extract<'a>(reg: &Regex, input: &'a str, group_index: usize) -> Option<Cow<'a, str>> {
    reg.captures(input)
        .and_then(|cap| cap.get(group_index).map(|m| Cow::Borrowed(m.as_str())))
}

pub trait Utf8NameSpaceImpl: AsUtf8 {
    /// Get the length of the string values.
    fn str_lengths(&self) -> UInt32Chunked {
        let ca = self.as_utf8();
        ca.apply_kernel_cast(&string_lengths)
    }

    /// Return a copy of the string left filled with ASCII '0' digits to make a string of length width.
    /// A leading sign prefix ('+'/'-') is handled by inserting the padding after the sign character
    /// rather than before.
    /// The original string is returned if width is less than or equal to `s.len()`.
    #[cfg(feature = "string_justify")]
    #[cfg_attr(docsrs, doc(cfg(feature = "string_justify")))]
    fn zfill<'a>(&'a self, alignment: usize) -> Utf8Chunked {
        let ca = self.as_utf8();

        let f = |s: &'a str| {
            let alignment = alignment.saturating_sub(s.len());
            if alignment == 0 {
                return Cow::Borrowed(s);
            }
            if let Some(stripped) = s.strip_prefix('-') {
                Cow::Owned(format!(
                    "-{:0alignment$}{value}",
                    0,
                    alignment = alignment,
                    value = stripped
                ))
            } else {
                Cow::Owned(format!(
                    "{:0alignment$}{value}",
                    0,
                    alignment = alignment,
                    value = s
                ))
            }
        };
        ca.apply(f)
    }

    /// Return the string left justified in a string of length width.
    /// Padding is done using the specified `fillchar`,
    /// The original string is returned if width is less than or equal to `s.len()`.
    #[cfg(feature = "string_justify")]
    #[cfg_attr(docsrs, doc(cfg(feature = "string_justify")))]
    fn ljust<'a>(&'a self, width: usize, fillchar: char) -> Utf8Chunked {
        let ca = self.as_utf8();

        let f = |s: &'a str| {
            let padding = width.saturating_sub(s.len());
            if padding == 0 {
                Cow::Borrowed(s)
            } else {
                let mut buf = String::with_capacity(width);
                buf.push_str(s);
                for _ in 0..padding {
                    buf.push(fillchar)
                }
                Cow::Owned(buf)
            }
        };
        ca.apply(f)
    }

    /// Return the string right justified in a string of length width.
    /// Padding is done using the specified `fillchar`,
    /// The original string is returned if width is less than or equal to `s.len()`.
    #[cfg(feature = "string_justify")]
    #[cfg_attr(docsrs, doc(cfg(feature = "string_justify")))]
    fn rjust<'a>(&'a self, width: usize, fillchar: char) -> Utf8Chunked {
        let ca = self.as_utf8();

        let f = |s: &'a str| {
            let padding = width.saturating_sub(s.len());
            if padding == 0 {
                Cow::Borrowed(s)
            } else {
                let mut buf = String::with_capacity(width);
                for _ in 0..padding {
                    buf.push(fillchar)
                }
                buf.push_str(s);
                Cow::Owned(buf)
            }
        };
        ca.apply(f)
    }

    /// Check if strings contain a regex pattern
    fn contains(&self, pat: &str) -> Result<BooleanChunked> {
        let ca = self.as_utf8();

        let reg = Regex::new(pat)?;
        let f = |s| reg.is_match(s);
        let mut out: BooleanChunked = if !ca.has_validity() {
            ca.into_no_null_iter().map(f).collect()
        } else {
            ca.into_iter().map(|opt_s| opt_s.map(f)).collect()
        };
        out.rename(ca.name());
        Ok(out)
    }

    /// Replace the leftmost (sub)string by a regex pattern
    fn replace(&self, pat: &str, val: &str) -> Result<Utf8Chunked> {
        let ca = self.as_utf8();
        let reg = Regex::new(pat)?;
        let f = |s| reg.replace(s, val);
        Ok(ca.apply(f))
    }

    /// Replace all (sub)strings by a regex pattern
    fn replace_all(&self, pat: &str, val: &str) -> Result<Utf8Chunked> {
        let ca = self.as_utf8();
        let reg = Regex::new(pat)?;
        let f = |s| reg.replace_all(s, val);
        Ok(ca.apply(f))
    }

    /// Extract the nth capture group from pattern
    fn extract(&self, pat: &str, group_index: usize) -> Result<Utf8Chunked> {
        let ca = self.as_utf8();
        let reg = Regex::new(pat)?;
        Ok(ca.apply_on_opt(|e| e.and_then(|input| f_regex_extract(&reg, input, group_index))))
    }

    /// Extract each successive non-overlapping regex match in an individual string as an array
    fn extract_all(&self, pat: &str) -> Result<ListChunked> {
        let ca = self.as_utf8();
        let reg = Regex::new(pat)?;

        let mut builder = ListUtf8ChunkedBuilder::new(ca.name(), ca.len(), ca.get_values_size());

        for opt_s in ca.into_iter() {
            match opt_s {
                None => builder.append_null(),
                Some(s) => {
                    let mut iter = reg.find_iter(s).map(|m| m.as_str()).peekable();
                    if iter.peek().is_some() {
                        builder.append_values_iter(iter);
                    } else {
                        builder.append_null()
                    }
                }
            }
        }
        Ok(builder.finish())
    }

    /// Count all successive non-overlapping regex matches.
    fn count_match(&self, pat: &str) -> Result<UInt32Chunked> {
        let ca = self.as_utf8();
        let reg = Regex::new(pat)?;

        let mut out: UInt32Chunked = ca
            .into_iter()
            .map(|opt_s| opt_s.map(|s| reg.find_iter(s).count() as u32))
            .collect();
        out.rename(ca.name());
        Ok(out)
    }

    /// Modify the strings to their lowercase equivalent
    #[must_use]
    fn to_lowercase(&self) -> Utf8Chunked {
        let ca = self.as_utf8();
        ca.apply(|s| str::to_lowercase(s).into())
    }

    /// Modify the strings to their uppercase equivalent
    #[must_use]
    fn to_uppercase(&self) -> Utf8Chunked {
        let ca = self.as_utf8();
        ca.apply(|s| str::to_uppercase(s).into())
    }

    /// Concat with the values from a second Utf8Chunked
    #[must_use]
    fn concat(&self, other: &Utf8Chunked) -> Utf8Chunked {
        let ca = self.as_utf8();
        ca + other
    }

    /// Slice the string values
    /// Determines a substring starting from `start` and with optional length `length` of each of the elements in `array`.
    /// `start` can be negative, in which case the start counts from the end of the string.
    fn str_slice(&self, start: i64, length: Option<u64>) -> Result<Utf8Chunked> {
        let ca = self.as_utf8();
        let chunks = ca
            .downcast_iter()
            .map(|c| Ok(substring(c, start, &length)?.into()))
            .collect::<arrow::error::Result<_>>()?;

        Ok(Utf8Chunked::from_chunks(ca.name(), chunks))
    }
}

impl Utf8NameSpaceImpl for Utf8Chunked {}
