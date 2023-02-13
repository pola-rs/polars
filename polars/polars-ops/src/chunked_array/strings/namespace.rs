use std::borrow::Cow;

#[cfg(feature = "string_encoding")]
use base64::engine::general_purpose;
#[cfg(feature = "string_encoding")]
use base64::Engine as _;
use polars_arrow::export::arrow::compute::substring::substring;
use polars_arrow::export::arrow::{self};
use polars_arrow::kernels::string::*;
use polars_core::export::num::Num;
use polars_core::export::regex::{escape, NoExpand, Regex};

use super::*;
#[cfg(feature = "string_encoding")]
use crate::chunked_array::binary::BinaryNameSpaceImpl;

fn f_regex_extract<'a>(reg: &Regex, input: &'a str, group_index: usize) -> Option<Cow<'a, str>> {
    reg.captures(input)
        .and_then(|cap| cap.get(group_index).map(|m| Cow::Borrowed(m.as_str())))
}

pub trait Utf8NameSpaceImpl: AsUtf8 {
    #[cfg(not(feature = "binary_encoding"))]
    fn hex_decode(&self) -> PolarsResult<Utf8Chunked> {
        panic!("activate 'dtype-binary' feature")
    }

    #[cfg(feature = "binary_encoding")]
    fn hex_decode(&self, strict: bool) -> PolarsResult<BinaryChunked> {
        let ca = self.as_utf8();
        ca.cast_unchecked(&DataType::Binary)?
            .binary()?
            .hex_decode(strict)
    }

    #[must_use]
    #[cfg(feature = "string_encoding")]
    fn hex_encode(&self) -> Utf8Chunked {
        let ca = self.as_utf8();
        ca.apply(|s| hex::encode(s).into())
    }

    #[cfg(not(feature = "binary_encoding"))]
    fn base64_decode(&self) -> PolarsResult<Utf8Chunked> {
        panic!("activate 'dtype-binary' feature")
    }

    #[cfg(feature = "binary_encoding")]
    fn base64_decode(&self, strict: bool) -> PolarsResult<BinaryChunked> {
        let ca = self.as_utf8();
        ca.cast_unchecked(&DataType::Binary)?
            .binary()?
            .base64_decode(strict)
    }

    #[must_use]
    #[cfg(feature = "string_encoding")]
    fn base64_encode(&self) -> Utf8Chunked {
        let ca = self.as_utf8();
        ca.apply(|s| general_purpose::STANDARD.encode(s).into())
    }

    #[cfg(feature = "string_from_radix")]
    // Parse a string number with base _radix_ into a decimal (i32)
    fn parse_int(&self, radix: Option<u32>) -> Int32Chunked {
        let ca = self.as_utf8();

        let f = |s: &str| <i32 as Num>::from_str_radix(s, radix.unwrap_or(2)).unwrap();
        ca.apply_cast_numeric(f)
    }

    /// Get the length of the string values as number of chars.
    fn str_n_chars(&self) -> UInt32Chunked {
        let ca = self.as_utf8();
        ca.apply_kernel_cast(&string_nchars)
    }

    /// Get the length of the string values as number of bytes.
    fn str_lengths(&self) -> UInt32Chunked {
        let ca = self.as_utf8();
        ca.apply_kernel_cast(&string_lengths)
    }

    /// Return a copy of the string left filled with ASCII '0' digits to make a string of length width.
    /// A leading sign prefix ('+'/'-') is handled by inserting the padding after the sign character
    /// rather than before.
    /// The original string is returned if width is less than or equal to `s.len()`.
    #[cfg(feature = "string_justify")]
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

    /// Check if strings contain a regex pattern.
    fn contains(&self, pat: &str) -> PolarsResult<BooleanChunked> {
        let ca = self.as_utf8();
        let reg = Regex::new(pat)?;
        let f = |s: &str| reg.is_match(s);
        let mut out: BooleanChunked = if !ca.has_validity() {
            ca.into_no_null_iter().map(f).collect()
        } else {
            ca.into_iter().map(|opt_s| opt_s.map(f)).collect()
        };
        out.rename(ca.name());
        Ok(out)
    }

    /// Check if strings contain a given literal
    fn contains_literal(&self, lit: &str) -> PolarsResult<BooleanChunked> {
        // note: benchmarking shows that the regex engine is actually
        // faster at finding literal matches than str::contains.
        // ref: https://github.com/pola-rs/polars/pull/6811
        self.contains(escape(lit).as_str())
    }

    /// Check if strings ends with a substring
    fn ends_with(&self, sub: &str) -> BooleanChunked {
        let ca = self.as_utf8();
        let f = |s: &str| s.ends_with(sub);
        let mut out: BooleanChunked = ca.into_iter().map(|opt_s| opt_s.map(f)).collect();
        out.rename(ca.name());
        out
    }

    /// Check if strings starts with a substring
    fn starts_with(&self, sub: &str) -> BooleanChunked {
        let ca = self.as_utf8();
        let f = |s: &str| s.starts_with(sub);
        let mut out: BooleanChunked = ca.into_iter().map(|opt_s| opt_s.map(f)).collect();
        out.rename(ca.name());
        out
    }

    /// Replace the leftmost regex-matched (sub)string with another string
    fn replace<'a>(&'a self, pat: &str, val: &str) -> PolarsResult<Utf8Chunked> {
        let reg = Regex::new(pat)?;
        let f = |s: &'a str| reg.replace(s, val);
        let ca = self.as_utf8();
        Ok(ca.apply(f))
    }

    /// Replace the leftmost literal (sub)string with another string
    fn replace_literal<'a>(&'a self, pat: &str, val: &str) -> PolarsResult<Utf8Chunked> {
        // note: benchmarking shows that using the regex engine for literal
        // replacement is faster than str::replacen in almost all cases.
        // ref: https://github.com/pola-rs/polars/pull/6777
        let reg = Regex::new(escape(pat).as_str())?;
        let f = |s: &'a str| reg.replace(s, NoExpand(val));
        let ca = self.as_utf8();
        Ok(ca.apply(f))
    }

    /// Replace all regex-matched (sub)strings with another string
    fn replace_all(&self, pat: &str, val: &str) -> PolarsResult<Utf8Chunked> {
        let ca = self.as_utf8();
        let reg = Regex::new(pat)?;
        Ok(ca.apply(|s| reg.replace_all(s, val)))
    }

    /// Replace all matching literal (sub)strings with another string
    fn replace_literal_all(&self, pat: &str, val: &str) -> PolarsResult<Utf8Chunked> {
        let ca = self.as_utf8();
        let reg = Regex::new(escape(pat).as_str())?;
        Ok(ca.apply(|s| reg.replace_all(s, NoExpand(val))))
    }

    /// Extract the nth capture group from pattern
    fn extract(&self, pat: &str, group_index: usize) -> PolarsResult<Utf8Chunked> {
        let ca = self.as_utf8();
        let reg = Regex::new(pat)?;
        Ok(ca.apply_on_opt(|e| e.and_then(|input| f_regex_extract(&reg, input, group_index))))
    }

    /// Extract each successive non-overlapping regex match in an individual string as an array
    fn extract_all(&self, pat: &str) -> PolarsResult<ListChunked> {
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

    /// Extract each successive non-overlapping regex match in an individual string as an array
    fn extract_all_many(&self, pat: &Utf8Chunked) -> PolarsResult<ListChunked> {
        let ca = self.as_utf8();
        if ca.len() != pat.len() {
            return Err(PolarsError::ComputeError(
                "pattern's length does not match that of the argument Series".into(),
            ));
        }

        let mut builder = ListUtf8ChunkedBuilder::new(ca.name(), ca.len(), ca.get_values_size());

        for (opt_s, opt_pat) in ca.into_iter().zip(pat.into_iter()) {
            match (opt_s, opt_pat) {
                (_, None) | (None, _) => builder.append_null(),
                (Some(s), Some(pat)) => {
                    let reg = Regex::new(pat)?;
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
    fn count_match(&self, pat: &str) -> PolarsResult<UInt32Chunked> {
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
    fn str_slice(&self, start: i64, length: Option<u64>) -> PolarsResult<Utf8Chunked> {
        let ca = self.as_utf8();
        let chunks = ca
            .downcast_iter()
            .map(|c| substring(c, start, &length))
            .collect::<arrow::error::Result<_>>()?;

        unsafe { Ok(Utf8Chunked::from_chunks(ca.name(), chunks)) }
    }
}

impl Utf8NameSpaceImpl for Utf8Chunked {}
