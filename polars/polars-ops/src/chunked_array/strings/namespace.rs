use std::borrow::Cow;

#[cfg(feature = "string_encoding")]
use base64::engine::general_purpose;
#[cfg(feature = "string_encoding")]
use base64::Engine as _;
use polars_arrow::export::arrow::compute::substring::substring;
use polars_arrow::export::arrow::{self};
use polars_arrow::kernels::string::*;
#[cfg(feature = "string_from_radix")]
use polars_core::export::num::Num;
use polars_core::export::regex::{escape, Regex};

use super::*;
#[cfg(feature = "binary_encoding")]
use crate::chunked_array::binary::BinaryNameSpaceImpl;

fn f_regex_extract<'a>(reg: &Regex, input: &'a str, group_index: usize) -> Option<Cow<'a, str>> {
    reg.captures(input)
        .and_then(|cap| cap.get(group_index).map(|m| Cow::Borrowed(m.as_str())))
}

pub trait Utf8NameSpaceImpl: AsUtf8 {
    #[cfg(not(feature = "binary_encoding"))]
    fn hex_decode(&self) -> PolarsResult<Utf8Chunked> {
        panic!("activate 'binary_encoding' feature")
    }

    #[cfg(feature = "binary_encoding")]
    fn hex_decode(&self, strict: bool) -> PolarsResult<BinaryChunked> {
        let ca = self.as_utf8();
        ca.as_binary().hex_decode(strict)
    }

    #[must_use]
    #[cfg(feature = "string_encoding")]
    fn hex_encode(&self) -> Utf8Chunked {
        let ca = self.as_utf8();
        ca.apply(|s| hex::encode(s).into())
    }

    #[cfg(not(feature = "binary_encoding"))]
    fn base64_decode(&self) -> PolarsResult<Utf8Chunked> {
        panic!("activate 'binary_encoding' feature")
    }

    #[cfg(feature = "binary_encoding")]
    fn base64_decode(&self, strict: bool) -> PolarsResult<BinaryChunked> {
        let ca = self.as_utf8();
        ca.as_binary().base64_decode(strict)
    }

    #[must_use]
    #[cfg(feature = "string_encoding")]
    fn base64_encode(&self) -> Utf8Chunked {
        let ca = self.as_utf8();
        ca.apply(|s| general_purpose::STANDARD.encode(s).into())
    }

    #[cfg(feature = "string_from_radix")]
    // Parse a string number with base _radix_ into a decimal (i32)
    fn parse_int(&self, radix: u32, strict: bool) -> PolarsResult<Int32Chunked> {
        use polars_arrow::utils::CustomIterTools;
        let ca = self.as_utf8();
        let f = |opt_s: Option<&str>| -> Option<i32> {
            opt_s.and_then(|s| <i32 as Num>::from_str_radix(s, radix).ok())
        };
        let out: Int32Chunked = ca.into_iter().map(f).collect_trusted();

        if strict && ca.null_count() != out.null_count() {
            let failure_mask = !ca.is_null() & out.is_null();
            let all_failures = ca.filter(&failure_mask)?;
            let n_failures = all_failures.len();
            let some_failures = all_failures.unique()?.slice(0, 10).sort(false);
            let some_error_msg = some_failures
                .get(0)
                .and_then(|s| <i32 as Num>::from_str_radix(s, radix).err())
                .map_or_else(
                    || unreachable!("failed to extract ParseIntError"),
                    |e| format!("{}", e),
                );
            polars_bail!(
                ComputeError:
                "strict integer parsing failed for {} value(s): {}; error message for the \
                first shown value: '{}' (consider non-strict parsing)",
                n_failures,
                some_failures.into_series().fmt_list(),
                some_error_msg
            );
        };

        Ok(out)
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
    fn zfill(&self, alignment: usize) -> Utf8Chunked {
        let ca = self.as_utf8();
        justify::zfill(ca, alignment)
    }

    /// Return the string left justified in a string of length width.
    /// Padding is done using the specified `fillchar`,
    /// The original string is returned if width is less than or equal to `s.len()`.
    #[cfg(feature = "string_justify")]
    fn ljust(&self, width: usize, fillchar: char) -> Utf8Chunked {
        let ca = self.as_utf8();
        justify::ljust(ca, width, fillchar)
    }

    /// Return the string right justified in a string of length width.
    /// Padding is done using the specified `fillchar`,
    /// The original string is returned if width is less than or equal to `s.len()`.
    #[cfg(feature = "string_justify")]
    fn rjust(&self, width: usize, fillchar: char) -> Utf8Chunked {
        let ca = self.as_utf8();
        justify::rjust(ca, width, fillchar)
    }

    /// Check if strings contain a regex pattern.
    fn contains(&self, pat: &str, strict: bool) -> PolarsResult<BooleanChunked> {
        let ca = self.as_utf8();

        let res_reg = Regex::new(pat);
        let opt_reg = if strict { Some(res_reg?) } else { res_reg.ok() };

        let mut out: BooleanChunked = match (opt_reg, ca.has_validity()) {
            (Some(reg), false) => ca
                .into_no_null_iter()
                .map(|s: &str| reg.is_match(s))
                .collect(),
            (Some(reg), true) => ca
                .into_iter()
                .map(|opt_s| opt_s.map(|s: &str| reg.is_match(s)))
                .collect(),
            (None, _) => ca.into_iter().map(|_| None).collect(),
        };
        out.rename(ca.name());
        Ok(out)
    }

    /// Check if strings contain a given literal
    fn contains_literal(&self, lit: &str) -> PolarsResult<BooleanChunked> {
        // note: benchmarking shows that the regex engine is actually
        // faster at finding literal matches than str::contains.
        // ref: https://github.com/pola-rs/polars/pull/6811
        self.contains(escape(lit).as_str(), true)
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
    fn replace_literal<'a>(&'a self, pat: &str, val: &str, n: usize) -> PolarsResult<Utf8Chunked> {
        let ca = self.as_utf8();
        if ca.is_empty() {
            return Ok(ca.clone());
        }

        // for single bytes we can replace on the whole values buffer
        if pat.len() == 1 && val.len() == 1 {
            let pat = pat.as_bytes()[0];
            let val = val.as_bytes()[0];
            return Ok(
                ca.apply_kernel(&|arr| Box::new(replace::replace_lit_n_char(arr, n, pat, val)))
            );
        }
        if pat.len() == val.len() {
            return Ok(
                ca.apply_kernel(&|arr| Box::new(replace::replace_lit_n_str(arr, n, pat, val)))
            );
        }

        // amortize allocation
        let mut buf = String::new();

        let f = move |s: &'a str| {
            buf.clear();
            let mut changed = false;

            // See: str.replacen
            let mut last_end = 0;
            for (start, part) in s.match_indices(pat).take(n) {
                changed = true;
                buf.push_str(unsafe { s.get_unchecked(last_end..start) });
                buf.push_str(val);
                last_end = start + part.len();
            }
            buf.push_str(unsafe { s.get_unchecked(last_end..s.len()) });

            if changed {
                // extend lifetime
                // lifetime is bound to 'a
                let slice = buf.as_str();
                unsafe { std::mem::transmute::<&str, &'a str>(slice) }
            } else {
                s
            }
        };
        Ok(ca.apply_mut(f))
    }

    /// Replace all regex-matched (sub)strings with another string
    fn replace_all(&self, pat: &str, val: &str) -> PolarsResult<Utf8Chunked> {
        let ca = self.as_utf8();
        let reg = Regex::new(pat)?;
        Ok(ca.apply(|s| reg.replace_all(s, val)))
    }

    /// Replace all matching literal (sub)strings with another string
    fn replace_literal_all<'a>(&'a self, pat: &str, val: &str) -> PolarsResult<Utf8Chunked> {
        let ca = self.as_utf8();
        if ca.is_empty() {
            return Ok(ca.clone());
        }
        // for single bytes we can replace on the whole values buffer
        if pat.len() == 1 && val.len() == 1 {
            let pat = pat.as_bytes()[0];
            let val = val.as_bytes()[0];
            return Ok(
                ca.apply_kernel(&|arr| Box::new(replace::replace_lit_single_char(arr, pat, val)))
            );
        }
        if pat.len() == val.len() {
            return Ok(ca.apply_kernel(&|arr| {
                Box::new(replace::replace_lit_n_str(arr, usize::MAX, pat, val))
            }));
        }

        // amortize allocation
        let mut buf = String::new();

        let f = move |s: &'a str| {
            buf.clear();
            let mut changed = false;

            // See: str.replace
            let mut last_end = 0;
            for (start, part) in s.match_indices(pat) {
                changed = true;
                buf.push_str(unsafe { s.get_unchecked(last_end..start) });
                buf.push_str(val);
                last_end = start + part.len();
            }
            buf.push_str(unsafe { s.get_unchecked(last_end..s.len()) });

            if changed {
                // extend lifetime
                // lifetime is bound to 'a
                let slice = buf.as_str();
                unsafe { std::mem::transmute::<&str, &'a str>(slice) }
            } else {
                s
            }
        };

        Ok(ca.apply_mut(f))
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
        polars_ensure!(
            ca.len() == pat.len(),
            ComputeError: "pattern's length: {} does not match that of the argument series: {}",
            pat.len(), ca.len(),
        );

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
        case::to_lowercase(ca)
    }

    /// Modify the strings to their uppercase equivalent
    #[must_use]
    fn to_uppercase(&self) -> Utf8Chunked {
        let ca = self.as_utf8();
        case::to_uppercase(ca)
    }

    /// Modify the strings to their titlecase equivalent
    #[must_use]
    #[cfg(feature = "nightly")]
    fn to_titlecase(&self) -> Utf8Chunked {
        let ca = self.as_utf8();
        case::to_titlecase(ca)
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
