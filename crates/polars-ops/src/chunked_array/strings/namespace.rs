use arrow::array::ValueSize;
use arrow::legacy::kernels::string::*;
#[cfg(feature = "string_encoding")]
use base64::engine::general_purpose;
#[cfg(feature = "string_encoding")]
use base64::Engine as _;
#[cfg(feature = "string_to_integer")]
use polars_core::export::num::Num;
use polars_core::export::regex::Regex;
use polars_core::prelude::arity::*;
use polars_utils::cache::FastFixedCache;
use regex::escape;

use super::*;
#[cfg(feature = "binary_encoding")]
use crate::chunked_array::binary::BinaryNameSpaceImpl;

// We need this to infer the right lifetimes for the match closure.
#[inline(always)]
fn infer_re_match<F>(f: F) -> F
where
    F: for<'a, 'b> FnMut(Option<&'a str>, Option<&'b str>) -> Option<bool>,
{
    f
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
        ca.apply_values(|s| hex::encode(s).into())
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
        ca.apply_values(|s| general_purpose::STANDARD.encode(s).into())
    }

    #[cfg(feature = "string_to_integer")]
    // Parse a string number with base _radix_ into a decimal (i64)
    fn to_integer(&self, base: u32, strict: bool) -> PolarsResult<Int64Chunked> {
        let ca = self.as_utf8();
        let f = |opt_s: Option<&str>| -> Option<i64> {
            opt_s.and_then(|s| <i64 as Num>::from_str_radix(s, base).ok())
        };
        let out: Int64Chunked = ca.apply_generic(f);

        if strict && ca.null_count() != out.null_count() {
            let failure_mask = !ca.is_null() & out.is_null();
            let all_failures = ca.filter(&failure_mask)?;
            let n_failures = all_failures.len();
            let some_failures = all_failures.unique()?.slice(0, 10).sort(false);
            let some_error_msg = some_failures
                .get(0)
                .and_then(|s| <i64 as Num>::from_str_radix(s, base).err())
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

    fn contains_chunked(
        &self,
        pat: &Utf8Chunked,
        literal: bool,
        strict: bool,
    ) -> PolarsResult<BooleanChunked> {
        let ca = self.as_utf8();
        match pat.len() {
            1 => match pat.get(0) {
                Some(pat) => {
                    if literal {
                        ca.contains_literal(pat)
                    } else {
                        ca.contains(pat, strict)
                    }
                },
                None => Ok(BooleanChunked::full_null(ca.name(), ca.len())),
            },
            _ => {
                if literal {
                    Ok(binary_elementwise_values(ca, pat, |src, pat| {
                        src.contains(pat)
                    }))
                } else if strict {
                    // A sqrt(n) regex cache is not too small, not too large.
                    let mut reg_cache = FastFixedCache::new((ca.len() as f64).sqrt() as usize);
                    try_binary_elementwise(ca, pat, |opt_src, opt_pat| match (opt_src, opt_pat) {
                        (Some(src), Some(pat)) => {
                            let reg = reg_cache.try_get_or_insert_with(pat, |p| Regex::new(p))?;
                            Ok(Some(reg.is_match(src)))
                        },
                        _ => Ok(None),
                    })
                } else {
                    // A sqrt(n) regex cache is not too small, not too large.
                    let mut reg_cache = FastFixedCache::new((ca.len() as f64).sqrt() as usize);
                    Ok(binary_elementwise(
                        ca,
                        pat,
                        infer_re_match(|src, pat| {
                            let reg = reg_cache.try_get_or_insert_with(pat?, |p| Regex::new(p));
                            Some(reg.ok()?.is_match(src?))
                        }),
                    ))
                }
            },
        }
    }

    /// Get the length of the string values as number of chars.
    fn str_len_chars(&self) -> UInt32Chunked {
        let ca = self.as_utf8();
        ca.apply_kernel_cast(&string_len_chars)
    }

    /// Get the length of the string values as number of bytes.
    fn str_len_bytes(&self) -> UInt32Chunked {
        let ca = self.as_utf8();
        ca.apply_kernel_cast(&string_len_bytes)
    }

    /// Pad the start of the string until it reaches the given length.
    ///
    /// Padding is done using the specified `fill_char`.
    /// Strings with length equal to or greater than the given length are
    /// returned as-is.
    #[cfg(feature = "string_pad")]
    fn pad_start(&self, length: usize, fill_char: char) -> Utf8Chunked {
        let ca = self.as_utf8();
        pad::pad_start(ca, length, fill_char)
    }

    /// Pad the end of the string until it reaches the given length.
    ///
    /// Padding is done using the specified `fill_char`.
    /// Strings with length equal to or greater than the given length are
    /// returned as-is.
    #[cfg(feature = "string_pad")]
    fn pad_end(&self, length: usize, fill_char: char) -> Utf8Chunked {
        let ca = self.as_utf8();
        pad::pad_end(ca, length, fill_char)
    }

    /// Pad the start of the string with zeros until it reaches the given length.
    ///
    /// A sign prefix (`-`) is handled by inserting the padding after the sign
    /// character rather than before.
    /// Strings with length equal to or greater than the given length are
    /// returned as-is.
    #[cfg(feature = "string_pad")]
    fn zfill(&self, length: usize) -> Utf8Chunked {
        let ca = self.as_utf8();
        pad::zfill(ca, length)
    }

    /// Check if strings contain a regex pattern.
    fn contains(&self, pat: &str, strict: bool) -> PolarsResult<BooleanChunked> {
        let ca = self.as_utf8();

        let res_reg = Regex::new(pat);
        let opt_reg = if strict { Some(res_reg?) } else { res_reg.ok() };

        let out: BooleanChunked = if let Some(reg) = opt_reg {
            ca.apply_values_generic(|s| reg.is_match(s))
        } else {
            BooleanChunked::full_null(ca.name(), ca.len())
        };
        Ok(out)
    }

    /// Check if strings contain a given literal
    fn contains_literal(&self, lit: &str) -> PolarsResult<BooleanChunked> {
        // note: benchmarking shows that the regex engine is actually
        // faster at finding literal matches than str::contains.
        // ref: https://github.com/pola-rs/polars/pull/6811
        self.contains(regex::escape(lit).as_str(), true)
    }

    /// Replace the leftmost regex-matched (sub)string with another string
    fn replace<'a>(&'a self, pat: &str, val: &str) -> PolarsResult<Utf8Chunked> {
        let reg = Regex::new(pat)?;
        let f = |s: &'a str| reg.replace(s, val);
        let ca = self.as_utf8();
        Ok(ca.apply_values(f))
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
        Ok(ca.apply_values(|s| reg.replace_all(s, val)))
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

        // Amortize allocation.
        let mut buf = String::new();

        let f = move |s: &'a str| {
            buf.clear();
            let mut changed = false;

            // See: str.replace.
            let mut last_end = 0;
            for (start, part) in s.match_indices(pat) {
                changed = true;
                buf.push_str(unsafe { s.get_unchecked(last_end..start) });
                buf.push_str(val);
                last_end = start + part.len();
            }
            buf.push_str(unsafe { s.get_unchecked(last_end..s.len()) });

            if changed {
                // Extend lifetime, lifetime is bound to 'a.
                let slice = buf.as_str();
                unsafe { std::mem::transmute::<&str, &'a str>(slice) }
            } else {
                s
            }
        };

        Ok(ca.apply_mut(f))
    }

    /// Extract the nth capture group from pattern.
    fn extract(&self, pat: &str, group_index: usize) -> PolarsResult<Utf8Chunked> {
        let ca = self.as_utf8();
        super::extract::extract_group(ca, pat, group_index)
    }

    /// Extract each successive non-overlapping regex match in an individual string as an array.
    fn extract_all(&self, pat: &str) -> PolarsResult<ListChunked> {
        let ca = self.as_utf8();
        let reg = Regex::new(pat)?;

        let mut builder = ListUtf8ChunkedBuilder::new(ca.name(), ca.len(), ca.get_values_size());
        for opt_s in ca.into_iter() {
            match opt_s {
                None => builder.append_null(),
                Some(s) => builder.append_values_iter(reg.find_iter(s).map(|m| m.as_str())),
            }
        }
        Ok(builder.finish())
    }

    fn strip_chars(&self, pat: &Series) -> PolarsResult<Utf8Chunked> {
        let ca = self.as_utf8();
        if pat.dtype() == &DataType::Null {
            Ok(ca.apply_generic(|opt_s| opt_s.map(|s| s.trim())))
        } else {
            Ok(strip_chars(ca, pat.utf8()?))
        }
    }

    fn strip_chars_start(&self, pat: &Series) -> PolarsResult<Utf8Chunked> {
        let ca = self.as_utf8();
        if pat.dtype() == &DataType::Null {
            return Ok(ca.apply_generic(|opt_s| opt_s.map(|s| s.trim_start())));
        } else {
            Ok(strip_chars_start(ca, pat.utf8()?))
        }
    }

    fn strip_chars_end(&self, pat: &Series) -> PolarsResult<Utf8Chunked> {
        let ca = self.as_utf8();
        if pat.dtype() == &DataType::Null {
            return Ok(ca.apply_generic(|opt_s| opt_s.map(|s| s.trim_end())));
        } else {
            Ok(strip_chars_end(ca, pat.utf8()?))
        }
    }

    fn strip_prefix(&self, prefix: &Utf8Chunked) -> Utf8Chunked {
        let ca = self.as_utf8();
        strip_prefix(ca, prefix)
    }

    fn strip_suffix(&self, suffix: &Utf8Chunked) -> Utf8Chunked {
        let ca = self.as_utf8();
        strip_suffix(ca, suffix)
    }

    #[cfg(feature = "dtype-struct")]
    fn split_exact(&self, by: &Utf8Chunked, n: usize) -> PolarsResult<StructChunked> {
        let ca = self.as_utf8();

        split_to_struct(ca, by, n + 1, |s, by| s.split(by))
    }

    #[cfg(feature = "dtype-struct")]
    fn split_exact_inclusive(&self, by: &Utf8Chunked, n: usize) -> PolarsResult<StructChunked> {
        let ca = self.as_utf8();

        split_to_struct(ca, by, n + 1, |s, by| s.split_inclusive(by))
    }

    #[cfg(feature = "dtype-struct")]
    fn splitn(&self, by: &Utf8Chunked, n: usize) -> PolarsResult<StructChunked> {
        let ca = self.as_utf8();

        split_to_struct(ca, by, n, |s, by| s.splitn(n, by))
    }

    fn split(&self, by: &Utf8Chunked) -> ListChunked {
        let ca = self.as_utf8();

        split_helper(ca, by, str::split)
    }

    fn split_inclusive(&self, by: &Utf8Chunked) -> ListChunked {
        let ca = self.as_utf8();

        split_helper(ca, by, str::split_inclusive)
    }

    /// Extract each successive non-overlapping regex match in an individual string as an array.
    fn extract_all_many(&self, pat: &Utf8Chunked) -> PolarsResult<ListChunked> {
        let ca = self.as_utf8();
        polars_ensure!(
            ca.len() == pat.len(),
            ComputeError: "pattern's length: {} does not match that of the argument series: {}",
            pat.len(), ca.len(),
        );

        // A sqrt(n) regex cache is not too small, not too large.
        let mut reg_cache = FastFixedCache::new((ca.len() as f64).sqrt() as usize);
        let mut builder = ListUtf8ChunkedBuilder::new(ca.name(), ca.len(), ca.get_values_size());
        binary_elementwise_for_each(ca, pat, |opt_s, opt_pat| match (opt_s, opt_pat) {
            (_, None) | (None, _) => builder.append_null(),
            (Some(s), Some(pat)) => {
                let reg = reg_cache.get_or_insert_with(pat, |p| Regex::new(p).unwrap());
                builder.append_values_iter(reg.find_iter(s).map(|m| m.as_str()));
            },
        });
        Ok(builder.finish())
    }

    #[cfg(feature = "extract_groups")]
    /// Extract all capture groups from pattern and return as a struct.
    fn extract_groups(&self, pat: &str, dtype: &DataType) -> PolarsResult<Series> {
        let ca = self.as_utf8();
        super::extract::extract_groups(ca, pat, dtype)
    }

    /// Count all successive non-overlapping regex matches.
    fn count_matches(&self, pat: &str, literal: bool) -> PolarsResult<UInt32Chunked> {
        let ca = self.as_utf8();
        let reg = if literal {
            Regex::new(escape(pat).as_str())?
        } else {
            Regex::new(pat)?
        };

        Ok(ca.apply_generic(|opt_s| opt_s.map(|s| reg.find_iter(s).count() as u32)))
    }

    /// Count all successive non-overlapping regex matches.
    fn count_matches_many(&self, pat: &Utf8Chunked, literal: bool) -> PolarsResult<UInt32Chunked> {
        let ca = self.as_utf8();
        polars_ensure!(
            ca.len() == pat.len(),
            ComputeError: "pattern's length: {} does not match that of the argument series: {}",
            pat.len(), ca.len(),
        );

        // A sqrt(n) regex cache is not too small, not too large.
        let mut reg_cache = FastFixedCache::new((ca.len() as f64).sqrt() as usize);
        let op = move |opt_s: Option<&str>, opt_pat: Option<&str>| -> PolarsResult<Option<u32>> {
            match (opt_s, opt_pat) {
                (Some(s), Some(pat)) => {
                    let reg = reg_cache.get_or_insert_with(pat, |p| {
                        if literal {
                            Regex::new(escape(p).as_str()).unwrap()
                        } else {
                            Regex::new(p).unwrap()
                        }
                    });
                    Ok(Some(reg.find_iter(s).count() as u32))
                },
                _ => Ok(None),
            }
        };

        let out: UInt32Chunked = try_binary_elementwise(ca, pat, op)?;

        Ok(out.with_name(ca.name()))
    }

    /// Modify the strings to their lowercase equivalent.
    #[must_use]
    fn to_lowercase(&self) -> Utf8Chunked {
        let ca = self.as_utf8();
        case::to_lowercase(ca)
    }

    /// Modify the strings to their uppercase equivalent.
    #[must_use]
    fn to_uppercase(&self) -> Utf8Chunked {
        let ca = self.as_utf8();
        case::to_uppercase(ca)
    }

    /// Modify the strings to their titlecase equivalent.
    #[must_use]
    #[cfg(feature = "nightly")]
    fn to_titlecase(&self) -> Utf8Chunked {
        let ca = self.as_utf8();
        case::to_titlecase(ca)
    }

    /// Concat with the values from a second Utf8Chunked.
    #[must_use]
    fn concat(&self, other: &Utf8Chunked) -> Utf8Chunked {
        let ca = self.as_utf8();
        ca + other
    }

    /// Reverses the string values
    #[must_use]
    fn str_reverse(&self) -> Utf8Chunked {
        let ca = self.as_utf8();
        reverse::reverse(ca)
    }

    /// Slice the string values.
    ///
    /// Determines a substring starting from `start` and with optional length `length` of each of the elements in `array`.
    /// `start` can be negative, in which case the start counts from the end of the string.
    fn str_slice(&self, start: i64, length: Option<u64>) -> Utf8Chunked {
        let ca = self.as_utf8();
        let iter = ca
            .downcast_iter()
            .map(|c| substring::utf8_substring(c, start, &length));
        Utf8Chunked::from_chunk_iter_like(ca, iter)
    }
}

impl Utf8NameSpaceImpl for Utf8Chunked {}
