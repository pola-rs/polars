use arrow::array::ValueSize;
use arrow::legacy::kernels::string::*;
#[cfg(feature = "string_encoding")]
use base64::Engine as _;
#[cfg(feature = "string_encoding")]
use base64::engine::general_purpose;
#[cfg(feature = "string_to_integer")]
use num_traits::Num;
use polars_core::prelude::arity::*;
use polars_utils::regex_cache::{compile_regex, with_regex_cache};

use super::*;
#[cfg(feature = "binary_encoding")]
use crate::chunked_array::binary::BinaryNameSpaceImpl;
#[cfg(feature = "string_normalize")]
use crate::prelude::strings::normalize::UnicodeForm;

// We need this to infer the right lifetimes for the match closure.
#[inline(always)]
fn infer_re_match<F>(f: F) -> F
where
    F: for<'a, 'b> FnMut(Option<&'a str>, Option<&'b str>) -> Option<bool>,
{
    f
}

pub trait StringNameSpaceImpl: AsString {
    #[cfg(not(feature = "binary_encoding"))]
    fn hex_decode(&self) -> PolarsResult<StringChunked> {
        panic!("activate 'binary_encoding' feature")
    }

    #[cfg(feature = "binary_encoding")]
    fn hex_decode(&self, strict: bool) -> PolarsResult<BinaryChunked> {
        let ca = self.as_string();
        ca.as_binary().hex_decode(strict)
    }

    #[must_use]
    #[cfg(feature = "string_encoding")]
    fn hex_encode(&self) -> StringChunked {
        let ca = self.as_string();
        ca.apply_values(|s| hex::encode(s).into())
    }

    #[cfg(not(feature = "binary_encoding"))]
    fn base64_decode(&self) -> PolarsResult<StringChunked> {
        panic!("activate 'binary_encoding' feature")
    }

    #[cfg(feature = "binary_encoding")]
    fn base64_decode(&self, strict: bool) -> PolarsResult<BinaryChunked> {
        let ca = self.as_string();
        ca.as_binary().base64_decode(strict)
    }

    #[must_use]
    #[cfg(feature = "string_encoding")]
    fn base64_encode(&self) -> StringChunked {
        let ca = self.as_string();
        ca.apply_values(|s| general_purpose::STANDARD.encode(s).into())
    }

    #[cfg(feature = "string_to_integer")]
    // Parse a string number with base _radix_ into a decimal (i64)
    fn to_integer(&self, base: &UInt32Chunked, strict: bool) -> PolarsResult<Int64Chunked> {
        let ca = self.as_string();

        polars_ensure!(
            ca.len() == base.len() || ca.len() == 1 || base.len() == 1,
            length_mismatch = "str.to_integer",
            ca.len(),
            base.len()
        );

        let f = |opt_s: Option<&str>, opt_base: Option<u32>| -> PolarsResult<Option<i64>> {
            let (Some(s), Some(base)) = (opt_s, opt_base) else {
                return Ok(None);
            };

            if !(2..=36).contains(&base) {
                polars_bail!(ComputeError: "`to_integer` called with invalid base '{base}'");
            }

            Ok(<i64 as Num>::from_str_radix(s, base).ok())
        };
        let out = broadcast_try_binary_elementwise(ca, base, f)?;
        if strict && ca.null_count() != out.null_count() {
            let failure_mask = ca.is_not_null() & out.is_null() & base.is_not_null();
            let n_failures = failure_mask.num_trues();
            if n_failures == 0 {
                return Ok(out);
            }

            let some_failures = if ca.len() == 1 {
                ca.clone()
            } else {
                let all_failures = ca.filter(&failure_mask)?;
                // `.unique()` does not necessarily preserve the original order.
                let unique_failures_args = all_failures.arg_unique()?;
                all_failures.take(&unique_failures_args.slice(0, 10))?
            };
            let some_error_msg = match base.len() {
                1 => {
                    // we can ensure that base is not null.
                    let base = base.get(0).unwrap();
                    some_failures
                        .get(0)
                        .and_then(|s| <i64 as Num>::from_str_radix(s, base).err())
                        .map_or_else(
                            || unreachable!("failed to extract ParseIntError"),
                            |e| format!("{}", e),
                        )
                },
                _ => {
                    let base_failures = base.filter(&failure_mask)?;
                    some_failures
                        .get(0)
                        .zip(base_failures.get(0))
                        .and_then(|(s, base)| <i64 as Num>::from_str_radix(s, base).err())
                        .map_or_else(
                            || unreachable!("failed to extract ParseIntError"),
                            |e| format!("{}", e),
                        )
                },
            };
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
        pat: &StringChunked,
        literal: bool,
        strict: bool,
    ) -> PolarsResult<BooleanChunked> {
        let ca = self.as_string();
        match (ca.len(), pat.len()) {
            (_, 1) => match pat.get(0) {
                Some(pat) => {
                    if literal {
                        ca.contains_literal(pat)
                    } else {
                        ca.contains(pat, strict)
                    }
                },
                None => Ok(BooleanChunked::full_null(ca.name().clone(), ca.len())),
            },
            (1, _) if ca.null_count() == 1 => Ok(BooleanChunked::full_null(
                ca.name().clone(),
                ca.len().max(pat.len()),
            )),
            _ => {
                if literal {
                    Ok(broadcast_binary_elementwise_values(ca, pat, |src, pat| {
                        src.contains(pat)
                    }))
                } else if strict {
                    with_regex_cache(|reg_cache| {
                        broadcast_try_binary_elementwise(ca, pat, |opt_src, opt_pat| {
                            match (opt_src, opt_pat) {
                                (Some(src), Some(pat)) => {
                                    let reg = reg_cache.compile(pat)?;
                                    Ok(Some(reg.is_match(src)))
                                },
                                _ => Ok(None),
                            }
                        })
                    })
                } else {
                    with_regex_cache(|reg_cache| {
                        Ok(broadcast_binary_elementwise(
                            ca,
                            pat,
                            infer_re_match(|src, pat| {
                                let reg = reg_cache.compile(pat?).ok()?;
                                Some(reg.is_match(src?))
                            }),
                        ))
                    })
                }
            },
        }
    }

    fn find_chunked(
        &self,
        pat: &StringChunked,
        literal: bool,
        strict: bool,
    ) -> PolarsResult<UInt32Chunked> {
        let ca = self.as_string();
        if pat.len() == 1 {
            return if let Some(pat) = pat.get(0) {
                if literal {
                    ca.find_literal(pat)
                } else {
                    ca.find(pat, strict)
                }
            } else {
                Ok(UInt32Chunked::full_null(ca.name().clone(), ca.len()))
            };
        } else if ca.len() == 1 && ca.null_count() == 1 {
            return Ok(UInt32Chunked::full_null(
                ca.name().clone(),
                ca.len().max(pat.len()),
            ));
        }
        if literal {
            Ok(broadcast_binary_elementwise(
                ca,
                pat,
                |src: Option<&str>, pat: Option<&str>| src?.find(pat?).map(|idx| idx as u32),
            ))
        } else {
            with_regex_cache(|reg_cache| {
                let matcher = |src: Option<&str>, pat: Option<&str>| -> PolarsResult<Option<u32>> {
                    if let (Some(src), Some(pat)) = (src, pat) {
                        let re = reg_cache.compile(pat)?;
                        return Ok(re.find(src).map(|m| m.start() as u32));
                    }
                    Ok(None)
                };
                broadcast_try_binary_elementwise(ca, pat, matcher)
            })
        }
    }

    /// Get the length of the string values as number of chars.
    fn str_len_chars(&self) -> UInt32Chunked {
        let ca = self.as_string();
        ca.apply_kernel_cast(&string_len_chars)
    }

    /// Get the length of the string values as number of bytes.
    fn str_len_bytes(&self) -> UInt32Chunked {
        let ca = self.as_string();
        ca.apply_kernel_cast(&utf8view_len_bytes)
    }

    /// Pad the start of the string until it reaches the given length.
    ///
    /// Padding is done using the specified `fill_char`.
    /// Strings with length equal to or greater than the given length are
    /// returned as-is.
    #[cfg(feature = "string_pad")]
    fn pad_start(&self, length: usize, fill_char: char) -> StringChunked {
        let ca = self.as_string();
        pad::pad_start(ca, length, fill_char)
    }

    /// Pad the end of the string until it reaches the given length.
    ///
    /// Padding is done using the specified `fill_char`.
    /// Strings with length equal to or greater than the given length are
    /// returned as-is.
    #[cfg(feature = "string_pad")]
    fn pad_end(&self, length: usize, fill_char: char) -> StringChunked {
        let ca = self.as_string();
        pad::pad_end(ca, length, fill_char)
    }

    /// Pad the start of the string with zeros until it reaches the given length.
    ///
    /// A sign prefix (`-`) is handled by inserting the padding after the sign
    /// character rather than before.
    /// Strings with length equal to or greater than the given length are
    /// returned as-is.
    #[cfg(feature = "string_pad")]
    fn zfill(&self, length: &UInt64Chunked) -> StringChunked {
        let ca = self.as_string();
        pad::zfill(ca, length)
    }

    /// Check if strings contain a regex pattern.
    fn contains(&self, pat: &str, strict: bool) -> PolarsResult<BooleanChunked> {
        let ca = self.as_string();
        let res_reg = polars_utils::regex_cache::compile_regex(pat);
        let opt_reg = if strict { Some(res_reg?) } else { res_reg.ok() };
        let out: BooleanChunked = if let Some(reg) = opt_reg {
            unary_elementwise_values(ca, |s| reg.is_match(s))
        } else {
            BooleanChunked::full_null(ca.name().clone(), ca.len())
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

    /// Return the index position of a literal substring in the target string.
    fn find_literal(&self, lit: &str) -> PolarsResult<UInt32Chunked> {
        self.find(regex::escape(lit).as_str(), true)
    }

    /// Return the index position of a regular expression substring in the target string.
    fn find(&self, pat: &str, strict: bool) -> PolarsResult<UInt32Chunked> {
        let ca = self.as_string();
        match polars_utils::regex_cache::compile_regex(pat) {
            Ok(rx) => Ok(unary_elementwise(ca, |opt_s| {
                opt_s.and_then(|s| rx.find(s)).map(|m| m.start() as u32)
            })),
            Err(_) if !strict => Ok(UInt32Chunked::full_null(ca.name().clone(), ca.len())),
            Err(e) => Err(PolarsError::ComputeError(
                format!("Invalid regular expression: {}", e).into(),
            )),
        }
    }

    /// Replace the leftmost regex-matched (sub)string with another string
    fn replace<'a>(&'a self, pat: &str, val: &str) -> PolarsResult<StringChunked> {
        let reg = polars_utils::regex_cache::compile_regex(pat)?;
        let f = |s: &'a str| reg.replace(s, val);
        let ca = self.as_string();
        Ok(ca.apply_values(f))
    }

    /// Replace the leftmost literal (sub)string with another string
    fn replace_literal<'a>(
        &'a self,
        pat: &str,
        val: &str,
        n: usize,
    ) -> PolarsResult<StringChunked> {
        let ca = self.as_string();
        if ca.is_empty() {
            return Ok(ca.clone());
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
    fn replace_all(&self, pat: &str, val: &str) -> PolarsResult<StringChunked> {
        let ca = self.as_string();
        let reg = polars_utils::regex_cache::compile_regex(pat)?;
        Ok(ca.apply_values(|s| reg.replace_all(s, val)))
    }

    /// Replace all matching literal (sub)strings with another string
    fn replace_literal_all<'a>(&'a self, pat: &str, val: &str) -> PolarsResult<StringChunked> {
        let ca = self.as_string();
        if ca.is_empty() {
            return Ok(ca.clone());
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
    fn extract(&self, pat: &StringChunked, group_index: usize) -> PolarsResult<StringChunked> {
        let ca = self.as_string();
        super::extract::extract_group(ca, pat, group_index)
    }

    /// Extract each successive non-overlapping regex match in an individual string as an array.
    fn extract_all(&self, pat: &str) -> PolarsResult<ListChunked> {
        let ca = self.as_string();
        let reg = polars_utils::regex_cache::compile_regex(pat)?;

        let mut builder =
            ListStringChunkedBuilder::new(ca.name().clone(), ca.len(), ca.get_values_size());
        for arr in ca.downcast_iter() {
            for opt_s in arr {
                match opt_s {
                    None => builder.append_null(),
                    Some(s) => builder.append_values_iter(reg.find_iter(s).map(|m| m.as_str())),
                }
            }
        }
        Ok(builder.finish())
    }

    fn strip_chars(&self, pat: &Column) -> PolarsResult<StringChunked> {
        let ca = self.as_string();
        if pat.dtype() == &DataType::Null {
            Ok(unary_elementwise(ca, |opt_s| opt_s.map(|s| s.trim())))
        } else {
            Ok(strip_chars(ca, pat.str()?))
        }
    }

    fn strip_chars_start(&self, pat: &Column) -> PolarsResult<StringChunked> {
        let ca = self.as_string();
        if pat.dtype() == &DataType::Null {
            Ok(unary_elementwise(ca, |opt_s| opt_s.map(|s| s.trim_start())))
        } else {
            Ok(strip_chars_start(ca, pat.str()?))
        }
    }

    fn strip_chars_end(&self, pat: &Column) -> PolarsResult<StringChunked> {
        let ca = self.as_string();
        if pat.dtype() == &DataType::Null {
            Ok(unary_elementwise(ca, |opt_s| opt_s.map(|s| s.trim_end())))
        } else {
            Ok(strip_chars_end(ca, pat.str()?))
        }
    }

    fn strip_prefix(&self, prefix: &StringChunked) -> StringChunked {
        let ca = self.as_string();
        strip_prefix(ca, prefix)
    }

    fn strip_suffix(&self, suffix: &StringChunked) -> StringChunked {
        let ca = self.as_string();
        strip_suffix(ca, suffix)
    }

    #[cfg(feature = "dtype-struct")]
    fn split_exact(&self, by: &StringChunked, n: usize) -> PolarsResult<StructChunked> {
        let ca = self.as_string();

        split_to_struct(ca, by, n + 1, str::split, false)
    }

    #[cfg(feature = "dtype-struct")]
    fn split_exact_inclusive(&self, by: &StringChunked, n: usize) -> PolarsResult<StructChunked> {
        let ca = self.as_string();

        split_to_struct(ca, by, n + 1, str::split_inclusive, false)
    }

    #[cfg(feature = "dtype-struct")]
    fn splitn(&self, by: &StringChunked, n: usize) -> PolarsResult<StructChunked> {
        let ca = self.as_string();

        split_to_struct(ca, by, n, |s, by| s.splitn(n, by), true)
    }

    fn split(&self, by: &StringChunked) -> PolarsResult<ListChunked> {
        let ca = self.as_string();
        split_helper(ca, by, str::split)
    }

    fn split_inclusive(&self, by: &StringChunked) -> PolarsResult<ListChunked> {
        let ca = self.as_string();
        split_helper(ca, by, str::split_inclusive)
    }

    /// Extract each successive non-overlapping regex match in an individual string as an array.
    fn extract_all_many(&self, pat: &StringChunked) -> PolarsResult<ListChunked> {
        let ca = self.as_string();
        polars_ensure!(
            ca.len() == pat.len(),
            ComputeError: "pattern's length: {} does not match that of the argument series: {}",
            pat.len(), ca.len(),
        );

        let mut builder =
            ListStringChunkedBuilder::new(ca.name().clone(), ca.len(), ca.get_values_size());
        with_regex_cache(|re_cache| {
            binary_elementwise_for_each(ca, pat, |opt_s, opt_pat| match (opt_s, opt_pat) {
                (_, None) | (None, _) => builder.append_null(),
                (Some(s), Some(pat)) => {
                    let re = re_cache.compile(pat).unwrap();
                    builder.append_values_iter(re.find_iter(s).map(|m| m.as_str()));
                },
            });
        });
        Ok(builder.finish())
    }

    #[cfg(feature = "extract_groups")]
    /// Extract all capture groups from pattern and return as a struct.
    fn extract_groups(&self, pat: &str, dtype: &DataType) -> PolarsResult<Series> {
        let ca = self.as_string();
        super::extract::extract_groups(ca, pat, dtype)
    }

    /// Count all successive non-overlapping regex matches.
    fn count_matches(&self, pat: &str, literal: bool) -> PolarsResult<UInt32Chunked> {
        let ca = self.as_string();
        if literal {
            Ok(unary_elementwise(ca, |opt_s| {
                opt_s.map(|s| s.matches(pat).count() as u32)
            }))
        } else {
            let re = compile_regex(pat)?;
            Ok(unary_elementwise(ca, |opt_s| {
                opt_s.map(|s| re.find_iter(s).count() as u32)
            }))
        }
    }

    /// Count all successive non-overlapping regex matches.
    fn count_matches_many(
        &self,
        pat: &StringChunked,
        literal: bool,
    ) -> PolarsResult<UInt32Chunked> {
        let ca = self.as_string();
        polars_ensure!(
            ca.len() == pat.len(),
            ComputeError: "pattern's length: {} does not match that of the argument series: {}",
            pat.len(), ca.len(),
        );

        let out: UInt32Chunked = if literal {
            broadcast_binary_elementwise(ca, pat, |s: Option<&str>, p: Option<&str>| {
                Some(s?.matches(p?).count() as u32)
            })
        } else {
            with_regex_cache(|re_cache| {
                let op = move |opt_s: Option<&str>,
                               opt_pat: Option<&str>|
                      -> PolarsResult<Option<u32>> {
                    match (opt_s, opt_pat) {
                        (Some(s), Some(pat)) => {
                            let reg = re_cache.compile(pat)?;
                            Ok(Some(reg.find_iter(s).count() as u32))
                        },
                        _ => Ok(None),
                    }
                };
                broadcast_try_binary_elementwise(ca, pat, op)
            })?
        };

        Ok(out.with_name(ca.name().clone()))
    }

    /// Modify the strings to their lowercase equivalent.
    #[must_use]
    fn to_lowercase(&self) -> StringChunked {
        let ca = self.as_string();
        case::to_lowercase(ca)
    }

    /// Modify the strings to their uppercase equivalent.
    #[must_use]
    fn to_uppercase(&self) -> StringChunked {
        let ca = self.as_string();
        case::to_uppercase(ca)
    }

    /// Modify the strings to their titlecase equivalent.
    #[must_use]
    #[cfg(feature = "nightly")]
    fn to_titlecase(&self) -> StringChunked {
        let ca = self.as_string();
        case::to_titlecase(ca)
    }

    /// Concat with the values from a second StringChunked.
    #[must_use]
    fn concat(&self, other: &StringChunked) -> StringChunked {
        let ca = self.as_string();
        ca + other
    }

    /// Normalizes the string values
    #[must_use]
    #[cfg(feature = "string_normalize")]
    fn str_normalize(&self, form: UnicodeForm) -> StringChunked {
        let ca = self.as_string();
        normalize::normalize(ca, form)
    }

    /// Reverses the string values
    #[must_use]
    #[cfg(feature = "string_reverse")]
    fn str_reverse(&self) -> StringChunked {
        let ca = self.as_string();
        reverse::reverse(ca)
    }

    /// Slice the string values.
    ///
    /// Determines a substring starting from `offset` and with length `length` of each of the elements in `array`.
    /// `offset` can be negative, in which case the start counts from the end of the string.
    fn str_slice(&self, offset: &Column, length: &Column) -> PolarsResult<StringChunked> {
        let ca = self.as_string();
        let offset = offset.cast(&DataType::Int64)?;
        // We strict cast, otherwise negative value will be treated as a valid length.
        let length = length.strict_cast(&DataType::UInt64)?;

        Ok(substring::substring(ca, offset.i64()?, length.u64()?))
    }

    /// Slice the first `n` values of the string.
    ///
    /// Determines a substring starting at the beginning of the string up to offset `n` of each
    /// element in `array`. `n` can be negative, in which case the slice ends `n` characters from
    /// the end of the string.
    fn str_head(&self, n: &Column) -> PolarsResult<StringChunked> {
        let ca = self.as_string();
        let n = n.strict_cast(&DataType::Int64)?;

        substring::head(ca, n.i64()?)
    }

    /// Slice the last `n` values of the string.
    ///
    /// Determines a substring starting at offset `n` of each element in `array`. `n` can be
    /// negative, in which case the slice begins `n` characters from the start of the string.
    fn str_tail(&self, n: &Column) -> PolarsResult<StringChunked> {
        let ca = self.as_string();
        let n = n.strict_cast(&DataType::Int64)?;

        substring::tail(ca, n.i64()?)
    }
    #[cfg(feature = "strings")]
    /// Escapes all regular expression meta characters in the string.
    fn str_escape_regex(&self) -> StringChunked {
        let ca = self.as_string();
        escape_regex::escape_regex(ca)
    }
}

impl StringNameSpaceImpl for StringChunked {}
