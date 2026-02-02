use arrow::array::ValueSize;
#[cfg(feature = "dtype-struct")]
use arrow::array::{MutableArray, MutableUtf8Array};
use polars_core::chunked_array::ops::arity::binary_elementwise_for_each;
use polars_core::prelude::*;
use polars_utils::regex_cache::compile_regex;
use regex::Regex;

pub struct SplitNChars<'a> {
    s: &'a str,
    n: usize,
    keep_remainder: bool,
}

impl<'a> Iterator for SplitNChars<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        let single_char_limit = if self.keep_remainder { 2 } else { 1 };
        if self.n >= single_char_limit {
            self.n -= 1;
            let ch = self.s.chars().next()?;
            let first;
            (first, self.s) = self.s.split_at(ch.len_utf8());
            Some(first)
        } else if self.n == 1 && !self.s.is_empty() {
            self.n -= 1;
            Some(self.s)
        } else {
            None
        }
    }
}

/// Splits a string into substrings consisting of single characters.
///
/// Returns at most n strings, where the last string is the entire remainder
/// of the string if keep_remainder is True, and just the nth character otherwise.
#[cfg(feature = "dtype-struct")]
fn splitn_chars(s: &str, n: usize, keep_remainder: bool) -> SplitNChars<'_> {
    SplitNChars {
        s,
        n,
        keep_remainder,
    }
}

/// Splits a string into substrings consisting of single characters.
fn split_chars(s: &str) -> SplitNChars<'_> {
    SplitNChars {
        s,
        n: usize::MAX,
        keep_remainder: false,
    }
}

#[cfg(feature = "dtype-struct")]
pub fn split_to_struct<'a, F, I>(
    ca: &'a StringChunked,
    by: &'a StringChunked,
    n: usize,
    op: F,
    keep_remainder: bool,
) -> PolarsResult<StructChunked>
where
    F: Fn(&'a str, &'a str) -> I,
    I: Iterator<Item = &'a str>,
{
    use polars_utils::format_pl_smallstr;

    let mut arrs = (0..n)
        .map(|_| MutableUtf8Array::<i64>::with_capacity(ca.len()))
        .collect::<Vec<_>>();

    if by.len() == 1 {
        if let Some(by) = by.get(0) {
            if by.is_empty() {
                ca.for_each(|opt_s| match opt_s {
                    None => {
                        for arr in &mut arrs {
                            arr.push_null()
                        }
                    },
                    Some(s) => {
                        let mut arr_iter = arrs.iter_mut();
                        splitn_chars(s, n, keep_remainder)
                            .zip(&mut arr_iter)
                            .for_each(|(splitted, arr)| arr.push(Some(splitted)));
                        // fill the remaining with null
                        for arr in arr_iter {
                            arr.push_null()
                        }
                    },
                });
            } else {
                ca.for_each(|opt_s| match opt_s {
                    None => {
                        for arr in &mut arrs {
                            arr.push_null()
                        }
                    },
                    Some(s) => {
                        let mut arr_iter = arrs.iter_mut();
                        op(s, by)
                            .zip(&mut arr_iter)
                            .for_each(|(splitted, arr)| arr.push(Some(splitted)));
                        // fill the remaining with null
                        for arr in arr_iter {
                            arr.push_null()
                        }
                    },
                });
            }
        } else {
            for arr in &mut arrs {
                arr.push_null()
            }
        }
    } else {
        binary_elementwise_for_each(ca, by, |opt_s, opt_by| match (opt_s, opt_by) {
            (Some(s), Some(by)) => {
                let mut arr_iter = arrs.iter_mut();
                if by.is_empty() {
                    splitn_chars(s, n, keep_remainder)
                        .zip(&mut arr_iter)
                        .for_each(|(splitted, arr)| arr.push(Some(splitted)));
                } else {
                    op(s, by)
                        .zip(&mut arr_iter)
                        .for_each(|(splitted, arr)| arr.push(Some(splitted)));
                };
                // fill the remaining with null
                for arr in arr_iter {
                    arr.push_null()
                }
            },
            _ => {
                for arr in &mut arrs {
                    arr.push_null()
                }
            },
        })
    }

    let fields = arrs
        .into_iter()
        .enumerate()
        .map(|(i, mut arr)| {
            Series::try_from((format_pl_smallstr!("field_{i}"), arr.as_box())).unwrap()
        })
        .collect::<Vec<_>>();

    StructChunked::from_series(ca.name().clone(), ca.len(), fields.iter())
}

pub fn split_helper<'a, F, I>(
    ca: &'a StringChunked,
    by: &'a StringChunked,
    op: F,
) -> PolarsResult<ListChunked>
where
    F: Fn(&'a str, &'a str) -> I,
    I: Iterator<Item = &'a str>,
{
    Ok(match (ca.len(), by.len()) {
        (a, b) if a == b => {
            let mut builder =
                ListStringChunkedBuilder::new(ca.name().clone(), ca.len(), ca.get_values_size());

            binary_elementwise_for_each(ca, by, |opt_s, opt_by| match (opt_s, opt_by) {
                (Some(s), Some(by)) => {
                    if by.is_empty() {
                        builder.append_values_iter(split_chars(s))
                    } else {
                        builder.append_values_iter(op(s, by))
                    }
                },
                _ => builder.append_null(),
            });

            builder.finish()
        },
        (1, _) => {
            if let Some(s) = ca.get(0) {
                let mut builder = ListStringChunkedBuilder::new(
                    by.name().clone(),
                    by.len(),
                    by.get_values_size(),
                );

                by.for_each(|opt_by| match opt_by {
                    Some(by) => builder.append_values_iter(op(s, by)),
                    _ => builder.append_null(),
                });
                builder.finish()
            } else {
                ListChunked::full_null_with_dtype(ca.name().clone(), ca.len(), &DataType::String)
            }
        },
        (_, 1) => {
            if let Some(by) = by.get(0) {
                let mut builder = ListStringChunkedBuilder::new(
                    ca.name().clone(),
                    ca.len(),
                    ca.get_values_size(),
                );

                if by.is_empty() {
                    ca.for_each(|opt_s| match opt_s {
                        Some(s) => builder.append_values_iter(split_chars(s)),
                        _ => builder.append_null(),
                    });
                } else {
                    ca.for_each(|opt_s| match opt_s {
                        Some(s) => builder.append_values_iter(op(s, by)),
                        _ => builder.append_null(),
                    });
                }
                builder.finish()
            } else {
                ListChunked::full_null_with_dtype(ca.name().clone(), ca.len(), &DataType::String)
            }
        },
        _ => polars_bail!(length_mismatch = "str.split", ca.len(), by.len()),
    })
}

#[inline]
fn split_inclusive<'a>(re: &'a Regex, s: &'a str) -> impl Iterator<Item = &'a str> + 'a {
    let mut it = re.find_iter(s);
    let mut last_end: usize = 0;
    let mut yielded_any = false;
    let mut done_tail = false;

    std::iter::from_fn(move || {
        if let Some(m) = it.next() {
            let end = m.end();
            let out = &s[last_end..end];
            last_end = end;
            yielded_any = true;
            return Some(out);
        }

        if done_tail {
            return None;
        }
        done_tail = true;

        if last_end < s.len() {
            Some(&s[last_end..])
        } else if !yielded_any {
            Some(s)
        } else {
            None
        }
    })
}

#[inline]
fn invalid_regex_err(pat: &str) -> PolarsError {
    polars_err!(ComputeError: "invalid regex pattern in str.split_regex: {}", pat)
}

#[inline]
fn append_split_compiled(
    builder: &mut ListStringChunkedBuilder,
    s: &str,
    re: &Regex,
    inclusive: bool,
) {
    if inclusive {
        builder.append_values_iter(split_inclusive(re, s));
    } else {
        builder.append_values_iter(re.split(s));
    }
}

#[inline]
fn append_split(
    builder: &mut ListStringChunkedBuilder,
    s: &str,
    pat: &str,
    inclusive: bool,
    strict: bool,
) -> PolarsResult<()> {
    if pat.is_empty() {
        builder.append_values_iter(split_chars(s));
        return Ok(());
    }

    match compile_regex(pat) {
        Ok(re) => {
            append_split_compiled(builder, s, &re, inclusive);
            Ok(())
        },
        Err(_) if strict => Err(invalid_regex_err(pat)),
        Err(_) => {
            builder.append_null();
            Ok(())
        },
    }
}

pub fn split_regex_helper(
    ca: &StringChunked,
    by: &StringChunked,
    inclusive: bool,
    strict: bool,
) -> PolarsResult<ListChunked> {
    use polars_utils::regex_cache::compile_regex;

    Ok(match (ca.len(), by.len()) {
        // elementwise: string[i] with pattern[i]
        (a, b) if a == b => {
            let mut builder =
                ListStringChunkedBuilder::new(ca.name().clone(), ca.len(), ca.get_values_size());

            for (opt_s, opt_pat) in ca.into_iter().zip(by.into_iter()) {
                match (opt_s, opt_pat) {
                    (Some(s), Some(pat)) => append_split(&mut builder, s, pat, inclusive, strict)?,
                    _ => builder.append_null(),
                }
            }

            builder.finish()
        },

        // scalar string with per-row patterns
        (1, _) => {
            if let Some(s0) = ca.get(0) {
                let mut builder = ListStringChunkedBuilder::new(
                    by.name().clone(),
                    by.len(),
                    by.get_values_size(),
                );

                for opt_pat in by.into_iter() {
                    match opt_pat {
                        Some(pat) => append_split(&mut builder, s0, pat, inclusive, strict)?,
                        None => builder.append_null(),
                    }
                }

                builder.finish()
            } else {
                ListChunked::full_null_with_dtype(ca.name().clone(), by.len(), &DataType::String)
            }
        },

        // per-row strings with scalar pattern
        (_, 1) => {
            if let Some(pat0) = by.get(0) {
                let mut builder = ListStringChunkedBuilder::new(
                    ca.name().clone(),
                    ca.len(),
                    ca.get_values_size(),
                );

                if pat0.is_empty() {
                    ca.for_each(|opt_s| match opt_s {
                        Some(s) => builder.append_values_iter(split_chars(s)),
                        None => builder.append_null(),
                    });
                    builder.finish()
                } else {
                    let re = match compile_regex(pat0) {
                        Ok(re) => re,
                        Err(_) if strict => return Err(invalid_regex_err(pat0)),
                        Err(_) => {
                            return Ok(ListChunked::full_null_with_dtype(
                                ca.name().clone(),
                                ca.len(),
                                &DataType::String,
                            ));
                        },
                    };

                    ca.for_each(|opt_s| match opt_s {
                        Some(s) => append_split_compiled(&mut builder, s, &re, inclusive),
                        None => builder.append_null(),
                    });

                    builder.finish()
                }
            } else {
                ListChunked::full_null_with_dtype(ca.name().clone(), ca.len(), &DataType::String)
            }
        },

        _ => polars_bail!(length_mismatch = "str.split_regex", ca.len(), by.len()),
    })
}
