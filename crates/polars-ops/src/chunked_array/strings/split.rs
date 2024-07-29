use std::iter::repeat;

use arrow::array::ValueSize;

use super::utils::{iter_with_view_and_buffers, subview_from_str};
use super::*;

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
    let mut arrs = (0..n)
        .map(|i| StringChunkedBuilder::new(format!("field_{i}").as_str(), ca.len()))
        .collect::<Vec<_>>();

    if by.len() != 1 {
        polars_ensure!(
            ca.len() == by.len(),
            ComputeError: "by's length: {} does not match that of the argument series: {}",
            by.len(), ca.len(),
        );
    }

    for ((opt_s, (&view, buffers)), opt_by) in
        iter_with_view_and_buffers(ca).zip(repeat(by).flatten())
    {
        match (opt_s, opt_by) {
            (Some(s), Some(by)) => {
                let mut arr_iter = arrs.iter_mut();
                if by.is_empty() {
                    splitn_chars(s, n, keep_remainder)
                        .zip(&mut arr_iter)
                        .for_each(|(splitted, arr)| {
                            arr.append_view(subview_from_str(s, splitted, view), buffers)
                        });
                } else {
                    op(s, by).zip(&mut arr_iter).for_each(|(splitted, arr)| {
                        arr.append_view(subview_from_str(s, splitted, view), buffers)
                    });
                };
                // fill the remaining with null
                for arr in arr_iter {
                    arr.append_null()
                }
            },
            _ => {
                for arr in &mut arrs {
                    arr.append_null()
                }
            },
        }
    }

    let fields = arrs
        .into_iter()
        .map(StringChunkedBuilder::finish)
        .map(StringChunked::into_series)
        .collect::<Vec<_>>();

    StructChunked::from_series(ca.name(), &fields)
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
    if by.len() == 1 {
        if by.get(0).is_none() {
            return Ok(ListChunked::full_null_with_dtype(
                ca.name(),
                ca.len(),
                &DataType::String,
            ));
        }
    } else {
        polars_ensure!(
            ca.len() == by.len(),
            ComputeError: "by's length: {} does not match that of the argument series: {}",
            by.len(), ca.len(),
        );
    }

    let mut builder = ListStringChunkedBuilder::new(ca.name(), ca.len(), ca.get_values_size());

    for ((opt_s, (&view, buffers)), opt_by) in
        iter_with_view_and_buffers(ca).zip(repeat(by).flatten())
    {
        match (opt_s, opt_by) {
            (Some(s), Some(by)) => {
                if by.is_empty() {
                    builder.append_views_iter(
                        split_chars(s).map(|subs| subview_from_str(s, subs, view)),
                        buffers,
                    );
                } else {
                    builder.append_views_iter(
                        op(s, by).map(|subs| subview_from_str(s, subs, view)),
                        buffers,
                    );
                }
            },
            _ => builder.append_null(),
        }
    }

    Ok(builder.finish())
}
