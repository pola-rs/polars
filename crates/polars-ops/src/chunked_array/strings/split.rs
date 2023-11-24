use arrow::array::ValueSize;
#[cfg(feature = "dtype-struct")]
use arrow::array::{MutableArray, MutableUtf8Array};
use polars_core::chunked_array::ops::arity::binary_elementwise_for_each;

use super::*;

#[cfg(feature = "dtype-struct")]
pub fn split_to_struct<'a, F, I>(
    ca: &'a Utf8Chunked,
    by: &'a Utf8Chunked,
    n: usize,
    op: F,
) -> PolarsResult<StructChunked>
where
    F: Fn(&'a str, &'a str) -> I,
    I: Iterator<Item = &'a str>,
{
    let mut arrs = (0..n)
        .map(|_| MutableUtf8Array::<i64>::with_capacity(ca.len()))
        .collect::<Vec<_>>();

    if by.len() == 1 {
        if let Some(by) = by.get(0) {
            ca.for_each(|opt_s| match opt_s {
                None => {
                    for arr in &mut arrs {
                        arr.push_null()
                    }
                },
                Some(s) => {
                    let mut arr_iter = arrs.iter_mut();
                    let split_iter = op(s, by);
                    (split_iter)
                        .zip(&mut arr_iter)
                        .for_each(|(splitted, arr)| arr.push(Some(splitted)));
                    // fill the remaining with null
                    for arr in arr_iter {
                        arr.push_null()
                    }
                },
            });
        } else {
            for arr in &mut arrs {
                arr.push_null()
            }
        }
    } else {
        binary_elementwise_for_each(ca, by, |opt_s, opt_by| match (opt_s, opt_by) {
            (Some(s), Some(by)) => {
                let mut arr_iter = arrs.iter_mut();
                let split_iter = op(s, by);
                (split_iter)
                    .zip(&mut arr_iter)
                    .for_each(|(splitted, arr)| arr.push(Some(splitted)));
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
            Series::try_from((format!("field_{i}").as_str(), arr.as_box())).unwrap()
        })
        .collect::<Vec<_>>();

    StructChunked::new(ca.name(), &fields)
}

pub fn split_helper<'a, F, I>(ca: &'a Utf8Chunked, by: &'a Utf8Chunked, op: F) -> ListChunked
where
    F: Fn(&'a str, &'a str) -> I,
    I: Iterator<Item = &'a str>,
{
    if by.len() == 1 {
        if let Some(by) = by.get(0) {
            let mut builder =
                ListUtf8ChunkedBuilder::new(ca.name(), ca.len(), ca.get_values_size());

            ca.for_each(|opt_s| match opt_s {
                Some(s) => {
                    let iter = op(s, by);
                    builder.append_values_iter(iter)
                },
                _ => builder.append_null(),
            });
            builder.finish()
        } else {
            ListChunked::full_null_with_dtype(ca.name(), ca.len(), &DataType::Utf8)
        }
    } else {
        let mut builder = ListUtf8ChunkedBuilder::new(ca.name(), ca.len(), ca.get_values_size());

        binary_elementwise_for_each(ca, by, |opt_s, opt_by| match (opt_s, opt_by) {
            (Some(s), Some(by)) => {
                let iter = op(s, by);
                builder.append_values_iter(iter);
            },
            _ => builder.append_null(),
        });

        builder.finish()
    }
}
