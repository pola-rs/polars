use std::fmt::{Display, Write};

use polars_arrow::array::default_arrays::FromDataUtf8;

use super::StrConcat;
use crate::prelude::*;

fn fmt_and_write<T: Display>(value: Option<T>, buf: &mut String) {
    match value {
        None => buf.push_str("null"),
        Some(v) => {
            write!(buf, "{v}").unwrap();
        }
    }
}

fn str_concat_impl<I, T>(mut iter: I, delimiter: &str, name: &str) -> Utf8Chunked
where
    I: Iterator<Item = Option<T>>,
    T: Display,
{
    let mut buf = String::with_capacity(iter.size_hint().0 * 5);

    if let Some(first) = iter.next() {
        fmt_and_write(first, &mut buf);

        for val in iter {
            buf.push_str(delimiter);
            fmt_and_write(val, &mut buf);
        }
    }
    buf.shrink_to_fit();
    let buf = buf.into_bytes();
    let offsets = vec![0, buf.len() as i64];
    let arr = unsafe { Utf8Array::from_data_unchecked_default(offsets.into(), buf.into(), None) };
    unsafe { Utf8Chunked::from_chunks(name, vec![Box::new(arr)]) }
}

impl<T> StrConcat for ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: Display,
{
    fn str_concat(&self, delimiter: &str) -> Utf8Chunked {
        let iter = self.into_iter();
        str_concat_impl(iter, delimiter, self.name())
    }
}

impl StrConcat for Utf8Chunked {
    fn str_concat(&self, delimiter: &str) -> Utf8Chunked {
        let iter = self.into_iter();
        str_concat_impl(iter, delimiter, self.name())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_str_concat() {
        let ca = Int32Chunked::new("foo", &[Some(1), None, Some(3)]);
        let out = ca.str_concat("-");

        let out = out.get(0);
        assert_eq!(out, Some("1-null-3"));
    }
}
