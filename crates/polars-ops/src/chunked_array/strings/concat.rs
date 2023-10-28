use arrow::array::Utf8Array;
use arrow::legacy::array::default_arrays::FromDataUtf8;
use polars_core::prelude::*;

// Vertically concatenate all strings in a Utf8Chunked.
pub fn str_concat(ca: &Utf8Chunked, delimiter: &str) -> Utf8Chunked {
    if ca.is_empty() {
        return Utf8Chunked::new(ca.name(), &[""]);
    }

    if ca.len() == 1 {
        return ca.clone();
    }

    // Calculate capacity.
    let null_str_len = 4;
    let capacity =
        ca.get_values_size() + ca.null_count() * null_str_len + delimiter.len() * (ca.len() - 1);

    let mut buf = String::with_capacity(capacity);
    let mut first = true;
    for arr in ca.downcast_iter() {
        for val in arr.into_iter() {
            if !first {
                buf.push_str(delimiter);
            }

            if let Some(s) = val {
                buf.push_str(s);
            } else {
                buf.push_str("null");
            }

            first = false;
        }
    }

    let buf = buf.into_bytes();
    let offsets = vec![0, buf.len() as i64];
    let arr = unsafe { Utf8Array::from_data_unchecked_default(offsets.into(), buf.into(), None) };
    Utf8Chunked::with_chunk(ca.name(), arr)
}

enum ColumnIter<I, T> {
    Iter(I),
    Broadcast(T),
}

/// Horizontally concatenate all strings.
///
/// Each array should have length 1 or a length equal to the maximum length.
pub fn hor_str_concat(cas: &[&Utf8Chunked], delimiter: &str) -> PolarsResult<Utf8Chunked> {
    if cas.is_empty() {
        return Ok(Utf8Chunked::full_null("", 0));
    }
    if cas.len() == 1 {
        return Ok(cas[0].clone());
    }

    // Calculate the post-broadcast length and ensure everything is consistent.
    let len = cas
        .iter()
        .map(|ca| ca.len())
        .filter(|l| *l != 1)
        .max()
        .unwrap_or(1);
    polars_ensure!(
        cas.iter().all(|ca| ca.len() == 1 || ca.len() == len),
        ComputeError: "all series in `hor_str_concat` should have equal or unit length"
    );

    // Calculate total capacity needed.
    let tot_strings_bytes: usize = cas
        .iter()
        .map(|ca| {
            let bytes = ca.get_values_size();
            if ca.len() == 1 {
                len * bytes
            } else {
                bytes
            }
        })
        .sum();
    let capacity = tot_strings_bytes + (cas.len() - 1) * delimiter.len() * len;
    let mut builder = Utf8ChunkedBuilder::new(cas[0].name(), len, capacity);

    // Broadcast if appropriate.
    let mut cols: Vec<_> = cas
        .iter()
        .map(|ca| {
            if ca.len() > 1 {
                ColumnIter::Iter(ca.into_iter())
            } else {
                ColumnIter::Broadcast(ca.get(0))
            }
        })
        .collect();

    // Build concatenated string.
    let mut buf = String::with_capacity(1024);
    for _row in 0..len {
        let mut has_null = false;
        for (i, col) in cols.iter_mut().enumerate() {
            if i > 0 {
                buf.push_str(delimiter);
            }

            let val = match col {
                ColumnIter::Iter(i) => i.next().unwrap(),
                ColumnIter::Broadcast(s) => *s,
            };
            if let Some(s) = val {
                buf.push_str(s);
            } else {
                has_null = true;
            }
        }

        if has_null {
            builder.append_null();
        } else {
            builder.append_value(&buf)
        }
        buf.clear();
    }

    Ok(builder.finish())
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_str_concat() {
        let ca = Int32Chunked::new("foo", &[Some(1), None, Some(3)]);
        let ca_str = ca.cast(&DataType::Utf8).unwrap();
        let out = str_concat(&ca_str.utf8().unwrap(), "-");

        let out = out.get(0);
        assert_eq!(out, Some("1-null-3"));
    }

    #[test]
    fn test_hor_str_concat() {
        let a = Utf8Chunked::new("a", &["foo", "bar"]);
        let b = Utf8Chunked::new("b", &["spam", "ham"]);

        let out = hor_str_concat(&[&a, &b], "_").unwrap();
        assert_eq!(Vec::from(&out), &[Some("foo_spam"), Some("bar_ham")]);

        let c = Utf8Chunked::new("b", &["literal"]);
        let out = hor_str_concat(&[&a, &b, &c], "_").unwrap();
        assert_eq!(
            Vec::from(&out),
            &[Some("foo_spam_literal"), Some("bar_ham_literal")]
        );
    }
}
