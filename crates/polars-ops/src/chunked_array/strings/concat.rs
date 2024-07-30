use arrow::array::{Utf8Array, ValueSize};
use arrow::compute::cast::utf8_to_utf8view;
use polars_core::prelude::arity::unary_elementwise;
use polars_core::prelude::*;

// Vertically concatenate all strings in a StringChunked.
pub fn str_join(ca: &StringChunked, delimiter: &str, ignore_nulls: bool) -> StringChunked {
    if ca.is_empty() {
        return StringChunked::new(ca.name(), &[""]);
    }

    // Propagate null value.
    if !ignore_nulls && ca.null_count() != 0 {
        return StringChunked::full_null(ca.name(), 1);
    }

    // Fast path for all nulls.
    if ignore_nulls && ca.null_count() == ca.len() {
        return StringChunked::new(ca.name(), &[""]);
    }

    if ca.len() == 1 {
        return ca.clone();
    }

    // Calculate capacity.
    let capacity = ca.get_values_size() + delimiter.len() * (ca.len() - 1);

    let mut buf = String::with_capacity(capacity);
    let mut first = true;
    ca.for_each(|val| {
        if let Some(val) = val {
            if !first {
                buf.push_str(delimiter);
            }
            buf.push_str(val);
            first = false;
        }
    });

    let buf = buf.into_bytes();
    assert!(capacity >= buf.len());
    let offsets = vec![0, buf.len() as i64];
    let arr = unsafe { Utf8Array::from_data_unchecked_default(offsets.into(), buf.into(), None) };
    // conversion is cheap with one value.
    let arr = utf8_to_utf8view(&arr);
    StringChunked::with_chunk(ca.name(), arr)
}

enum ColumnIter<I, T> {
    Iter(I),
    Broadcast(T),
}

/// Horizontally concatenate all strings.
///
/// Each array should have length 1 or a length equal to the maximum length.
pub fn hor_str_concat(
    cas: &[&StringChunked],
    delimiter: &str,
    ignore_nulls: bool,
) -> PolarsResult<StringChunked> {
    if cas.is_empty() {
        return Ok(StringChunked::full_null("", 0));
    }
    if cas.len() == 1 {
        let ca = cas[0];
        return if !ignore_nulls || ca.null_count() == 0 {
            Ok(ca.clone())
        } else {
            Ok(unary_elementwise(ca, |val| Some(val.unwrap_or(""))))
        };
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

    let mut builder = StringChunkedBuilder::new(cas[0].name(), len);

    // Broadcast if appropriate.
    let mut cols: Vec<_> = cas
        .iter()
        .map(|ca| match ca.len() {
            0 => ColumnIter::Broadcast(None),
            1 => ColumnIter::Broadcast(ca.get(0)),
            _ => ColumnIter::Iter(ca.iter()),
        })
        .collect();

    // Build concatenated string.
    let mut buf = String::with_capacity(1024);
    for _row in 0..len {
        let mut has_null = false;
        let mut found_not_null_value = false;
        for col in cols.iter_mut() {
            let val = match col {
                ColumnIter::Iter(i) => i.next().unwrap(),
                ColumnIter::Broadcast(s) => *s,
            };

            if has_null && !ignore_nulls {
                // We know that the result must be null, but we can't just break out of the loop,
                // because all cols iterator has to be moved correctly.
                continue;
            }

            if let Some(s) = val {
                if found_not_null_value {
                    buf.push_str(delimiter);
                }
                buf.push_str(s);
                found_not_null_value = true;
            } else {
                has_null = true;
            }
        }

        if !ignore_nulls && has_null {
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
        let ca_str = ca.cast(&DataType::String).unwrap();
        let out = str_join(ca_str.str().unwrap(), "-", true);

        let out = out.get(0);
        assert_eq!(out, Some("1-3"));
    }

    #[test]
    fn test_hor_str_concat() {
        let a = StringChunked::new("a", &["foo", "bar"]);
        let b = StringChunked::new("b", &["spam", "ham"]);

        let out = hor_str_concat(&[&a, &b], "_", true).unwrap();
        assert_eq!(Vec::from(&out), &[Some("foo_spam"), Some("bar_ham")]);

        let c = StringChunked::new("b", &["literal"]);
        let out = hor_str_concat(&[&a, &b, &c], "_", true).unwrap();
        assert_eq!(
            Vec::from(&out),
            &[Some("foo_spam_literal"), Some("bar_ham_literal")]
        );
    }
}
