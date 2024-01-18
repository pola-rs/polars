use std::fmt::Write;

use polars_core::prelude::ArrayChunked;

use super::*;

fn join_literal(ca: &ArrayChunked, separator: &str) -> PolarsResult<StringChunked> {
    let DataType::Array(_, width) = ca.dtype() else {
        unreachable!()
    };

    let mut buf = String::with_capacity(128);
    let mut builder = StringChunkedBuilder::new(ca.name(), ca.len());

    ca.for_each_amortized(|opt_s| {
        let opt_val = opt_s.map(|s| {
            // make sure that we don't write values of previous iteration
            buf.clear();
            let ca = s.as_ref().str().unwrap();

            let iter = ca.into_iter().map(|opt_v| opt_v.unwrap_or("null"));

            for val in iter {
                buf.write_str(val).unwrap();
                buf.write_str(separator).unwrap();
            }
            // last value should not have a separator, so slice that off
            // saturating sub because there might have been nothing written.
            &buf[..buf.len().saturating_sub(separator.len())]
        });
        builder.append_option(opt_val)
    });
    Ok(builder.finish())
}

fn join_many(ca: &ArrayChunked, separator: &StringChunked) -> PolarsResult<StringChunked> {
    let mut buf = String::new();
    let mut builder = StringChunkedBuilder::new(ca.name(), ca.len());

    ca.amortized_iter()
        .zip(separator)
        .for_each(|(opt_s, opt_sep)| match opt_sep {
            Some(separator) => {
                let opt_val = opt_s.map(|s| {
                    // make sure that we don't write values of previous iteration
                    buf.clear();
                    let ca = s.as_ref().str().unwrap();
                    let iter = ca.into_iter().map(|opt_v| opt_v.unwrap_or("null"));

                    for val in iter {
                        buf.write_str(val).unwrap();
                        buf.write_str(separator).unwrap();
                    }
                    // last value should not have a separator, so slice that off
                    // saturating sub because there might have been nothing written.
                    &buf[..buf.len().saturating_sub(separator.len())]
                });
                builder.append_option(opt_val)
            },
            _ => builder.append_null(),
        });
    Ok(builder.finish())
}

/// In case the inner dtype [`DataType::String`], the individual items will be joined into a
/// single string separated by `separator`.
pub fn array_join(ca: &ArrayChunked, separator: &StringChunked) -> PolarsResult<StringChunked> {
    match ca.inner_dtype() {
        DataType::String => match separator.len() {
            1 => match separator.get(0) {
                Some(separator) => join_literal(ca, separator),
                _ => Ok(StringChunked::full_null(ca.name(), ca.len())),
            },
            _ => join_many(ca, separator),
        },
        dt => polars_bail!(op = "`array.join`", got = dt, expected = "String"),
    }
}
