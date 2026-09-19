use super::*;
use crate::chunked_array::list::namespace::join_one_list;

fn join_literal(
    ca: &ArrayChunked,
    separator: &str,
    ignore_nulls: bool,
) -> PolarsResult<StringChunked> {
    let DataType::Array(_, _) = ca.dtype() else {
        unreachable!()
    };

    if let Some(length) = ca.repeats_one_list() {
        let mut buf = String::with_capacity(128);
        let one = ca.amortized_iter().next().flatten();
        let joined = one.and_then(|s| join_one_list(s.as_ref(), separator, ignore_nulls, &mut buf));

        let name = ca.name().clone();
        return Ok(match joined {
            Some(joined) => StringChunked::full(name, joined, length),
            None => StringChunked::full_null(name, length),
        });
    }

    let mut buf = String::with_capacity(128);
    let mut builder = StringChunkedBuilder::new(ca.name().clone(), ca.len());

    ca.for_each_amortized(|opt_s| {
        let opt_val =
            opt_s.and_then(|s| join_one_list(s.as_ref(), separator, ignore_nulls, &mut buf));
        builder.append_option(opt_val)
    });
    Ok(builder.finish())
}

fn join_many(
    ca: &ArrayChunked,
    separator: &StringChunked,
    ignore_nulls: bool,
) -> PolarsResult<StringChunked> {
    polars_ensure!(
        ca.len() == separator.len(),
        length_mismatch = "arr.join",
        ca.len(),
        separator.len()
    );

    if ca.repeats_one_list().is_some()
        && let Some(separator) = separator.scalar_value()
    {
        return match separator {
            Some(separator) => join_literal(ca, separator, ignore_nulls),
            None => Ok(StringChunked::full_null(ca.name().clone(), ca.len())),
        };
    }

    let mut buf = String::new();
    let mut builder = StringChunkedBuilder::new(ca.name().clone(), ca.len());

    { ca.amortized_iter() }
        .zip(separator.iter())
        .for_each(|(opt_s, opt_sep)| match opt_sep {
            Some(separator) => {
                let opt_val = opt_s
                    .and_then(|s| join_one_list(s.as_ref(), separator, ignore_nulls, &mut buf));
                builder.append_option(opt_val)
            },
            _ => builder.append_null(),
        });
    Ok(builder.finish())
}

/// In case the inner dtype [`DataType::String`], the individual items will be joined into a
/// single string separated by `separator`.
pub fn array_join(
    ca: &ArrayChunked,
    separator: &StringChunked,
    ignore_nulls: bool,
) -> PolarsResult<StringChunked> {
    match ca.inner_dtype() {
        DataType::String => match separator.len() {
            1 => match separator.get(0) {
                Some(separator) => join_literal(ca, separator, ignore_nulls),
                _ => Ok(StringChunked::full_null(ca.name().clone(), ca.len())),
            },
            _ => join_many(ca, separator, ignore_nulls),
        },
        dt => polars_bail!(op = "`array.join`", got = dt, expected = "String"),
    }
}
