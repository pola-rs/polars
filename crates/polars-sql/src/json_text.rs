//! The JSONPath that the `->`, `->>`, `#>` and `#>>` operators read from JSON text.

use std::fmt::Write;

use polars_core::prelude::*;
use polars_lazy::dsl::{Expr, lit};
use polars_plan::plans::{DynLiteralValue, LiteralValue};

/// The value that `->` or `->>` (or with `path_array`, `#>` or `#>>`) reads with `rhs` from
/// `lhs`, JSON text; `None` if `rhs` is not a literal path. A key that starts with `$` is a
/// JSONPath, as in DuckDB, and an integer is an array index.
pub(crate) fn json_text_access(
    lhs: Expr,
    rhs: &Expr,
    path_array: bool,
) -> PolarsResult<Option<Expr>> {
    let Expr::Literal(lv) = rhs else {
        return Ok(None);
    };
    let mut path = String::from("$");
    if let LiteralValue::Dyn(DynLiteralValue::Int(idx)) = lv
        && !path_array
    {
        write!(path, "[{idx}]").unwrap();
    } else {
        let Some(key) = lv.extract_str() else {
            return Ok(None);
        };
        if path_array {
            for elem in text_array_elements(key) {
                // A NULL step reads nothing, as in PostgreSQL.
                let Some(elem) = elem else {
                    let null = Expr::Literal(LiteralValue::untyped_null());
                    return Ok(Some(null.cast(DataType::String)));
                };
                match elem.parse::<i64>() {
                    Ok(idx) => write!(path, "[{idx}]").unwrap(),
                    Err(_) => push_key(&mut path, &elem)?,
                }
            }
        } else if let Some(rest) = key.strip_prefix('$') {
            push_duckdb_path(&mut path, rest, key)?;
        } else {
            push_key(&mut path, key)?;
        }
    }
    Ok(Some(lhs.str().json_path_match(lit(path))))
}

fn push_key(path: &mut String, key: &str) -> PolarsResult<()> {
    polars_ensure!(
        !key.contains('\\'),
        SQLInterface: "JSON keys with a backslash are not supported (found '{}')", key
    );
    write!(path, "['{}']", key.replace('\'', "\\'")).unwrap();
    Ok(())
}

/// Add the steps of a DuckDB path such as `$.a[0]."b c"[#-1]`, given without its `$`, to `path`.
fn push_duckdb_path(path: &mut String, mut rest: &str, full: &str) -> PolarsResult<()> {
    let unsupported = || polars_err!(SQLInterface: "unsupported JSON path '{}'; only keys and array indexes are supported", full);
    while !rest.is_empty() {
        if let Some(step) = rest.strip_prefix('.') {
            let (key, next) = match step.strip_prefix('"') {
                Some(quoted) => {
                    // A backslash takes the next character as is.
                    let mut key = String::new();
                    let mut chars = quoted.char_indices();
                    let end = loop {
                        match chars.next() {
                            Some((i, '"')) => break i,
                            Some((_, '\\')) => key.extend(chars.next().map(|(_, c)| c)),
                            Some((_, c)) => key.push(c),
                            None => return Err(unsupported()),
                        }
                    };
                    (key, &quoted[end + 1..])
                },
                None => {
                    let end = step.find(['.', '[']).unwrap_or(step.len());
                    (step[..end].to_string(), &step[end..])
                },
            };
            if key.is_empty() || key == "*" {
                return Err(unsupported());
            }
            push_key(path, &key)?;
            rest = next;
        } else if let Some(step) = rest.strip_prefix('[') {
            let end = step.find(']').ok_or_else(unsupported)?;
            let idx = match step[..end].strip_prefix('#') {
                // `#-n` is the n-th element from the end.
                Some(from_end) => from_end.parse::<i64>().ok().filter(|idx| *idx < 0),
                None => step[..end].parse::<i64>().ok().filter(|idx| *idx >= 0),
            };
            write!(path, "[{}]", idx.ok_or_else(unsupported)?).unwrap();
            rest = &step[end + 1..];
        } else {
            return Err(unsupported());
        }
    }
    Ok(())
}

/// The elements of a PostgreSQL text array literal such as `{a,"b,c",NULL}`, where an unquoted
/// NULL is `None`; the braces are optional.
fn text_array_elements(s: &str) -> Vec<Option<String>> {
    fn finish(mut elem: String, kept: usize, literal: bool) -> Option<String> {
        let len = elem.trim_end().len().max(kept);
        elem.truncate(len);
        (literal || !elem.eq_ignore_ascii_case("NULL")).then_some(elem)
    }

    let s = s.trim();
    let inner = s
        .strip_prefix('{')
        .and_then(|s| s.strip_suffix('}'))
        .unwrap_or(s);
    if inner.trim().is_empty() {
        return vec![];
    }
    let mut out = vec![];
    let mut elem = String::new();
    // The length of `elem` that ends in quoted or escaped text, which is not trimmed.
    let mut kept = 0;
    let mut literal = false;
    let mut in_quotes = false;
    let mut chars = inner.chars();
    loop {
        match chars.next() {
            Some('"') => {
                in_quotes = !in_quotes;
                literal = true;
                kept = elem.len();
            },
            Some('\\') => {
                elem.extend(chars.next());
                literal = true;
                kept = elem.len();
            },
            Some(c) if c.is_whitespace() && !in_quotes && elem.is_empty() => {},
            Some(',') if !in_quotes => {
                out.push(finish(std::mem::take(&mut elem), kept, literal));
                kept = 0;
                literal = false;
            },
            Some(c) => {
                elem.push(c);
                if in_quotes {
                    kept = elem.len();
                }
            },
            None => {
                out.push(finish(elem, kept, literal));
                return out;
            },
        }
    }
}
