//! The JSONPath that the `->`, `->>`, `#>` and `#>>` operators read from JSON text.

use std::fmt::Write;

use polars_core::prelude::*;
use polars_lazy::dsl::Expr;
use polars_plan::plans::{DynLiteralValue, LiteralValue};

/// The JSONPath that `->` or `->>` (or with `path_array`, `#>` or `#>>`) reads with `rhs`;
/// `None` if `rhs` is not a literal path. A key that starts with `$` is a JSONPath, as in DuckDB,
/// and an integer is an array index.
pub(crate) fn json_path(rhs: &Expr, path_array: bool) -> PolarsResult<Option<String>> {
    let Expr::Literal(lv) = rhs else {
        return Ok(None);
    };
    let mut path = String::from("$");
    if let LiteralValue::Dyn(DynLiteralValue::Int(idx)) = lv
        && !path_array
    {
        write!(path, "[{idx}]").unwrap();
        return Ok(Some(path));
    }
    let Some(key) = lv.extract_str() else {
        return Ok(None);
    };
    if path_array {
        for elem in text_array_elements(key) {
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
    Ok(Some(path))
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
                    let end = quoted.find('"').ok_or_else(unsupported)?;
                    (&quoted[..end], &quoted[end + 1..])
                },
                None => {
                    let end = step.find(['.', '[']).unwrap_or(step.len());
                    (&step[..end], &step[end..])
                },
            };
            if key.is_empty() || key == "*" {
                return Err(unsupported());
            }
            push_key(path, key)?;
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

/// The elements of a PostgreSQL text array literal such as `{a,"b,c"}`; the braces are optional.
fn text_array_elements(s: &str) -> Vec<String> {
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
    let mut in_quotes = false;
    let mut quoted = false;
    let mut chars = inner.chars();
    loop {
        match chars.next() {
            Some('"') => {
                in_quotes = !in_quotes;
                quoted = true;
            },
            Some('\\') => elem.extend(chars.next()),
            Some(c) if c.is_whitespace() && !in_quotes && (quoted || elem.is_empty()) => {},
            Some(',') if !in_quotes => {
                let len = if quoted {
                    elem.len()
                } else {
                    elem.trim_end().len()
                };
                elem.truncate(len);
                out.push(std::mem::take(&mut elem));
                quoted = false;
            },
            Some(c) => elem.push(c),
            None => {
                let len = if quoted {
                    elem.len()
                } else {
                    elem.trim_end().len()
                };
                elem.truncate(len);
                out.push(elem);
                return out;
            },
        }
    }
}
