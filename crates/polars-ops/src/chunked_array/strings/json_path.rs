use std::borrow::Cow;

use arrow::array::ValueSize;
use jsonpath_lib::PathCompiled;
use polars_core::prelude::arity::{broadcast_try_binary_elementwise, unary_elementwise};
use serde_json::Value;

use super::*;

pub fn extract_json(expr: &PathCompiled, json_str: &str) -> Option<String> {
    serde_json::from_str(json_str).ok().and_then(|value| {
        // TODO: a lot of heap allocations here. Improve json path by adding a take?
        let result = expr.select(&value).ok()?;
        let first = *result.first()?;

        match first {
            Value::String(s) => Some(s.clone()),
            Value::Null => None,
            v => Some(v.to_string()),
        }
    })
}

/// Returns a string of the most specific value given the compiled JSON path expression.
/// This avoids creating a list to represent individual elements so that they can be
/// selected directly.
pub fn select_json<'a>(expr: &PathCompiled, json_str: &'a str) -> Option<Cow<'a, str>> {
    serde_json::from_str(json_str).ok().and_then(|value| {
        // TODO: a lot of heap allocations here. Improve json path by adding a take?
        let result = expr.select(&value).ok()?;

        let result_str = match result.len() {
            0 => None,
            1 => serde_json::to_string(&result[0]).ok(),
            _ => serde_json::to_string(&result).ok(),
        };

        result_str.map(Cow::Owned)
    })
}

pub trait Utf8JsonPathImpl: AsString {
    /// Extract json path, first match
    /// Refer to <https://goessner.net/articles/JsonPath/>
    fn json_path_match(&self, json_path: &StringChunked) -> PolarsResult<StringChunked> {
        let ca = self.as_string();
        match (ca.len(), json_path.len()) {
            (_, 1) => {
                // SAFETY: `json_path` was verified to have exactly 1 element.
                let opt_path = unsafe { json_path.get_unchecked(0) };
                let out = if let Some(path) = opt_path {
                    let pat = PathCompiled::compile(path).map_err(
                        |e| polars_err!(ComputeError: "error compiling JSON path expression {}", e),
                    )?;
                    unary_elementwise(ca, |opt_s| opt_s.and_then(|s| extract_json(&pat, s)))
                } else {
                    StringChunked::full_null(ca.name(), ca.len())
                };
                Ok(out)
            },
            (len_ca, len_path) if len_ca == 1 || len_ca == len_path => {
                broadcast_try_binary_elementwise(ca, json_path, |opt_str, opt_path| {
                    match (opt_str, opt_path) {
                    (Some(str_val), Some(path)) => {
                        PathCompiled::compile(path)
                            .map_err(|e| polars_err!(ComputeError: "error compiling JSON path expression {}", e))
                            .map(|path| extract_json(&path, str_val))
                    },
                    _ => Ok(None),
                }
                })
            },
            (len_ca, len_path) => {
                polars_bail!(ComputeError: "The length of `ca` and `json_path` should either 1 or the same, but `{}`, `{}` founded", len_ca, len_path)
            },
        }
    }

    /// Returns the inferred DataType for JSON values for each row
    /// in the StringChunked, with an optional number of rows to inspect.
    /// When None is passed for the number of rows, all rows are inspected.
    fn json_infer(&self, number_of_rows: Option<usize>) -> PolarsResult<DataType> {
        let ca = self.as_string();
        let values_iter = ca
            .iter()
            .map(|x| x.unwrap_or("null"))
            .take(number_of_rows.unwrap_or(ca.len()));

        polars_json::ndjson::infer_iter(values_iter)
            .map(|d| DataType::from(&d))
            .map_err(|e| polars_err!(ComputeError: "error inferring JSON: {}", e))
    }

    /// Extracts a typed-JSON value for each row in the StringChunked
    fn json_decode(
        &self,
        dtype: Option<DataType>,
        infer_schema_len: Option<usize>,
    ) -> PolarsResult<Series> {
        let ca = self.as_string();
        let dtype = match dtype {
            Some(dt) => dt,
            None => ca.json_infer(infer_schema_len)?,
        };
        let buf_size = ca.get_values_size() + ca.null_count() * "null".len();
        let iter = ca.iter().map(|x| x.unwrap_or("null"));

        let array = polars_json::ndjson::deserialize::deserialize_iter(
            iter,
            dtype.to_arrow(CompatLevel::newest()),
            buf_size,
            ca.len(),
        )
        .map_err(|e| polars_err!(ComputeError: "error deserializing JSON: {}", e))?;
        Series::try_from(("", array))
    }

    fn json_path_select(&self, json_path: &str) -> PolarsResult<StringChunked> {
        let pat = PathCompiled::compile(json_path)
            .map_err(|e| polars_err!(ComputeError: "error compiling JSONpath expression: {}", e))?;
        Ok(self
            .as_string()
            .apply(|opt_s| opt_s.and_then(|s| select_json(&pat, s))))
    }

    fn json_path_extract(
        &self,
        json_path: &str,
        dtype: Option<DataType>,
        infer_schema_len: Option<usize>,
    ) -> PolarsResult<Series> {
        let selected_json = self.as_string().json_path_select(json_path)?;
        selected_json.json_decode(dtype, infer_schema_len)
    }
}

impl Utf8JsonPathImpl for StringChunked {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_select() {
        let json_str = r#"{"a":1,"b":{"c":"hello"},"d":[{"e":0},{"e":2},{"e":null}]}"#;

        let compile = |s| PathCompiled::compile(s).unwrap();
        let some_cow = |s: &str| Some(Cow::Owned(s.to_string()));

        assert_eq!(select_json(&compile("$"), json_str), some_cow(json_str));
        assert_eq!(select_json(&compile("$.a"), json_str), some_cow("1"));
        assert_eq!(
            select_json(&compile("$.b.c"), json_str),
            some_cow(r#""hello""#)
        );
        assert_eq!(select_json(&compile("$.d[0].e"), json_str), some_cow("0"));
        assert_eq!(
            select_json(&compile("$.d[2].e"), json_str),
            some_cow("null")
        );
        assert_eq!(
            select_json(&compile("$.d[:].e"), json_str),
            some_cow("[0,2,null]")
        );
    }

    #[test]
    fn test_json_infer() {
        let s = Series::new(
            "json",
            [
                None,
                Some(r#"{"a": 1, "b": [{"c": 0}, {"c": 1}]}"#),
                Some(r#"{"a": 2, "b": [{"c": 2}, {"c": 5}]}"#),
                None,
            ],
        );
        let ca = s.str().unwrap();

        let inner_dtype = DataType::Struct(vec![Field::new("c", DataType::Int64)]);
        let expected_dtype = DataType::Struct(vec![
            Field::new("a", DataType::Int64),
            Field::new("b", DataType::List(Box::new(inner_dtype))),
        ]);

        assert_eq!(ca.json_infer(None).unwrap(), expected_dtype);
        // Infereing with the first row will only see None
        assert_eq!(ca.json_infer(Some(1)).unwrap(), DataType::Null);
        assert_eq!(ca.json_infer(Some(2)).unwrap(), expected_dtype);
    }

    #[test]
    fn test_json_decode() {
        let s = Series::new(
            "json",
            [
                None,
                Some(r#"{"a": 1, "b": "hello"}"#),
                Some(r#"{"a": 2, "b": "goodbye"}"#),
                None,
            ],
        );
        let ca = s.str().unwrap();

        let expected_series = StructChunked::from_series(
            "",
            &[
                Series::new("a", &[None, Some(1), Some(2), None]),
                Series::new("b", &[None, Some("hello"), Some("goodbye"), None]),
            ],
        )
        .unwrap()
        .with_outer_validity_chunked(BooleanChunked::new("", [false, true, true, false]))
        .into_series();
        let expected_dtype = expected_series.dtype().clone();

        assert!(ca
            .json_decode(None, None)
            .unwrap()
            .equals_missing(&expected_series));
        assert!(ca
            .json_decode(Some(expected_dtype), None)
            .unwrap()
            .equals_missing(&expected_series));
    }

    #[test]
    fn test_json_path_select() {
        let s = Series::new(
            "json",
            [
                None,
                Some(r#"{"a":1,"b":[{"c":0},{"c":1}]}"#),
                Some(r#"{"a":2,"b":[{"c":2},{"c":5}]}"#),
                None,
            ],
        );
        let ca = s.str().unwrap();

        assert!(ca
            .json_path_select("$")
            .unwrap()
            .into_series()
            .equals_missing(&s));

        let b_series = Series::new(
            "json",
            [
                None,
                Some(r#"[{"c":0},{"c":1}]"#),
                Some(r#"[{"c":2},{"c":5}]"#),
                None,
            ],
        );
        assert!(ca
            .json_path_select("$.b")
            .unwrap()
            .into_series()
            .equals_missing(&b_series));

        let c_series = Series::new("json", [None, Some(r#"[0,1]"#), Some(r#"[2,5]"#), None]);
        assert!(ca
            .json_path_select("$.b[:].c")
            .unwrap()
            .into_series()
            .equals_missing(&c_series));
    }

    #[test]
    fn test_json_path_extract() {
        let s = Series::new(
            "json",
            [
                None,
                Some(r#"{"a":1,"b":[{"c":0},{"c":1}]}"#),
                Some(r#"{"a":2,"b":[{"c":2},{"c":5}]}"#),
                None,
            ],
        );
        let ca = s.str().unwrap();

        let c_series = Series::new(
            "",
            [
                None,
                Some(Series::new("", &[0, 1])),
                Some(Series::new("", &[2, 5])),
                None,
            ],
        );

        assert!(ca
            .json_path_extract("$.b[:].c", None, None)
            .unwrap()
            .into_series()
            .equals_missing(&c_series));
    }
}
