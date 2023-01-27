use std::borrow::Cow;

use arrow::io::ndjson;
use jsonpath_lib::PathCompiled;
use serde_json::Value;

use super::*;

pub fn extract_json<'a>(expr: &PathCompiled, json_str: &'a str) -> Option<Cow<'a, str>> {
    serde_json::from_str(json_str).ok().and_then(|value| {
        // TODO: a lot of heap allocations here. Improve json path by adding a take?
        let result = expr.select(&value).ok()?;
        let first = *result.get(0)?;

        match first {
            Value::String(s) => Some(Cow::Owned(s.clone())),
            Value::Null => None,
            v => Some(Cow::Owned(v.to_string())),
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

pub trait Utf8JsonPathImpl: AsUtf8 {
    /// Extract json path, first match
    /// Refer to <https://goessner.net/articles/JsonPath/>
    fn json_path_match(&self, json_path: &str) -> PolarsResult<Utf8Chunked> {
        match PathCompiled::compile(json_path) {
            Ok(pat) => Ok(self
                .as_utf8()
                .apply_on_opt(|opt_s| opt_s.and_then(|s| extract_json(&pat, s)))),
            Err(e) => Err(PolarsError::ComputeError(
                format!("error compiling JSONpath expression {e:?}").into(),
            )),
        }
    }

    /// Returns the infered DataType for JSON values for each row
    /// in the Utf8Chunked, with an optional number of rows to inspect.
    /// When None is passed for the number of rows, all rows are inspected.
    fn json_infer(&self, number_of_rows: Option<usize>) -> PolarsResult<DataType> {
        let ca = self.as_utf8();
        let values_iter = ca
            .into_iter()
            .map(|x| x.unwrap_or("null"))
            .take(number_of_rows.unwrap_or(ca.len()));

        ndjson::read::infer_iter(values_iter)
            .map(|d| DataType::from(&d))
            .map_err(|e| PolarsError::ComputeError(format!("error infering JSON {e:?}").into()))
    }

    /// Extracts a typed-JSON value for each row in the Utf8Chunked
    fn json_extract(&self, dtype: Option<DataType>) -> PolarsResult<Series> {
        let ca = self.as_utf8();
        let dtype = match dtype {
            Some(dt) => dt,
            None => ca.json_infer(None)?,
        };

        let iter = ca.into_iter().map(|x| x.unwrap_or("null"));

        let array = ndjson::read::deserialize_iter(iter, dtype.to_arrow()).map_err(|e| {
            PolarsError::ComputeError(format!("error deserializing JSON {e:?}").into())
        })?;

        Series::try_from(("", array))
    }

    fn json_path_select(&self, json_path: &str) -> PolarsResult<Utf8Chunked> {
        match PathCompiled::compile(json_path) {
            Ok(pat) => Ok(self
                .as_utf8()
                .apply_on_opt(|opt_s| opt_s.and_then(|s| select_json(&pat, s)))),
            Err(e) => Err(PolarsError::ComputeError(
                format!("error compiling JSONpath expression {e:?}").into(),
            )),
        }
    }

    fn json_path_extract(&self, json_path: &str, dtype: Option<DataType>) -> PolarsResult<Series> {
        let selected_json = self.as_utf8().json_path_select(json_path)?;
        selected_json.json_extract(dtype)
    }
}

impl Utf8JsonPathImpl for Utf8Chunked {}

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
        let ca = s.utf8().unwrap();

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
    fn test_json_extract() {
        let s = Series::new(
            "json",
            [
                None,
                Some(r#"{"a": 1, "b": "hello"}"#),
                Some(r#"{"a": 2, "b": "goodbye"}"#),
                None,
            ],
        );
        let ca = s.utf8().unwrap();

        let expected_series = StructChunked::new(
            "",
            &[
                Series::new("a", &[None, Some(1), Some(2), None]),
                Series::new("b", &[None, Some("hello"), Some("goodbye"), None]),
            ],
        )
        .unwrap()
        .into_series();
        let expected_dtype = expected_series.dtype().clone();

        assert!(ca
            .json_extract(None)
            .unwrap()
            .series_equal_missing(&expected_series));
        assert!(ca
            .json_extract(Some(expected_dtype))
            .unwrap()
            .series_equal_missing(&expected_series));
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
        let ca = s.utf8().unwrap();

        assert!(ca
            .json_path_select("$")
            .unwrap()
            .into_series()
            .series_equal_missing(&s));

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
            .series_equal_missing(&b_series));

        let c_series = Series::new("json", [None, Some(r#"[0,1]"#), Some(r#"[2,5]"#), None]);
        assert!(ca
            .json_path_select("$.b[:].c")
            .unwrap()
            .into_series()
            .series_equal_missing(&c_series));
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
        let ca = s.utf8().unwrap();

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
            .json_path_extract("$.b[:].c", None)
            .unwrap()
            .into_series()
            .series_equal_missing(&c_series));
    }
}
