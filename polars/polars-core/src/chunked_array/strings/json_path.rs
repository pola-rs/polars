use crate::prelude::*;
use jsonpath_lib::PathCompiled;
use serde_json::Value;
use std::borrow::Cow;
use arrow::io::ndjson;


#[cfg(feature = "extract_jsonpath")]
fn extract_json<'a>(expr: &PathCompiled, json_str: &'a str) -> Option<Cow<'a, str>> {
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

#[cfg(feature = "extract_jsonpath")]
fn select_json<'a>(expr: &PathCompiled, json_str: &'a str) -> Option<Cow<'a, str>> {
    serde_json::from_str(json_str).ok().and_then(|value| {
        // TODO: a lot of heap allocations here. Improve json path by adding a take?
        let result = expr.select(&value).ok()?;

        let result_str = match result.len() {
            0 => None,
            1 => serde_json::to_string(&result[0]).ok(),
            _ => serde_json::to_string(&result).ok(),
        };

        match result_str {
            Some(s) => Some(Cow::Owned(s.clone())),
            None => None,
        }
    })
}

#[cfg(feature = "extract_jsonpath")]
impl Utf8Chunked {
    /// Extract json path, first match
    /// Refer to <https://goessner.net/articles/JsonPath/>
    pub fn json_path_match(&self, json_path: &str) -> Result<Utf8Chunked> {
        match PathCompiled::compile(json_path) {
            Ok(pat) => Ok(self.apply_on_opt(|opt_s| opt_s.and_then(|s| extract_json(&pat, s)))),
            Err(e) => Err(PolarsError::ComputeError(
                format!("error compiling JSONpath expression {:?}", e).into(),
            )),
        }
    }

    /// Returns the infered DataType for JSON values for each row
    /// in the Utf8Chunked, with an optional number of rows to inspect.
    /// When None is passed for the number of rows, all rows are inspected.
    pub fn json_infer(&self, number_of_rows: Option<usize>) -> Result<DataType> {
        let values_iter = self
            .into_iter()
            .map(|x| x.unwrap_or("null"))
            .take(number_of_rows.unwrap_or(self.len()));

        ndjson::read::infer_iter(values_iter)
            .map(|d| DataType::from(&d))
            .map_err(|e| PolarsError::ComputeError(
                format!("error infering JSON {:?}", e).into(),
            ))
    }


    /// Extracts a JSON value for each row in the Utf8Chunked
    pub fn json_extract(&self, dtype: Option<DataType>) -> Result<Series> {
        let dtype = match dtype {
            Some(dt) => dt,
            None => self.json_infer(None)?,
        };

        let iter = self
            .into_iter()
            .map(|x| x.unwrap_or("null"));

        let array = ndjson::read::deserialize_iter(iter, dtype.to_arrow())
            .map_err(|e| PolarsError::ComputeError(
                format!("error deserializing JSON {:?}", e).into(),
            ))?;

        Series::try_from(("", array))
    }

    pub fn json_path_select(&self, json_path: &str) -> Result<Utf8Chunked> {
        match PathCompiled::compile(json_path) {
            Ok(pat) => Ok(self.apply_on_opt(|opt_s| opt_s.and_then(|s| select_json(&pat, s)))),
            Err(e) => Err(PolarsError::ComputeError(
                format!("error compiling JSONpath expression {:?}", e).into(),
            )),
        }
    }

    pub fn json_path_extract(&self, json_path: &str, dtype: Option<DataType>) -> Result<Series> {
        let selected_json = self.json_path_select(json_path)?;
        selected_json.json_extract(dtype)
    }
}
