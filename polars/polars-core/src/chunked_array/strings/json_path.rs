use crate::prelude::*;
use jsonpath_lib::PathCompiled;
use serde_json::Value;
use std::borrow::Cow;
use indexmap::set::IndexSet as HashSet;
use arrow::io::{json, ndjson};
use arrow::datatypes::DataType as ArrowDataType;


#[cfg(feature = "extract_jsonpath")]
fn extract_json<'a>(expr: &PathCompiled, json_str: &'a str) -> Option<Cow<'a, str>> {
    serde_json::from_str(json_str).ok().and_then(|value| {
        // TODO: a lot of heap allocations here. Improve json path by adding a take?
        let result = expr.select(&value).ok()?;

        let result_str = match result.len() {
            0 => None,
            1 => serde_json::to_string(&result[0]).ok(),
            _ => serde_json::to_string(&result).ok(),
        };
        //let first = *result.get(0)?;

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
        // rechunk to have a continuous array
        self.rechunk();
        let chunk = &self.chunks()[0];
        let utf8_array = chunk.as_any().downcast_ref::<Utf8Array<i64>>().unwrap();
        utf8_array
            .json_infer(number_of_rows)
            .map(|d| DataType::from(&d))
            .map_err(|e| PolarsError::ComputeError(
                format!("error infering JSON {:?}", e).into(),
            ))
    }


    /// Extracts a JSON value for each row in the Utf8Chunked
    pub fn json_deserialize(&self, data_type: DataType) -> Result<Series> {
        // rechunk to have a continuous array
        self.rechunk();
        let chunk = &self.chunks()[0];
        let utf8_array = chunk.as_any().downcast_ref::<Utf8Array<i64>>().unwrap();
        let array = utf8_array
            .json_deserialize(data_type.to_arrow())
            .map_err(|e| PolarsError::ComputeError(
                format!("error deserializing JSON {:?}", e).into(),
            ))?;

        Series::try_from(("", array))
    }

    pub fn json_path_extract(&self, json_path: &str) -> Result<Series> {
        let expr = Compiled::compile(json_path)
            .map_err(|e| PolarsError::ComputeError(
                format!("error compiling JSONpath expression {:?}", e).into(),
            ))?;

        let selected_json = self.apply_on_opt(|opt_s| opt_s.and_then(|s| extract_json(&expr, s)));

        let data_type = selected_json.json_infer(None)?;
        selected_json.json_deserialize(data_type)
    }
}
