use std::borrow::Cow;

use jsonpath_lib::PathCompiled;
use serde_json::Value;

use crate::prelude::*;

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

impl Utf8Chunked {
    /// Extract json path, first match
    /// Refer to <https://goessner.net/articles/JsonPath/>
    #[cfg(feature = "extract_jsonpath")]
    pub fn json_path_match(&self, json_path: &str) -> PolarsResult<Utf8Chunked> {
        match PathCompiled::compile(json_path) {
            Ok(pat) => Ok(self.apply_on_opt(|opt_s| opt_s.and_then(|s| extract_json(&pat, s)))),
            Err(e) => Err(PolarsError::ComputeError(
                format!("error compiling JSONpath expression {:?}", e).into(),
            )),
        }
    }
}
