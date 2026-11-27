//! JSON parsing without `serde_json`'s default recursion limit of 128, which valid inputs exceed:
//! each nested struct takes three levels in schema JSON, and each `and`/`or` one level in a row
//! filter.
use serde::de::DeserializeOwned;

use crate::iceberg::error::{IcebergResult, err_invalid_data, err_not_implemented};

/// Deeper inputs are reported as unsupported, so that Polars falls back to PyIceberg.
const MAX_DEPTH: usize = 512;
/// Inputs up to this depth are parsed on the calling thread, deeper ones on a thread with a stack
/// of [`DEEP_STACK_SIZE`].
const SHALLOW_DEPTH: usize = 128;
const DEEP_STACK_SIZE: usize = 64 << 20;

pub fn from_slice<T: DeserializeOwned + Send>(bytes: &[u8], what: &str) -> IcebergResult<T> {
    let depth = max_depth(bytes);
    if depth > MAX_DEPTH {
        return Err(err_not_implemented(format!(
            "{what} nested deeper than {MAX_DEPTH} levels"
        )));
    }

    let parse = || {
        let mut de = serde_json::Deserializer::from_slice(bytes);
        de.disable_recursion_limit();
        T::deserialize(&mut de)
            .and_then(|v| de.end().map(|()| v))
            .map_err(|e| err_invalid_data(format!("invalid {what}: {e}")))
    };
    if depth <= SHALLOW_DEPTH {
        return parse();
    }
    std::thread::scope(|s| {
        std::thread::Builder::new()
            .name("polars-iceberg-json".into())
            .stack_size(DEEP_STACK_SIZE)
            .spawn_scoped(s, parse)
            .map_err(|e| err_invalid_data(format!("failed to spawn thread: {e}")))?
            .join()
            .unwrap_or_else(|p| std::panic::resume_unwind(p))
    })
}

/// Maximum array/object nesting depth. Malformed input is left to the parser to report.
fn max_depth(bytes: &[u8]) -> usize {
    let (mut depth, mut max) = (0usize, 0usize);
    let (mut in_str, mut escaped) = (false, false);
    for &b in bytes {
        if in_str {
            match (escaped, b) {
                (true, _) => escaped = false,
                (false, b'\\') => escaped = true,
                (false, b'"') => in_str = false,
                _ => {},
            }
            continue;
        }
        match b {
            b'"' => in_str = true,
            b'[' | b'{' => {
                depth += 1;
                max = max.max(depth);
            },
            b']' | b'}' => depth = depth.saturating_sub(1),
            _ => {},
        }
    }
    max
}

#[cfg(test)]
mod tests {
    use serde_json::Value;

    use super::*;

    fn nested(depth: usize) -> String {
        format!("{}{}", "[".repeat(depth), "]".repeat(depth))
    }

    #[test]
    fn depth_ignores_brackets_in_strings() {
        assert_eq!(max_depth(br#"{"a": "[{\"[", "b": [[]]}"#), 3);
    }

    #[test]
    fn parses_beyond_default_recursion_limit() {
        assert!(from_slice::<Value>(nested(MAX_DEPTH).as_bytes(), "test JSON").is_ok());
    }

    #[test]
    fn too_deep_is_unsupported() {
        let err = from_slice::<Value>(nested(MAX_DEPTH + 1).as_bytes(), "test JSON").unwrap_err();
        assert!(err.message().contains("unsupported"), "{}", err.message());
    }

    #[test]
    fn trailing_characters_are_rejected() {
        assert!(from_slice::<Value>(b"{} x", "test JSON").is_err());
    }
}
