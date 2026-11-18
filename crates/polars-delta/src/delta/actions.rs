//! Actions remain untyped until snapshot metadata determines physical names and logical types.
use polars_utils::aliases::PlHashSet;
use serde_json::Value;

use super::error::{Result, invalid};
#[derive(Debug, Clone)]
pub struct Action {
    pub kind: String,
    pub value: Value,
}
pub fn parse_json(bytes: &[u8]) -> Result<Vec<Action>> {
    let mut out = Vec::new();
    for line in bytes
        .split(|b| *b == b'\n')
        .filter(|l| !l.iter().all(u8::is_ascii_whitespace))
    {
        let value: Value = serde_json::from_slice(line).map_err(invalid)?;
        if let Some(a) = from_row(value)? {
            out.push(a)
        }
    }
    Ok(out)
}
pub fn from_row(row: Value) -> Result<Option<Action>> {
    let object = row
        .as_object()
        .ok_or_else(|| invalid("action must be object"))?;
    let mut populated = object.iter().filter(|(_, v)| !v.is_null());
    let Some((kind, value)) = populated.next() else {
        return Err(invalid("empty action"));
    };
    if populated.next().is_some() {
        return Err(invalid("multiple actions in one row"));
    }
    Ok(Some(Action {
        kind: kind.clone(),
        value: value.clone(),
    }))
}
pub fn string<'a>(v: &'a Value, key: &str) -> Result<&'a str> {
    v.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| invalid(format!("missing/invalid {key}")))
}
pub fn u64_value(v: &Value, key: &str) -> Result<u64> {
    v.get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| invalid(format!("missing/invalid {key}")))
}
pub fn logical_key(v: &Value) -> Result<(String, Option<String>)> {
    let path = string(v, "path")?.to_owned();
    let dv = v
        .get("deletionVector")
        .filter(|d| !d.is_null())
        .map(|d| {
            let typ = string(d, "storageType")?;
            let value = string(d, "pathOrInlineDv")?;
            Ok(match d.get("offset").filter(|o| !o.is_null()) {
                Some(o) => format!(
                    "{typ}{value}@{}",
                    o.as_u64()
                        .ok_or_else(|| invalid("negative/invalid DV offset"))?
                ),
                None => format!("{typ}{value}"),
            })
        })
        .transpose()?;
    Ok((path, dv))
}
pub fn validate(actions: &[Action], checkpoint: bool, sidecar: bool, version: i64) -> Result<()> {
    let mut files = PlHashSet::default();
    let (mut metadata, mut protocol, mut cp, mut sidecars) = (0, 0, 0, 0);
    for a in actions {
        if !checkpoint && !sidecar && matches!(a.kind.as_str(), "checkpointMetadata" | "sidecar") {
            return Err(invalid("checkpoint-only action in Delta commit"));
        }
        match a.kind.as_str() {
            "add" | "remove" => {
                if !files.insert(logical_key(&a.value)?) {
                    return Err(invalid("duplicate add/remove logical file in action set"));
                }
                if a.kind == "add" {
                    u64_value(&a.value, "size")?;
                    if !a.value.get("partitionValues").is_some_and(Value::is_object) {
                        return Err(invalid("missing add.partitionValues"));
                    }
                }
            },
            "metaData" => metadata += 1,
            "protocol" => protocol += 1,
            "checkpointMetadata" => {
                cp += 1;
                if a.value.get("version").and_then(Value::as_i64) != Some(version) {
                    return Err(invalid("checkpointMetadata version mismatch"));
                }
            },
            "sidecar" => {
                sidecars += 1;
                string(&a.value, "path")?;
                u64_value(&a.value, "sizeInBytes")?;
            },
            _ => {},
        }
        if sidecar && !matches!(a.kind.as_str(), "add" | "remove") {
            return Err(invalid("non-file action in sidecar"));
        }
    }
    if metadata > 1 || protocol > 1 || cp > 1 {
        return Err(invalid("duplicate metadata/protocol/checkpointMetadata"));
    }
    if checkpoint && !sidecar && (metadata != 1 || protocol != 1) {
        return Err(invalid(
            "checkpoint must contain exactly one metadata and protocol",
        ));
    }
    if sidecars > 0 && (!files.is_empty() || cp != 1) {
        return Err(invalid(
            "sidecar checkpoint has embedded file actions or no checkpointMetadata",
        ));
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn keys_distinguish_offset_absence() {
        let a = serde_json::json!({"path":"a","deletionVector":{"storageType":"p","pathOrInlineDv":"dv"}});
        let mut b = a.clone();
        b["deletionVector"]["offset"] = 0.into();
        assert_ne!(logical_key(&a).unwrap(), logical_key(&b).unwrap());
    }
    #[test]
    fn rejects_duplicate() {
        let a=parse_json(b"{\"add\":{\"path\":\"a\",\"size\":1,\"partitionValues\":{}}}\n{\"remove\":{\"path\":\"a\"}}\n").unwrap();
        assert!(validate(&a, false, false, 0).is_err());
    }
    #[test]
    fn rejects_checkpoint_actions_in_commits() {
        for action in [
            serde_json::json!({"checkpointMetadata":{"version":0}}),
            serde_json::json!({"sidecar":{"path":"x.parquet","sizeInBytes":1}}),
        ] {
            let actions = parse_json(action.to_string().as_bytes()).unwrap();
            assert!(validate(&actions, false, false, 0).is_err());
        }
    }
    #[test]
    fn rejects_two_actions() {
        assert!(parse_json(b"{\"add\":{},\"remove\":{}}").is_err())
    }
}
