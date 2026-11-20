//! Delta log actions, parsed into typed records without building a JSON tree per action.
//! Add and remove actions are the bulk of a log; the small actions (metadata, protocol, ...) stay
//! `Value`s because snapshot metadata decides what the rest means.
use std::borrow::Cow;
use std::fmt;

use polars_utils::aliases::PlHashSet;
use serde::Deserialize;
use serde::de::{self, Deserializer, IgnoredAny, MapAccess, Visitor};
use serde_json::Value;

use super::error::{Result, invalid};
/// A deletion vector descriptor, as in the log.
#[derive(Debug, Clone)]
pub struct DvRec {
    pub storage_type: Box<str>,
    pub path_or_inline: Box<str>,
    pub offset: Option<u64>,
    pub size_in_bytes: Option<u64>,
    pub cardinality: Option<u64>,
}
#[derive(Debug, Clone)]
pub struct AddRec {
    pub path: Box<str>,
    pub size: u64,
    pub partition_values: Vec<(Box<str>, Option<Box<str>>)>,
    /// The `stats` JSON text.
    pub stats: Option<Box<str>>,
    pub dv: Option<DvRec>,
}
#[derive(Debug, Clone)]
pub struct RemoveRec {
    pub path: Box<str>,
    pub dv: Option<DvRec>,
}
/// Identity of a deletion vector, the part that distinguishes versions of one data file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DvId<'a> {
    pub storage_type: &'a str,
    pub path_or_inline: &'a str,
    /// Absent and zero are different identities.
    pub offset: Option<u64>,
}
/// A logical file: data file plus deletion vector. Adds reconcile by path; removes by this key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Key<'a> {
    pub path: &'a str,
    pub dv: Option<DvId<'a>>,
}
impl DvRec {
    pub fn id(&self) -> DvId<'_> {
        DvId {
            storage_type: &self.storage_type,
            path_or_inline: &self.path_or_inline,
            offset: self.offset,
        }
    }
}
/// Everything one JSON log file (a commit or a JSON checkpoint) holds.
#[derive(Debug, Default)]
pub struct Commit {
    pub adds: Vec<AddRec>,
    pub removes: Vec<RemoveRec>,
    pub metadata: Vec<Value>,
    pub protocol: Vec<Value>,
    pub checkpoint_metadata: Vec<Value>,
    pub sidecars: Vec<Value>,
    /// `commitInfo.inCommitTimestamp` of each `commitInfo` action.
    pub commit_info: Vec<Option<i64>>,
}
impl Commit {
    /// Add and remove actions with the same logical file are contradictory.
    pub fn check_file_keys(&self) -> Result<()> {
        let mut keys = PlHashSet::default();
        for a in &self.adds {
            let key = Key {
                path: &a.path,
                dv: a.dv.as_ref().map(DvRec::id),
            };
            if !keys.insert(key) {
                return Err(invalid("duplicate add/remove logical file in action set"));
            }
        }
        for r in &self.removes {
            let key = Key {
                path: &r.path,
                dv: r.dv.as_ref().map(DvRec::id),
            };
            if !keys.insert(key) {
                return Err(invalid("duplicate add/remove logical file in action set"));
            }
        }
        Ok(())
    }
    /// The checks that apply to a commit: no checkpoint-only actions, at most one of each
    /// singleton, no contradictory file actions.
    pub fn validate_commit(&self, files: bool) -> Result<()> {
        if !self.checkpoint_metadata.is_empty() || !self.sidecars.is_empty() {
            return Err(invalid("checkpoint-only action in Delta commit"));
        }
        if self.metadata.len() > 1 || self.protocol.len() > 1 {
            return Err(invalid("duplicate metadata/protocol/checkpointMetadata"));
        }
        if files {
            self.check_file_keys()?;
        }
        Ok(())
    }
}
/// A string that borrows from the input when it has no escapes.
struct BCow<'a>(Cow<'a, str>);
impl<'de> Deserialize<'de> for BCow<'de> {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = BCow<'de>;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a string")
            }
            fn visit_borrowed_str<E: de::Error>(
                self,
                v: &'de str,
            ) -> std::result::Result<BCow<'de>, E> {
                Ok(BCow(Cow::Borrowed(v)))
            }
            fn visit_str<E: de::Error>(self, v: &str) -> std::result::Result<BCow<'de>, E> {
                Ok(BCow(Cow::Owned(v.to_owned())))
            }
            fn visit_string<E: de::Error>(self, v: String) -> std::result::Result<BCow<'de>, E> {
                Ok(BCow(Cow::Owned(v)))
            }
        }
        d.deserialize_str(V)
    }
}
/// A partition value: a string or null (other scalars are read as their text).
struct PvValue<'a>(Option<Cow<'a, str>>);
impl<'de> Deserialize<'de> for PvValue<'de> {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = PvValue<'de>;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a string or null")
            }
            fn visit_borrowed_str<E: de::Error>(
                self,
                v: &'de str,
            ) -> std::result::Result<Self::Value, E> {
                Ok(PvValue(Some(Cow::Borrowed(v))))
            }
            fn visit_str<E: de::Error>(self, v: &str) -> std::result::Result<Self::Value, E> {
                Ok(PvValue(Some(Cow::Owned(v.to_owned()))))
            }
            fn visit_string<E: de::Error>(self, v: String) -> std::result::Result<Self::Value, E> {
                Ok(PvValue(Some(Cow::Owned(v))))
            }
            fn visit_unit<E: de::Error>(self) -> std::result::Result<Self::Value, E> {
                Ok(PvValue(None))
            }
            fn visit_none<E: de::Error>(self) -> std::result::Result<Self::Value, E> {
                Ok(PvValue(None))
            }
            fn visit_bool<E: de::Error>(self, v: bool) -> std::result::Result<Self::Value, E> {
                Ok(PvValue(Some(Cow::Owned(v.to_string()))))
            }
            fn visit_i64<E: de::Error>(self, v: i64) -> std::result::Result<Self::Value, E> {
                Ok(PvValue(Some(Cow::Owned(v.to_string()))))
            }
            fn visit_u64<E: de::Error>(self, v: u64) -> std::result::Result<Self::Value, E> {
                Ok(PvValue(Some(Cow::Owned(v.to_string()))))
            }
            fn visit_f64<E: de::Error>(self, v: f64) -> std::result::Result<Self::Value, E> {
                Ok(PvValue(Some(Cow::Owned(v.to_string()))))
            }
        }
        d.deserialize_any(V)
    }
}
type Pairs<'a> = Vec<(Cow<'a, str>, Option<Cow<'a, str>>)>;
struct PartitionValues<'a>(Pairs<'a>);
impl<'de> Deserialize<'de> for PartitionValues<'de> {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = PartitionValues<'de>;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a map of strings")
            }
            fn visit_map<A: MapAccess<'de>>(
                self,
                mut map: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                let mut out = Vec::with_capacity(map.size_hint().unwrap_or(0).min(64));
                while let Some(key) = map.next_key::<BCow>()? {
                    out.push((key.0, map.next_value::<PvValue>()?.0));
                }
                Ok(PartitionValues(out))
            }
        }
        d.deserialize_map(V)
    }
}
struct RawDv<'a> {
    storage_type: Option<Cow<'a, str>>,
    path_or_inline: Option<Cow<'a, str>>,
    offset: Option<u64>,
    size_in_bytes: Option<u64>,
    cardinality: Option<u64>,
}
impl<'de> Deserialize<'de> for RawDv<'de> {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = RawDv<'de>;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a deletion vector descriptor")
            }
            fn visit_map<A: MapAccess<'de>>(
                self,
                mut map: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                let mut dv = RawDv {
                    storage_type: None,
                    path_or_inline: None,
                    offset: None,
                    size_in_bytes: None,
                    cardinality: None,
                };
                while let Some(key) = map.next_key::<BCow>()? {
                    match &*key.0 {
                        "storageType" => {
                            dv.storage_type = map.next_value::<Option<BCow>>()?.map(|v| v.0)
                        },
                        "pathOrInlineDv" => {
                            dv.path_or_inline = map.next_value::<Option<BCow>>()?.map(|v| v.0)
                        },
                        "offset" => dv.offset = map.next_value()?,
                        "sizeInBytes" => dv.size_in_bytes = map.next_value()?,
                        "cardinality" => dv.cardinality = map.next_value()?,
                        _ => {
                            map.next_value::<IgnoredAny>()?;
                        },
                    }
                }
                Ok(dv)
            }
        }
        d.deserialize_map(V)
    }
}
impl RawDv<'_> {
    fn own(self) -> Result<DvRec> {
        Ok(DvRec {
            storage_type: self
                .storage_type
                .ok_or_else(|| invalid("missing/invalid storageType"))?
                .into_owned()
                .into_boxed_str(),
            path_or_inline: self
                .path_or_inline
                .ok_or_else(|| invalid("missing/invalid pathOrInlineDv"))?
                .into_owned()
                .into_boxed_str(),
            offset: self.offset,
            size_in_bytes: self.size_in_bytes,
            cardinality: self.cardinality,
        })
    }
}
struct RawFile<'a> {
    path: Option<Cow<'a, str>>,
    size: Option<u64>,
    partition_values: Option<Pairs<'a>>,
    stats: Option<Cow<'a, str>>,
    dv: Option<RawDv<'a>>,
}
impl<'de> Deserialize<'de> for RawFile<'de> {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = RawFile<'de>;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a file action")
            }
            fn visit_map<A: MapAccess<'de>>(
                self,
                mut map: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                let mut file = RawFile {
                    path: None,
                    size: None,
                    partition_values: None,
                    stats: None,
                    dv: None,
                };
                while let Some(key) = map.next_key::<BCow>()? {
                    match &*key.0 {
                        "path" => file.path = map.next_value::<Option<BCow>>()?.map(|v| v.0),
                        "size" => file.size = map.next_value()?,
                        "partitionValues" => {
                            file.partition_values =
                                map.next_value::<Option<PartitionValues>>()?.map(|v| v.0)
                        },
                        "stats" => file.stats = map.next_value::<Option<BCow>>()?.map(|v| v.0),
                        "deletionVector" => file.dv = map.next_value()?,
                        _ => {
                            map.next_value::<IgnoredAny>()?;
                        },
                    }
                }
                Ok(file)
            }
        }
        d.deserialize_map(V)
    }
}
fn boxed(s: Cow<'_, str>) -> Box<str> {
    s.into_owned().into_boxed_str()
}
impl RawFile<'_> {
    fn add(self) -> Result<AddRec> {
        Ok(AddRec {
            path: boxed(self.path.ok_or_else(|| invalid("missing/invalid path"))?),
            size: self.size.ok_or_else(|| invalid("missing/invalid size"))?,
            partition_values: self
                .partition_values
                .ok_or_else(|| invalid("missing add.partitionValues"))?
                .into_iter()
                .map(|(k, v)| (boxed(k), v.map(boxed)))
                .collect(),
            stats: self.stats.map(boxed),
            dv: self.dv.map(RawDv::own).transpose()?,
        })
    }
    fn remove(self) -> Result<RemoveRec> {
        Ok(RemoveRec {
            path: boxed(self.path.ok_or_else(|| invalid("missing/invalid path"))?),
            dv: self.dv.map(RawDv::own).transpose()?,
        })
    }
}
/// The `inCommitTimestamp` of a `commitInfo` action; its other fields are ignored.
struct CommitInfo(Option<i64>);
impl<'de> Deserialize<'de> for CommitInfo {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = CommitInfo;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("commitInfo")
            }
            fn visit_map<A: MapAccess<'de>>(
                self,
                mut map: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                let mut ict = None;
                while let Some(key) = map.next_key::<BCow>()? {
                    if &*key.0 == "inCommitTimestamp" {
                        // Anything but an integer counts as missing.
                        ict = match map.next_value::<Value>()? {
                            Value::Number(n) => n.as_i64(),
                            _ => None,
                        };
                    } else {
                        map.next_value::<IgnoredAny>()?;
                    }
                }
                Ok(CommitInfo(ict))
            }
        }
        d.deserialize_map(V)
    }
}
/// One log line: exactly one populated action.
struct Line<'a> {
    add: Option<RawFile<'a>>,
    remove: Option<RawFile<'a>>,
    metadata: Option<Value>,
    protocol: Option<Value>,
    checkpoint_metadata: Option<Value>,
    sidecar: Option<Value>,
    commit_info: Option<CommitInfo>,
    populated: usize,
}
impl<'de> Deserialize<'de> for Line<'de> {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = Line<'de>;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("an action object")
            }
            fn visit_map<A: MapAccess<'de>>(
                self,
                mut map: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                let mut line = Line {
                    add: None,
                    remove: None,
                    metadata: None,
                    protocol: None,
                    checkpoint_metadata: None,
                    sidecar: None,
                    commit_info: None,
                    populated: 0,
                };
                while let Some(key) = map.next_key::<BCow>()? {
                    let populated = match &*key.0 {
                        "add" => {
                            line.add = map.next_value()?;
                            line.add.is_some()
                        },
                        "remove" => {
                            line.remove = map.next_value()?;
                            line.remove.is_some()
                        },
                        "metaData" => {
                            line.metadata = map.next_value()?;
                            line.metadata.is_some()
                        },
                        "protocol" => {
                            line.protocol = map.next_value()?;
                            line.protocol.is_some()
                        },
                        "checkpointMetadata" => {
                            line.checkpoint_metadata = map.next_value()?;
                            line.checkpoint_metadata.is_some()
                        },
                        "sidecar" => {
                            line.sidecar = map.next_value()?;
                            line.sidecar.is_some()
                        },
                        "commitInfo" => {
                            line.commit_info = map.next_value()?;
                            line.commit_info.is_some()
                        },
                        _ => map.next_value::<Option<IgnoredAny>>()?.is_some(),
                    };
                    line.populated += usize::from(populated);
                }
                Ok(line)
            }
        }
        d.deserialize_map(V)
    }
}
/// Parse a JSON log file. With `skip_files`, add and remove lines are skipped unread (so are
/// not validated): enough for metadata, protocol and timestamps.
pub fn parse_json(bytes: &[u8], skip_files: bool) -> Result<Commit> {
    let mut out = Commit::default();
    for line in bytes.split(|b| *b == b'\n') {
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        if skip_files && (line.starts_with(b"{\"add\":") || line.starts_with(b"{\"remove\":")) {
            continue;
        }
        let mut de = serde_json::Deserializer::from_slice(line);
        let parsed = Line::deserialize(&mut de)
            .and_then(|l| de.end().map(|()| l))
            .map_err(|e| {
                if e.is_data() || e.is_syntax() || e.is_eof() {
                    invalid(format!("invalid action: {e}"))
                } else {
                    invalid(e)
                }
            })?;
        match parsed.populated {
            0 => return Err(invalid("empty action")),
            1 => {},
            _ => return Err(invalid("multiple actions in one row")),
        }
        if let Some(f) = parsed.add {
            if !skip_files {
                out.adds.push(f.add()?);
            }
        } else if let Some(f) = parsed.remove {
            if !skip_files {
                out.removes.push(f.remove()?);
            }
        } else if let Some(v) = parsed.metadata {
            out.metadata.push(v)
        } else if let Some(v) = parsed.protocol {
            out.protocol.push(v)
        } else if let Some(v) = parsed.checkpoint_metadata {
            out.checkpoint_metadata.push(v)
        } else if let Some(v) = parsed.sidecar {
            out.sidecars.push(v)
        } else if let Some(c) = parsed.commit_info {
            out.commit_info.push(c.0)
        }
    }
    Ok(out)
}
/// The `inCommitTimestamp` of the commit's `commitInfo`, reading lines up to it.
pub fn in_commit_timestamp(bytes: &[u8]) -> Result<Option<i64>> {
    for line in bytes.split(|b| *b == b'\n') {
        if line.iter().all(u8::is_ascii_whitespace)
            || line.starts_with(b"{\"add\":")
            || line.starts_with(b"{\"remove\":")
        {
            continue;
        }
        let commit = parse_json(line, true)?;
        if let Some(ict) = commit.commit_info.first() {
            return Ok(*ict);
        }
    }
    Err(invalid("missing inCommitTimestamp"))
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
#[cfg(test)]
mod tests {
    use super::*;
    fn key_of(json: &str) -> (Option<DvId<'static>>, usize) {
        let c = Box::leak(Box::new(parse_json(json.as_bytes(), false).unwrap()));
        (c.adds[0].dv.as_ref().map(DvRec::id), c.adds.len())
    }
    #[test]
    fn keys_distinguish_offset_absence() {
        let line = |dv: &str| {
            format!(
                r#"{{"add":{{"path":"a","size":1,"partitionValues":{{}},"deletionVector":{dv}}}}}"#
            )
        };
        let a = key_of(&line(r#"{"storageType":"p","pathOrInlineDv":"dv"}"#));
        let b = key_of(&line(
            r#"{"storageType":"p","pathOrInlineDv":"dv","offset":0}"#,
        ));
        assert_ne!(a.0, b.0);
    }
    #[test]
    fn rejects_duplicate() {
        let a = parse_json(
            b"{\"add\":{\"path\":\"a\",\"size\":1,\"partitionValues\":{}}}\n{\"remove\":{\"path\":\"a\"}}\n",
            false,
        )
        .unwrap();
        assert!(a.validate_commit(true).is_err());
    }
    #[test]
    fn rejects_checkpoint_actions_in_commits() {
        for action in [
            r#"{"checkpointMetadata":{"version":0}}"#,
            r#"{"sidecar":{"path":"x.parquet","sizeInBytes":1}}"#,
        ] {
            let actions = parse_json(action.as_bytes(), false).unwrap();
            assert!(actions.validate_commit(true).is_err());
        }
    }
    #[test]
    fn rejects_two_actions() {
        assert!(parse_json(b"{\"add\":{},\"remove\":{}}", false).is_err());
        assert!(parse_json(b"{\"add\":null}", false).is_err());
        assert!(parse_json(b"{}", false).is_err());
        // Unknown actions count as populated; null ones do not.
        assert!(parse_json(b"{\"txn\":{\"appId\":\"x\"},\"add\":null}", false).is_ok());
        assert!(parse_json(b"{\"txn\":{\"appId\":\"x\"},\"cdc\":{}}", false).is_err());
    }
    #[test]
    fn escaped_and_typed_fields() {
        let c = parse_json(
            br#"{"add":{"path":"p%3D1\/a\u00e4.parquet","size":7,"partitionValues":{"k":"v","n":null,"i":5},"stats":"{\"numRecords\":1}","tags":{"x":"y"}}}"#,
            false,
        )
        .unwrap();
        let a = &c.adds[0];
        assert_eq!(&*a.path, "p%3D1/aä.parquet");
        assert_eq!(a.size, 7);
        assert_eq!(a.partition_values.len(), 3);
        assert_eq!(a.partition_values[1].1, None);
        assert_eq!(a.partition_values[2].1.as_deref(), Some("5"));
        assert_eq!(a.stats.as_deref(), Some("{\"numRecords\":1}"));
        assert!(
            parse_json(
                br#"{"add":{"path":"a","size":-1,"partitionValues":{}}}"#,
                false
            )
            .is_err()
        );
        assert!(parse_json(br#"{"add":{"path":"a","partitionValues":{}}}"#, false).is_err());
        assert!(parse_json(br#"{"add":{"path":"a","size":1}}"#, false).is_err());
        assert!(
            parse_json(
                br#"{"add":{"path":"a","size":1,"partitionValues":{}}} x"#,
                false
            )
            .is_err()
        );
    }
    #[test]
    fn skipped_file_lines_are_not_parsed() {
        let c = parse_json(
            b"{\"add\":garbage}\n{\"protocol\":{\"minReaderVersion\":1}}",
            true,
        )
        .unwrap();
        assert_eq!(c.protocol.len(), 1);
        assert_eq!(
            in_commit_timestamp(
                b"{\"add\":garbage}\n{\"commitInfo\":{\"inCommitTimestamp\":9,\"x\":[1]}}"
            )
            .unwrap(),
            Some(9)
        );
        assert!(in_commit_timestamp(b"{\"add\":{}}").is_err());
    }
}
