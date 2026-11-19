use std::collections::BTreeMap;

use futures::{StreamExt, TryStreamExt};
use polars_io_ext_ffi::delta_v1::Request;
use polars_utils::aliases::PlHashSet;
use serde_json::Value;

use super::actions::{self, Action};
use super::checkpoint;
use super::error::{Result, input, invalid};
use super::host::{Object, Storage};
#[derive(Debug)]
pub struct Snapshot {
    pub version: i64,
    pub metadata: Value,
    pub protocol: Value,
    pub files: Vec<Value>,
}
#[derive(Debug, Clone, PartialEq)]
enum Name {
    Commit(i64),
    Single(i64, bool),
    Part(i64, u32, u32),
}
fn version(s: &str) -> Option<i64> {
    (s.len() == 20 && s.bytes().all(|b| b.is_ascii_digit()))
        .then(|| s.parse().ok())
        .flatten()
}
fn name(s: &str) -> Option<Name> {
    let p: Vec<_> = s.split('.').collect();
    let v = version(p.first()?)?;
    match p.as_slice() {
        [_, "json"] => Some(Name::Commit(v)),
        [_, "checkpoint", "parquet"] => Some(Name::Single(v, false)),
        [_, "checkpoint", uuid, ext] if uuid.len() == 36 && matches!(*ext, "json" | "parquet") => {
            Some(Name::Single(v, true))
        },
        [_, "checkpoint", part, total, "parquet"] => {
            Some(Name::Part(v, part.parse().ok()?, total.parse().ok()?))
        },
        _ => None,
    }
}
#[derive(Default)]
struct Inventory {
    commits: BTreeMap<i64, Object>,
    checkpoints: BTreeMap<i64, Vec<Vec<Object>>>,
}
impl Inventory {
    fn new(objects: Vec<Object>) -> Result<Self> {
        let mut out = Self::default();
        let mut parts: BTreeMap<(i64, u32), BTreeMap<u32, Object>> = BTreeMap::new();
        for o in objects {
            let base = o.url.rsplit('/').next().unwrap_or("");
            match name(base) {
                Some(Name::Commit(v)) => {
                    if out.commits.insert(v, o).is_some() {
                        return Err(invalid("duplicate commit version listing"));
                    }
                },
                Some(Name::Single(v, _)) => out.checkpoints.entry(v).or_default().push(vec![o]),
                Some(Name::Part(v, p, n)) if p > 0 && p <= n && n > 0 => {
                    parts.entry((v, n)).or_default().insert(p, o);
                },
                Some(Name::Part(..)) | None => {},
            }
        }
        for ((v, n), p) in parts {
            if p.len() == n as usize && (1..=n).all(|i| p.contains_key(&i)) {
                out.checkpoints
                    .entry(v)
                    .or_default()
                    .push(p.into_values().collect());
            }
        }
        Ok(out)
    }
    fn latest(&self) -> Option<i64> {
        self.commits
            .last_key_value()
            .map(|(v, _)| *v)
            .into_iter()
            .chain(self.checkpoints.last_key_value().map(|(v, _)| *v))
            .max()
    }
}
pub fn absolute(root: &str, path: &str) -> Result<String> {
    let bytes = path.as_bytes();
    for (i, byte) in bytes.iter().enumerate() {
        if *byte == b'%'
            && !bytes
                .get(i + 1..i + 3)
                .is_some_and(|escape| escape.iter().all(u8::is_ascii_hexdigit))
        {
            return Err(invalid("invalid percent escape in Delta path URI"));
        }
    }
    if let Ok(url) = url::Url::parse(path) {
        return transport_path(&url);
    }
    // File URIs describe an escaped filesystem root. Subsequent roots, including
    // sidecar roots produced here, are raw Polars paths and must not be decoded.
    let local_root;
    let root = if let Ok(url) = url::Url::parse(root) {
        if url.scheme() == "file" {
            local_root = transport_path(&url)?;
            local_root.as_str()
        } else {
            root
        }
    } else {
        root
    };
    if let Some((scheme, rest)) = root.split_once("://") {
        let (authority, key) = rest.split_once('/').unwrap_or((rest, ""));
        // The root is already a Polars object path. Protect its literal escapes
        // and URI delimiters while resolving the URI-escaped Delta child.
        let key = key
            .replace('%', "%25")
            .replace('?', "%3F")
            .replace('#', "%23");
        let base = format!(
            "{scheme}://{authority}/{key}{}",
            if key.ends_with('/') { "" } else { "/" }
        );
        let url = url::Url::parse(&base)
            .map_err(invalid)?
            .join(path)
            .map_err(invalid)?;
        return transport_path(&url);
    }
    let decoded = percent_encoding::percent_decode_str(path)
        .decode_utf8()
        .map_err(invalid)?;
    let path = std::path::Path::new(decoded.as_ref());
    Ok(if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::path::Path::new(root).join(path)
    }
    .to_string_lossy()
    .into_owned())
}
// Polars storage and scan paths contain raw object keys, unlike Delta action URIs.
fn transport_path(url: &url::Url) -> Result<String> {
    if url.scheme() == "file" {
        return url
            .to_file_path()
            .map_err(|_| invalid("invalid local file URI"))?
            .into_os_string()
            .into_string()
            .map_err(|_| invalid("non-UTF-8 local file URI"));
    }
    let path = percent_encoding::percent_decode_str(url.path())
        .decode_utf8()
        .map_err(invalid)?;
    Ok(format!("{}{}", &url[..url::Position::BeforePath], path))
}
async fn read_actions(storage: &Storage, url: &str, request: &Request) -> Result<Vec<Action>> {
    if url.ends_with(".parquet") {
        checkpoint::read(storage, url, request, None).await
    } else {
        let bytes = storage.get(url).await?;
        // Small commits cost less to parse inline; larger commits can use the
        // bounded commit-read window concurrently without blocking the IO task.
        if bytes.as_slice().len() >= 32 * 1024 {
            tokio::task::spawn_blocking(move || actions::parse_json(bytes.as_slice()))
                .await
                .map_err(invalid)?
        } else {
            actions::parse_json(bytes.as_slice())
        }
    }
}
async fn snapshot(
    storage: &Storage,
    inv: &Inventory,
    root: &str,
    target: i64,
    threads: usize,
    request: &Request,
) -> Result<Snapshot> {
    let checkpoint = inv.checkpoints.range(..=target).next_back();
    let base = checkpoint.map_or(-1, |(v, _)| *v);
    let mut checkpoint_actions = Vec::new();
    if let Some((v, groups)) = checkpoint {
        let group = &groups[0];
        for o in group {
            checkpoint_actions.extend(read_actions(storage, &o.url, request).await?)
        }
        actions::validate(&checkpoint_actions, true, false, *v)?;
        let uuid = matches!(
            name(group[0].url.rsplit('/').next().unwrap()),
            Some(Name::Single(_, true))
        );
        let cp_count = checkpoint_actions
            .iter()
            .filter(|a| a.kind == "checkpointMetadata")
            .count();
        if cp_count == 1
            && !checkpoint_actions
                .iter()
                .find(|a| a.kind == "protocol")
                .and_then(|a| a.value.get("readerFeatures"))
                .and_then(Value::as_array)
                .is_some_and(|f| f.iter().any(|f| f == "v2Checkpoint"))
        {
            return Err(invalid("V2 checkpoint missing v2Checkpoint feature"));
        }
        if uuid && cp_count != 1 {
            return Err(invalid("UUID checkpoint missing checkpointMetadata"));
        }
        if group.len() > 1 && cp_count > 0 {
            return Err(invalid("multipart checkpoint cannot use V2 spec"));
        }
        let sidecars = checkpoint_actions
            .iter()
            .filter(|a| a.kind == "sidecar")
            .map(|a| actions::string(&a.value, "path").map(str::to_owned))
            .collect::<Result<Vec<_>>>()?;
        let inherited_metadata = checkpoint_actions
            .iter()
            .find(|a| a.kind == "metaData")
            .map(|a| a.value.clone());
        let sidecar_root = absolute(root, "_delta_log/_sidecars/")?;
        let decoded = futures::stream::iter(sidecars.into_iter().map(|s| {
            let storage = storage.clone();
            let root = sidecar_root.clone();
            let inherited_metadata = inherited_metadata.clone();
            async move {
                let path = absolute(&root, &s)?;
                let actions =
                    checkpoint::read(&storage, &path, request, inherited_metadata.as_ref()).await?;
                actions::validate(&actions, false, true, *v)?;
                Ok::<_, polars_io_ext_ffi::common::FfiError>(actions)
            }
        }))
        .buffered(threads)
        .try_collect::<Vec<_>>()
        .await?;
        for a in decoded {
            checkpoint_actions.extend(a)
        }
        // Detect duplicates across multipart groups and sidecars as well as within them.
        let mut keys = PlHashSet::default();
        for a in &checkpoint_actions {
            if matches!(a.kind.as_str(), "add" | "remove")
                && !keys.insert(actions::logical_key(&a.value)?)
            {
                return Err(invalid("duplicate checkpoint logical file"));
            }
        }
    }
    let count = target
        .checked_sub(base)
        .ok_or_else(|| invalid("version overflow"))?;
    if count < 0 {
        return Err(input("unknown version"));
    }
    // Bound iteration by the inventory, rather than an arbitrarily huge user version.
    let commits = if base == target {
        Vec::new()
    } else {
        inv.commits
            .range((base + 1)..=target)
            .map(|(v, o)| (*v, o.clone()))
            .collect::<Vec<_>>()
    };
    if commits.len() as i64 != count {
        return Err(invalid(format!(
            "missing Delta commit in range {}..={target}",
            base + 1
        )));
    }
    let mut commits = futures::stream::iter(commits.into_iter().rev().map(|(v, o)| {
        let storage = storage.clone();
        async move {
            let a = read_actions(&storage, &o.url, request).await?;
            actions::validate(&a, false, false, v)?;
            Ok::<_, polars_io_ext_ffi::common::FfiError>(a)
        }
    }))
    .buffered(threads);
    let mut seen = PlHashSet::default();
    let mut seen_add_paths = PlHashSet::default();
    let mut files = Vec::new();
    let (mut metadata, mut protocol) = (None, None);
    let mut apply = |actions: Vec<Action>| -> Result<()> {
        for a in actions {
            match a.kind.as_str() {
                "add" => {
                    let key = actions::logical_key(&a.value)?;
                    // Adds reconcile by path; tombstones reconcile by exact logical key.
                    // Mark every newest add even when a later remove already suppressed it.
                    let newest_add = seen_add_paths.insert(key.0.clone());
                    if seen.insert(key) && newest_add {
                        files.push(a.value);
                    }
                },
                "remove" => {
                    seen.insert(actions::logical_key(&a.value)?);
                },
                "metaData" if metadata.is_none() => metadata = Some(a.value),
                "protocol" if protocol.is_none() => protocol = Some(a.value),
                _ => {},
            }
        }
        Ok(())
    };
    while let Some(actions) = commits.next().await {
        apply(actions?)?;
    }
    apply(checkpoint_actions)?;
    // A deterministic order independent of concurrent listing and decoding.
    files.sort_by(|a, b| a["path"].as_str().cmp(&b["path"].as_str()));
    Ok(Snapshot {
        version: target,
        metadata: metadata.ok_or_else(|| invalid("snapshot missing metadata"))?,
        protocol: protocol.ok_or_else(|| invalid("snapshot missing protocol"))?,
        files,
    })
}
pub async fn load(storage: &Storage, request: &Request) -> Result<Snapshot> {
    let log_root = absolute(&request.table_uri, "_delta_log/")?;
    // The mutable hint only narrows listing; validate it against a complete
    // checkpoint and fall back to a full inventory for stale/corrupt hints.
    // Timestamp and historical requests need earlier history, so use full listing.
    let hint = if request.version.is_none() && request.timestamp_ms.is_none() {
        match storage
            .get(&absolute(
                &request.table_uri,
                "_delta_log/_last_checkpoint",
            )?)
            .await
        {
            Ok(bytes) => serde_json::from_slice::<Value>(bytes.as_slice())
                .ok()
                .and_then(|v| v.get("version").and_then(Value::as_i64))
                .filter(|v| *v >= 0),
            Err(_) => None,
        }
    } else {
        None
    };
    let inv = if let Some(hint) = hint {
        let start = format!("{log_root}{hint:020}");
        let inv = Inventory::new(storage.list_after(&log_root, Some(&start)).await?)?;
        if inv.checkpoints.contains_key(&hint) {
            inv
        } else {
            Inventory::new(storage.list(&log_root).await?)?
        }
    } else {
        Inventory::new(storage.list(&log_root).await?)?
    };
    let latest = inv
        .latest()
        .ok_or_else(|| input("not a Delta table: no commits or complete checkpoints"))?;
    let threads = request.max_threads.unwrap_or(8).clamp(1, 16) as usize;
    if let Some(target) = request.version {
        if target > latest {
            return Err(input(format!(
                "Delta version {target} exceeds latest version {latest}"
            )));
        }
        return snapshot(storage, &inv, &request.table_uri, target, threads, request).await;
    }
    let current = snapshot(storage, &inv, &request.table_uri, latest, threads, request).await?;
    let Some(timestamp) = request.timestamp_ms else {
        return Ok(current);
    };
    let config = current
        .metadata
        .get("configuration")
        .and_then(Value::as_object);
    let ict_enabled = config
        .and_then(|c| c.get("delta.enableInCommitTimestamps"))
        .and_then(Value::as_str)
        == Some("true");
    let enablement = config
        .and_then(|c| c.get("delta.inCommitTimestampEnablementVersion"))
        .and_then(Value::as_str)
        .map(str::parse::<i64>)
        .transpose()
        .map_err(invalid)?
        .unwrap_or(0);
    let mut previous = None;
    let mut chosen = None;
    for (version, object) in &inv.commits {
        let time = if ict_enabled && *version >= enablement {
            let actions = read_actions(storage, &object.url, request).await?;
            actions
                .iter()
                .find(|a| a.kind == "commitInfo")
                .and_then(|a| a.value.get("inCommitTimestamp"))
                .and_then(Value::as_i64)
                .ok_or_else(|| invalid("missing inCommitTimestamp"))?
        } else {
            let mtime = object
                .modified_ms
                .ok_or_else(|| invalid("timestamp travel needs commit modification time"))?;
            previous.map_or(mtime, |p: i64| mtime.max(p.saturating_add(1)))
        };
        previous = Some(time);
        if time <= timestamp {
            chosen = Some(*version)
        } else {
            break;
        }
    }
    let target = chosen.ok_or_else(|| input("timestamp precedes available Delta history"))?;
    if target == latest {
        Ok(current)
    } else {
        snapshot(storage, &inv, &request.table_uri, target, threads, request).await
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn object(name: &str) -> Object {
        Object {
            url: format!("/t/_delta_log/{name}"),
            modified_ms: Some(0),
        }
    }
    #[test]
    fn multipart_requires_all_parts() {
        let inv = Inventory::new(vec![object(
            "00000000000000000003.checkpoint.0000000001.0000000002.parquet",
        )])
        .unwrap();
        assert!(inv.checkpoints.is_empty());
    }
    #[test]
    fn checkpoint_names() {
        assert_eq!(
            name("00000000000000000003.checkpoint.parquet"),
            Some(Name::Single(3, false))
        );
        assert_eq!(
            name("00000000000000000003.checkpoint.12345678-1234-1234-1234-123456789abc.json"),
            Some(Name::Single(3, true))
        );
        assert!(name("00000000000000000003.crc").is_none())
    }
    #[test]
    fn malformed_path_escapes() {
        for path in ["a%", "a%2", "a%zz", "s3://bucket/a%G0"] {
            assert!(absolute("/table", path).is_err());
            assert!(absolute("s3://bucket/table", path).is_err());
        }
        assert!(absolute("/table", "a%FF").is_err());
    }
    #[test]
    fn action_paths_decode_once_for_polars_storage() {
        for (root, expected_root) in [
            ("/table%20literal", "/table%20literal"),
            ("file:///table%2520literal", "/table%20literal"),
            ("s3://bucket/table%20literal", "s3://bucket/table%20literal"),
        ] {
            for (encoded, expected) in [
                ("p=a%20b/a.parquet", "p=a b/a.parquet"),
                ("p=100%2525/a.parquet", "p=100%25/a.parquet"),
                (
                    "p=%25C3%25A4%252F%25C3%25B6/a.parquet",
                    "p=%C3%A4%2F%C3%B6/a.parquet",
                ),
                ("p=%C3%A4%2F%C3%B6/a.parquet", "p=ä/ö/a.parquet"),
                ("p=a%2520b/a.parquet", "p=a%20b/a.parquet"),
            ] {
                assert_eq!(
                    absolute(root, encoded).unwrap(),
                    format!("{expected_root}/{expected}")
                );
            }
            let sidecar_root = absolute(root, "_delta_log/_sidecars/").unwrap();
            assert_eq!(
                absolute(&sidecar_root, "a%2520.parquet").unwrap(),
                format!("{expected_root}/_delta_log/_sidecars/a%20.parquet")
            );
        }
        assert_eq!(
            absolute("/ignored", "file:///t/p=100%2525/a.parquet").unwrap(),
            "/t/p=100%25/a.parquet"
        );
        assert_eq!(
            absolute("/ignored", "s3://bucket/t/p=100%2525/a.parquet").unwrap(),
            "s3://bucket/t/p=100%25/a.parquet"
        );
    }
    #[test]
    fn escaped_local_paths() {
        assert_eq!(
            absolute("/t", "a%20b%2520.parquet").unwrap(),
            "/t/a b%20.parquet"
        );
        assert_eq!(
            absolute("s3://bucket/table", "a%20b.parquet").unwrap(),
            "s3://bucket/table/a b.parquet"
        );
        assert_eq!(
            absolute("/table", "/some/unqualified/with%20space/p%40%23h").unwrap(),
            "/some/unqualified/with space/p@#h"
        );
        assert_eq!(
            absolute("file:///table", "/some/unqualified/with%20space/p%40%23h").unwrap(),
            "/some/unqualified/with space/p@#h"
        );
        assert_eq!(
            absolute("s3://bucket/table", "/absolute/file.parquet").unwrap(),
            "s3://bucket/absolute/file.parquet"
        );
        assert_eq!(
            absolute("/table", "s3://other/absolute/file.parquet").unwrap(),
            "s3://other/absolute/file.parquet"
        )
    }
}
