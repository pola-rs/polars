use std::collections::BTreeMap;

use futures::StreamExt;
use polars_io_ext_ffi::common::FfiError;
use polars_io_ext_ffi::delta_v1::Request;
use serde_json::Value;

use super::actions::{self, Commit};
use super::checkpoint::{self, Checkpoint};
use super::error::{Result, input, invalid};
use super::files::{self, Rows, Source};
use super::host::{Object, Storage};
use super::snapshot::Snapshot;
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
        // A signed HTTP URL authenticates by its query: pass it on whole, still escaped.
        if matches!(url.scheme(), "http" | "https") && url.query().is_some() {
            return Ok(url.to_string());
        }
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
    // A rooted path without a drive letter is not absolute on Windows, but it is
    // never relative to the table.
    if path.is_absolute() || decoded.starts_with('/') {
        return Ok(decoded.into_owned());
    }
    // Join with `/` on every platform (Windows accepts it too), so the result is
    // the same string the log listing prefixes use.
    let separated = root.ends_with('/') || (cfg!(windows) && root.ends_with('\\'));
    Ok(format!(
        "{root}{}{decoded}",
        if separated || root.is_empty() {
            ""
        } else {
            "/"
        }
    ))
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
/// Files above this size are parsed on a blocking thread, so that the IO task keeps going.
const PARSE_OFFLOAD_BYTES: usize = 32 * 1024;
/// A JSON log file. With `files` false its add and remove lines are not read.
async fn read_json(storage: &Storage, url: &str, files: bool) -> Result<Commit> {
    let bytes = storage.get(url).await?;
    if bytes.as_slice().len() >= PARSE_OFFLOAD_BYTES {
        tokio::task::spawn_blocking(move || actions::parse_json(bytes.as_slice(), !files))
            .await
            .map_err(invalid)?
    } else {
        actions::parse_json(bytes.as_slice(), !files)
    }
}
async fn read_commit(storage: &Storage, url: &str, files: bool) -> Result<Commit> {
    let commit = read_json(storage, url, files).await?;
    commit.validate_commit(files)?;
    Ok(commit)
}
async fn read_checkpoint_file(storage: &Storage, url: &str) -> Result<Checkpoint> {
    if url.ends_with(".parquet") {
        return checkpoint::read_actions(storage, url).await;
    }
    let commit = read_json(storage, url, false).await?;
    Ok(Checkpoint {
        metadata: commit.metadata,
        protocol: commit.protocol,
        checkpoint_metadata: commit.checkpoint_metadata,
        sidecars: commit.sidecars,
    })
}
fn merge(into: &mut Checkpoint, from: Checkpoint) {
    into.metadata.extend(from.metadata);
    into.protocol.extend(from.protocol);
    into.checkpoint_metadata.extend(from.checkpoint_metadata);
    into.sidecars.extend(from.sidecars);
}
/// The checks of a checkpoint file set (the parts of one checkpoint, without sidecars).
fn validate_checkpoint(cp: &Checkpoint, version: i64) -> Result<()> {
    if cp.metadata.len() > 1 || cp.protocol.len() > 1 || cp.checkpoint_metadata.len() > 1 {
        return Err(invalid("duplicate metadata/protocol/checkpointMetadata"));
    }
    if cp.metadata.len() != 1 || cp.protocol.len() != 1 {
        return Err(invalid(
            "checkpoint must contain exactly one metadata and protocol",
        ));
    }
    if let Some(m) = cp.checkpoint_metadata.first() {
        if m.get("version").and_then(Value::as_i64) != Some(version) {
            return Err(invalid("checkpointMetadata version mismatch"));
        }
    }
    for s in &cp.sidecars {
        actions::string(s, "path")?;
        actions::u64_value(s, "sizeInBytes")?;
    }
    if !cp.sidecars.is_empty() && cp.checkpoint_metadata.is_empty() {
        return Err(invalid("sidecar checkpoint has no checkpointMetadata"));
    }
    Ok(())
}
/// Read the latest raw metadata when a newer schema prevents Kernel from selecting an
/// older timestamp. File actions and the schema string do not need to be decoded here.
async fn latest_raw_metadata(
    storage: &Storage,
    inv: &Inventory,
    target: i64,
    threads: usize,
) -> Result<Value> {
    let checkpoint = inv.checkpoints.range(..=target).next_back();
    let base = checkpoint.map_or(-1, |(v, _)| *v);
    let mut cp = Checkpoint::default();
    if let Some((v, groups)) = checkpoint {
        let group = &groups[0];
        for o in group {
            merge(&mut cp, read_checkpoint_file(storage, &o.url).await?);
        }
        validate_checkpoint(&cp, *v)?;
        let uuid = matches!(
            name(group[0].url.rsplit('/').next().unwrap()),
            Some(Name::Single(_, true))
        );
        let cp_count = cp.checkpoint_metadata.len();
        if cp_count == 1
            && !cp.protocol[0]
                .get("readerFeatures")
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
    }
    let commits = commit_range(&inv.commits, base, target)?;
    let mut commits = futures::stream::iter(
        commits
            .into_iter()
            .rev()
            .map(|(_, o)| read_commit(storage, &o.url, false)),
    )
    .buffered(threads);
    let (mut metadata, mut protocol) = (None, None);
    while let Some(commit) = commits.next().await {
        let commit = commit?;
        if metadata.is_none() {
            metadata = commit.metadata.into_iter().next();
        }
        if protocol.is_none() {
            protocol = commit.protocol.into_iter().next();
        }
        if metadata.is_some() && protocol.is_some() {
            break;
        }
    }
    let metadata = metadata
        .or_else(|| cp.metadata.into_iter().next())
        .ok_or_else(|| invalid("snapshot missing metadata"))?;
    protocol
        .or_else(|| cp.protocol.into_iter().next())
        .ok_or_else(|| invalid("snapshot missing protocol"))?;
    Ok(metadata)
}
fn commit_range(
    commits: &BTreeMap<i64, Object>,
    base: i64,
    target: i64,
) -> Result<Vec<(i64, &Object)>> {
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
        commits
            .range((base + 1)..=target)
            .map(|(v, o)| (*v, o))
            .collect::<Vec<_>>()
    };
    if commits.len() as i64 != count {
        return Err(invalid(format!(
            "missing Delta commit in range {}..={target}",
            base + 1
        )));
    }
    Ok(commits)
}
/// The version a timestamp selects: the newest commit whose timestamp is at or before it.
///
/// A commit's timestamp is its `inCommitTimestamp` from the version the table enabled them
/// (`ict_from`), and its file modification time before that, raised to one millisecond past the
/// previous commit's time so that times strictly increase. In-commit timestamps increase by
/// the protocol, so the first version past the target is found by binary search; modification
/// times come with the listing, so they cost no reads.
///
/// `ict_from` is `(enablement version, enablement timestamp)`. A timestamp at or after the
/// enablement timestamp lies in the in-commit era, however the older commits' modification
/// times were set, so those are not consulted.
pub(super) async fn version_at(
    storage: &Storage,
    commits: &BTreeMap<i64, Object>,
    timestamp: i64,
    ict_from: Option<(i64, Option<i64>)>,
) -> Result<i64> {
    let in_commit_era = ict_from.is_some_and(|(_, at)| at.is_some_and(|at| timestamp >= at));
    let ict_from = ict_from.map(|(version, _)| version);
    let mut chosen = None;
    let mut previous: Option<i64> = None;
    let mut in_commit = Vec::new();
    for (version, object) in commits {
        if ict_from.is_some_and(|from| *version >= from) {
            in_commit.push((*version, object));
            continue;
        }
        if in_commit_era {
            continue;
        }
        let mtime = object
            .modified_ms
            .ok_or_else(|| invalid("timestamp travel needs commit modification time"))?;
        let time = previous.map_or(mtime, |p| mtime.max(p.saturating_add(1)));
        previous = Some(time);
        if time > timestamp {
            return chosen.ok_or_else(|| input("timestamp precedes available Delta history"));
        }
        chosen = Some(*version);
    }
    // The first commit past the target, among those with in-commit timestamps.
    let (mut lo, mut hi) = (0, in_commit.len());
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let bytes = storage.get(&in_commit[mid].1.url).await?;
        let time = actions::in_commit_timestamp(bytes.as_slice())?
            .ok_or_else(|| invalid("missing inCommitTimestamp"))?;
        if time <= timestamp {
            lo = mid + 1
        } else {
            hi = mid
        }
    }
    if lo > 0 {
        chosen = Some(in_commit[lo - 1].0);
    }
    chosen.ok_or_else(|| input("timestamp precedes available Delta history"))
}

async fn version_at_with_metadata(
    storage: &Storage,
    commits: &BTreeMap<i64, Object>,
    timestamp: i64,
    metadata: &Value,
) -> Result<i64> {
    let config = metadata.get("configuration").and_then(Value::as_object);
    let ict_from = if config
        .and_then(|c| c.get("delta.enableInCommitTimestamps"))
        .and_then(Value::as_str)
        == Some("true")
    {
        let property = |name: &str| {
            config
                .and_then(|c| c.get(name))
                .and_then(Value::as_str)
                .map(str::parse::<i64>)
                .transpose()
                .map_err(invalid)
        };
        Some((
            property("delta.inCommitTimestampEnablementVersion")?.unwrap_or(0),
            property("delta.inCommitTimestampEnablementTimestamp")?,
        ))
    } else {
        None
    };
    version_at(storage, commits, timestamp, ict_from).await
}

pub(super) async fn time_travel_version(
    storage: &Storage,
    request: &Request,
    timestamp: i64,
    latest: Option<(i64, &Value)>,
) -> Result<i64> {
    let root = &request.table_uri;
    let log_root = absolute(root, "_delta_log/")?;
    let mut inventory = Inventory::new(storage.list(&log_root).await?)?;
    if let Some((version, metadata)) = latest {
        // Ignore commits published after the latest snapshot supplied its time-travel properties.
        inventory.commits.retain(|v, _| *v <= version);
        return version_at_with_metadata(storage, &inventory.commits, timestamp, metadata).await;
    }
    let latest = inventory
        .latest()
        .ok_or_else(|| input("not a Delta table: no commits or complete checkpoints"))?;
    let threads = request.max_threads.unwrap_or(8).clamp(1, 16) as usize;
    let metadata = latest_raw_metadata(storage, &inventory, latest, threads).await?;
    version_at_with_metadata(storage, &inventory.commits, timestamp, &metadata).await
}

/// Recover writer provenance for a checkpoint partition from the complete commit history.
pub async fn replay_commits(storage: &Storage, request: &Request, target: i64) -> Result<Snapshot> {
    let log_root = absolute(&request.table_uri, "_delta_log/")?;
    let inv = Inventory::new(storage.list(&log_root).await?)?;
    let threads = request.max_threads.unwrap_or(8).clamp(1, 16) as usize;
    let commits = commit_range(&inv.commits, -1, target)?;
    let mut commits =
        futures::stream::iter(commits.into_iter().rev().map(|(version, o)| async move {
            let mut commit = read_commit(storage, &o.url, true).await?;
            let timestamp_partitions = commit.timestamp_partitions(version);
            Ok::<_, FfiError>((commit, timestamp_partitions))
        }))
        .buffered(threads);
    let (mut metadata, mut protocol) = (None, None);
    let mut rows = Vec::new();
    while let Some(commit) = commits.next().await {
        let (commit, timestamp_partitions) = commit?;
        if metadata.is_none() {
            metadata = commit.metadata.into_iter().next();
        }
        if protocol.is_none() {
            protocol = commit.protocol.into_iter().next();
        }
        rows.push(Rows {
            adds: commit.adds,
            removes: commit.removes,
            timestamp_partitions,
        });
    }
    let metadata = metadata.ok_or_else(|| invalid("snapshot missing metadata"))?;
    let protocol = protocol.ok_or_else(|| invalid("snapshot missing protocol"))?;
    let files = files::reconcile_commits(&rows);
    Ok(Snapshot {
        version: target,
        metadata,
        protocol,
        sources: rows.into_iter().map(Source::Rows).collect(),
        files,
    })
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
            #[cfg(unix)]
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
        #[cfg(unix)]
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
    fn signed_http_urls_keep_their_query() {
        let signed = "https://host.example/t/a%20b.parquet?sig=a%2Bb&exp=1";
        assert_eq!(absolute("s3://bucket/table", signed).unwrap(), signed);
        // Without a query the object key is raw, like other absolute URIs.
        assert_eq!(
            absolute("s3://bucket/table", "https://host.example/t/a%20b.parquet").unwrap(),
            "https://host.example/t/a b.parquet"
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
        // `file:///table` has no drive letter, so it is not a Windows path.
        #[cfg(unix)]
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
