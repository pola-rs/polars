use std::collections::BTreeMap;

use futures::{StreamExt, TryStreamExt};
use polars_io_ext_ffi::common::FfiError;
use polars_io_ext_ffi::delta_v1::Request;
use serde_json::Value;

use super::actions::{self, Commit, TimestampPartitions};
use super::checkpoint::{self, Checkpoint};
use super::error::{Result, input, invalid};
use super::files::{self, Rows, Source};
use super::host::{Object, Storage};
/// One version of the table. File actions are left where they were read: `files` lists the
/// surviving adds as `(source, row)`, sorted by path. A snapshot read without files has none.
pub struct Snapshot {
    pub version: i64,
    pub metadata: Value,
    pub protocol: Value,
    pub sources: Vec<Source>,
    pub files: Vec<(u32, u32)>,
}
impl Snapshot {
    pub fn file_at_path(&self, path: &str) -> Option<(&Source, usize)> {
        let index = self
            .files
            .binary_search_by(|&(source, row)| {
                self.sources[source as usize].path(row as usize).cmp(path)
            })
            .ok()?;
        let (source, row) = self.files[index];
        Some((&self.sources[source as usize], row as usize))
    }
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
/// One file of a checkpoint (Parquet or JSON).
async fn read_checkpoint_file(
    storage: &Storage,
    url: &str,
    request: &Request,
    inherited_metadata: Option<&Value>,
    files: bool,
) -> Result<Checkpoint> {
    if url.ends_with(".parquet") {
        return checkpoint::read(storage, url, request, inherited_metadata, files).await;
    }
    let commit = read_json(storage, url, files).await?;
    if files {
        commit.check_file_keys()?;
    }
    Ok(Checkpoint {
        metadata: commit.metadata,
        protocol: commit.protocol,
        checkpoint_metadata: commit.checkpoint_metadata,
        sidecars: commit.sidecars,
        has_removes: !commit.removes.is_empty(),
        // The removes are tombstones: see `checkpoint::read`.
        sources: if files {
            vec![Source::Rows(Rows {
                adds: commit.adds,
                removes: Vec::new(),
                timestamp_partitions: TimestampPartitions::Checkpoint,
            })]
        } else {
            Vec::new()
        },
    })
}
fn merge(into: &mut Checkpoint, from: Checkpoint) {
    into.metadata.extend(from.metadata);
    into.protocol.extend(from.protocol);
    into.checkpoint_metadata.extend(from.checkpoint_metadata);
    into.sidecars.extend(from.sidecars);
    into.sources.extend(from.sources);
    into.has_removes |= from.has_removes;
}
/// The checks of a checkpoint file set (the parts of one checkpoint, without sidecars).
fn validate_checkpoint(cp: &Checkpoint, files: bool, version: i64) -> Result<()> {
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
    // A V2 checkpoint holds all its file actions inline or all in sidecars (PROTOCOL.md,
    // "V2 Spec"): a mix is invalid, so only a manifest without them may list sidecars.
    if !cp.sidecars.is_empty()
        && (cp.checkpoint_metadata.len() != 1
            || (files && (cp.has_removes || cp.sources.iter().any(files::has_adds))))
    {
        return Err(invalid(
            "sidecar checkpoint has embedded file actions or no checkpointMetadata",
        ));
    }
    Ok(())
}
async fn snapshot(
    storage: &Storage,
    inv: &Inventory,
    root: &str,
    target: i64,
    threads: usize,
    request: &Request,
    read_files: bool,
) -> Result<Snapshot> {
    let checkpoint = inv.checkpoints.range(..=target).next_back();
    let base = checkpoint.map_or(-1, |(v, _)| *v);
    let mut cp = Checkpoint::default();
    if let Some((v, groups)) = checkpoint {
        let group = &groups[0];
        if group.len() > 1 && read_files {
            // The table metadata is in one part only and tells which statistics columns every
            // part's adds need: read the small actions of all parts first, then the files.
            let heads = futures::future::try_join_all(
                group
                    .iter()
                    .map(|o| read_checkpoint_file(storage, &o.url, request, None, false)),
            )
            .await?;
            let inherited = heads.iter().find_map(|h| h.metadata.first()).cloned();
            let inherited = inherited.as_ref();
            let full = futures::future::try_join_all(
                group
                    .iter()
                    .map(|o| read_checkpoint_file(storage, &o.url, request, inherited, true)),
            )
            .await?;
            for (head, full) in heads.into_iter().zip(full) {
                merge(
                    &mut cp,
                    Checkpoint {
                        sources: full.sources,
                        has_removes: full.has_removes,
                        ..head
                    },
                );
            }
        } else {
            for o in group {
                merge(
                    &mut cp,
                    read_checkpoint_file(storage, &o.url, request, None, read_files).await?,
                )
            }
        }
        validate_checkpoint(&cp, read_files, *v)?;
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
        if read_files {
            let sidecars = cp
                .sidecars
                .iter()
                .map(|s| actions::string(s, "path").map(str::to_owned))
                .collect::<Result<Vec<_>>>()?;
            let inherited_metadata = cp.metadata[0].clone();
            let sidecar_root = absolute(root, "_delta_log/_sidecars/")?;
            let decoded = futures::stream::iter(sidecars.into_iter().map(|s| {
                let storage = storage.clone();
                let root = sidecar_root.clone();
                let inherited_metadata = &inherited_metadata;
                async move {
                    let path = absolute(&root, &s)?;
                    let sidecar =
                        checkpoint::read(&storage, &path, request, Some(inherited_metadata), true)
                            .await?;
                    if !sidecar.metadata.is_empty()
                        || !sidecar.protocol.is_empty()
                        || !sidecar.checkpoint_metadata.is_empty()
                        || !sidecar.sidecars.is_empty()
                    {
                        return Err(invalid("non-file action in sidecar"));
                    }
                    Ok::<_, FfiError>(sidecar.sources)
                }
            }))
            .buffered(threads)
            .try_collect::<Vec<_>>()
            .await?;
            cp.sources.extend(decoded.into_iter().flatten());
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
    let mut commits = futures::stream::iter(commits.into_iter().rev().map(|(version, o)| {
        let storage = storage.clone();
        async move {
            let mut commit = read_commit(&storage, &o.url, read_files).await?;
            let timestamp_partitions = if read_files {
                commit.timestamp_partitions(version)
            } else {
                TimestampPartitions::Unknown
            };
            Ok::<_, FfiError>((commit, timestamp_partitions))
        }
    }))
    .buffered(threads);
    let (mut metadata, mut protocol) = (None, None);
    let mut sources = Vec::new();
    while let Some(commit) = commits.next().await {
        let (commit, timestamp_partitions) = commit?;
        // The newest action wins.
        if metadata.is_none() {
            metadata = commit.metadata.into_iter().next();
        }
        if protocol.is_none() {
            protocol = commit.protocol.into_iter().next();
        }
        if read_files {
            sources.push(Source::Rows(Rows {
                adds: commit.adds,
                removes: commit.removes,
                timestamp_partitions,
            }));
        } else if metadata.is_some() && protocol.is_some() {
            // Without file actions the newest metadata and protocol are all that is needed.
            break;
        }
    }
    drop(commits);
    let metadata = metadata
        .or_else(|| cp.metadata.first().cloned())
        .ok_or_else(|| invalid("snapshot missing metadata"))?;
    let protocol = protocol
        .or_else(|| cp.protocol.first().cloned())
        .ok_or_else(|| invalid("snapshot missing protocol"))?;
    // Checkpoint files come after every commit: older than all of them.
    sources.extend(cp.sources);
    // A deterministic order independent of concurrent listing and decoding.
    let files = files::reconcile(&sources)?;
    Ok(Snapshot {
        version: target,
        metadata,
        protocol,
        sources,
        files,
    })
}
/// The version a timestamp selects: the newest commit whose timestamp is at or before it.
///
/// A commit's timestamp is its `inCommitTimestamp` from the version the table enabled them
/// (`ict_from`), and its file modification time before that, raised to one millisecond past the
/// previous commit's time so that times strictly increase. In-commit timestamps increase by
/// the protocol, so the first version past the target is found by binary search; modification
/// times come with the listing, so they cost no reads.
pub(super) async fn version_at(
    storage: &Storage,
    commits: &BTreeMap<i64, Object>,
    timestamp: i64,
    ict_from: Option<i64>,
) -> Result<i64> {
    let mut chosen = None;
    let mut previous: Option<i64> = None;
    let mut in_commit = Vec::new();
    for (version, object) in commits {
        if ict_from.is_some_and(|from| *version >= from) {
            in_commit.push((*version, object));
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
/// Exceptional replay for checkpoint partitions whose writer zone was discarded. The ordinary
/// snapshot validator requires the full commit range 0..=target once checkpoints are excluded.
pub async fn replay_commits(storage: &Storage, request: &Request, target: i64) -> Result<Snapshot> {
    let root = &request.table_uri;
    let log_root = absolute(root, "_delta_log/")?;
    let mut inv = Inventory::new(storage.list(&log_root).await?)?;
    inv.checkpoints.clear();
    let threads = request.max_threads.unwrap_or(8).clamp(1, 16) as usize;
    snapshot(storage, &inv, root, target, threads, request, true).await
}
pub async fn load(storage: &Storage, request: &Request) -> Result<Snapshot> {
    let read_files = !request.schema_only;
    let log_root = absolute(&request.table_uri, "_delta_log/")?;
    // The mutable hint only narrows listing; validate it against a complete
    // checkpoint and fall back to a full inventory for stale/corrupt hints.
    // A timestamp request needs the whole history, and a version before the hint's earlier
    // history.
    let hint = if request.timestamp_ms.is_none() {
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
                .filter(|v| *v >= 0 && request.version.is_none_or(|target| *v <= target)),
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
    let root = &request.table_uri;
    if let Some(target) = request.version {
        if target > latest {
            return Err(input(format!(
                "Delta version {target} exceeds latest version {latest}"
            )));
        }
        return snapshot(storage, &inv, root, target, threads, request, read_files).await;
    }
    let Some(timestamp) = request.timestamp_ms else {
        return snapshot(storage, &inv, root, latest, threads, request, read_files).await;
    };
    // The newest metadata says whether and from when commits carry timestamps.
    let current = snapshot(storage, &inv, root, latest, threads, request, false).await?;
    let config = current
        .metadata
        .get("configuration")
        .and_then(Value::as_object);
    let ict_from = if config
        .and_then(|c| c.get("delta.enableInCommitTimestamps"))
        .and_then(Value::as_str)
        == Some("true")
    {
        Some(
            config
                .and_then(|c| c.get("delta.inCommitTimestampEnablementVersion"))
                .and_then(Value::as_str)
                .map(str::parse::<i64>)
                .transpose()
                .map_err(invalid)?
                .unwrap_or(0),
        )
    } else {
        None
    };
    let target = version_at(storage, &inv.commits, timestamp, ict_from).await?;
    if target == latest && !read_files {
        return Ok(current);
    }
    snapshot(storage, &inv, root, target, threads, request, read_files).await
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
