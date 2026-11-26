//! In-process host exercises the actual C ABI with deterministic object mtimes.
use std::collections::BTreeMap;
use std::ffi::c_void;

use polars_arrow::array::{Array, StructArray, Utf8ViewArray};
use polars_arrow::ffi;
use polars_io_ext_ffi::common::{
    FfiBuf, FfiError, FfiErrorKind, FfiFuture, FfiOption, FfiResult, FfiStr, FfiVec, FutureExt,
};
use polars_io_ext_ffi::delta_v1::{self, ObjectMeta, OutputHeader, Request, StorageHandle};
use serde_json::{Value, json};

#[path = "test_fixtures.rs"]
mod test_fixtures;
#[path = "test_timestamp_fixtures.rs"]
mod test_timestamp_fixtures;
#[derive(Default)]
struct Store(
    BTreeMap<String, (Vec<u8>, Option<i64>)>,
    std::sync::atomic::AtomicUsize,
    std::sync::Mutex<Vec<String>>,
    std::sync::atomic::AtomicBool,
);
impl Store {
    fn put(&mut self, path: &str, actions: &[Value]) {
        self.0.insert(
            format!("/table/_delta_log/{path}"),
            (
                actions
                    .iter()
                    .map(Value::to_string)
                    .collect::<Vec<_>>()
                    .join("\n")
                    .into_bytes(),
                Some(1000 + path[..20].parse::<i64>().unwrap_or(0) * 1000),
            ),
        );
    }
    fn host(&self) -> delta_v1::Host {
        delta_v1::Host {
            ctx: self as *const Self as *const c_void,
            get_storage,
            storage_get: get,
            storage_head: head,
            storage_list: list,
            storage_get_range: range,
            storage_release: release,
            log: log_message,
        }
    }
}
fn ready<T: Send + 'static>(v: Result<T, FfiError>) -> FfiFuture<FfiResult<T>> {
    async move { v.into() }.into_ffi()
}
unsafe fn store(s: StorageHandle) -> &'static Store {
    unsafe { &*(s.0 as *const Store) }
}
unsafe extern "C" fn get_storage(
    ctx: *const c_void,
    _: FfiStr,
) -> FfiFuture<FfiResult<StorageHandle>> {
    ready(Ok(StorageHandle(ctx)))
}
unsafe extern "C" fn get(s: StorageHandle, url: FfiStr) -> FfiFuture<FfiResult<FfiBuf>> {
    unsafe { store(s) }
        .2
        .lock()
        .unwrap()
        .push(url.as_str().unwrap().to_owned());
    let data = unsafe { store(s) }
        .0
        .get(url.as_str().unwrap())
        .map(|v| FfiBuf::from_vec(v.0.clone()))
        .ok_or_else(|| {
            FfiError::new(
                FfiErrorKind::NOT_FOUND,
                format!("missing {}", url.as_str().unwrap()),
            )
        });
    ready(data)
}
unsafe extern "C" fn head(s: StorageHandle, url: FfiStr) -> FfiFuture<FfiResult<u64>> {
    ready(
        unsafe { store(s) }
            .0
            .get(url.as_str().unwrap())
            .map(|v| v.0.len() as u64)
            .ok_or_else(|| FfiError::new(FfiErrorKind::NOT_FOUND, "missing object")),
    )
}
unsafe extern "C" fn list(
    s: StorageHandle,
    prefix: FfiStr,
    start: FfiOption<FfiStr>,
) -> FfiFuture<FfiResult<FfiVec<ObjectMeta>>> {
    let prefix = prefix.as_str().unwrap();
    let start = start.into_option().map(|s| s.as_str().unwrap());
    ready(Ok(FfiVec::from_vec(
        unsafe { store(s) }
            .0
            .iter()
            .filter(|(k, _)| {
                k.strip_prefix(prefix).is_some_and(|s| !s.contains('/'))
                    && start.is_none_or(|s| k.as_str() > s)
            })
            .map(|(k, v)| ObjectMeta {
                url: FfiBuf::from_string(k.clone()),
                size: v.0.len() as u64,
                modified_ms: v.1.into(),
            })
            .collect(),
    )))
}
unsafe extern "C" fn range(
    s: StorageHandle,
    url: FfiStr,
    start: u64,
    end: u64,
) -> FfiFuture<FfiResult<FfiBuf>> {
    if unsafe { store(s) }
        .3
        .swap(false, std::sync::atomic::Ordering::Relaxed)
    {
        return ready(Err(FfiError::new(
            FfiErrorKind::OBJECT_CHANGED,
            "checkpoint replaced after HEAD",
        )));
    }
    unsafe { store(s) }
        .1
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    ready(
        unsafe { store(s) }
            .0
            .get(url.as_str().unwrap())
            .and_then(|v| v.0.get(start as usize..end as usize))
            .map(|v| FfiBuf::from_vec(v.to_vec()))
            .ok_or_else(|| FfiError::new(FfiErrorKind::IO, "invalid range")),
    )
}
unsafe extern "C" fn release(_: StorageHandle) {}
unsafe extern "C" fn log_message(_: *const c_void, _: FfiStr) {}
fn metadata() -> Value {
    json!({"metaData":{"id":"id","format":{"provider":"parquet","options":{}},"schemaString":json!({"type":"struct","fields":[{"name":"x","type":"long","nullable":true,"metadata":{}}]}).to_string(),"partitionColumns":[],"configuration":{}}})
}
fn protocol() -> Value {
    json!({"protocol":{"minReaderVersion":3,"minWriterVersion":7,"readerFeatures":["deletionVectors","v2Checkpoint"],"writerFeatures":["deletionVectors","v2Checkpoint"]}})
}
fn add(path: &str) -> Value {
    json!({"add":{"path":path,"size":10,"partitionValues":{},"stats":"{\"numRecords\":5,\"minValues\":{\"x\":1},\"maxValues\":{\"x\":5},\"nullCount\":{\"x\":0}}"}})
}
fn initial() -> Store {
    let mut s = Store::default();
    s.put(
        "00000000000000000000.json",
        &[protocol(), metadata(), add("a.parquet")],
    );
    s
}
fn planned(s: &Store, mut r: Request) -> Result<(OutputHeader, Vec<String>), FfiError> {
    r.table_uri = "/table".into();
    let host = s.host();
    let output = r
        .with_ffi(|r| unsafe { super::plan(&host, r) })
        .into_result()?;
    let header = OutputHeader::from_ffi(&output.header);
    let stream: ffi::ArrowArrayStream = unsafe { output.files.transmute_into() };
    let mut reader = unsafe { ffi::ArrowArrayStreamReader::try_new(Box::new(stream)) }.unwrap();
    let mut paths = Vec::new();
    while let Some(batch) = unsafe { reader.next() } {
        let batch = batch.unwrap();
        let a = batch.as_any().downcast_ref::<StructArray>().unwrap();
        let fields = match a.dtype() {
            polars_arrow::datatypes::ArrowDataType::Struct(f) => f,
            _ => unreachable!(),
        };
        let index = fields.iter().position(|f| f.name == "path").unwrap();
        let a = a.values()[index]
            .as_any()
            .downcast_ref::<Utf8ViewArray>()
            .unwrap();
        paths.extend(a.values_iter().map(str::to_owned));
    }
    Ok((header, paths))
}
#[test]
fn persisted_dv_offsets_default_zero_in_output() {
    use polars_arrow::array::PrimitiveArray;

    for offset in [None, Some(0), Some(1)] {
        let mut s = initial();
        let mut a = dv_add("a.parquet", "dv.bin");
        if let Some(offset) = offset {
            a["add"]["deletionVector"]["offset"] = json!(offset);
        }
        s.put("00000000000000000000.json", &[protocol(), metadata(), a]);
        let request = Request {
            table_uri: "/table".into(),
            ..Request::default()
        };
        let host = s.host();
        let output = request
            .with_ffi(|r| unsafe { super::plan(&host, r) })
            .into_result()
            .unwrap();
        let stream: ffi::ArrowArrayStream = unsafe { output.files.transmute_into() };
        let mut reader = unsafe { ffi::ArrowArrayStreamReader::try_new(Box::new(stream)) }.unwrap();
        let batch = unsafe { reader.next() }.unwrap().unwrap();
        let files = batch.as_any().downcast_ref::<StructArray>().unwrap();
        let field = |array: &StructArray, name: &str| {
            let polars_arrow::datatypes::ArrowDataType::Struct(fields) = array.dtype() else {
                unreachable!()
            };
            fields.iter().position(|f| f.name == name).unwrap()
        };
        let dv = files.values()[field(files, "deletion_vector")]
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();
        let offsets = dv.values()[field(dv, "offset")]
            .as_any()
            .downcast_ref::<PrimitiveArray<u64>>()
            .unwrap();
        assert_eq!(offsets.value(0), offset.unwrap_or(0));
    }
}

#[test]
fn replay_and_time_travel() {
    let mut s = initial();
    s.put(
        "00000000000000000001.json",
        &[json!({"remove":{"path":"a.parquet"}}), add("b.parquet")],
    );
    assert_eq!(
        planned(&s, Request::default()).unwrap().1,
        vec!["/table/b.parquet"]
    );
    assert_eq!(
        planned(
            &s,
            Request {
                version: Some(0),
                ..Request::default()
            }
        )
        .unwrap()
        .1,
        vec!["/table/a.parquet"]
    );
    assert_eq!(
        planned(
            &s,
            Request {
                timestamp_ms: Some(1500),
                ..Request::default()
            }
        )
        .unwrap()
        .0
        .resolved_version,
        0
    )
}

#[test]
fn timestamp_travel_can_pass_a_later_malformed_schema() {
    let mut store = initial();
    let mut malformed = metadata();
    malformed["metaData"]["schemaString"] = json!("{");
    store.put(&version_name(1), &[malformed]);
    let request = |timestamp| Request {
        timestamp_ms: Some(timestamp),
        ..Request::default()
    };
    let (header, paths) = planned(&store, request(1500)).unwrap();
    assert_eq!(header.resolved_version, 0);
    assert_eq!(paths, ["/table/a.parquet"]);
    assert!(planned(&store, request(2500)).is_err());
}
#[test]
fn request_limit_never_loses_files() {
    let mut s = initial();
    s.put("00000000000000000001.json", &[add("b.parquet")]);
    let (h, p) = planned(
        &s,
        Request {
            limit: Some(1),
            use_metadata_statistics: true,
            ..Request::default()
        },
    )
    .unwrap();
    assert_eq!(p.len(), 2);
    assert_eq!(h.row_count, Some((10, 0)))
}
#[test]
fn gap_is_error() {
    let mut s = initial();
    s.put("00000000000000000002.json", &[add("b.parquet")]);
    assert!(planned(&s, Request::default()).is_err())
}
#[test]
fn json_v2_checkpoint() {
    let mut s = Store::default();
    s.put(
        "00000000000000000005.checkpoint.12345678-1234-1234-1234-123456789abc.json",
        &[
            protocol(),
            metadata(),
            json!({"checkpointMetadata":{"version":5}}),
            add("a.parquet"),
        ],
    );
    assert_eq!(
        planned(&s, Request::default()).unwrap().0.resolved_version,
        5
    )
}
#[test]
fn duplicate_commit_is_error() {
    let mut s = initial();
    s.put(
        "00000000000000000001.json",
        &[add("b.parquet"), add("b.parquet")],
    );
    assert!(planned(&s, Request::default()).is_err())
}
#[test]
fn inline_dv_counts() {
    let mut s = Store::default();
    let mut a = add("a.parquet");
    a["add"]["stats"] = "{\"numRecords\":30}".into();
    a["add"]["deletionVector"] = json!({"storageType":"i","pathOrInlineDv":"^Bg9^0rr910000000000iXQKl0rr91000f55c8Xg0@@D72lkbi5=-{L","sizeInBytes":44,"cardinality":6});
    s.put("00000000000000000000.json", &[protocol(), metadata(), a]);
    assert_eq!(
        planned(
            &s,
            Request {
                use_metadata_statistics: true,
                fast_deletion_count: true,
                ..Request::default()
            }
        )
        .unwrap()
        .0
        .row_count,
        Some((30, 6))
    )
}
/// Optional external differential manifest: [{root,version,files:[relative path,...]}].
#[test]
fn differential_fixtures() {
    let Ok(path) = std::env::var("POLARS_DELTA_FIXTURE_MANIFEST") else {
        return;
    };
    let cases: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let mut failures = Vec::new();
    for case in cases.as_array().unwrap() {
        let root = case["root"].as_str().unwrap();
        let mut store = Store::default();
        fn visit(dir: &std::path::Path, store: &mut Store) {
            for entry in std::fs::read_dir(dir).unwrap() {
                let p = entry.unwrap().path();
                if p.is_dir() {
                    visit(&p, store)
                } else {
                    let meta = std::fs::metadata(&p).unwrap();
                    let mtime = meta
                        .modified()
                        .ok()
                        .and_then(|v| v.duration_since(std::time::UNIX_EPOCH).ok())
                        .map(|v| v.as_millis() as i64);
                    store.0.insert(
                        p.to_string_lossy().into_owned(),
                        (std::fs::read(&p).unwrap(), mtime),
                    );
                }
            }
        }
        visit(&std::path::Path::new(root).join("_delta_log"), &mut store);
        let host = store.host();
        let req = Request {
            table_uri: root.to_owned(),
            version: case["version"].as_i64(),
            use_metadata_statistics: true,
            ..Request::default()
        };
        let output = match req
            .with_ffi(|r| unsafe { super::plan(&host, r) })
            .into_result()
        {
            Ok(output) => output,
            Err(e)
                if root.ends_with("table-with-geo")
                    && e.kind() == FfiErrorKind::NOT_IMPLEMENTED =>
            {
                println!("fixture expected unsupported {root}: {e:?}");
                continue;
            },
            Err(e) => {
                failures.push(format!("{root}: {e:?}"));
                continue;
            },
        };
        let header = OutputHeader::from_ffi(&output.header);
        assert_eq!(header.resolved_version, case["version"].as_i64().unwrap());
        let stream: ffi::ArrowArrayStream = unsafe { output.files.transmute_into() };
        let mut reader = unsafe { ffi::ArrowArrayStreamReader::try_new(Box::new(stream)) }.unwrap();
        let mut paths = Vec::new();
        while let Some(batch) = unsafe { reader.next() } {
            let batch = batch.unwrap();
            let a = batch.as_any().downcast_ref::<StructArray>().unwrap();
            let p = a.values()[0]
                .as_any()
                .downcast_ref::<Utf8ViewArray>()
                .unwrap();
            paths.extend(p.values_iter().map(str::to_owned));
        }
        let mut expected = case["files"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| p.as_str().unwrap().to_owned())
            .collect::<Vec<_>>();
        paths.sort();
        expected.sort();
        // delta-rs 1.6.6 incorrectly treats scheme-less absolute add paths as
        // table-relative. Spark DeltaFileOperations.absolutePath and kernel's
        // Url::join both retain the leading slash (RFC 2396 absolute-path).
        // Assert the exact known fixture result rather than exempting mismatches.
        if [
            "canonicalized-paths-special-a",
            "canonicalized-paths-special-b",
        ]
        .iter()
        .any(|name| root.ends_with(&format!("/{name}/delta")))
        {
            assert!(matches!(case["version"].as_i64(), Some(0 | 1)));
            assert_eq!(
                expected,
                [format!("{root}/some/unqualified/with space/p@#h")]
            );
            expected = vec!["/some/unqualified/with space/p@#h".to_owned()];
        }
        if paths != expected {
            failures.push(format!(
                "{root}: paths mismatch actual={paths:?} expected={expected:?}"
            ));
        }
        println!(
            "fixture {root}: version {} files {}",
            header.resolved_version,
            paths.len()
        );
    }
    for failure in &failures {
        println!("FIXTURE FAILURE {failure}");
    }
    assert!(failures.is_empty(), "{} fixture failures", failures.len());
}
fn dv_add(path: &str, id: &str) -> Value {
    let mut a = add(path);
    a["add"]["deletionVector"] = json!({"storageType":"p","pathOrInlineDv":format!("/table/{id}"),"sizeInBytes":44,"cardinality":1});
    a
}
#[test]
fn dv_replacement_unordered() {
    for reverse in [false, true] {
        let mut s = Store::default();
        s.put(
            "00000000000000000000.json",
            &[protocol(), metadata(), dv_add("a.parquet", "old")],
        );
        let mut actions = vec![
            json!({"remove":{"path":"a.parquet","deletionVector":{"storageType":"p","pathOrInlineDv":"/table/old","sizeInBytes":44,"cardinality":1}}}),
            dv_add("a.parquet", "new"),
        ];
        if reverse {
            actions.reverse()
        }
        s.put("00000000000000000001.json", &actions);
        assert_eq!(
            planned(&s, Request::default()).unwrap().1,
            vec!["/table/a.parquet"]
        );
    }
}
#[test]
#[ignore = "Writer omitted the matching remove for the old deletion vector."]
fn newest_add_path_wins() {
    let mut s = Store::default();
    s.put(
        "00000000000000000000.json",
        &[protocol(), metadata(), dv_add("a.parquet", "old")],
    );
    s.put("00000000000000000001.json", &[dv_add("a.parquet", "new")]);
    assert_eq!(planned(&s, Request::default()).unwrap().1.len(), 1);
    s.put("00000000000000000002.json",&[json!({"remove":{"path":"a.parquet","deletionVector":{"storageType":"p","pathOrInlineDv":"/table/new"}}})]);
    assert!(planned(&s, Request::default()).unwrap().1.is_empty())
}
#[test]
fn drop_readd_without_dv() {
    let mut s = initial();
    s.put(
        "00000000000000000001.json",
        &[json!({"remove":{"path":"a.parquet"}})],
    );
    s.put("00000000000000000002.json", &[add("a.parquet")]);
    assert_eq!(
        planned(&s, Request::default()).unwrap().1,
        vec!["/table/a.parquet"]
    )
}
#[test]
fn ict_uses_enablement_boundary() {
    let mut s = initial();
    let mut m = metadata();
    m["metaData"]["configuration"] = json!({"delta.enableInCommitTimestamps":"true","delta.inCommitTimestampEnablementVersion":"1","delta.inCommitTimestampEnablementTimestamp":"5000"});
    s.put(
        "00000000000000000001.json",
        &[
            m,
            json!({"commitInfo":{"inCommitTimestamp":5000}}),
            add("b.parquet"),
        ],
    );
    assert_eq!(
        planned(
            &s,
            Request {
                timestamp_ms: Some(3000),
                ..Request::default()
            }
        )
        .unwrap()
        .0
        .resolved_version,
        0
    );
    assert_eq!(
        planned(
            &s,
            Request {
                timestamp_ms: Some(5000),
                ..Request::default()
            }
        )
        .unwrap()
        .0
        .resolved_version,
        1
    )
}

#[test]
fn checkpoint_rejects_metadata_and_add_in_one_row_across_passes() {
    let mut store = Store::default();
    store.0.insert(
        "/table/_delta_log/00000000000000000000.checkpoint.parquet".to_owned(),
        (fixture("mixed-root.checkpoint.parquet"), Some(1000)),
    );
    // The collision is in the second row group, after independently valid roots.
    let error = planned(&store, Request::default()).unwrap_err();
    assert!(
        format!("{error:?}").contains("multiple actions in one row"),
        "{error:?}"
    );
}

#[test]
fn wide_checkpoint_projects_statistics_and_skips_binary_parsed_partitions() {
    let bytes = fixture("wide-binary.checkpoint.parquet");
    for hint in [
        None,
        Some("{invalid"),
        Some("{\"version\":0}"),
        Some("{\"version\":999}"),
    ] {
        let mut store = Store::default();
        store.0.insert(
            "/table/_delta_log/00000000000000000000.checkpoint.parquet".to_owned(),
            (bytes.clone(), Some(1000)),
        );
        if let Some(hint) = hint {
            store.0.insert(
                "/table/_delta_log/_last_checkpoint".to_owned(),
                (hint.as_bytes().to_vec(), Some(1000)),
            );
        }
        let (header, paths) = planned(
            &store,
            Request {
                use_metadata_statistics: true,
                statistics_columns: Some(vec!["c127".to_owned()]),
                ..Request::default()
            },
        )
        .unwrap();
        assert_eq!(header.row_count, Some((5, 0)));
        assert_eq!(paths, ["/table/a.parquet"]);
        let requests = store.1.load(std::sync::atomic::Ordering::Relaxed);
        assert!(
            requests <= 16,
            "one-column projection made {requests} range requests for128columns"
        );
        println!(
            "wide checkpoint:128columns requested1, hint={hint:?}, range_requests={requests}, row_count=5"
        );
    }
}

#[test]
#[ignore = "DK-UNINDEXED-CHECKPOINT-STATS: selecting an incompatible extra bound can drop every typed stat"]
fn wide_checkpoint_exports_requested_stats_outside_indexed_set() {
    let store = checkpoint_store(&[(
        "00000000000000000000.checkpoint.parquet",
        fixture("wide-binary.checkpoint.parquet"),
    )]);
    assert_eq!(
        planned_stats(&store, &["c127", "p", "missing"], "c127_min"),
        [("/table/a.parquet".to_owned(), Some(1))]
    );
}

#[test]
fn unindexed_checkpoint_bound_stays_unknown() {
    let store = checkpoint_store(&[(
        "00000000000000000000.checkpoint.parquet",
        fixture("wide-binary.checkpoint.parquet"),
    )]);
    assert_eq!(
        planned_stats(&store, &["c0", "c127"], "c0_min"),
        [("/table/a.parquet".to_owned(), Some(1))]
    );
    assert_eq!(
        planned_stats(&store, &["c0", "c127"], "c127_min"),
        [("/table/a.parquet".to_owned(), None)]
    );
}

/// Planned files' `(path, value)` for one `stats` child (`t_min`, `t_max`), as `Option<i128>`
/// (booleans as 0 or 1, doubles as their bit patterns).
fn planned_stats(s: &Store, columns: &[&str], child: &str) -> Vec<(String, Option<i128>)> {
    use polars_arrow::array::{BooleanArray, PrimitiveArray};
    let mut r = Request {
        use_metadata_statistics: true,
        statistics_columns: Some(columns.iter().map(|c| c.to_string()).collect()),
        ..Request::default()
    };
    r.table_uri = "/table".into();
    let host = s.host();
    let output = r
        .with_ffi(|r| unsafe { super::plan(&host, r) })
        .into_result()
        .unwrap();
    let stream: ffi::ArrowArrayStream = unsafe { output.files.transmute_into() };
    let mut reader = unsafe { ffi::ArrowArrayStreamReader::try_new(Box::new(stream)) }.unwrap();
    let mut out = Vec::new();
    while let Some(batch) = unsafe { reader.next() } {
        let batch = batch.unwrap();
        let a = batch.as_any().downcast_ref::<StructArray>().unwrap();
        let column = |name: &str, a: &StructArray| {
            let polars_arrow::datatypes::ArrowDataType::Struct(f) = a.dtype() else {
                unreachable!()
            };
            a.values()[f.iter().position(|f| f.name == name).unwrap()].clone()
        };
        let paths = column("path", a);
        let paths = paths.as_any().downcast_ref::<Utf8ViewArray>().unwrap();
        let stats = column("stats", a);
        let stats = stats.as_any().downcast_ref::<StructArray>().unwrap();
        let v = column(child, stats);
        for i in 0..a.len() {
            let value = if v.is_null(i) {
                None
            } else if let Some(v) = v.as_any().downcast_ref::<PrimitiveArray<i64>>() {
                Some(v.value(i) as i128)
            } else if let Some(v) = v.as_any().downcast_ref::<BooleanArray>() {
                Some(i128::from(v.value(i)))
            } else if let Some(v) = v.as_any().downcast_ref::<PrimitiveArray<f64>>() {
                Some(i128::from(v.value(i).to_bits()))
            } else {
                Some(
                    v.as_any()
                        .downcast_ref::<PrimitiveArray<i128>>()
                        .unwrap()
                        .value(i),
                )
            };
            out.push((paths.value(i).to_owned(), value));
        }
    }
    out.sort();
    out
}
fn single_column_metadata(ty: &str) -> Value {
    json!({"metaData":{"id":"id","format":{"provider":"parquet","options":{}},"schemaString":json!({"type":"struct","fields":[{"name":"t","type":ty,"nullable":true,"metadata":{}}]}).to_string(),"partitionColumns":[],"configuration":{}}})
}

fn timestamp_checkpoint(unit: &str) -> Vec<u8> {
    fixture(&format!("timestamp-{unit}.checkpoint.parquet"))
}

fn timestamp_checkpoint_store(bytes: &[u8]) -> Store {
    let mut store = Store::default();
    store.0.insert(
        "/table/_delta_log/00000000000000000000.checkpoint.parquet".to_owned(),
        (bytes.to_vec(), Some(1000)),
    );
    store
}

fn check_checkpoint_timestamp_bounds(unit: &str, bytes: &[u8], with_b: bool) {
    // 2024-01-01T00:00:00Z, 2024-01-02T00:00:00Z, 1500-01-01T00:00:00Z, 1500-01-02T00:00:00Z.
    let (a_lo, a_hi) = (1_704_067_200_000_000i128, 1_704_153_600_000_000i128);
    let (b_lo, b_hi) = (-14_831_769_600_000_000i128, -14_831_683_200_000_000i128);
    // Maximum bounds are widened by 1ms (Delta truncates timestamp maxima to ms).
    let mut min = vec![("/table/a.parquet".to_owned(), Some(a_lo))];
    let mut max = vec![("/table/a.parquet".to_owned(), Some(a_hi + 1000))];
    if with_b {
        min.push(("/table/b.parquet".to_owned(), Some(b_lo)));
        max.push(("/table/b.parquet".to_owned(), Some(b_hi + 1000)));
    }
    let store = timestamp_checkpoint_store(bytes);
    assert_eq!(planned_stats(&store, &["t"], "t_min"), min, "{unit} min");
    assert_eq!(planned_stats(&store, &["t"], "t_max"), max, "{unit} max");
}

#[test]
fn checkpoint_timestamp_statistics_are_microseconds_in_every_parquet_unit() {
    for (unit, bytes, with_b) in [
        ("ms", timestamp_checkpoint("ms"), true),
        ("us", timestamp_checkpoint("us"), true),
        ("ns", timestamp_checkpoint("ns"), false),
    ] {
        check_checkpoint_timestamp_bounds(unit, &bytes, with_b);
    }
}

#[test]
#[ignore = "DK-INT96: Kernel 0.30.1 wraps old INT96 values before casting. Rerun and remove this skip when the stock reader decodes INT96 losslessly."]
fn checkpoint_int96_timestamp_statistics_are_microseconds() {
    check_checkpoint_timestamp_bounds("int96", &timestamp_checkpoint("int96"), true);
}

#[test]
fn checkpoint_int96_timestamp_bounds_are_unknown() {
    let store = timestamp_checkpoint_store(&timestamp_checkpoint("int96"));
    let unknown = [
        ("/table/a.parquet".to_owned(), None),
        ("/table/b.parquet".to_owned(), None),
    ];
    assert_eq!(planned_stats(&store, &["t"], "t_min"), unknown);
    assert_eq!(planned_stats(&store, &["t"], "t_max"), unknown);
}

#[test]
fn json_decimal_statistics_are_exact() {
    // 19 significant digits: not representable in f64.
    let stats = |min: &str, max: &str| {
        json!({"add":{"path":"a.parquet","size":10,"partitionValues":{},"stats":
            format!("{{\"numRecords\":5,\"minValues\":{{\"t\":{min}}},\"maxValues\":{{\"t\":{max}}},\"nullCount\":{{\"t\":0}}}}")}})
    };
    let mut store = Store::default();
    store.put(
        "00000000000000000000.json",
        &[
            protocol(),
            single_column_metadata("decimal(20,2)"),
            stats("12345678901234567.89", "-1.5E+3"),
        ],
    );
    assert_eq!(
        planned_stats(&store, &["t"], "t_min"),
        [("/table/a.parquet".to_owned(), Some(1234567890123456789))]
    );
    assert_eq!(
        planned_stats(&store, &["t"], "t_max"),
        [("/table/a.parquet".to_owned(), Some(-150000))]
    );
}

impl Store {
    fn put_raw(&mut self, path: &str, bytes: &[u8], mtime: i64) {
        self.0.insert(
            format!("/table/_delta_log/{path}"),
            (bytes.to_vec(), Some(mtime)),
        );
    }
    fn commit_gets(&self) -> usize {
        self.2
            .lock()
            .unwrap()
            .iter()
            .filter(|u| u.ends_with(".json") && !u.ends_with("_last_checkpoint"))
            .count()
    }
}
fn version_name(v: i64) -> String {
    format!("{v:020}.json")
}

#[test]
fn schema_only_reads_no_file_actions() {
    let mut s = Store::default();
    s.put("00000000000000000000.json", &[protocol(), metadata()]);
    // File actions that cannot be parsed: a schema-only plan must not look at them.
    s.put_raw(&version_name(1), b"{\"add\":garbage}\n", 2000);
    let request = Request {
        schema_only: true,
        use_metadata_statistics: true,
        ..Request::default()
    };
    let (header, paths) = planned(&s, request.clone()).unwrap();
    assert!(paths.is_empty());
    assert_eq!(header.resolved_version, 1);
    assert_eq!(header.row_count, None);
    assert!(planned(&s, Request::default()).is_err());

    // Only lines with the canonical prefix are skipped. A noncanonical line still has to parse.
    s.put_raw(&version_name(1), b" {\"add\":garbage}\n", 2000);
    assert!(planned(&s, request.clone()).is_err());
    s.put_raw(&version_name(1), b" {\"add\":{},\"protocol\":{}}\n", 2000);
    let error = planned(&s, request).unwrap_err();
    assert!(format!("{error:?}").contains("multiple actions in one row"));
}

#[test]
fn schema_only_stops_at_the_newest_metadata_and_protocol() {
    let mut s = Store::default();
    s.put(
        "00000000000000000000.json",
        &[protocol(), metadata(), add("a.parquet")],
    );
    for v in 1..20 {
        let mut actions = vec![add(&format!("f{v}.parquet"))];
        if v == 19 {
            actions.extend([protocol(), metadata()]);
        }
        s.put(&version_name(v), &actions);
    }
    let request = Request {
        schema_only: true,
        max_threads: Some(2),
        ..Request::default()
    };
    let (header, _) = planned(&s, request).unwrap();
    assert_eq!(header.resolved_version, 19);
    // Newest first, with at most the read-ahead window in flight.
    assert!(s.commit_gets() <= 4, "{}", s.commit_gets());
    let before = s.commit_gets();
    // Without them in the newest commits, the older ones are read.
    s.put(&version_name(19), &[add("g.parquet")]);
    let request = Request {
        schema_only: true,
        ..Request::default()
    };
    planned(&s, request).unwrap();
    assert_eq!(s.commit_gets() - before, 20);
}

#[test]
fn schema_only_reads_checkpoint_metadata_without_file_columns() {
    let mut store = Store::default();
    store.0.insert(
        "/table/_delta_log/00000000000000000000.checkpoint.parquet".to_owned(),
        (fixture("wide-binary.checkpoint.parquet"), Some(1000)),
    );
    let (header, paths) = planned(
        &store,
        Request {
            schema_only: true,
            ..Request::default()
        },
    )
    .unwrap();
    assert!(paths.is_empty());
    assert_eq!(header.resolved_version, 0);
    let schema_only = store.1.load(std::sync::atomic::Ordering::Relaxed);
    planned(&store, Request::default()).unwrap();
    let full = store.1.load(std::sync::atomic::Ordering::Relaxed) - schema_only;
    assert!(schema_only < full, "{schema_only} vs {full}");
}

#[test]
fn schema_only_rejects_protocol_and_metadata_in_a_later_checkpoint_row_group() {
    let mut store = Store::default();
    store.put_raw(
        "00000000000000000000.checkpoint.parquet",
        &fixture("rg-mixed-protocol-metadata.checkpoint.parquet"),
        1000,
    );
    let error = planned(
        &store,
        Request {
            schema_only: true,
            ..Request::default()
        },
    )
    .unwrap_err();
    assert!(
        format!("{error:?}").contains("multiple actions in one row"),
        "{error:?}"
    );
}

#[test]
fn schema_only_rejects_duplicate_actions_across_checkpoint_parts() {
    let mut store = Store::default();
    let part = &fixture("rg-meta-only.checkpoint.parquet");
    store.put_raw(
        "00000000000000000000.checkpoint.0000000001.0000000002.parquet",
        part,
        1000,
    );
    store.put_raw(
        "00000000000000000000.checkpoint.0000000002.0000000002.parquet",
        part,
        1000,
    );
    let error = planned(
        &store,
        Request {
            schema_only: true,
            ..Request::default()
        },
    )
    .unwrap_err();
    assert!(
        format!("{error:?}").contains("duplicate metadata/protocol/checkpointMetadata"),
        "{error:?}"
    );
}

/// The linear scan the planner used before binary search, as the oracle. A timestamp before every
/// commit selects version 0.
fn linear_version_at(times: &[(i64, Option<i64>)], timestamp: i64) -> Option<i64> {
    let (mut previous, mut chosen) = (None, None);
    for (version, (mtime, ict)) in times.iter().enumerate() {
        let time = match ict {
            Some(ict) => *ict,
            None => previous.map_or(*mtime, |p: i64| (*mtime).max(p + 1)),
        };
        previous = Some(time);
        if time <= timestamp {
            chosen = Some(version as i64)
        } else {
            break;
        }
    }
    chosen.or(Some(0))
}
fn version_at_store(
    times: &[(i64, Option<i64>)],
    timestamp: i64,
    ict_from: Option<(i64, Option<i64>)>,
) -> (Option<i64>, usize) {
    let mut s = Store::default();
    let mut commits = BTreeMap::new();
    for (v, (mtime, ict)) in times.iter().enumerate() {
        let v = v as i64;
        let mut actions = vec![add(&format!("f{v}.parquet"))];
        if let Some(ict) = ict {
            actions.insert(
                0,
                json!({"commitInfo":{"inCommitTimestamp":ict,"operation":"WRITE"}}),
            );
        }
        s.put(&version_name(v), &actions);
        s.0.get_mut(&format!("/table/_delta_log/{}", version_name(v)))
            .unwrap()
            .1 = Some(*mtime);
        commits.insert(
            v,
            super::host::Object {
                url: format!("/table/_delta_log/{}", version_name(v)),
                modified_ms: Some(*mtime),
            },
        );
    }
    let vtable = s.host();
    let host = unsafe { super::host::Host::from_raw(&vtable, false) }.unwrap();
    let found = super::runtime::block_on(None, async {
        let storage = host.get_storage("/table").await?;
        super::log::version_at(&storage, &commits, timestamp, ict_from).await
    });
    (found.ok(), s.commit_gets())
}

#[test]
fn timestamp_travel_matches_the_linear_scan() {
    // Modification times that go backwards are raised past the previous commit's time.
    let mtimes = [5000, 4000, 4000, 9000, 9000, 9500, 20000];
    let times = mtimes.iter().map(|m| (*m, None)).collect::<Vec<_>>();
    for timestamp in 0..21000 {
        assert_eq!(
            version_at_store(&times, timestamp, None).0,
            linear_version_at(&times, timestamp),
            "mtimes, timestamp {timestamp}"
        );
    }
    // In-commit timestamps from version 3: earlier versions keep adjusted modification times,
    // and in-commit timestamps are used as written (they may be below a raised time).
    let mut times = (0..3).map(|v| (3000 - v * 1000, None)).collect::<Vec<_>>();
    times.extend((3..40).map(|v| (50, Some(10_000 + v * 100))));
    for timestamp in (0..15000).step_by(7) {
        assert_eq!(
            version_at_store(&times, timestamp, Some((3, None))).0,
            linear_version_at(&times, timestamp),
            "in-commit, timestamp {timestamp}"
        );
    }
}

#[test]
fn timestamp_travel_trusts_in_commit_timestamps_from_the_enablement_timestamp() {
    // Version 1's file time was set after the in-commit timestamps of versions 2 and 3.
    let times = [
        (1000, None),
        (4000, None),
        (50, Some(2000)),
        (50, Some(3000)),
    ];
    for (timestamp, expected) in [(3000, Some(3)), (2500, Some(2)), (2000, Some(2))] {
        assert_eq!(
            version_at_store(&times, timestamp, Some((2, Some(2000)))).0,
            expected,
            "{timestamp}"
        );
    }
    // Before the enablement timestamp the modification times still decide.
    assert_eq!(
        version_at_store(&times, 1500, Some((2, Some(2000)))).0,
        Some(0)
    );
    // Before every commit the earliest version is selected.
    assert_eq!(
        version_at_store(&times, 500, Some((2, Some(2000)))).0,
        Some(0)
    );
}

#[test]
fn timestamp_travel_before_the_first_commit_clamps_to_the_earliest_version() {
    let mtimes = [(5000, None), (6000, None), (7000, None)];
    assert_eq!(version_at_store(&mtimes, 0, None).0, Some(0));
    assert_eq!(version_at_store(&mtimes, 4999, None).0, Some(0));
    let ict = [(50, Some(5000)), (50, Some(6000))];
    assert_eq!(version_at_store(&ict, 0, Some((0, None))).0, Some(0));
    assert_eq!(version_at_store(&ict, 0, Some((0, Some(5000)))).0, Some(0));
}

#[test]
fn timestamp_travel_reads_logarithmically_many_commits() {
    let times = (0..200)
        .map(|v| (1000 + v, Some(100_000 + v * 10)))
        .collect::<Vec<_>>();
    for timestamp in [99_999, 100_000, 100_005, 101_000, 101_990, 105_000] {
        let (found, reads) = version_at_store(&times, timestamp, Some((0, None)));
        assert_eq!(found, linear_version_at(&times, timestamp), "{timestamp}");
        assert!(reads <= 9, "{reads} commit reads for {timestamp}");
    }
}

#[test]
fn timestamp_travel_at_the_latest_version_replays_once_with_files() {
    let mut s = initial();
    s.put(
        "00000000000000000001.json",
        &[json!({"remove":{"path":"a.parquet"}}), add("b.parquet")],
    );
    let (header, paths) = planned(
        &s,
        Request {
            timestamp_ms: Some(10_000),
            ..Request::default()
        },
    )
    .unwrap();
    assert_eq!(header.resolved_version, 1);
    assert_eq!(paths, vec!["/table/b.parquet"]);
}

fn v2_manifest(files: &[Value]) -> Vec<Value> {
    let mut actions = vec![
        protocol(),
        metadata(),
        json!({"checkpointMetadata":{"version":5}}),
        json!({"sidecar":{"path":"sidecar.parquet","sizeInBytes":1}}),
    ];
    actions.extend_from_slice(files);
    actions
}
fn v2_store(manifest: &[Value]) -> Store {
    let mut s = Store::default();
    s.put(
        "00000000000000000005.checkpoint.12345678-1234-1234-1234-123456789abc.json",
        manifest,
    );
    s.0.insert(
        "/table/_delta_log/_sidecars/sidecar.parquet".to_owned(),
        (fixture("v2-sidecar.sidecar.parquet"), Some(1000)),
    );
    s
}

#[test]
fn v2_manifest_with_sidecars_reads_them() {
    let s = v2_store(&v2_manifest(&[]));
    let (header, paths) = planned(&s, Request::default()).unwrap();
    assert_eq!(header.resolved_version, 5);
    assert_eq!(paths, ["/table/a.parquet", "/table/b.parquet"]);
}

#[test]
fn v2_manifest_reads_a_sidecar_with_no_action_or_host_mtime() {
    let mut s = v2_store(&v2_manifest(&[]));
    s.0.get_mut("/table/_delta_log/_sidecars/sidecar.parquet")
        .unwrap()
        .1 = None;
    let (_, paths) = planned(&s, Request::default()).unwrap();
    assert_eq!(paths, ["/table/a.parquet", "/table/b.parquet"]);
}

#[test]
fn v2_sidecar_rejects_standalone_metadata_actions() {
    let mut manifest = v2_manifest(&[]);
    // A file URI needs a drive letter on Windows, and the host lists the path it resolves to.
    let root = if cfg!(windows) { "C:\\table" } else { "/table" };
    let uri = url::Url::from_file_path(
        std::path::Path::new(root).join("_delta_log/_sidecars/sidecar.parquet"),
    )
    .unwrap();
    let path = uri
        .to_file_path()
        .unwrap()
        .into_os_string()
        .into_string()
        .unwrap();
    let bytes = fixture("rg-spread.checkpoint.parquet");
    manifest.last_mut().unwrap()["sidecar"] = json!({
        "path": uri.as_str(),
        "sizeInBytes": bytes.len(),
        "modificationTime": 1000,
    });
    let mut store = v2_store(&manifest);
    store.0.insert(path, (bytes, Some(1000)));
    let error = planned(&store, Request::default()).unwrap_err();
    assert!(
        error.message().contains("non-file action in sidecar"),
        "{error:?}"
    );
}

#[test]
fn v2_inline_manifest_rejects_duplicate_logical_files() {
    let dv = json!({"storageType":"p","pathOrInlineDv":"/table/vector.bin","offset":1,"sizeInBytes":4,"cardinality":1});
    let mut added = add("a.parquet");
    added["add"]["deletionVector"] = dv.clone();
    let removed = json!({"remove":{"path":"a.parquet","deletionVector":dv}});
    let mut manifest = v2_manifest(&[added, removed]);
    manifest.retain(|action| action.get("sidecar").is_none());
    let error = planned(&v2_store(&manifest), Request::default()).unwrap_err();
    assert!(
        error
            .message()
            .contains("duplicate add/remove logical file"),
        "{error:?}"
    );
}

#[test]
fn json_commit_rejects_add_and_remove_on_one_row() {
    let mut s = initial();
    let mut mixed = add("b.parquet");
    mixed["remove"] = json!({"path":"a.parquet"});
    s.put(&version_name(1), &[mixed]);
    let error = planned(&s, Request::default()).unwrap_err();
    assert!(
        error.message().contains("multiple actions in one row"),
        "{error:?}"
    );
}

#[test]
fn json_commit_rejects_metadata_and_add_on_one_row() {
    let mut store = initial();
    let mut mixed = add("b.parquet");
    mixed["metaData"] = metadata()["metaData"].clone();
    store.put(&version_name(1), &[mixed]);
    let error = planned(&store, Request::default()).unwrap_err();
    assert!(
        error.message().contains("multiple actions in one row"),
        "{error:?}"
    );
}

#[test]
fn v2_manifest_cannot_mix_inline_and_sidecar_file_actions() {
    // PROTOCOL.md, "V2 Spec": all add and remove actions inline, or all in sidecars.
    for inline in [add("c.parquet"), json!({"remove":{"path":"c.parquet"}})] {
        let s = v2_store(&v2_manifest(&[inline]));
        let error = planned(&s, Request::default()).unwrap_err();
        assert!(
            format!("{error:?}").contains("embedded file actions"),
            "{error:?}"
        );
    }
}

#[test]
fn unusable_checkpoint_statistics_types_leave_only_that_column_unknown() {
    let mut store = Store::default();
    store.0.insert(
        "/table/_delta_log/00000000000000000000.checkpoint.parquet".to_owned(),
        (fixture("binary-stats.checkpoint.parquet"), Some(1000)),
    );
    assert_eq!(
        planned_stats(&store, &["t", "b"], "t_min"),
        [("/table/a.parquet".to_owned(), Some(1))]
    );
    assert_eq!(
        planned_stats(&store, &["t", "b"], "t_max"),
        [("/table/a.parquet".to_owned(), Some(5))]
    );
    let (header, _) = planned(
        &store,
        Request {
            use_metadata_statistics: true,
            statistics_columns: Some(vec!["t".to_owned(), "b".to_owned()]),
            ..Request::default()
        },
    )
    .unwrap();
    assert_eq!(header.row_count, Some((5, 0)));
}

#[test]
fn one_bad_decimal_leaves_only_that_files_bound_unknown() {
    let file = |path: &str, min: &str| {
        json!({"add":{"path":path,"size":10,"partitionValues":{},"stats":
            format!("{{\"numRecords\":5,\"minValues\":{{\"t\":{min}}},\"maxValues\":{{\"t\":9.50}},\"nullCount\":{{\"t\":0}}}}")}})
    };
    let mut store = Store::default();
    store.put(
        "00000000000000000000.json",
        &[
            protocol(),
            single_column_metadata("decimal(5,2)"),
            file("a.parquet", "1.25"),
            // Three decimals do not fit scale 2.
            file("b.parquet", "1.001"),
            file("c.parquet", "-3.5"),
        ],
    );
    assert_eq!(
        planned_stats(&store, &["t"], "t_min"),
        [
            ("/table/a.parquet".to_owned(), Some(125)),
            ("/table/b.parquet".to_owned(), None),
            ("/table/c.parquet".to_owned(), Some(-350)),
        ]
    );
    assert_eq!(
        planned_stats(&store, &["t"], "t_max"),
        [
            ("/table/a.parquet".to_owned(), Some(950)),
            ("/table/b.parquet".to_owned(), Some(950)),
            ("/table/c.parquet".to_owned(), Some(950)),
        ]
    );
}

#[test]
fn statistics_of_other_columns_are_not_parsed() {
    // Statistics text that is invalid beyond the requested columns still serves the others.
    let mut store = Store::default();
    store.put(
        "00000000000000000000.json",
        &[
            protocol(),
            metadata(),
            json!({"add":{"path":"a.parquet","size":10,"partitionValues":{},"stats":
                "{\"numRecords\":5,\"minValues\":{\"y\":[1,{\"x\":2}],\"x\":4},\"maxValues\":{\"x\":9}}"}}),
        ],
    );
    assert_eq!(
        planned_stats(&store, &["x"], "x_min"),
        [("/table/a.parquet".to_owned(), Some(4))]
    );
}

fn fixture(name: &str) -> Vec<u8> {
    if name.starts_with("partition-timestamp-") || name.starts_with("timestamp-") {
        test_timestamp_fixtures::bytes(name)
    } else {
        test_fixtures::bytes(name)
    }
}

fn checkpoint_store(files: &[(&str, Vec<u8>)]) -> Store {
    let mut store = Store::default();
    for (name, bytes) in files {
        store.0.insert(
            format!("/table/_delta_log/{name}"),
            (bytes.clone(), Some(1000)),
        );
    }
    store
}

#[test]
fn checkpoint_generation_mismatch_restarts_planning() {
    let store = checkpoint_store(&[(
        "00000000000000000000.checkpoint.parquet",
        fixture("rg-spread.checkpoint.parquet"),
    )]);
    store.3.store(true, std::sync::atomic::Ordering::Relaxed);
    let (header, paths) = planned(&store, Request::default()).unwrap();
    assert_eq!(header.resolved_version, 0);
    assert_eq!(paths.len(), 6);
    // `_last_checkpoint` is read again only when the complete plan restarts.
    assert_eq!(
        store.2.lock().unwrap().as_slice(),
        ["/table/_delta_log/_last_checkpoint"; 2]
    );
}
const DAY_MICROS: i128 = 86_400_000_000;

#[test]
#[ignore = "DK-NTZ-PROTOCOL: timestampNtz is undeclared"]
fn widened_date_checkpoint_bounds_are_days_in_microseconds() {
    let store = checkpoint_store(&[(
        "00000000000000000000.checkpoint.parquet",
        fixture("date-widened.checkpoint.parquet"),
    )]);
    assert_eq!(
        planned_stats(&store, &["x"], "x_min"),
        [("/table/a.parquet".to_owned(), Some(18262 * DAY_MICROS))]
    );
    // Timestamp maxima are raised by a millisecond (they are truncated to milliseconds).
    assert_eq!(
        planned_stats(&store, &["x"], "x_max"),
        [(
            "/table/a.parquet".to_owned(),
            Some(18292 * DAY_MICROS + 1000)
        )]
    );
}

#[test]
#[ignore = "DK-NTZ-PROTOCOL: timestampNtz is undeclared"]
fn widened_date_json_bounds_are_midnights() {
    let mut store = Store::default();
    let column = json!({"name":"x","type":"timestamp_ntz","nullable":true,"metadata":
        {"delta.typeChanges":[{"fromType":"date","toType":"timestamp_ntz"}]}});
    let metadata = json!({"metaData":{"id":"id","format":{"provider":"parquet","options":{}},
        "schemaString":json!({"type":"struct","fields":[column]}).to_string(),
        "partitionColumns":[],"configuration":{}}});
    store.put(
        "00000000000000000000.json",
        &[
            protocol(),
            metadata,
            json!({"add":{"path":"a.parquet","size":10,"partitionValues":{},"stats":
                "{\"numRecords\":5,\"minValues\":{\"x\":\"2020-01-01\"},\"maxValues\":{\"x\":\"2020-01-31\"},\"nullCount\":{\"x\":0}}"}}),
        ],
    );
    assert_eq!(
        planned_stats(&store, &["x"], "x_min"),
        [("/table/a.parquet".to_owned(), Some(18262 * DAY_MICROS))]
    );
}

fn enable_date_widening(store: &mut Store, typed_checkpoint: bool) {
    let mut configuration = json!({"delta.enableTypeWidening": "true"});
    if typed_checkpoint {
        configuration["delta.checkpoint.writeStatsAsJson"] = json!("false");
        configuration["delta.checkpoint.writeStatsAsStruct"] = json!("true");
    }
    store.put(
        "00000000000000000001.json",
        &[
            json!({"protocol":{"minReaderVersion":3,"minWriterVersion":7,
                "readerFeatures":["timestampNtz","typeWidening"],
                "writerFeatures":["appendOnly","invariants","timestampNtz","typeWidening"]}}),
            json!({"metaData":{"id":if typed_checkpoint {"before-date-widening"} else {"id"},
                "format":{"provider":"parquet","options":{}},"schemaString":
                json!({"type":"struct","fields":[
                    {"name":"x","type":"timestamp_ntz","nullable":true,
                     "metadata":{"delta.typeChanges":[
                         {"fromType":"date","toType":"timestamp_ntz","tableVersion":1}]}}
                ]}).to_string(),"partitionColumns":[],"configuration":configuration}}),
        ],
    );
}

fn assert_widened_date_bounds(store: &Store) {
    assert_eq!(
        planned_stats(store, &["x"], "x_min"),
        [("/table/a.parquet".to_owned(), Some(18262 * DAY_MICROS))]
    );
    assert_eq!(
        planned_stats(store, &["x"], "x_max"),
        [(
            "/table/a.parquet".to_owned(),
            Some(18292 * DAY_MICROS + 1000)
        )]
    );
}

#[test]
fn widened_date_json_bounds_with_declared_features() {
    let mut store = Store::default();
    store.put(
        "00000000000000000000.json",
        &[
            json!({"protocol":{"minReaderVersion":1,"minWriterVersion":2}}),
            json!({"metaData":{"id":"id","format":{"provider":"parquet","options":{}},
                "schemaString":json!({"type":"struct","fields":[
                    {"name":"x","type":"date","nullable":true,"metadata":{}}
                ]}).to_string(),"partitionColumns":[],"configuration":{}}}),
            json!({"add":{"path":"a.parquet","size":10,"modificationTime":1000,"dataChange":true,
                "partitionValues":{},"stats":
                "{\"numRecords\":5,\"minValues\":{\"x\":\"2020-01-01\"},\"maxValues\":{\"x\":\"2020-01-31\"},\"nullCount\":{\"x\":0}}"}}),
        ],
    );
    enable_date_widening(&mut store, false);
    assert_widened_date_bounds(&store);
}

#[test]
fn widened_float_json_bounds_drop_values_that_are_not_exact_floats() {
    let mut store = Store::default();
    let stats = |min: &str, max: &str| {
        format!(
            r#"{{"numRecords":1,"minValues":{{"x":{min}}},"maxValues":{{"x":{max}}},"nullCount":{{"x":0}}}}"#
        )
    };
    let add = |path: &str, stats: String| {
        json!({"add":{"path":path,"size":10,"modificationTime":1000,"dataChange":true,
            "partitionValues":{},"stats":stats}})
    };
    store.put(
        "00000000000000000000.json",
        &[
            json!({"protocol":{"minReaderVersion":1,"minWriterVersion":2}}),
            json!({"metaData":{"id":"id","format":{"provider":"parquet","options":{}},
                "schemaString":json!({"type":"struct","fields":[
                    {"name":"x","type":"float","nullable":true,"metadata":{}}
                ]}).to_string(),"partitionColumns":[],"configuration":{}}}),
            // The shortest text of the FLOAT -0.1, the exact widened value and 0.5.
            add("short.parquet", stats("-0.1", "-0.1")),
            add(
                "exact.parquet",
                stats("-0.10000000149011612", "-0.10000000149011612"),
            ),
            add("half.parquet", stats("0.5", "0.5")),
        ],
    );
    store.put(
        "00000000000000000001.json",
        &[
            json!({"protocol":{"minReaderVersion":3,"minWriterVersion":7,
                "readerFeatures":["typeWidening"],"writerFeatures":["typeWidening"]}}),
            json!({"metaData":{"id":"id","format":{"provider":"parquet","options":{}},
                "schemaString":json!({"type":"struct","fields":[
                    {"name":"x","type":"double","nullable":true,
                     "metadata":{"delta.typeChanges":[
                         {"fromType":"float","toType":"double","tableVersion":1}]}}
                ]}).to_string(),"partitionColumns":[],
                "configuration":{"delta.enableTypeWidening":"true"}}}),
        ],
    );
    let exact = f64::from(-0.1f32).to_bits();
    for child in ["x_min", "x_max"] {
        assert_eq!(
            planned_stats(&store, &["x"], child),
            [
                ("/table/exact.parquet".to_owned(), Some(i128::from(exact))),
                (
                    "/table/half.parquet".to_owned(),
                    Some(i128::from(0.5f64.to_bits()))
                ),
                ("/table/short.parquet".to_owned(), None),
            ],
            "{child}"
        );
    }
}

fn assert_widened_float_bounds(store: &Store, short: Option<f64>) {
    let bits = |v: f64| Some(i128::from(v.to_bits()));
    for child in ["x_min", "x_max"] {
        assert_eq!(
            planned_stats(store, &["x"], child),
            [
                ("/table/exact.parquet".to_owned(), bits(f64::from(-0.1f32))),
                ("/table/half.parquet".to_owned(), bits(0.5)),
                ("/table/short.parquet".to_owned(), short.and_then(bits)),
            ],
            "{child}"
        );
    }
}

#[test]
fn widened_float_checkpoint_keeps_float_bounds_exact_as_doubles() {
    // Kernel coerces the typed FLOAT bounds of an old checkpoint to DOUBLE: all are floats.
    let mut store = checkpoint_store(&[(
        "00000000000000000000.checkpoint.parquet",
        fixture("before-float-widening.checkpoint.parquet"),
    )]);
    store.put(
        "00000000000000000001.json",
        &[
            json!({"protocol":{"minReaderVersion":3,"minWriterVersion":7,
                "readerFeatures":["typeWidening"],"writerFeatures":["typeWidening"]}}),
            json!({"metaData":{"id":"before-float-widening","format":{"provider":"parquet","options":{}},
                "schemaString":json!({"type":"struct","fields":[
                    {"name":"x","type":"double","nullable":true,
                     "metadata":{"delta.typeChanges":[
                         {"fromType":"float","toType":"double","tableVersion":1}]}}
                ]}).to_string(),"partitionColumns":[],
                "configuration":{"delta.enableTypeWidening":"true",
                    "delta.checkpoint.writeStatsAsJson":"false",
                    "delta.checkpoint.writeStatsAsStruct":"true"}}}),
        ],
    );
    assert_widened_float_bounds(&store, Some(f64::from(-0.1f32)));
}

#[test]
fn widened_float_checkpoint_drops_double_bounds_that_are_not_exact_floats() {
    let store = checkpoint_store(&[(
        "00000000000000000001.checkpoint.parquet",
        fixture("after-float-widening.checkpoint.parquet"),
    )]);
    assert_widened_float_bounds(&store, None);
}

fn checkpoint_before_date_widening() -> Store {
    let mut store = checkpoint_store(&[(
        "00000000000000000000.checkpoint.parquet",
        fixture("before-date-widening.checkpoint.parquet"),
    )]);
    enable_date_widening(&mut store, true);
    store
}

#[test]
#[ignore = "DK-NTZ-CHECKPOINT-STATS: Kernel drops old typed date bounds"]
fn widened_date_checkpoint_bounds_with_declared_features() {
    assert_widened_date_bounds(&checkpoint_before_date_widening());
}

#[test]
fn widened_date_checkpoint_keeps_file_with_unknown_statistics() {
    // Kernel 0.30.1 drops the entire typed statistics struct when a DATE bound
    // would need to widen to TIMESTAMP_NTZ.
    let store = checkpoint_before_date_widening();
    for bound in ["x_min", "x_max"] {
        assert_eq!(
            planned_stats(&store, &["x"], bound),
            [("/table/a.parquet".to_owned(), None)]
        );
    }
    let (header, _) = planned(
        &store,
        Request {
            use_metadata_statistics: true,
            ..Request::default()
        },
    )
    .unwrap();
    assert_eq!(header.row_count, None);
}

#[test]
fn widened_date_checkpoint_bounds_after_rewriting() {
    let store = checkpoint_store(&[(
        "00000000000000000001.checkpoint.parquet",
        fixture("after-date-widening.checkpoint.parquet"),
    )]);
    assert_widened_date_bounds(&store);
}

#[test]
fn checkpoint_rows_without_typed_statistics_use_their_text() {
    let store = checkpoint_store(&[(
        "00000000000000000000.checkpoint.parquet",
        fixture("partial-parsed.checkpoint.parquet"),
    )]);
    assert_eq!(
        planned_stats(&store, &["x"], "x_min"),
        [
            ("/table/a.parquet".to_owned(), Some(1)),
            ("/table/b.parquet".to_owned(), Some(10)),
            ("/table/c.parquet".to_owned(), Some(20)),
        ]
    );
    let (header, _) = planned(
        &store,
        Request {
            use_metadata_statistics: true,
            statistics_columns: Some(vec!["x".to_owned()]),
            ..Request::default()
        },
    )
    .unwrap();
    assert_eq!(header.row_count, Some((14, 0)));
}

#[test]
fn json_boolean_statistics_survive_replay_after_a_typed_checkpoint() {
    let mut store = checkpoint_store(&[(
        "00000000000000000000.checkpoint.parquet",
        fixture("partial-parsed.checkpoint.parquet"),
    )]);
    let mut new_metadata = metadata();
    new_metadata["metaData"]["schemaString"] = json!(
        json!({"type":"struct","fields":[
            {"name":"x","type":"long","nullable":true,"metadata":{}},
            {"name":"b","type":"boolean","nullable":true,"metadata":{}}
        ]})
        .to_string()
    );
    let mut new_add = add("d.parquet");
    new_add["add"]["stats"] = json!(
        r#"{"numRecords":5,"minValues":{"b":false},"maxValues":{"b":false},"nullCount":{"b":0}}"#
    );
    store.put(&version_name(1), &[new_metadata, new_add]);
    assert_eq!(
        planned_stats(&store, &["x", "b"], "b_max"),
        [
            ("/table/a.parquet".to_owned(), None),
            ("/table/b.parquet".to_owned(), None),
            ("/table/c.parquet".to_owned(), None),
            ("/table/d.parquet".to_owned(), Some(0)),
        ]
    );
    assert_eq!(
        planned_stats(&store, &["x", "b"], "x_min"),
        [
            ("/table/a.parquet".to_owned(), Some(1)),
            ("/table/b.parquet".to_owned(), Some(10)),
            ("/table/c.parquet".to_owned(), Some(20)),
            ("/table/d.parquet".to_owned(), None),
        ]
    );
    let (header, _) = planned(
        &store,
        Request {
            use_metadata_statistics: true,
            ..Request::default()
        },
    )
    .unwrap();
    assert_eq!(header.row_count, Some((19, 0)));
}

fn planned_partitions(s: &Store, column: &str) -> Result<Vec<Option<String>>, FfiError> {
    let r = Request {
        table_uri: "/table".into(),
        ..Request::default()
    };
    let host = s.host();
    let output = r
        .with_ffi(|r| unsafe { super::plan(&host, r) })
        .into_result()?;
    let stream: ffi::ArrowArrayStream = unsafe { output.files.transmute_into() };
    let mut reader = unsafe { ffi::ArrowArrayStreamReader::try_new(Box::new(stream)) }.unwrap();
    let mut out = Vec::new();
    while let Some(batch) = unsafe { reader.next() } {
        let batch = batch.unwrap();
        let a = batch.as_any().downcast_ref::<StructArray>().unwrap();
        let get = |a: &StructArray, name: &str| {
            let polars_arrow::datatypes::ArrowDataType::Struct(f) = a.dtype() else {
                unreachable!()
            };
            a.values()[f.iter().position(|f| f.name == name).unwrap()].clone()
        };
        let pv = get(a, "partition_values");
        let pv = pv.as_any().downcast_ref::<StructArray>().unwrap();
        let v = get(pv, column);
        let v = v.as_any().downcast_ref::<Utf8ViewArray>().unwrap();
        out.extend(v.iter().map(|s| s.map(str::to_owned)));
    }
    Ok(out)
}

#[test]
fn byte_array_partition_values_are_text() {
    let store = checkpoint_store(&[(
        "00000000000000000000.checkpoint.parquet",
        fixture("binary-partition.checkpoint.parquet"),
    )]);
    assert_eq!(
        planned_partitions(&store, "p").unwrap(),
        [Some("north".to_owned())]
    );
    let store = checkpoint_store(&[(
        "00000000000000000000.checkpoint.parquet",
        fixture("binary-partition-invalid.checkpoint.parquet"),
    )]);
    let error = planned_partitions(&store, "p").unwrap_err();
    assert!(format!("{error:?}").contains("UTF-8"), "{error:?}");
}

#[test]
fn every_part_of_a_multipart_checkpoint_gets_the_table_metadata() {
    let store = checkpoint_store(&[
        (
            "00000000000000000000.checkpoint.0000000001.0000000002.parquet",
            fixture("multipart-1.checkpoint.parquet"),
        ),
        (
            "00000000000000000000.checkpoint.0000000002.0000000002.parquet",
            fixture("multipart-2.checkpoint.parquet"),
        ),
    ]);
    assert_eq!(
        planned_stats(&store, &["x"], "x_min"),
        [
            ("/table/a.parquet".to_owned(), Some(1)),
            ("/table/b.parquet".to_owned(), Some(10))
        ]
    );
}

/// `(path, x_min, x_max)` of every planned file, in plan order.
fn planned_bounds(store: &Store) -> Vec<(String, Option<i128>, Option<i128>)> {
    let min = planned_stats(store, &["x"], "x_min");
    let max = planned_stats(store, &["x"], "x_max");
    assert_eq!(min.len(), max.len());
    min.into_iter()
        .zip(max)
        .map(|((p, lo), (q, hi))| {
            assert_eq!(p, q);
            (p, lo, hi)
        })
        .collect()
}

fn bounds(spec: &[(&str, i128)]) -> Vec<(String, Option<i128>, Option<i128>)> {
    spec.iter()
        .map(|(p, lo)| (format!("/table/{p}.parquet"), Some(*lo), Some(lo + 5)))
        .collect()
}

#[test]
fn multi_row_group_checkpoint_keeps_every_add_in_order() {
    for (name, expected) in [
        (
            "rg-spread",
            bounds(&[
                ("a", 0),
                ("b", 10),
                ("c", 20),
                ("d", 30),
                ("e", 40),
                ("f", 50),
            ]),
        ),
        // The first row group holds only metadata and protocol.
        (
            "rg-meta-only",
            bounds(&[("a", 0), ("b", 10), ("c", 20), ("d", 30)]),
        ),
        // The `stats` text is read only for the row group with file "c".
        (
            "rg-stats-text",
            bounds(&[("a", 0), ("b", 10), ("c", 20), ("d", 30), ("e", 40)]),
        ),
    ] {
        let store = checkpoint_store(&[(
            "00000000000000000000.checkpoint.parquet",
            match name {
                "rg-spread" => fixture("rg-spread.checkpoint.parquet"),
                "rg-meta-only" => fixture("rg-meta-only.checkpoint.parquet"),
                _ => fixture("rg-stats-text.checkpoint.parquet"),
            },
        )]);
        assert_eq!(planned_bounds(&store), expected, "{name}");
        let (header, paths) = planned(
            &store,
            Request {
                use_metadata_statistics: true,
                statistics_columns: Some(vec!["x".to_owned()]),
                ..Request::default()
            },
        )
        .unwrap();
        assert_eq!(paths.len(), expected.len(), "{name}");
        assert_eq!(
            header.row_count,
            Some((5 * expected.len() as u64, 0)),
            "{name}"
        );
    }
}

#[test]
fn multi_row_group_checkpoint_reconciles_with_later_commits() {
    let mut store = checkpoint_store(&[(
        "00000000000000000000.checkpoint.parquet",
        fixture("rg-spread.checkpoint.parquet"),
    )]);
    store.put(
        "00000000000000000001.json",
        &[
            json!({"remove":{"path":"c.parquet","deletionTimestamp":1,"dataChange":true}}),
            json!({"remove":{"path":"f.parquet","deletionTimestamp":1,"dataChange":true}}),
            add("z.parquet"),
        ],
    );
    let mut paths = planned_bounds(&store)
        .into_iter()
        .map(|(p, _, _)| p)
        .collect::<Vec<_>>();
    paths.sort();
    assert_eq!(
        paths,
        ["a", "b", "d", "e", "z"].map(|p| format!("/table/{p}.parquet"))
    );
}

#[test]
fn multipart_checkpoint_with_several_row_groups_per_part() {
    let store = checkpoint_store(&[
        (
            "00000000000000000000.checkpoint.0000000001.0000000002.parquet",
            fixture("rg-multipart-1.checkpoint.parquet"),
        ),
        (
            "00000000000000000000.checkpoint.0000000002.0000000002.parquet",
            fixture("rg-multipart-2.checkpoint.parquet"),
        ),
    ]);
    let mut got = planned_bounds(&store);
    got.sort();
    assert_eq!(
        got,
        bounds(&[
            ("a", 0),
            ("b", 10),
            ("c", 20),
            ("d", 30),
            ("e", 40),
            ("f", 50),
            ("g", 60),
            ("h", 70),
            ("i", 80)
        ])
    );
}

#[test]
fn checkpoint_uses_stored_timestamp_partition_instant() {
    assert_stored_timestamp_partition_instants(
        &checkpoint_store(&[(
            "00000000000000000000.checkpoint.parquet",
            fixture("partition-timestamp-typed.checkpoint.parquet"),
        )]),
        &[Some(1_577_908_800_000_000)],
    );
}

#[test]
fn checkpoint_uses_stored_timestamp_for_mapped_nullable_partition() {
    assert_stored_timestamp_partition_instants(
        &checkpoint_store(&[(
            "00000000000000000000.checkpoint.parquet",
            fixture("partition-timestamp-mapped.checkpoint.parquet"),
        )]),
        &[Some(1_577_908_800_000_000), None],
    );
}

#[test]
#[ignore = "DK-INT96: Kernel decodes stored partition instants as nanoseconds before casting"]
fn checkpoint_uses_stored_int96_timestamp_partition_instant() {
    let store = timestamp_checkpoint_store(&fixture(
        "partition-timestamp-int96-wall.checkpoint.parquet",
    ));
    assert_stored_timestamp_partition_instants(&store, &[Some(-14_831_697_600_000_000)]);
}

#[test]
fn checkpoint_does_not_trust_a_wrapped_int96_partition_instant() {
    let store = timestamp_checkpoint_store(&fixture(
        "partition-timestamp-int96-wall.checkpoint.parquet",
    ));
    let error = planned(&store, Request::default()).unwrap_err();
    assert!(
        error
            .message()
            .contains("cannot recover checkpoint timestamp partition writer zone"),
        "{error:?}"
    );
}

#[test]
fn checkpoint_uses_explicit_partition_offset_when_int96_is_unavailable() {
    let store = timestamp_checkpoint_store(&fixture(
        "partition-timestamp-int96-offset.checkpoint.parquet",
    ));
    assert_stored_timestamp_partition_instants(&store, &[Some(-14_831_697_600_000_000)]);
}

fn mixed_layout_timestamp_partition_checkpoint() -> Store {
    checkpoint_store(&[
        (
            "00000000000000000000.checkpoint.0000000001.0000000002.parquet",
            fixture("partition-timestamp-multipart-raw.checkpoint.parquet"),
        ),
        (
            "00000000000000000000.checkpoint.0000000002.0000000002.parquet",
            fixture("partition-timestamp-multipart-typed.checkpoint.parquet"),
        ),
    ])
}

#[test]
#[ignore = "DK-TYPED-PARTITION-MIXED-LAYOUT: Kernel uses the first checkpoint part to select typed partition values"]
fn multipart_checkpoint_reads_a_stored_instant_after_a_raw_only_part() {
    assert_stored_timestamp_partition_instants(
        &mixed_layout_timestamp_partition_checkpoint(),
        &[Some(1_577_908_800_000_000); 2],
    );
}

#[test]
fn multipart_checkpoint_does_not_assume_a_zone_after_a_raw_only_part() {
    let error = planned(
        &mixed_layout_timestamp_partition_checkpoint(),
        Request::default(),
    )
    .unwrap_err();
    assert!(
        error
            .message()
            .contains("cannot recover checkpoint timestamp partition writer zone"),
        "{error:?}"
    );
}

fn assert_stored_timestamp_partition_instants(store: &Store, expected: &[Option<i64>]) {
    use polars_arrow::array::PrimitiveArray;
    use polars_arrow::datatypes::{ArrowDataType, TimeUnit};

    use super::files::child;

    let host = store.host();
    let output = Request {
        table_uri: "/table".into(),
        ..Request::default()
    }
    .with_ffi(|request| unsafe { super::plan(&host, request) })
    .into_result()
    .unwrap();
    let stream: ffi::ArrowArrayStream = unsafe { output.files.transmute_into() };
    let mut reader = unsafe { ffi::ArrowArrayStreamReader::try_new(Box::new(stream)) }.unwrap();
    let batch = unsafe { reader.next() }.unwrap().unwrap();
    let files = batch.as_any().downcast_ref::<StructArray>().unwrap();
    let partitions = child(files, "partition_values")
        .unwrap()
        .as_any()
        .downcast_ref::<StructArray>()
        .unwrap();
    let timestamp = child(partitions, "t")
        .unwrap()
        .as_any()
        .downcast_ref::<PrimitiveArray<i64>>()
        .unwrap();
    assert_eq!(
        timestamp.dtype(),
        &ArrowDataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()))
    );
    // The raw wall time says 12:00. The checkpoint stores 20:00 UTC.
    let actual = timestamp
        .iter()
        .map(|value| value.copied())
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
    assert!(unsafe { reader.next() }.is_none());
}

#[test]
fn checkpoint_rejects_raw_timestamp_partition_without_writer_zone() {
    let store = checkpoint_store(&[(
        "00000000000000000000.checkpoint.parquet",
        fixture("partition-timestamp-raw.checkpoint.parquet"),
    )]);
    let error = planned(&store, Request::default()).unwrap_err();
    assert!(
        error
            .message()
            .contains("cannot recover checkpoint timestamp partition writer zone"),
        "{error:?}"
    );
}
