use std::collections::BTreeMap;
use std::ffi::c_void;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use delta_kernel::Snapshot;
use delta_kernel::arrow::array::{Int64Array, StringArray, StructArray};
use delta_kernel::arrow::buffer::NullBuffer;
use delta_kernel::arrow::record_batch::RecordBatch;
use delta_kernel::parquet::arrow::ArrowWriter;
use delta_kernel::schema::{DataType, StructField, StructType};
use polars_io_ext_ffi::common::{
    FfiBuf, FfiFuture, FfiOption, FfiResult, FfiStr, FfiVec, FutureExt,
};
use polars_io_ext_ffi::delta_v1::{self, ObjectMeta, StorageHandle};

use super::super::host::Host;
use super::*;

#[derive(Debug, PartialEq)]
enum Call {
    Get(String),
    Head(String),
    Range(String),
    List,
    Release,
}

#[derive(Default)]
struct RecordingHost {
    files: BTreeMap<String, Vec<u8>>,
    calls: Mutex<Vec<Call>>,
    replace_on_range: AtomicBool,
}

impl RecordingHost {
    fn run<T>(&self, root: &str, run: impl FnOnce(&dyn Engine, Url) -> Result<T>) -> Result<T> {
        let vtable = delta_v1::Host {
            ctx: self as *const Self as *const c_void,
            get_storage,
            storage_get: get,
            storage_head: head,
            storage_get_range: range,
            storage_list: list,
            storage_release: release,
            log: log_message,
        };
        let host = unsafe { Host::from_raw(&vtable, false) }?;
        let storage = runtime::block_on(None, host.get_storage(root))?;
        with_engine(&storage, root, false, run)
    }

    fn record(&self, call: Call) {
        self.calls.lock().unwrap().push(call);
    }

    fn file(&self, url: &str) -> Result<&[u8]> {
        self.files
            .get(url)
            .map(Vec::as_slice)
            .ok_or_else(|| FfiError::new(FfiErrorKind::NOT_FOUND, format!("missing {url}")))
    }
}

fn ready<T: Send + 'static>(result: Result<T>) -> FfiFuture<FfiResult<T>> {
    async move { result.into() }.into_ffi()
}

unsafe fn recording_host<'a>(handle: StorageHandle) -> &'a RecordingHost {
    unsafe { &*handle.0.cast() }
}

unsafe extern "C" fn get_storage(
    ctx: *const c_void,
    _: FfiStr,
) -> FfiFuture<FfiResult<StorageHandle>> {
    ready(Ok(StorageHandle(ctx)))
}

unsafe extern "C" fn get(handle: StorageHandle, url: FfiStr) -> FfiFuture<FfiResult<FfiBuf>> {
    let host = unsafe { recording_host(handle) };
    let url = url.as_str().unwrap();
    host.record(Call::Get(url.into()));
    ready(host.file(url).map(|data| FfiBuf::from_vec(data.to_vec())))
}

unsafe extern "C" fn head(handle: StorageHandle, url: FfiStr) -> FfiFuture<FfiResult<u64>> {
    let host = unsafe { recording_host(handle) };
    let url = url.as_str().unwrap();
    host.record(Call::Head(url.into()));
    ready(host.file(url).map(|data| data.len() as u64))
}

unsafe extern "C" fn range(
    handle: StorageHandle,
    url: FfiStr,
    start: u64,
    end: u64,
) -> FfiFuture<FfiResult<FfiBuf>> {
    let host = unsafe { recording_host(handle) };
    let url = url.as_str().unwrap();
    host.record(Call::Range(url.into()));
    if host.replace_on_range.swap(false, Ordering::Relaxed) {
        return ready(Err(FfiError::new(
            FfiErrorKind::OBJECT_CHANGED,
            "replaced after HEAD",
        )));
    }
    ready(host.file(url).and_then(|data| {
        data.get(start as usize..end as usize)
            .map(|data| FfiBuf::from_vec(data.to_vec()))
            .ok_or_else(|| FfiError::new(FfiErrorKind::IO, "invalid test range"))
    }))
}

unsafe extern "C" fn list(
    handle: StorageHandle,
    prefix: FfiStr,
    start: FfiOption<FfiStr>,
) -> FfiFuture<FfiResult<FfiVec<ObjectMeta>>> {
    let host = unsafe { recording_host(handle) };
    let prefix = prefix.as_str().unwrap();
    let start = start.into_option().map(|start| start.as_str().unwrap());
    host.record(Call::List);
    ready(Ok(FfiVec::from_vec(
        host.files
            .iter()
            .filter(|(name, _)| {
                name.strip_prefix(prefix)
                    .is_some_and(|relative| !relative.contains('/'))
                    && start.is_none_or(|start| name.as_str() > start)
            })
            .map(|(name, data)| ObjectMeta {
                url: FfiBuf::from_string(name.clone()),
                size: data.len() as u64,
                modified_ms: Some(1_234).into(),
            })
            .collect(),
    )))
}

unsafe extern "C" fn release(handle: StorageHandle) {
    unsafe { recording_host(handle) }.record(Call::Release);
}

unsafe extern "C" fn log_message(_: *const c_void, _: FfiStr) {}

#[test]
fn json_snapshot_masks_crc_and_revokes_the_lease() {
    let root = "/table literal%23";
    let commit = format!("{root}/_delta_log/00000000000000000000.json");
    let crc = format!("{root}/_delta_log/00000000000000000000.crc");
    let mut host = RecordingHost::default();
    host.files.insert(
        commit.clone(),
        br#"{"protocol":{"minReaderVersion":1,"minWriterVersion":2}}
{"metaData":{"id":"table","schemaString":"{\"type\":\"struct\",\"fields\":[{\"name\":\"x\",\"type\":\"long\",\"nullable\":true,\"metadata\":{}}]}","partitionColumns":[],"format":{"provider":"parquet","options":{}},"configuration":{}}}
"#
        .to_vec(),
    );
    host.files.insert(crc.clone(), b"not a Delta CRC".to_vec());

    let (escaped, location) = host
        .run(root, |engine, root| {
            let snapshot = Snapshot::builder_for(root.clone()).build(engine).unwrap();
            assert_eq!(snapshot.version(), 0);
            let handler = engine.storage_handler();
            let log_root = root.join("_delta_log/").unwrap();
            let list: Vec<_> = handler
                .list_from(&log_root)
                .unwrap()
                .collect::<KernelResult<_>>()
                .unwrap();
            assert_eq!(list.len(), 1);
            assert_eq!(list[0].size, host.files[&commit].len() as u64);
            assert_eq!(list[0].last_modified, 1_234);
            let crc = log_root.join("00000000000000000000.crc").unwrap();
            let mut out = handler.read_files(vec![(crc, None)]).unwrap();
            assert!(matches!(
                out.next().unwrap(),
                Err(KernelError::FileNotFound(_))
            ));
            Ok((handler, list[0].location.clone()))
        })
        .unwrap();
    assert!(escaped.head(&location).is_err());
    let calls = host.calls.lock().unwrap();
    assert!(calls.contains(&Call::Get(commit)));
    assert!(!calls.contains(&Call::Get(crc)));
    assert_eq!(calls.last(), Some(&Call::Release));
}

#[test]
fn signed_parquet_footer_and_body_use_the_same_host_pin() {
    // This object exists only behind the test host callbacks.
    let signed = "https://127.0.0.1:9/checkpoint.parquet?X-Amz-Signature=test";
    let batch =
        RecordBatch::try_from_iter([("x", Arc::new(Int64Array::from(vec![1])) as _)]).unwrap();
    let mut data = Vec::new();
    let mut writer = ArrowWriter::try_new(&mut data, batch.schema(), None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    let file = FileMeta::new(Url::parse(signed).unwrap(), 0, data.len() as u64);
    let mut host = RecordingHost::default();
    host.files.insert(signed.into(), data);

    host.run("/table", |engine, _| {
        let parquet = engine.parquet_handler();
        parquet.read_parquet_footer(&file).unwrap();
        let schema = StructType::try_new([StructField::nullable("x", DataType::LONG)]).unwrap();
        let mut batches = parquet
            .read_parquet_files(std::slice::from_ref(&file), Arc::new(schema), None)
            .unwrap();
        assert_eq!(batches.next().unwrap().unwrap().len(), 1);
        assert!(batches.next().is_none());
        Ok(())
    })
    .unwrap();
    let calls = std::mem::take(&mut *host.calls.lock().unwrap());
    assert_eq!(calls.first(), Some(&Call::Head(signed.into())));
    assert_eq!(calls.last(), Some(&Call::Release));
    assert!(calls.len() >= 4);
    assert!(
        calls[1..calls.len() - 1]
            .iter()
            .all(|call| *call == Call::Range(signed.into()))
    );

    host.replace_on_range.store(true, Ordering::Relaxed);
    let error = host
        .run("/table", |engine, _| {
            let error = engine
                .parquet_handler()
                .read_parquet_footer(&file)
                .unwrap_err();
            let error = host_error(&error).unwrap();
            assert_eq!(error.kind(), FfiErrorKind::OBJECT_CHANGED);
            Ok(())
        })
        .unwrap_err();
    assert_eq!(error.kind(), FfiErrorKind::OBJECT_CHANGED);
    assert_eq!(error.message(), "replaced after HEAD");
}

#[test]
fn parquet_manifest_rejects_sidecar_and_inline_actions_on_separate_rows() {
    let sidecar = RecordBatch::try_from_iter([
        (
            "path",
            Arc::new(StringArray::from(vec![Some("unopened.parquet"), None])) as _,
        ),
        (
            "sizeInBytes",
            Arc::new(Int64Array::from(vec![Some(1), None])) as _,
        ),
        (
            "modificationTime",
            Arc::new(Int64Array::from(vec![Some(1), None])) as _,
        ),
    ])
    .unwrap();
    let sidecar = StructArray::try_new(
        sidecar.schema().fields().clone(),
        sidecar.columns().to_vec(),
        Some(NullBuffer::from(vec![true, false])),
    )
    .unwrap();
    let add = RecordBatch::try_from_iter([(
        "path",
        Arc::new(StringArray::from(vec![None, Some("inline.parquet")])) as _,
    )])
    .unwrap();
    let add = StructArray::try_new(
        add.schema().fields().clone(),
        add.columns().to_vec(),
        Some(NullBuffer::from(vec![false, true])),
    )
    .unwrap();
    let batch = RecordBatch::try_from_iter([
        ("sidecar", Arc::new(sidecar) as _),
        ("add", Arc::new(add) as _),
    ])
    .unwrap();
    let mut data = Vec::new();
    let mut writer = ArrowWriter::try_new(&mut data, batch.schema(), None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    let name = "_delta_log/00000000000000000000.checkpoint.parquet";
    let size = data.len() as u64;
    let mut host = RecordingHost::default();
    host.files.insert(format!("/table/{name}"), data);
    let schema = StructType::try_new([StructField::nullable(
        "sidecar",
        StructType::try_new([
            StructField::not_null("path", DataType::STRING),
            StructField::not_null("sizeInBytes", DataType::LONG),
            StructField::not_null("modificationTime", DataType::LONG),
        ])
        .unwrap(),
    )])
    .unwrap();
    host.run("/table", |engine, root| {
        let file = FileMeta::new(root.join(name).unwrap(), 0, size);
        let result = engine
            .parquet_handler()
            .read_parquet_files(&[file], Arc::new(schema), None);
        let error = result.err().unwrap();
        assert!(
            error.to_string().contains("embedded file actions"),
            "{error}"
        );
        Ok(())
    })
    .unwrap();
    assert!(!host.calls.lock().unwrap().iter().any(|call| match call {
        Call::Get(url) | Call::Head(url) | Call::Range(url) => url.contains("_sidecars"),
        _ => false,
    }));
}
