use std::fmt;
use std::mem::ManuallyDrop;
use std::sync::Arc;

use ::iceberg::encryption::{GeneratedKey, KeyManagementClient, SensitiveBytes};
use ::iceberg::io::FileIO;
use ::iceberg::scan::ArrowRecordBatchStream;
use ::iceberg::spec::TableMetadata;
use ::iceberg::table::Table;
use ::iceberg::{Error, ErrorKind, NamespaceIdent, Result, Runtime, TableIdent};
use arrow_array::RecordBatch;
use futures::StreamExt;
use polars_core::prelude::{DataFrame, IntoColumn, PolarsResult, Series};
use polars_core::utils::arrow::ffi;
use polars_error::polars_err;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::PyDataFrame;
use crate::error::PyPolarsErr;
use crate::exceptions::ComputeError;

pub struct PythonKmsClient {
    inner: Arc<Py<PyAny>>,
}

impl PythonKmsClient {
    pub fn new(inner: Py<PyAny>) -> Self {
        Self {
            inner: Arc::new(inner),
        }
    }

    fn call(&self, method: &str, key: &[u8], wrapping_key_id: &str) -> Result<Vec<u8>> {
        Python::attach(|py| {
            self.inner
                .call_method1(py, method, (PyBytes::new(py, key), wrapping_key_id))
                .and_then(|out| out.extract::<Vec<u8>>(py))
                .map_err(|_| {
                    Error::new(ErrorKind::Unexpected, format!("Python KMS {method} failed"))
                })
        })
    }
}

impl fmt::Debug for PythonKmsClient {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PythonKmsClient").finish_non_exhaustive()
    }
}

#[async_trait::async_trait]
impl KeyManagementClient for PythonKmsClient {
    async fn wrap_key(&self, key: &[u8], wrapping_key_id: &str) -> Result<Vec<u8>> {
        self.call("wrap_key", key, wrapping_key_id)
    }

    async fn unwrap_key(
        &self,
        wrapped_key: &[u8],
        wrapping_key_id: &str,
    ) -> Result<SensitiveBytes> {
        self.call("unwrap_key", wrapped_key, wrapping_key_id)
            .map(SensitiveBytes::new)
    }

    fn supports_key_generation(&self) -> bool {
        false
    }

    async fn generate_key(&self, _wrapping_key_id: &str) -> Result<GeneratedKey> {
        Err(Error::new(
            ErrorKind::FeatureUnsupported,
            "Python KMS client does not support server-side key generation",
        ))
    }
}

#[pyclass]
pub struct IcebergBatchIterator {
    state: tokio::sync::Mutex<IcebergBatchIteratorState>,
}

struct IcebergBatchIteratorState {
    stream: ArrowRecordBatchStream,
    remaining: Option<usize>,
}

#[pymethods]
impl IcebergBatchIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&self, py: Python<'_>) -> PyResult<Option<PyDataFrame>> {
        let batch: PyResult<Option<RecordBatch>> = py.detach(|| {
            polars_core::runtime::ASYNC.block_on(async {
                let mut state = self.state.lock().await;
                let remaining = state.remaining;
                if remaining == Some(0) {
                    return Ok(None);
                }

                let Some(batch) = state.stream.next().await else {
                    return Ok(None);
                };
                let mut batch = batch.map_err(to_py_err)?;

                if let Some(remaining) = remaining {
                    if batch.num_rows() > remaining {
                        batch = batch.slice(0, remaining);
                    }
                    state.remaining = Some(remaining - batch.num_rows());
                }

                Ok(Some(batch))
            })
        });

        let Some(batch) = batch? else {
            return Ok(None);
        };

        record_batch_to_dataframe(batch)
            .map(PyDataFrame::new)
            .map(Some)
            .map_err(|error| PyErr::from(PyPolarsErr::from(error)))
    }
}

#[pyfunction]
pub fn _scan_iceberg_rust(
    py: Python<'_>,
    metadata_location: String,
    kms_client: Py<PyAny>,
    snapshot_id: Option<i64>,
    columns: Option<Vec<String>>,
    n_rows: Option<usize>,
    batch_size: Option<usize>,
) -> PyResult<Py<IcebergBatchIterator>> {
    let stream = py
        .detach(|| {
            polars_core::runtime::ASYNC.block_on(async move {
                let file_io = FileIO::new_with_fs();
                let metadata = TableMetadata::read_from(&file_io, &metadata_location).await?;
                let table = Table::builder()
                    .metadata(metadata)
                    .metadata_location(metadata_location)
                    .identifier(TableIdent::new(
                        NamespaceIdent::new("polars".into()),
                        "encrypted_scan".into(),
                    ))
                    .file_io(file_io)
                    .kms_client(Arc::new(PythonKmsClient::new(kms_client)))
                    .runtime(Runtime::try_current()?)
                    .readonly(true)
                    .build()?;

                let mut scan = table.scan().with_batch_size(batch_size);
                if let Some(snapshot_id) = snapshot_id {
                    scan = scan.snapshot_id(snapshot_id);
                }
                if let Some(columns) = columns {
                    scan = scan.select(columns);
                }

                scan.build()?.to_arrow().await
            })
        })
        .map_err(to_py_err)?;

    Py::new(
        py,
        IcebergBatchIterator {
            state: tokio::sync::Mutex::new(IcebergBatchIteratorState {
                stream,
                remaining: n_rows,
            }),
        },
    )
}

fn to_py_err(error: Error) -> PyErr {
    ComputeError::new_err(error.to_string())
}

fn record_batch_to_dataframe(batch: RecordBatch) -> PolarsResult<DataFrame> {
    if batch.num_columns() == 0 {
        return Ok(DataFrame::empty_with_height(batch.num_rows()));
    }

    let columns = batch
        .schema()
        .fields()
        .iter()
        .zip(batch.columns())
        .map(|(field, array)| {
            let array = arrow_array::ffi::FFI_ArrowArray::new(&array.to_data());
            let schema = arrow_array::ffi::FFI_ArrowSchema::try_from(field.as_ref()).map_err(
                |error| polars_err!(ComputeError: "failed to export Arrow field: {error}"),
            )?;

            let array = unsafe { move_ffi_array(array) };
            let schema = unsafe { move_ffi_schema(schema) };
            let field = unsafe { ffi::import_field_from_c(&schema) }?;
            let array = unsafe { ffi::import_array_from_c(array, field.dtype.clone()) }?;

            Series::try_from((&field, array)).map(IntoColumn::into_column)
        })
        .collect::<PolarsResult<Vec<_>>>()?;

    DataFrame::new(batch.num_rows(), columns)
}

unsafe fn move_ffi_array(array: arrow_array::ffi::FFI_ArrowArray) -> ffi::ArrowArray {
    let array = ManuallyDrop::new(array);
    assert_eq!(
        std::mem::size_of::<arrow_array::ffi::FFI_ArrowArray>(),
        std::mem::size_of::<ffi::ArrowArray>()
    );
    let ptr: *const arrow_array::ffi::FFI_ArrowArray = &*array;
    unsafe { std::ptr::read(ptr.cast()) }
}

unsafe fn move_ffi_schema(schema: arrow_array::ffi::FFI_ArrowSchema) -> ffi::ArrowSchema {
    let schema = ManuallyDrop::new(schema);
    assert_eq!(
        std::mem::size_of::<arrow_array::ffi::FFI_ArrowSchema>(),
        std::mem::size_of::<ffi::ArrowSchema>()
    );
    let ptr: *const arrow_array::ffi::FFI_ArrowSchema = &*schema;
    unsafe { std::ptr::read(ptr.cast()) }
}

#[cfg(test)]
mod tests {
    use arrow_array::{ArrayRef, Int64Array, StringArray};

    use super::*;

    #[pyclass]
    struct TestKms;

    #[pymethods]
    impl TestKms {
        fn wrap_key(&self, key: &[u8], wrapping_key_id: &str) -> Vec<u8> {
            [wrapping_key_id.as_bytes(), key].concat()
        }

        fn unwrap_key(&self, wrapped_key: &[u8], wrapping_key_id: &str) -> PyResult<Vec<u8>> {
            wrapped_key
                .strip_prefix(wrapping_key_id.as_bytes())
                .map(<[u8]>::to_vec)
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "invalid wrapping key: {wrapped_key:?}"
                    ))
                })
        }
    }

    #[test]
    fn test_python_kms_roundtrip() {
        Python::initialize();
        let client =
            Python::attach(|py| PythonKmsClient::new(Py::new(py, TestKms).unwrap().into_any()));

        let wrapped = polars_core::runtime::ASYNC
            .block_on(client.wrap_key(b"secret", "master"))
            .unwrap();
        let unwrapped = polars_core::runtime::ASYNC
            .block_on(client.unwrap_key(&wrapped, "master"))
            .unwrap();

        assert_eq!(unwrapped.as_bytes(), b"secret");
        assert_eq!(format!("{client:?}"), "PythonKmsClient { .. }");
    }

    #[test]
    fn test_python_kms_error_does_not_expose_key_material() {
        Python::initialize();
        let client =
            Python::attach(|py| PythonKmsClient::new(Py::new(py, TestKms).unwrap().into_any()));

        let err = polars_core::runtime::ASYNC
            .block_on(client.unwrap_key(b"sensitive", "missing"))
            .unwrap_err();
        let message = err.to_string();

        assert!(message.contains("Python KMS unwrap_key failed"));
        assert!(!message.contains("sensitive"));
    }

    #[test]
    fn test_record_batch_to_dataframe() {
        let batch = RecordBatch::try_from_iter([
            ("id", Arc::new(Int64Array::from(vec![1, 2, 3])) as ArrayRef),
            (
                "value",
                Arc::new(StringArray::from(vec!["a", "b", "c"])) as ArrayRef,
            ),
        ])
        .unwrap();

        let dataframe = record_batch_to_dataframe(batch).unwrap();

        assert_eq!(dataframe.shape(), (3, 2));
        assert_eq!(
            dataframe
                .column("id")
                .unwrap()
                .i64()
                .unwrap()
                .into_no_null_iter()
                .collect::<Vec<_>>(),
            [1, 2, 3]
        );
        assert_eq!(
            dataframe
                .column("value")
                .unwrap()
                .str()
                .unwrap()
                .iter()
                .collect::<Vec<_>>(),
            [Some("a"), Some("b"), Some("c")]
        );
    }
}
