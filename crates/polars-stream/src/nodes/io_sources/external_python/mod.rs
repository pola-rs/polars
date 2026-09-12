use std::sync::{Arc, OnceLock};

use async_trait::async_trait;
use parking_lot::Mutex;
use polars_async::executor::{JoinHandle, TaskPriority};
use polars_async::{ASYNC, executor};
use polars_core::frame::DataFrame;
use polars_core::schema::SchemaRef;
use polars_error::{PolarsResult, polars_bail, polars_err};
use polars_io::RowIndex;
use polars_io::cloud::CloudOptions;
use polars_plan::dsl::ScanSource;
use polars_utils::index::idxsize_try_from;
use polars_utils::pl_str::PlSmallStr;
use polars_utils::python_function::PythonObject;
use polars_utils::python_thread_pool::PyThreadPool;
use polars_utils::{IdxSize, python_interns};
use pyo3::pybacked::PyBackedStr;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyAnyMethods, PyBytes, PyDict, PyList, PyString};
use pyo3::{Bound, Py, PyAny, PyResult, Python, intern, pyclass, pymethods};

use crate::morsel::{Morsel, MorselSeq, SourceToken};
use crate::nodes::io_sources::multi_scan::reader_interface::builder::FileReaderBuilder;
use crate::nodes::io_sources::multi_scan::reader_interface::capabilities::ReaderCapabilities;
use crate::nodes::io_sources::multi_scan::reader_interface::output::{
    FileReaderOutputRecv, FileReaderOutputSend,
};
use crate::nodes::io_sources::multi_scan::reader_interface::{
    BeginReadArgs, FileReader, FileReaderCallbacks, Projection,
};

pub static PY_EXTERNAL_READER_VTABLE: OnceLock<PyExternalReaderVTable> = OnceLock::new();

fn py_external_reader_vtable() -> &'static PyExternalReaderVTable {
    PY_EXTERNAL_READER_VTABLE
        .get()
        .unwrap_or_else(|| panic!("PY_EXTERNAL_READER_VTABLE not initialized"))
}

pub struct PyExternalReaderVTable {
    pub extract_schema: fn(py: Python<'_>, schema: Py<PyAny>) -> PolarsResult<SchemaRef>,
    /// Returns: false if send failed.
    pub enter_polars_send_df:
        fn(py: Python<'_>, df: DataFrame, tx: &ExternalPythonReaderDataFrameTx) -> bool,
    pub extract_df: fn(py: Python<'_>, py_df: Bound<'_, PyAny>) -> PyResult<DataFrame>,
}

pub struct PythonFileReaderBuilder {
    builder: polars_io::external_reader::python::PythonFileReaderBuilder,
    threadpool: Arc<PyThreadPool>,
    capabilities: PyOnceLock<ReaderCapabilities>,
}

impl std::fmt::Debug for PythonFileReaderBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PythonFileReaderBuilder")
            .field("builder", &self.builder)
            .field("threadpool", &self.threadpool)
            .finish()
    }
}

impl PythonFileReaderBuilder {
    pub fn new(
        builder: polars_io::external_reader::python::PythonFileReaderBuilder,
        threadpool: Arc<PyThreadPool>,
    ) -> Self {
        Self {
            builder,
            threadpool,
            capabilities: PyOnceLock::new(),
        }
    }

    fn builder(&self) -> &Arc<PythonObject> {
        self.builder.builder()
    }
}

impl FileReaderBuilder for PythonFileReaderBuilder {
    fn reader_name(&self) -> PolarsResult<PlSmallStr> {
        ASYNC.block_in_place(|| {
            Python::attach(|py| {
                PolarsResult::Ok(PlSmallStr::from_str(
                    &self
                        .builder()
                        .getattr(py, python_interns::DUNDER_CLASS.get(py))?
                        .getattr(py, python_interns::DUNDER_NAME.get(py))?
                        .extract::<PyBackedStr>(py)?,
                ))
            })
        })
    }

    fn is_external_python_reader(&self) -> bool {
        true
    }

    fn is_external_python_reader_with_filter_support(&self) -> PolarsResult<bool> {
        let mask = ReaderCapabilities::FULL_FILTER | ReaderCapabilities::PARTIAL_FILTER;

        Ok(!(self.reader_capabilities()? & mask).is_empty())
    }

    fn reader_capabilities(&self) -> PolarsResult<ReaderCapabilities> {
        ASYNC
            .block_in_place(|| {
                Python::attach(|py| {
                    let mut capabilities = ReaderCapabilities::empty();

                    self.capabilities.get_or_try_init(py, || {
                        let reader_capabilities = self
                            .builder()
                            .call_method0(py, intern!(py, "reader_capabilities"))?;

                        if reader_capabilities
                            .getattr(py, intern!(py, "row_index"))?
                            .extract::<bool>(py)?
                        {
                            capabilities |= ReaderCapabilities::ROW_INDEX;
                        }

                        if reader_capabilities
                            .getattr(py, intern!(py, "pre_slice"))?
                            .extract::<bool>(py)?
                        {
                            capabilities |= ReaderCapabilities::PRE_SLICE;
                        }

                        if reader_capabilities
                            .getattr(py, intern!(py, "negative_pre_slice"))?
                            .extract::<bool>(py)?
                        {
                            capabilities |= ReaderCapabilities::NEGATIVE_PRE_SLICE;
                        }

                        let supported_filter =
                            reader_capabilities.getattr(py, intern!(py, "supported_filter"))?;

                        if !supported_filter.is_none(py) {
                            match &*supported_filter.extract::<PyBackedStr>(py)? {
                                "partial" => capabilities |= ReaderCapabilities::PARTIAL_FILTER,
                                "full" => {
                                    capabilities |= ReaderCapabilities::PARTIAL_FILTER
                                        | ReaderCapabilities::FULL_FILTER
                                },
                                x => polars_bail!(
                                    ComputeError:
                                    "unknown value for supported_filter: '{x}'; \
                                    expected one of: ('partial', 'full')"
                                ),
                            }
                        }

                        Ok(capabilities)
                    })
                })
            })
            .copied()
    }

    fn build_file_reader(
        &self,
        source: polars_plan::prelude::ScanSource,
        _cloud_options: Option<Arc<CloudOptions>>,
        _scan_source_idx: usize,
    ) -> PolarsResult<Box<dyn FileReader>> {
        ASYNC.block_in_place(|| {
            Python::attach(|py| {
                let py_source = match source {
                    ScanSource::Path(path) => PyString::new(py, path.as_str()).into_any(),
                    ScanSource::Buffer(buffer) => PyBytes::new(py, buffer.as_slice()).into_any(),
                    ScanSource::File(_) => polars_bail!(
                        ComputeError:
                        "unsupported: scan_external_reader with opened file sources"
                    ),
                };

                let reader = self.builder().call_method1(
                    py,
                    intern!(py, "build_file_reader"),
                    (py_source,),
                )?;

                PolarsResult::Ok(Box::new(PythonFileReader {
                    inner: Arc::new(PythonFileReaderInner {
                        reader: PythonObject(reader),
                        n_rows_cache: Mutex::new((0, 0)),
                    }),
                    threadpool: Arc::clone(&self.threadpool),
                }) as Box<dyn FileReader>)
            })
        })
    }
}

#[derive(Clone)]
pub struct PythonFileReader {
    inner: Arc<PythonFileReaderInner>,
    threadpool: Arc<PyThreadPool>,
}

struct PythonFileReaderInner {
    reader: PythonObject,
    n_rows_cache: Mutex<(u64, u64)>,
}

impl PythonFileReader {
    fn n_rows_in_file_cached(&self, limit: Option<u64>) -> PolarsResult<u64> {
        let within_cached_limit = {
            let (_, prev_limit) = { *self.inner.n_rows_cache.lock() };
            prev_limit < limit.unwrap_or(u64::MAX)
        };

        if within_cached_limit {
            let n_rows = ASYNC.block_in_place(|| {
                Python::attach(|py| {
                    ASYNC.block_in_place(|| {
                        self.inner
                            .reader
                            .call_method1(py, intern!(py, "n_rows_in_file"), (limit,))?
                            .extract::<u64>(py)
                    })
                })
            })?;

            *self.inner.n_rows_cache.lock() = (n_rows, limit.unwrap_or(u64::MAX));
        }

        let (n_rows, _) = { *self.inner.n_rows_cache.lock() };
        Ok(n_rows)
    }
}

#[async_trait]
impl FileReader for PythonFileReader {
    async fn initialize(&mut self) -> PolarsResult<()> {
        ASYNC.block_in_place(|| {
            Python::attach(|py| {
                self.inner
                    .reader
                    .call_method0(py, intern!(py, "fetch_metadata"))?;
                Ok(())
            })
        })
    }

    async fn file_schema(&mut self) -> PolarsResult<SchemaRef> {
        ASYNC.block_in_place(|| {
            Python::attach(|py| {
                (py_external_reader_vtable().extract_schema)(
                    py,
                    self.inner.reader.call_method0(py, intern!(py, "schema"))?,
                )
            })
        })
    }

    async fn n_rows_in_file(&mut self) -> PolarsResult<IdxSize> {
        let n_rows = self.n_rows_in_file_cached(None)?;

        idxsize_try_from(n_rows).map_err(|_| {
            polars_err!(
                ComputeError:
                "PythonFileReader: number of rows in file ({n_rows}) exceeds \
                IdxSize limit; please use bigidx"
            )
        })
    }

    fn begin_read(
        &mut self,
        args: BeginReadArgs,
    ) -> PolarsResult<(FileReaderOutputRecv, JoinHandle<PolarsResult<()>>)> {
        let (mut tx, rx) = FileReaderOutputSend::new_serial();

        let BeginReadArgs {
            projection: Projection::Plain(projected_schema),
            row_index,
            pre_slice,

            num_pipelines: _,
            disable_morsel_split: _,
            last_morsel_pipelines: _,
            callbacks:
                FileReaderCallbacks {
                    mut file_schema_tx,
                    mut n_rows_in_file_tx,
                    mut row_position_on_end_tx,
                },

            predicate,
            cast_columns_policy: _,
            extra_columns_policy: _,
            missing_columns_policy: _,
        } = args
        else {
            panic!("unsupported args: {:?}", args)
        };

        let mut slf = self.clone();
        let filters = predicate.map(|x| x.py_filter_exprs.unwrap());

        let handle = executor::spawn(TaskPriority::Low, async move {
            if let Some(file_schema_tx) = file_schema_tx.take() {
                let file_schema = slf.file_schema().await?;
                let _ = file_schema_tx.send(file_schema);
            }

            if (pre_slice.is_some() || filters.is_some())
                && (n_rows_in_file_tx.is_some() || row_position_on_end_tx.is_some())
            {
                let n_rows =
                    slf.n_rows_in_file_cached(pre_slice.clone().map(|x| x.end_position() as u64))?;
                let n_rows = idxsize_try_from(n_rows).unwrap_or(IdxSize::MAX);

                if let Some(n_rows_in_file_tx) = n_rows_in_file_tx.take() {
                    let _ = n_rows_in_file_tx.send(n_rows);
                }

                if let Some(row_position_on_end_tx) = row_position_on_end_tx.take() {
                    let _ = row_position_on_end_tx.send(n_rows);
                }
            }

            let (py_df_tx, mut py_df_rx) = tokio::sync::mpsc::channel(1);
            let py_df_tx = ExternalPythonReaderDataFrameTx { tx: py_df_tx };

            let mut total_n_rows: u64 = 0;

            let dfs_pass_handle =
                executor::AbortOnDropHandle::new(executor::spawn(TaskPriority::Low, async move {
                    let source_token = SourceToken::new();

                    let mut i: u64 = 0;
                    while let Some(df) = py_df_rx.recv().await {
                        total_n_rows = total_n_rows.saturating_add(df.height() as _);

                        if tx
                            .send_morsel(Morsel::new_unregistered(
                                df,
                                MorselSeq::new(i),
                                source_token.clone(),
                            ))
                            .await
                            .is_err()
                        {
                            break;
                        };
                        i += 1;

                        if source_token.stop_requested() {
                            break;
                        }
                    }

                    let total_n_rows = idxsize_try_from(total_n_rows).unwrap_or(IdxSize::MAX);

                    if let Some(n_rows_in_file_tx) = n_rows_in_file_tx.take() {
                        let _ = n_rows_in_file_tx.send(total_n_rows);
                    }

                    if let Some(row_position_on_end_tx) = row_position_on_end_tx.take() {
                        let _ = row_position_on_end_tx.send(total_n_rows);
                    }
                }));

            ASYNC
                .spawn_blocking(move || {
                    Python::attach(|py| {
                        let kwargs = PyDict::new(py);

                        let columns =
                            PyList::new(py, projected_schema.iter_names().map(|x| x.as_str()))?;
                        kwargs.set_item(intern!(py, "columns"), columns)?;

                        let row_index = row_index
                            .as_ref()
                            .map(|RowIndex { name, offset }| (name.as_str(), *offset));
                        kwargs.set_item(intern!(py, "row_index"), row_index)?;

                        let pre_slice = pre_slice.map(|x| x.to_signed_offset_len());
                        kwargs.set_item(intern!(py, "pre_slice"), pre_slice)?;

                        if let Some(filters) = filters {
                            kwargs.set_item(intern!(py, "filters"), filters.as_ref())?;
                        } else {
                            kwargs.set_item(intern!(py, "filters"), ())?;
                        };

                        let py_df_iterator = slf.inner.reader.call_method(
                            py,
                            intern!(py, "collect_batches"),
                            (),
                            Some(&kwargs),
                        )?;

                        slf.threadpool.spawn_call(
                            py,
                            py_send_dfs_loop_fn(py),
                            (py_df_iterator, py_df_tx),
                            None,
                        )?;

                        PolarsResult::Ok(())
                    })
                })
                .await
                .unwrap()?;

            dfs_pass_handle.await;

            Ok(())
        });

        Ok((rx, handle))
    }
}

fn py_send_dfs_loop_fn(py: Python<'_>) -> &'static Py<PyAny> {
    static SEND_DFS: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

    SEND_DFS.get_or_init(py, || {
        py.import("polars.io.external_reader._external_reader")
            .unwrap()
            .getattr("send_dfs")
            .unwrap()
            .unbind()
    })
}

#[pyclass]
pub struct ExternalPythonReaderDataFrameTx {
    tx: tokio::sync::mpsc::Sender<DataFrame>,
}

impl ExternalPythonReaderDataFrameTx {
    pub fn send_df_(&self, df: DataFrame) -> bool {
        self.tx.blocking_send(df).is_ok()
    }
}

#[pymethods]
impl ExternalPythonReaderDataFrameTx {
    fn send_df(&self, py: Python<'_>, py_df: Bound<'_, PyAny>) -> PyResult<bool> {
        let PyExternalReaderVTable {
            extract_df,
            enter_polars_send_df,
            ..
        } = py_external_reader_vtable();

        let df = (extract_df)(py, py_df)?;

        Ok((enter_polars_send_df)(py, df, self))
    }
}
