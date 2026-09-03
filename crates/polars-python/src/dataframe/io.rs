use std::io::BufWriter;
use std::num::NonZeroUsize;
use std::sync::Arc;

use polars::io::RowIndex;
#[cfg(feature = "avro")]
use polars::io::avro::AvroCompression;
use polars::prelude::*;
use pyo3::prelude::*;

use super::PyDataFrame;
use crate::conversion::Wrap;
use crate::file::{get_file_like, get_mmap_bytes_reader};
use crate::prelude::PyCompatLevel;
use crate::utils::EnterPolarsExt;

#[pymethods]
impl PyDataFrame {
    #[staticmethod]
    #[cfg(feature = "json")]
    #[pyo3(signature = (py_f, infer_schema_length, schema, schema_overrides))]
    pub fn read_json(
        py: Python<'_>,
        py_f: Bound<PyAny>,
        infer_schema_length: Option<usize>,
        schema: Option<Wrap<Schema>>,
        schema_overrides: Option<Wrap<Schema>>,
    ) -> PyResult<Self> {
        assert!(infer_schema_length != Some(0));
        let mmap_bytes_r = get_mmap_bytes_reader(&py_f)?;

        py.enter_polars_df(move || {
            let mut reader = JsonReader::new(mmap_bytes_r)
                .with_json_format(JsonFormat::Json)
                .infer_schema_len(infer_schema_length.and_then(NonZeroUsize::new));

            if let Some(schema) = schema {
                reader = reader.with_schema(Arc::new(schema.0));
            }

            if let Some(schema) = schema_overrides.as_ref() {
                reader = reader.with_schema_overwrite(&schema.0);
            }

            reader.finish()
        })
    }

    #[staticmethod]
    #[cfg(feature = "ipc_streaming")]
    #[pyo3(signature = (py_f, columns, projection, n_rows, row_index, rechunk))]
    pub fn read_ipc_stream(
        py: Python<'_>,
        py_f: Bound<PyAny>,
        columns: Option<Vec<String>>,
        projection: Option<Vec<usize>>,
        n_rows: Option<usize>,
        row_index: Option<(String, IdxSize)>,
        rechunk: bool,
    ) -> PyResult<Self> {
        let row_index = row_index.map(|(name, offset)| RowIndex {
            name: name.into(),
            offset,
        });
        let mmap_bytes_r = get_mmap_bytes_reader(&py_f)?;
        py.enter_polars_df(move || {
            IpcStreamReader::new(mmap_bytes_r)
                .with_projection(projection)
                .with_columns(columns)
                .with_n_rows(n_rows)
                .with_row_index(row_index)
                .set_rechunk(rechunk)
                .finish()
        })
    }

    #[staticmethod]
    #[cfg(feature = "avro")]
    #[pyo3(signature = (py_f, columns, projection, n_rows))]
    pub fn read_avro(
        py: Python<'_>,
        py_f: Py<PyAny>,
        columns: Option<Vec<String>>,
        projection: Option<Vec<usize>>,
        n_rows: Option<usize>,
    ) -> PyResult<Self> {
        use polars::io::avro::AvroReader;

        let file = get_file_like(py_f, false)?;
        py.enter_polars_df(move || {
            AvroReader::new(file)
                .with_projection(projection)
                .with_columns(columns)
                .with_n_rows(n_rows)
                .finish()
        })
    }

    #[cfg(feature = "json")]
    pub fn write_json(&self, py: Python<'_>, py_f: Py<PyAny>) -> PyResult<()> {
        let file = BufWriter::new(get_file_like(py_f, true)?);
        py.enter_polars(|| {
            // TODO: Cloud support

            JsonWriter::new(file)
                .with_json_format(JsonFormat::Json)
                .finish(&mut self.df.write())
        })
    }

    #[cfg(feature = "ipc_streaming")]
    pub fn write_ipc_stream(
        &self,
        py: Python<'_>,
        py_f: Py<PyAny>,
        compression: Wrap<Option<IpcCompression>>,
        compat_level: PyCompatLevel,
    ) -> PyResult<()> {
        let mut buf = get_file_like(py_f, true)?;
        py.enter_polars(|| {
            IpcStreamWriter::new(&mut buf)
                .with_compression(compression.0)
                .with_compat_level(compat_level.0)
                .finish(&mut self.df.write())
        })
    }

    #[cfg(feature = "avro")]
    #[pyo3(signature = (py_f, compression, name))]
    pub fn write_avro(
        &self,
        py: Python<'_>,
        py_f: Py<PyAny>,
        compression: Wrap<Option<AvroCompression>>,
        name: String,
    ) -> PyResult<()> {
        use polars::io::avro::AvroWriter;
        let mut buf = get_file_like(py_f, true)?;
        py.enter_polars(|| {
            AvroWriter::new(&mut buf)
                .with_compression(compression.0)
                .with_name(name)
                .finish(&mut self.df.write())
        })
    }
}
