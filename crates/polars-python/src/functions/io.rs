use std::io::BufReader;

#[cfg(any(feature = "ipc", feature = "parquet"))]
use polars::prelude::ArrowSchema;
use polars::prelude::CloudScheme;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::conversion::Wrap;
use crate::error::PyPolarsErr;
use crate::file::{EitherRustPythonFile, get_either_file};
use crate::io::cloud_options::OptPyCloudOptions;

#[cfg(feature = "ipc")]
#[pyfunction]
pub fn read_ipc_schema(py: Python<'_>, py_f: Py<PyAny>) -> PyResult<Bound<'_, PyDict>> {
    use arrow::io::ipc::read::read_file_metadata;
    let metadata = match get_either_file(py_f, false)? {
        EitherRustPythonFile::Rust(r) => {
            read_file_metadata(&mut BufReader::new(r)).map_err(PyPolarsErr::from)?
        },
        EitherRustPythonFile::Py(mut r) => read_file_metadata(&mut r).map_err(PyPolarsErr::from)?,
    };

    let dict = PyDict::new(py);
    fields_to_pydict(&metadata.schema, &dict)?;
    Ok(dict)
}

#[cfg(feature = "parquet")]
#[pyfunction]
pub fn read_parquet_metadata(
    py: Python,
    py_f: Py<PyAny>,
    storage_options: OptPyCloudOptions,
    credential_provider: Option<Py<PyAny>>,
) -> PyResult<Py<PyDict>> {
    use std::io::Cursor;

    use polars_error::feature_gated;
    use polars_io::pl_async::get_runtime;
    use polars_parquet::read::read_metadata;
    use polars_parquet::read::schema::read_custom_key_value_metadata;

    use crate::file::{PythonScanSourceInput, get_python_scan_source_input};

    let metadata = match get_python_scan_source_input(py_f, false)? {
        PythonScanSourceInput::Buffer(buf) => {
            read_metadata(&mut Cursor::new(buf)).map_err(PyPolarsErr::from)?
        },
        PythonScanSourceInput::Path(p) => {
            let cloud_options = storage_options.extract_opt_cloud_options(
                CloudScheme::from_path(p.as_str()),
                credential_provider,
            )?;

            if p.has_scheme() {
                feature_gated!("cloud", {
                    use polars::prelude::ParquetObjectStore;
                    use polars_error::PolarsResult;

                    py.detach(|| {
                        get_runtime().block_on(async {
                            let mut reader =
                                ParquetObjectStore::from_uri(p, cloud_options.as_ref(), None)
                                    .await?;
                            let result = reader.get_metadata().await?;
                            PolarsResult::Ok((**result).clone())
                        })
                    })
                })
                .map_err(PyPolarsErr::from)?
            } else {
                let file = polars_utils::open_file(p.as_std_path()).map_err(PyPolarsErr::from)?;
                read_metadata(&mut BufReader::new(file)).map_err(PyPolarsErr::from)?
            }
        },
        PythonScanSourceInput::File(f) => {
            read_metadata(&mut BufReader::new(f)).map_err(PyPolarsErr::from)?
        },
    };

    let key_value_metadata = read_custom_key_value_metadata(metadata.key_value_metadata());
    let dict = PyDict::new(py);
    for (key, value) in key_value_metadata.into_iter() {
        dict.set_item(key.as_str(), value.as_str())?;
    }
    Ok(dict.unbind())
}

/// Decode a parquet footer, optionally apply `FileMetadata::pruned`, then
/// bincode-encode and return the byte length of the wire form.
///
/// Exposed for out-of-tree measurement of the IR-plan-borne metadata wire
/// form (the `bincode(FileMetadata)` blob shipped to workers in distributed
/// execution); no caller in py-polars itself.
///
/// `projection = None` ⇒ encode the full `FileMetadata`. `projection =
/// Some(cols)` ⇒ apply `pruned(cols, predicate)`. Local files only.
#[cfg(all(feature = "parquet", feature = "json"))]
#[pyfunction]
pub fn _bench_parquet_metadata_bincode_size(
    path: &str,
    projection: Option<Vec<String>>,
    predicate: Vec<String>,
) -> PyResult<usize> {
    use polars_parquet::read::read_metadata;
    use polars_utils::pl_serialize;
    use polars_utils::pl_str::PlSmallStr;

    let file = std::fs::File::open(path).map_err(|e| PyPolarsErr::Other(e.to_string()))?;
    let metadata = read_metadata(&mut BufReader::new(file)).map_err(PyPolarsErr::from)?;

    // Match the IR-plan serializer's framing format.
    let bytes = match projection {
        None => {
            pl_serialize::serialize_to_bytes::<_, false>(&metadata).map_err(PyPolarsErr::from)?
        },
        Some(keep) => {
            let keep_pl: Vec<PlSmallStr> = keep.into_iter().map(PlSmallStr::from).collect();
            let pred_pl: Vec<PlSmallStr> = predicate.into_iter().map(PlSmallStr::from).collect();
            let pruned = metadata
                .pruned(&keep_pl, &pred_pl)
                .map_err(|e| PyPolarsErr::Other(e.to_string()))?;
            pl_serialize::serialize_to_bytes::<_, false>(&pruned).map_err(PyPolarsErr::from)?
        },
    };
    Ok(bytes.len())
}

/// Decode a parquet footer, apply `FileMetadata::pruned(projection, predicate)`,
/// and return the result as a JSON string. Format-agnostic custom serde lets
/// the same wire DTOs emit JSON for inspection or bincode for dispatch.
///
/// Used by py-polars tests to assert structural prune behavior (only kept
/// columns survive, stats only on predicate columns). Local files only.
#[cfg(all(feature = "parquet", feature = "json"))]
#[pyfunction]
pub fn _bench_parquet_metadata_pruned_json(
    path: &str,
    projection: Vec<String>,
    predicate: Vec<String>,
) -> PyResult<String> {
    use polars_parquet::read::read_metadata;
    use polars_utils::pl_str::PlSmallStr;

    let file = std::fs::File::open(path).map_err(|e| PyPolarsErr::Other(e.to_string()))?;
    let metadata = read_metadata(&mut BufReader::new(file)).map_err(PyPolarsErr::from)?;

    let keep: Vec<PlSmallStr> = projection.into_iter().map(PlSmallStr::from).collect();
    let pred: Vec<PlSmallStr> = predicate.into_iter().map(PlSmallStr::from).collect();
    let pruned = metadata
        .pruned(&keep, &pred)
        .map_err(|e| PyPolarsErr::Other(e.to_string()))?;

    serde_json::to_string(&pruned).map_err(|e| PyPolarsErr::Other(e.to_string()).into())
}

#[cfg(any(feature = "ipc", feature = "parquet"))]
fn fields_to_pydict(schema: &ArrowSchema, dict: &Bound<'_, PyDict>) -> PyResult<()> {
    for field in schema.iter_values() {
        let dt = Wrap(polars::prelude::DataType::from_arrow_field(field));
        dict.set_item(field.name.as_str(), &dt)?;
    }
    Ok(())
}

#[cfg(feature = "clipboard")]
#[pyfunction]
pub fn read_clipboard_string() -> PyResult<String> {
    use arboard;
    let mut clipboard =
        arboard::Clipboard::new().map_err(|e| PyPolarsErr::Other(format!("{e}")))?;
    let result = clipboard
        .get_text()
        .map_err(|e| PyPolarsErr::Other(format!("{e}")))?;
    Ok(result)
}

#[cfg(feature = "clipboard")]
#[pyfunction]
pub fn write_clipboard_string(s: &str) -> PyResult<()> {
    use arboard;
    let mut clipboard =
        arboard::Clipboard::new().map_err(|e| PyPolarsErr::Other(format!("{e}")))?;
    clipboard
        .set_text(s)
        .map_err(|e| PyPolarsErr::Other(format!("{e}")))?;
    Ok(())
}
