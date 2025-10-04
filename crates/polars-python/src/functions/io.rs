use std::io::BufReader;

#[cfg(any(feature = "ipc", feature = "parquet"))]
use polars::prelude::ArrowSchema;
use polars::prelude::PlPathRef;
use polars_io::cloud::CloudOptions;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::conversion::Wrap;
use crate::error::PyPolarsErr;
use crate::file::{EitherRustPythonFile, get_either_file};

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
    storage_options: Option<Vec<(String, String)>>,
    credential_provider: Option<Py<PyAny>>,
    retries: usize,
) -> PyResult<Bound<PyDict>> {
    use std::io::Cursor;

    use polars_error::feature_gated;
    use polars_io::pl_async::get_runtime;
    use polars_parquet::read::read_metadata;
    use polars_parquet::read::schema::read_custom_key_value_metadata;
    use polars_utils::plpath::PlPath;

    use crate::file::{PythonScanSourceInput, get_python_scan_source_input};

    let metadata = match get_python_scan_source_input(py_f, false)? {
        PythonScanSourceInput::Buffer(buf) => {
            read_metadata(&mut Cursor::new(buf)).map_err(PyPolarsErr::from)?
        },
        PythonScanSourceInput::Path(p) => {
            let cloud_options = parse_cloud_options(
                Some(p.as_ref()),
                storage_options,
                credential_provider,
                retries,
            )?;
            match p {
                PlPath::Local(local) => {
                    let file = polars_utils::open_file(&local).map_err(PyPolarsErr::from)?;
                    read_metadata(&mut BufReader::new(file)).map_err(PyPolarsErr::from)?
                },
                PlPath::Cloud(_) => {
                    use polars::prelude::ParquetObjectStore;
                    use polars_error::PolarsResult;

                    feature_gated!("cloud", {
                        get_runtime().block_on(async {
                            let mut reader = ParquetObjectStore::from_uri(
                                p.as_ref(),
                                cloud_options.as_ref(),
                                None,
                            )
                            .await?;
                            let result = reader.get_metadata().await?;
                            PolarsResult::Ok((**result).clone())
                        })
                    })
                    .map_err(PyPolarsErr::from)?
                },
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
    Ok(dict)
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

pub fn parse_cloud_options<'a>(
    first_path: Option<PlPathRef<'a>>,
    storage_options: Option<Vec<(String, String)>>,
    credential_provider: Option<Py<PyAny>>,
    retries: usize,
) -> PyResult<Option<CloudOptions>> {
    let result = if let Some(first_path) = first_path {
        #[cfg(feature = "cloud")]
        {
            use polars_io::cloud::credential_provider::PlCredentialProvider;

            use crate::prelude::parse_cloud_options;

            let first_path_url = first_path.to_str();
            let cloud_options =
                parse_cloud_options(first_path_url, storage_options.unwrap_or_default())?;

            Some(
                cloud_options
                    .with_max_retries(retries)
                    .with_credential_provider(
                        credential_provider.map(PlCredentialProvider::from_python_builder),
                    ),
            )
        }

        #[cfg(not(feature = "cloud"))]
        {
            None
        }
    } else {
        None
    };
    Ok(result)
}
