use std::hash::Hash;
use std::sync::Arc;

use polars_core::frame::DataFrame;
use polars_core::prelude::Column;
use polars_error::PolarsResult;
use polars_io::hive::HivePathFormatter;
use polars_io::utils::file::Writeable;
use polars_utils::pl_str::PlSmallStr;

use crate::prelude::PlanCallback;

#[derive(Debug)]
pub struct FileProviderArgs {
    pub index_in_partition: usize,
    /// Will always have a height of 1.
    pub partition_keys: Arc<DataFrame>,
}

pub enum FileProviderReturn {
    Path(String),
    Writeable(Writeable),
}

pub type FileProviderFunction = PlanCallback<FileProviderArgs, FileProviderReturn>;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Hash, PartialEq)]
pub enum FileProviderType {
    Hive(HivePathProvider),
    Function(FileProviderFunction),
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Hash, PartialEq)]
pub struct HivePathProvider {
    pub extension: PlSmallStr,
}

impl FileProviderType {
    pub fn get_path_or_file(&self, args: FileProviderArgs) -> PolarsResult<FileProviderReturn> {
        match self {
            Self::Hive(v) => v.get_path(args).map(FileProviderReturn::Path),
            Self::Function(v) => v.get_path_or_file(args),
        }
    }
}

impl HivePathProvider {
    pub fn get_path(&self, args: FileProviderArgs) -> PolarsResult<String> {
        use std::fmt::Write;

        let HivePathProvider { extension } = self;

        let FileProviderArgs {
            index_in_partition,
            partition_keys,
        } = args;

        let mut partition_parts = String::new();

        let partition_keys: &[Column] = partition_keys.columns();

        write!(
            &mut partition_parts,
            "{}",
            HivePathFormatter::new(partition_keys)
        )
        .unwrap();

        assert!(index_in_partition <= 0xffff_ffff);

        write!(&mut partition_parts, "{index_in_partition:08x}.{extension}").unwrap();

        Ok(partition_parts)
    }
}

impl FileProviderFunction {
    pub fn get_path_or_file(&self, args: FileProviderArgs) -> PolarsResult<FileProviderReturn> {
        match self {
            Self::Rust(func) => (func)(args),
            #[cfg(feature = "python")]
            Self::Python(object) => pyo3::Python::attach(|py| {
                use polars_error::PolarsError;
                use pyo3::intern;
                use pyo3::types::{PyAnyMethods, PyDict};

                let FileProviderArgs {
                    index_in_partition,
                    partition_keys,
                } = args;

                let convert_registry =
                    polars_utils::python_convert_registry::get_python_convert_registry();

                let partition_keys = convert_registry
                    .to_py
                    .df_to_wrapped_pydf(partition_keys.as_ref())
                    .map_err(PolarsError::from)?;

                let kwargs = PyDict::new(py);
                kwargs.set_item(intern!(py, "index_in_partition"), index_in_partition)?;
                kwargs.set_item(intern!(py, "partition_keys"), partition_keys)?;

                let args_dataclass = convert_registry.py_file_provider_args_dataclass().call(
                    py,
                    (),
                    Some(&kwargs),
                )?;

                let out = object.call1(py, (args_dataclass,))?;
                let out = (convert_registry.from_py.file_provider_result)(out)?;
                let out: FileProviderReturn = *out.downcast().unwrap();

                PolarsResult::Ok(out)
            }),
        }
    }
}
