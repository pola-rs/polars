use std::hash::Hash;
use std::sync::Arc;

use polars_core::frame::DataFrame;
use polars_core::prelude::{Column, DataType};
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
    Iceberg(IcebergPathProvider),
    Function(FileProviderFunction),
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Hash, PartialEq)]
pub struct HivePathProvider {
    pub extension: PlSmallStr,
}

impl FileProviderType {
    /// Get a mutable reference to the file part prefix for this file provider.
    ///
    /// File part prefixes are inserted after the partition prefix, before the file part number.
    ///
    /// # Returns
    /// Returns `None` if this file provider does not support attaching file part prefixes.
    pub fn file_part_prefix_mut(&mut self) -> Option<&mut String> {
        use FileProviderType::*;

        match self {
            Iceberg(p) => Some(p.file_part_prefix_mut()),
            Hive(_) | Function(_) => None,
        }
    }

    pub fn get_path_or_file(&self, args: FileProviderArgs) -> PolarsResult<FileProviderReturn> {
        use FileProviderType::*;

        match self {
            Hive(p) => p.get_path(args).map(FileProviderReturn::Path),
            Iceberg(p) => p.get_path(args).map(FileProviderReturn::Path),
            Function(p) => p.get_path_or_file(args),
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

        let mut path = String::new();

        let partition_keys: &[Column] = partition_keys.columns();

        write!(&mut path, "{}", HivePathFormatter::new(partition_keys)).unwrap();

        assert!(index_in_partition <= 0xffff_ffff);

        write!(&mut path, "{index_in_partition:08x}.{extension}").unwrap();

        Ok(path)
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Hash, PartialEq)]
pub struct IcebergPathProvider {
    pub extension: PlSmallStr,
    pub file_part_prefix: String,
}

impl IcebergPathProvider {
    pub fn file_part_prefix_mut(&mut self) -> &mut String {
        &mut self.file_part_prefix
    }

    /// # Panics
    /// Panics if `self.file_part_prefix` is `None`.
    pub fn get_path(&self, args: FileProviderArgs) -> PolarsResult<String> {
        use std::fmt::Write;

        let IcebergPathProvider {
            extension,
            file_part_prefix,
        } = self;

        assert!(!file_part_prefix.is_empty());

        let FileProviderArgs {
            index_in_partition,
            partition_keys,
        } = args;

        let mut partition_keys_hash = None;

        if partition_keys.width() != 0 {
            let mut hasher = blake3::Hasher::new();

            for column in partition_keys.columns() {
                let column = column.cast(&DataType::String).unwrap();

                let value = column.str().unwrap().get(0);

                hasher.update(&[value.is_some() as u8]);
                hasher.update(value.unwrap_or_default().as_bytes());
            }

            partition_keys_hash = Some(hasher.finalize().to_hex());
        }

        let partition_key_prefix: &str = partition_keys_hash.as_ref().map_or("", |x| &x[..32]);

        let mut path = String::with_capacity(
            partition_key_prefix.len() + file_part_prefix.len() + 8 + 1 + extension.len(),
        );

        assert!(index_in_partition <= 0xffff_ffff);

        write!(
            &mut path,
            "{partition_key_prefix}{file_part_prefix}{index_in_partition:08x}.{extension}"
        )
        .unwrap();

        Ok(path)
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
