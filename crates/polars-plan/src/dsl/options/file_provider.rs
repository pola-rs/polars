use std::hash::Hash;
use std::io::Cursor;
use std::sync::Arc;

use polars_core::frame::DataFrame;
use polars_core::prelude::Column;
use polars_core::prelude::row_encode::_get_rows_encoded_ca_unordered;
use polars_error::PolarsResult;
use polars_io::hive::HivePathFormatter;
use polars_io::utils::file::Writable;
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
    Writable(Writable),
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
    pub layout: IcebergPathProviderLayout,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Hash, PartialEq)]
pub enum IcebergPathProviderLayout {
    Simple,
    ObjectStorage { partitioned_paths: bool },
}

impl IcebergPathProvider {
    pub fn file_part_prefix_mut(&mut self) -> &mut String {
        &mut self.file_part_prefix
    }

    /// # Panics
    /// Panics if `self.file_part_prefix` is empty.
    pub fn get_path(&self, args: FileProviderArgs) -> PolarsResult<String> {
        use std::fmt::Write;

        let IcebergPathProvider {
            extension,
            file_part_prefix,
            layout,
        } = self;

        assert!(!file_part_prefix.is_empty());

        let FileProviderArgs {
            index_in_partition,
            partition_keys,
        } = args;

        let mut partition_keys_hash = None;

        if partition_keys.width() != 0 {
            let encoded =
                _get_rows_encoded_ca_unordered(PlSmallStr::EMPTY, partition_keys.columns())?;
            partition_keys_hash = Some(blake3::hash(encoded.get(0).unwrap()).to_hex());
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

        Ok(match layout {
            IcebergPathProviderLayout::Simple => path,
            IcebergPathProviderLayout::ObjectStorage { partitioned_paths } => {
                object_storage_path(path, *partitioned_paths)
            },
        })
    }
}

fn object_storage_path(path: String, partitioned_paths: bool) -> String {
    let hash = murmur3::murmur3_32(&mut Cursor::new(path.as_bytes()), 0).unwrap();
    let separator = if partitioned_paths { '/' } else { '-' };

    format!(
        "{:04b}/{:04b}/{:04b}/{:08b}{separator}{path}",
        (hash >> 16) & 0xf,
        (hash >> 12) & 0xf,
        (hash >> 8) & 0xf,
        hash & 0xff,
    )
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use polars_core::frame::DataFrame;
    use polars_core::prelude::Column;

    use super::{
        FileProviderArgs, IcebergPathProvider, IcebergPathProviderLayout, object_storage_path,
    };

    fn iceberg_path(partition_keys: DataFrame) -> String {
        IcebergPathProvider {
            extension: "parquet".into(),
            file_part_prefix: "part".to_owned(),
            layout: IcebergPathProviderLayout::Simple,
        }
        .get_path(FileProviderArgs {
            index_in_partition: 0,
            partition_keys: Arc::new(partition_keys),
        })
        .unwrap()
    }

    #[test]
    fn iceberg_object_storage_paths_match_reference_hashes() {
        for (file_name, expected_hash) in [
            ("a", "0101/0110/1001/10110010"),
            ("b", "1110/0111/1110/00000011"),
            ("c", "0010/1101/0110/01011111"),
            ("d", "1001/0001/0100/01110011"),
        ] {
            assert_eq!(
                object_storage_path(file_name.to_owned(), true),
                format!("{expected_hash}/{file_name}")
            );
        }

        assert_eq!(
            object_storage_path("test.parquet".to_owned(), false),
            "0110/1010/0011/11101000-test.parquet"
        );
    }

    #[test]
    fn iceberg_partition_paths_support_arbitrary_binary() {
        let binary_a = Column::new("key".into(), [&b"\xff"[..]]);
        let binary_b = Column::new("key".into(), [&b"\xfe"[..]]);

        let path_a = iceberg_path(DataFrame::new(1, vec![binary_a]).unwrap());
        let path_b = iceberg_path(DataFrame::new(1, vec![binary_b]).unwrap());

        assert_ne!(path_a, path_b);
    }

    #[test]
    fn iceberg_compound_partition_paths_are_unambiguous() {
        let keys_a = DataFrame::new(
            1,
            vec![
                Column::new("first".into(), ["a"]),
                Column::new("second".into(), ["\u{1}b"]),
            ],
        )
        .unwrap();
        let keys_b = DataFrame::new(
            1,
            vec![
                Column::new("first".into(), ["a\u{1}"]),
                Column::new("second".into(), ["b"]),
            ],
        )
        .unwrap();

        assert_ne!(iceberg_path(keys_a), iceberg_path(keys_b));
    }
}
