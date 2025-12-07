use std::fmt::Write;
use std::sync::Arc;

use polars_core::prelude::{Column, DataType};
use polars_error::PolarsResult;
use polars_io::cloud::CloudOptions;
use polars_io::pl_async;
use polars_io::utils::HIVE_VALUE_ENCODE_CHARSET;
use polars_io::utils::file::Writeable;
use polars_plan::dsl::sink2::{FileProviderReturn, FileProviderType};
use polars_plan::prelude::sink2::FileProviderArgs;
use polars_utils::plpath::PlPath;

pub struct FileProvider {
    pub base_path: PlPath,
    pub cloud_options: Option<Arc<CloudOptions>>,
    pub provider_type: FileProviderType,
}

impl FileProvider {
    pub async fn open_file(&self, args: FileProviderArgs) -> PolarsResult<Writeable> {
        let provided_path: String = match &self.provider_type {
            FileProviderType::Hive { extension } => {
                let FileProviderArgs {
                    index_in_partition,
                    partition_keys,
                } = args;

                let mut partition_parts = String::new();

                let partition_keys: &[Column] = partition_keys.get_columns();

                write!(
                    &mut partition_parts,
                    "{}",
                    HivePathFormatter::new(partition_keys)
                )
                .unwrap();

                assert!(index_in_partition <= 0xffff_ffff);

                write!(&mut partition_parts, "{index_in_partition:08x}.{extension}").unwrap();

                partition_parts
            },

            FileProviderType::Function(f) => {
                let f = f.clone();

                let out = pl_async::get_runtime()
                    .spawn_blocking(move || f.get_file(args))
                    .await
                    .unwrap()?;

                match out {
                    FileProviderReturn::Path(p) => p,
                    FileProviderReturn::Writeable(v) => return Ok(v),
                }
            },

            FileProviderType::Legacy(_) => unreachable!(),
        };

        let path = self.base_path.as_ref().join(&provided_path);
        let path = path.as_ref();

        if let Some(path) = path.as_local_path().and_then(|p| p.parent()) {
            // Ignore errors from directory creation - the `Writeable::try_new()` below will raise
            // appropriate errors.
            let _ = tokio::fs::DirBuilder::new()
                .recursive(true)
                .create(path)
                .await;
        }

        Writeable::try_new(path, self.cloud_options.as_deref())
    }
}

/// # Panics
/// The `Display` impl of this will panic if a column has non-unit length.
pub struct HivePathFormatter<'a> {
    keys: &'a [Column],
}

impl<'a> HivePathFormatter<'a> {
    pub fn new(keys: &'a [Column]) -> Self {
        Self { keys }
    }
}

impl std::fmt::Display for HivePathFormatter<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for column in self.keys {
            assert_eq!(column.len(), 1);
            let column = column.cast(&DataType::String).unwrap();

            let key = column.name();
            let value = percent_encoding::percent_encode(
                column
                    .str()
                    .unwrap()
                    .get(0)
                    .unwrap_or("__HIVE_DEFAULT_PARTITION__")
                    .as_bytes(),
                HIVE_VALUE_ENCODE_CHARSET,
            );

            write!(f, "{key}={value}/")?
        }

        Ok(())
    }
}
