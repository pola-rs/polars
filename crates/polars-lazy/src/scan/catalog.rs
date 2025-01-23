use polars_core::error::{feature_gated, polars_bail, PolarsResult};
use polars_io::catalog::schema::table_info_to_schemas;
use polars_io::catalog::unity::models::{DataSourceFormat, TableInfo};
use polars_io::cloud::CloudOptions;

use crate::frame::LazyFrame;

impl LazyFrame {
    pub fn scan_catalog_table(
        table_info: &TableInfo,
        cloud_options: Option<CloudOptions>,
    ) -> PolarsResult<Self> {
        let Some(data_source_format) = &table_info.data_source_format else {
            polars_bail!(ComputeError: "scan_catalog_table requires Some(_) for data_source_format")
        };

        let Some(storage_location) = table_info.storage_location.as_deref() else {
            polars_bail!(ComputeError: "scan_catalog_table requires Some(_) for storage_location")
        };

        match data_source_format {
            DataSourceFormat::Parquet => feature_gated!("parquet", {
                use polars_io::HiveOptions;

                use crate::frame::ScanArgsParquet;
                let (schema, hive_schema) = table_info_to_schemas(table_info)?;

                let args = ScanArgsParquet {
                    schema,
                    cloud_options,
                    hive_options: HiveOptions {
                        schema: hive_schema,
                        ..Default::default()
                    },
                    ..Default::default()
                };

                Self::scan_parquet(storage_location, args)
            }),
            DataSourceFormat::Csv => feature_gated!("csv", {
                use crate::frame::{LazyCsvReader, LazyFileListReader};
                let (schema, _) = table_info_to_schemas(table_info)?;

                LazyCsvReader::new(storage_location)
                    .with_schema(schema)
                    .finish()
            }),
            v => polars_bail!(
                ComputeError:
                "not yet supported data_source_format: {:?}",
                v
            ),
        }
    }
}
