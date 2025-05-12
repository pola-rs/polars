use polars_core::prelude::*;
use polars_io::{HiveOptions, RowIndex};
use polars_utils::slice_enum::Slice;

use crate::prelude::*;

#[derive(Clone)]
pub struct ScanArgsAnonymous {
    pub infer_schema_length: Option<usize>,
    pub schema: Option<SchemaRef>,
    pub skip_rows: Option<usize>,
    pub n_rows: Option<usize>,
    pub row_index: Option<RowIndex>,
    pub name: &'static str,
}

impl Default for ScanArgsAnonymous {
    fn default() -> Self {
        Self {
            infer_schema_length: None,
            skip_rows: None,
            n_rows: None,
            schema: None,
            row_index: None,
            name: "ANONYMOUS SCAN",
        }
    }
}
impl LazyFrame {
    pub fn anonymous_scan(
        function: Arc<dyn AnonymousScan>,
        args: ScanArgsAnonymous,
    ) -> PolarsResult<Self> {
        let schema = match args.schema {
            Some(s) => s,
            None => function.schema(args.infer_schema_length)?,
        };

        let mut lf: LazyFrame = DslBuilder::anonymous_scan(
            function,
            AnonymousScanOptions {
                skip_rows: args.skip_rows,
                fmt_str: args.name,
            },
            UnifiedScanArgs {
                schema: Some(schema),
                cloud_options: None,
                hive_options: HiveOptions::new_disabled(),
                rechunk: false,
                cache: false,
                glob: false,
                projection: None,
                row_index: None,
                pre_slice: args.n_rows.map(|len| Slice::Positive { offset: 0, len }),
                cast_columns_policy: CastColumnsPolicy::ERROR_ON_MISMATCH,
                missing_columns_policy: MissingColumnsPolicy::Raise,
                include_file_paths: None,
            },
        )?
        .build()
        .into();

        if let Some(rc) = args.row_index {
            lf = lf.with_row_index(rc.name.clone(), Some(rc.offset))
        };

        Ok(lf)
    }
}
