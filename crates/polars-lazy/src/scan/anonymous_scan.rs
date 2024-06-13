use polars_core::prelude::*;
use polars_io::RowIndex;

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
        let mut lf: LazyFrame = DslBuilder::anonymous_scan(
            function,
            args.schema,
            args.infer_schema_length,
            args.skip_rows,
            args.n_rows,
            args.name,
        )?
        .build()
        .into();

        if let Some(rc) = args.row_index {
            lf = lf.with_row_index(&rc.name, Some(rc.offset))
        };

        Ok(lf)
    }
}
