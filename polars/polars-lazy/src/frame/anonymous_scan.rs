use crate::prelude::*;
use polars_core::prelude::*;
use polars_io::RowCount;

#[derive(Clone)]
pub struct ScanArgsAnonymous {
    pub skip_rows: Option<usize>,
    pub n_rows: Option<usize>,
    pub infer_schema_length: Option<usize>,
    pub schema: Option<Schema>,
    pub row_count: Option<RowCount>,
    pub name: &'static str,
}

impl Default for ScanArgsAnonymous {
    fn default() -> Self {
        Self {
            skip_rows: None,
            n_rows: None,
            infer_schema_length: None,
            schema: None,
            row_count: None,
            name: "ANONYMOUS SCAN",
        }
    }
}
impl LazyFrame {
    pub fn anonymous_scan(
        function: Arc<dyn AnonymousScan>,
        args: ScanArgsAnonymous,
    ) -> Result<Self> {
        let mut lf: LazyFrame = LogicalPlanBuilder::anonymous_scan(
            function,
            args.schema,
            args.infer_schema_length,
            args.skip_rows,
            args.n_rows,
            args.name,
        )?
        .build()
        .into();

        if let Some(n_rows) = args.n_rows {
            lf = lf.slice(0, n_rows as IdxSize);
        };

        if let Some(rc) = args.row_count {
            lf = lf.with_row_count(&rc.name, Some(rc.offset))
        };

        Ok(lf)
    }
}
