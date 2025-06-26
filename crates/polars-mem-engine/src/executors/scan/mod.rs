#[cfg(feature = "python")]
mod python_scan;

use std::mem;

use polars_plan::global::_set_n_rows_for_scan;
use polars_utils::slice_enum::Slice;

#[cfg(feature = "python")]
pub(crate) use self::python_scan::*;
use super::*;
use crate::ScanPredicate;
use crate::prelude::*;

/// Producer of an in memory DataFrame
pub struct DataFrameExec {
    pub(crate) df: Arc<DataFrame>,
    pub(crate) projection: Option<Vec<PlSmallStr>>,
}

impl Executor for DataFrameExec {
    fn execute(&mut self, _state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        let df = mem::take(&mut self.df);
        let mut df = Arc::try_unwrap(df).unwrap_or_else(|df| (*df).clone());

        // projection should be before selection as those are free
        // TODO: this is only the case if we don't create new columns
        if let Some(projection) = &self.projection {
            df = df.select(projection.iter().cloned())?;
        }

        Ok(match _set_n_rows_for_scan(None) {
            Some(limit) => df.head(Some(limit)),
            None => df,
        })
    }
}

pub(crate) struct AnonymousScanExec {
    pub(crate) function: Arc<dyn AnonymousScan>,
    pub(crate) unified_scan_args: Box<UnifiedScanArgs>,
    pub(crate) file_info: FileInfo,
    pub(crate) predicate: Option<ScanPredicate>,
    pub(crate) output_schema: Option<SchemaRef>,
    pub(crate) predicate_has_windows: bool,
}

impl Executor for AnonymousScanExec {
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        let mut args = AnonymousScanArgs {
            n_rows: self.unified_scan_args.pre_slice.clone().map(|x| {
                assert!(matches!(x, Slice::Positive { offset: 0, .. }));

                x.len()
            }),
            with_columns: self.unified_scan_args.projection.clone(),
            schema: self.file_info.schema.clone(),
            output_schema: self.output_schema.clone(),
            predicate: None,
        };
        if self.predicate.is_some() {
            state.insert_has_window_function_flag()
        }

        match (self.function.allows_predicate_pushdown(), &self.predicate) {
            (true, Some(predicate)) => state.record(
                || {
                    args.predicate = predicate.predicate.as_expression().cloned();
                    self.function.scan(args)
                },
                "anonymous_scan".into(),
            ),
            (false, Some(predicate)) => state.record(
                || {
                    let mut df = self.function.scan(args)?;
                    let s = predicate.predicate.evaluate(&df, state)?;
                    if self.predicate_has_windows {
                        state.clear_window_expr_cache()
                    }
                    let mask = s.bool().map_err(
                        |_| polars_err!(ComputeError: "filter predicate was not of type boolean"),
                    )?;
                    df = df.filter(mask)?;

                    Ok(df)
                },
                "anonymous_scan".into(),
            ),
            _ => state.record(|| self.function.scan(args), "anonymous_scan".into()),
        }
    }
}
