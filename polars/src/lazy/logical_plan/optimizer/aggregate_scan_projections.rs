use crate::lazy::prelude::*;
use crate::prelude::*;
use ahash::RandomState;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

fn process_with_columns(
    path: &str,
    with_columns: &Option<Vec<String>>,
    columns: &mut HashMap<String, HashSet<String, RandomState>, RandomState>,
) {
    if let Some(with_columns) = &with_columns {
        let cols = columns
            .entry(path.to_string())
            .or_insert_with(|| HashSet::with_capacity_and_hasher(256, RandomState::default()));
        cols.extend(with_columns.iter().cloned());
    }
}

/// Aggregate all the columns used in csv scans and make sure that all columns are scanned in on go.
/// Due to self joins there can be multiple Scans of the same file in a LP. We already cache the scans
/// in the PhysicalPlan, but we need to make sure that the first scan has all the columns needed.
pub struct AggScanProjection {}

impl AggScanProjection {
    /// Hashmap
    ///     keys: file path
    ///     values: Projected column names
    pub(crate) fn agg_projection(
        &self,
        logical_plan: &LogicalPlan,
        columns: &mut HashMap<String, HashSet<String, RandomState>, RandomState>,
    ) {
        use LogicalPlan::*;
        match logical_plan {
            Selection { input, .. } => {
                self.agg_projection(input, columns);
            }
            Cache { input } => {
                self.agg_projection(input, columns);
            }
            CsvScan {
                path, with_columns, ..
            } => {
                process_with_columns(&path, &with_columns, columns);
            }
            #[cfg(feature = "parquet")]
            ParquetScan {
                path, with_columns, ..
            } => {
                process_with_columns(&path, &with_columns, columns);
            }
            DataFrameScan { .. } => (),
            Projection { input, .. } => {
                self.agg_projection(input, columns);
            }
            LocalProjection { input, .. } => {
                self.agg_projection(input, columns);
            }
            Sort { input, .. } => {
                self.agg_projection(input, columns);
            }
            Explode { input, .. } => {
                self.agg_projection(input, columns);
            }
            Distinct { input, .. } => {
                self.agg_projection(input, columns);
            }
            Aggregate { input, .. } => {
                self.agg_projection(input, columns);
            }
            Join {
                input_left,
                input_right,
                ..
            } => {
                self.agg_projection(input_left, columns);
                self.agg_projection(input_right, columns);
            }
            HStack { input, .. } => {
                self.agg_projection(input, columns);
            }
        }
    }

    fn finish_rewrite(
        &self,
        mut lp: LogicalPlan,
        path: &str,
        with_columns: Option<Vec<String>>,
        columns: &HashMap<String, HashSet<String, RandomState>, RandomState>,
    ) -> LogicalPlan {
        // if the original projection is less than the new one. Also project locally
        if let Some(with_columns) = with_columns {
            let agg = columns.get(path).unwrap();
            if with_columns.len() < agg.len() {
                lp = LogicalPlanBuilder::from(lp)
                    .project(
                        with_columns
                            .into_iter()
                            .map(|s| Expr::Column(Arc::new(s)))
                            .collect(),
                    )
                    .build();
            }
        }
        lp
    }

    fn rewrite_plan(
        &self,
        logical_plan: LogicalPlan,
        columns: &HashMap<String, HashSet<String, RandomState>, RandomState>,
    ) -> Result<LogicalPlan> {
        use LogicalPlan::*;
        match logical_plan {
            Cache { input } => {
                let input = Box::new(self.rewrite_plan(*input, columns)?);
                Ok(Cache { input })
            }
            Selection { input, predicate } => {
                let input = Box::new(self.rewrite_plan(*input, columns)?);
                Ok(Selection { input, predicate })
            }
            #[cfg(feature = "parquet")]
            ParquetScan {
                path,
                schema,
                predicate,
                with_columns,
                stop_after_n_rows,
                cache,
            } => {
                let new_with_columns = match columns.get(&path) {
                    Some(agg) => Some(agg.iter().cloned().collect()),
                    None => None,
                };
                let lp = ParquetScan {
                    path: path.clone(),
                    schema,
                    with_columns: new_with_columns,
                    predicate,
                    stop_after_n_rows,
                    cache,
                };
                Ok(self.finish_rewrite(lp, &path, with_columns, columns))
            }
            CsvScan {
                path,
                schema,
                has_header,
                delimiter,
                ignore_errors,
                skip_rows,
                stop_after_n_rows,
                predicate,
                with_columns,
                cache,
            } => {
                let new_with_columns = match columns.get(&path) {
                    Some(agg) => Some(agg.iter().cloned().collect()),
                    None => None,
                };
                let lp = CsvScan {
                    path: path.clone(),
                    schema,
                    has_header,
                    delimiter,
                    ignore_errors,
                    skip_rows,
                    stop_after_n_rows,
                    with_columns: new_with_columns,
                    predicate,
                    cache,
                };
                Ok(self.finish_rewrite(lp, &path, with_columns, columns))
            }
            DataFrameScan { .. } => Ok(logical_plan),
            Projection {
                expr,
                input,
                schema,
            } => {
                let input = Box::new(self.rewrite_plan(*input, columns)?);
                Ok(Projection {
                    expr,
                    input,
                    schema,
                })
            }
            LocalProjection {
                expr,
                input,
                schema,
            } => {
                let input = Box::new(self.rewrite_plan(*input, columns)?);
                Ok(LocalProjection {
                    expr,
                    input,
                    schema,
                })
            }
            Sort {
                input,
                by_column,
                reverse,
            } => {
                let input = Box::new(self.rewrite_plan(*input, columns)?);
                Ok(Sort {
                    input,
                    by_column,
                    reverse,
                })
            }
            Explode { input, column } => {
                let input = Box::new(self.rewrite_plan(*input, columns)?);
                Ok(Explode { input, column })
            }
            Distinct {
                input,
                maintain_order,
                subset,
            } => {
                let input = self.rewrite_plan(*input, columns)?;
                Ok(Distinct {
                    input: Box::new(input),
                    maintain_order,
                    subset,
                })
            }
            Aggregate {
                input,
                keys,
                aggs,
                schema,
            } => {
                let input = Box::new(self.rewrite_plan(*input, columns)?);
                Ok(Aggregate {
                    input,
                    keys,
                    aggs,
                    schema,
                })
            }
            Join {
                input_left,
                input_right,
                schema,
                how,
                left_on,
                right_on,
                allow_par,
                force_par,
            } => {
                let input_left = Box::new(self.rewrite_plan(*input_left, columns)?);
                let input_right = Box::new(self.rewrite_plan(*input_right, columns)?);
                Ok(Join {
                    input_left,
                    input_right,
                    schema,
                    how,
                    left_on,
                    right_on,
                    allow_par,
                    force_par,
                })
            }
            HStack { input, exprs, .. } => {
                let input = self.rewrite_plan(*input, columns)?;
                Ok(LogicalPlanBuilder::from(input).with_columns(exprs).build())
            }
        }
    }
}

impl Optimize for AggScanProjection {
    fn optimize(&self, logical_plan: LogicalPlan) -> Result<LogicalPlan> {
        // First aggregate all the columns projected in scan
        let mut agg = HashMap::with_capacity_and_hasher(32, RandomState::default());
        self.agg_projection(&logical_plan, &mut agg);
        // and then make sure that all scans of the same files have the same columns. Such that the one that get executed first has all the columns.
        // The first scan gets cached
        self.rewrite_plan(logical_plan, &agg)
    }
}
