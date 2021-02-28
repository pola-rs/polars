use crate::logical_plan::ALogicalPlanBuilder;
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

/// Aggregate all the projections in an LP
pub(crate) fn agg_projection(
    logical_plan: &LogicalPlan,
    columns: &mut HashMap<String, HashSet<String, RandomState>, RandomState>,
) {
    use LogicalPlan::*;
    match logical_plan {
        Slice { input, .. } => {
            agg_projection(input, columns);
        }
        Selection { input, .. } => {
            agg_projection(input, columns);
        }
        Cache { input } => {
            agg_projection(input, columns);
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
            agg_projection(input, columns);
        }
        LocalProjection { input, .. } => {
            agg_projection(input, columns);
        }
        Sort { input, .. } => {
            agg_projection(input, columns);
        }
        Explode { input, .. } => {
            agg_projection(input, columns);
        }
        Distinct { input, .. } => {
            agg_projection(input, columns);
        }
        Aggregate { input, .. } => {
            agg_projection(input, columns);
        }
        Join {
            input_left,
            input_right,
            ..
        } => {
            agg_projection(input_left, columns);
            agg_projection(input_right, columns);
        }
        HStack { input, .. } => {
            agg_projection(input, columns);
        }
        Melt { input, .. } => {
            agg_projection(input, columns);
        }
        Udf { input, .. } => {
            agg_projection(input, columns);
        }
    }
}

/// Aggregate all the columns used in csv scans and make sure that all columns are scanned in one go.
/// Due to self joins there can be multiple Scans of the same file in a LP. We already cache the scans
/// in the PhysicalPlan, but we need to make sure that the first scan has all the columns needed.
pub struct AggScanProjection {
    pub columns: HashMap<String, HashSet<String, RandomState>, RandomState>,
}

impl AggScanProjection {
    fn finish_rewrite(
        &self,
        mut lp: ALogicalPlan,
        expr_arena: &mut Arena<AExpr>,
        lp_arena: &mut Arena<ALogicalPlan>,
        path: &str,
        with_columns: Option<Vec<String>>,
    ) -> ALogicalPlan {
        // if the original projection is less than the new one. Also project locally
        if let Some(with_columns) = with_columns {
            let agg = self.columns.get(path).unwrap();
            if with_columns.len() < agg.len() {
                let node = lp_arena.add(lp);
                lp = ALogicalPlanBuilder::new(node, expr_arena, lp_arena)
                    .project(
                        with_columns
                            .into_iter()
                            .map(|s| Expr::Column(Arc::new(s)))
                            .collect(),
                    )
                    .into_lp();
            }
        }
        lp
    }
}

impl OptimizationRule for AggScanProjection {
    fn optimize_plan(
        &mut self,
        lp_arena: &mut Arena<ALogicalPlan>,
        expr_arena: &mut Arena<AExpr>,
        node: Node,
    ) -> Option<ALogicalPlan> {
        let lp = lp_arena.get_mut(node);
        use ALogicalPlan::*;
        match lp {
            #[cfg(feature = "parquet")]
            ParquetScan { .. } => {
                let lp = std::mem::take(lp);
                if let ALogicalPlan::ParquetScan {
                    path,
                    schema,
                    predicate,
                    aggregate,
                    with_columns,
                    stop_after_n_rows,
                    cache,
                } = lp
                {
                    let new_with_columns = self
                        .columns
                        .get(&path)
                        .map(|agg| agg.iter().cloned().collect());
                    // prevent infinite loop
                    if with_columns == new_with_columns {
                        let lp = ALogicalPlan::ParquetScan {
                            path,
                            schema,
                            predicate,
                            aggregate,
                            with_columns,
                            stop_after_n_rows,
                            cache,
                        };
                        lp_arena.assign(node, lp);
                        return None;
                    }

                    let lp = ParquetScan {
                        path: path.clone(),
                        schema,
                        with_columns: new_with_columns,
                        predicate,
                        aggregate,
                        stop_after_n_rows,
                        cache,
                    };
                    Some(self.finish_rewrite(lp, expr_arena, lp_arena, &path, with_columns))
                } else {
                    unreachable!()
                }
            }
            CsvScan { .. } => {
                let lp = std::mem::take(lp);
                if let ALogicalPlan::CsvScan {
                    path,
                    schema,
                    has_header,
                    delimiter,
                    ignore_errors,
                    skip_rows,
                    stop_after_n_rows,
                    predicate,
                    aggregate,
                    with_columns,
                    cache,
                } = lp
                {
                    let new_with_columns = self
                        .columns
                        .get(&path)
                        .map(|agg| agg.iter().cloned().collect());
                    if with_columns == new_with_columns {
                        let lp = ALogicalPlan::CsvScan {
                            path,
                            schema,
                            has_header,
                            delimiter,
                            ignore_errors,
                            skip_rows,
                            stop_after_n_rows,
                            predicate,
                            aggregate,
                            with_columns,
                            cache,
                        };
                        lp_arena.assign(node, lp);
                        return None;
                    }
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
                        aggregate,
                        cache,
                    };
                    Some(self.finish_rewrite(lp, expr_arena, lp_arena, &path, with_columns))
                } else {
                    unreachable!()
                }
            }
            _ => None,
        }
    }
}
