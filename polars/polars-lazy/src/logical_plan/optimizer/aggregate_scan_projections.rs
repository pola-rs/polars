use crate::logical_plan::optimizer::stack_opt::OptimizationRule;
use crate::logical_plan::ALogicalPlanBuilder;
use crate::prelude::*;
use polars_core::datatypes::{PlHashMap, PlHashSet};
use polars_core::prelude::Schema;
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn process_with_columns(
    path: &Path,
    with_columns: &Option<Vec<String>>,
    columns: &mut PlHashMap<PathBuf, PlHashSet<(usize, String)>>,
    schema: &Schema,
) {
    let cols = columns
        .entry(path.to_owned())
        .or_insert_with(PlHashSet::new);

    match with_columns {
        // add only the projected columns
        Some(with_columns) => {
            cols.extend(with_columns.iter().enumerate().map(|t| (t.0, t.1.clone())));
        }
        // no projection, so we must take all columns
        None => {
            cols.extend(
                schema
                    .iter_names()
                    .enumerate()
                    .map(|t| (t.0, t.1.to_string())),
            );
        }
    }
}

/// Aggregate all the projections in an LP
pub(crate) fn agg_projection(
    root: Node,
    // The hashmap maps files to a hashset over column names. (There is a usize to be able to sort them later)
    columns: &mut PlHashMap<PathBuf, PlHashSet<(usize, String)>>,
    lp_arena: &Arena<ALogicalPlan>,
) {
    use ALogicalPlan::*;
    match lp_arena.get(root) {
        #[cfg(feature = "csv-file")]
        CsvScan {
            path,
            options,
            schema,
            ..
        } => {
            process_with_columns(path, &options.with_columns, columns, schema);
        }
        #[cfg(feature = "parquet")]
        ParquetScan {
            path,
            options,
            schema,
            ..
        } => {
            process_with_columns(path, &options.with_columns, columns, schema);
        }
        #[cfg(feature = "ipc")]
        IpcScan {
            path,
            options,
            schema,
            ..
        } => {
            process_with_columns(path, &options.with_columns, columns, schema);
        }
        DataFrameScan { .. } => (),
        lp => {
            for input in lp.get_inputs() {
                agg_projection(input, columns, lp_arena)
            }
        }
    }
}

/// Aggregate all the columns used in csv scans and make sure that all columns are scanned in one go.
/// Due to self joins there can be multiple Scans of the same file in a LP. We already cache the scans
/// in the PhysicalPlan, but we need to make sure that the first scan has all the columns needed.
pub struct AggScanProjection {
    pub columns: PlHashMap<PathBuf, PlHashSet<(usize, String)>>,
}

impl AggScanProjection {
    fn finish_rewrite(
        &self,
        mut lp: ALogicalPlan,
        expr_arena: &mut Arena<AExpr>,
        lp_arena: &mut Arena<ALogicalPlan>,
        path: &Path,
        with_columns: Option<Vec<String>>,
    ) -> ALogicalPlan {
        // if the original projection is less than the new one. Also project locally
        if let Some(with_columns) = with_columns {
            let agg = self.columns.get(path).unwrap();
            if with_columns.len() < agg.len() {
                let node = lp_arena.add(lp);

                let projections = with_columns
                    .into_iter()
                    .map(|s| expr_arena.add(AExpr::Column(Arc::from(s))))
                    .collect();

                lp = ALogicalPlanBuilder::new(node, expr_arena, lp_arena)
                    .project(projections)
                    .build();
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
        match lp {
            #[cfg(feature = "ipc")]
            ALogicalPlan::IpcScan { .. } => {
                let lp = std::mem::take(lp);
                if let ALogicalPlan::IpcScan {
                    path,
                    schema,
                    output_schema,
                    predicate,
                    aggregate,
                    mut options,
                } = lp
                {
                    let with_columns = self.columns.get(&path).map(|agg| {
                        let mut columns = agg.iter().cloned().collect::<Vec<_>>();
                        // make sure that the columns are sorted because they come from a hashmap
                        columns.sort_unstable_by_key(|k| k.0);
                        columns.into_iter().map(|k| k.1).collect()
                    });
                    // prevent infinite loop
                    if options.with_columns == with_columns {
                        let lp = ALogicalPlan::IpcScan {
                            path,
                            schema,
                            output_schema,
                            predicate,
                            aggregate,
                            options,
                        };
                        lp_arena.replace(node, lp);
                        return None;
                    }

                    options.with_columns = with_columns;
                    let lp = ALogicalPlan::IpcScan {
                        path: path.clone(),
                        schema,
                        output_schema,
                        predicate,
                        aggregate,
                        options: options.clone(),
                    };
                    Some(self.finish_rewrite(lp, expr_arena, lp_arena, &path, options.with_columns))
                } else {
                    unreachable!()
                }
            }
            #[cfg(feature = "parquet")]
            ALogicalPlan::ParquetScan { .. } => {
                let lp = std::mem::take(lp);
                if let ALogicalPlan::ParquetScan {
                    path,
                    schema,
                    output_schema,
                    predicate,
                    aggregate,
                    mut options,
                } = lp
                {
                    let mut with_columns = self.columns.get(&path).map(|agg| {
                        let mut columns = agg.iter().cloned().collect::<Vec<_>>();
                        // make sure that the columns are sorted because they come from a hashmap
                        columns.sort_unstable_by_key(|k| k.0);
                        columns.into_iter().map(|k| k.1).collect()
                    });
                    // prevent infinite loop
                    if options.with_columns == with_columns {
                        let lp = ALogicalPlan::ParquetScan {
                            path,
                            schema,
                            output_schema,
                            predicate,
                            aggregate,
                            options,
                        };
                        lp_arena.replace(node, lp);
                        return None;
                    }
                    std::mem::swap(&mut options.with_columns, &mut with_columns);

                    let lp = ALogicalPlan::ParquetScan {
                        path: path.clone(),
                        schema,
                        output_schema,
                        predicate,
                        aggregate,
                        options,
                    };
                    Some(self.finish_rewrite(lp, expr_arena, lp_arena, &path, with_columns))
                } else {
                    unreachable!()
                }
            }
            #[cfg(feature = "csv-file")]
            ALogicalPlan::CsvScan { .. } => {
                let lp = std::mem::take(lp);
                if let ALogicalPlan::CsvScan {
                    path,
                    schema,
                    output_schema,
                    mut options,
                    predicate,
                    aggregate,
                } = lp
                {
                    let with_columns = self.columns.get(&path).map(|agg| {
                        let mut columns = agg.iter().cloned().collect::<Vec<_>>();
                        // make sure that the columns are sorted because they come from a hashmap
                        columns.sort_unstable_by_key(|k| k.0);
                        columns.into_iter().map(|k| k.1).collect()
                    });
                    if options.with_columns == with_columns {
                        let lp = ALogicalPlan::CsvScan {
                            path,
                            schema,
                            output_schema,
                            options,
                            predicate,
                            aggregate,
                        };
                        lp_arena.replace(node, lp);
                        return None;
                    }
                    options.with_columns = with_columns;
                    let lp = ALogicalPlan::CsvScan {
                        path: path.clone(),
                        schema,
                        output_schema,
                        options: options.clone(),
                        predicate,
                        aggregate,
                    };
                    Some(self.finish_rewrite(lp, expr_arena, lp_arena, &path, options.with_columns))
                } else {
                    unreachable!()
                }
            }
            _ => None,
        }
    }
}
