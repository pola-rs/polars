use crate::logical_plan::optimizer::stack_opt::OptimizationRule;
use crate::logical_plan::ALogicalPlanBuilder;
use crate::prelude::*;
use polars_core::datatypes::PlHashMap;
use polars_core::prelude::{PlIndexSet, Schema};
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn process_with_columns(
    path: &Path,
    with_columns: &Option<Arc<Vec<String>>>,
    columns: &mut PlHashMap<PathBuf, PlIndexSet<String>>,
    schema: &Schema,
) {
    let cols = columns
        .entry(path.to_owned())
        .or_insert_with(|| PlIndexSet::with_capacity_and_hasher(32, Default::default()));

    match with_columns {
        // add only the projected columns
        Some(with_columns) => cols.extend(with_columns.iter().cloned()),
        // no projection, so we must take all columns
        None => {
            cols.extend(schema.iter_names().map(|t| t.to_string()));
        }
    }
}

/// Aggregate all the projections in an LP
pub(crate) fn agg_projection(
    root: Node,
    // The hashmap maps files to a hashset over column names.
    columns: &mut PlHashMap<PathBuf, PlIndexSet<String>>,
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
    columns: PlHashMap<PathBuf, Arc<Vec<String>>>,
}

impl AggScanProjection {
    pub(crate) fn new(columns: PlHashMap<PathBuf, PlIndexSet<String>>) -> Self {
        let new_columns_mapping = columns
            .into_iter()
            .map(|(k, agg)| {
                let columns = agg.iter().cloned().collect::<Vec<_>>();
                (k, Arc::new(columns))
            })
            .collect();
        Self {
            columns: new_columns_mapping,
        }
    }

    fn finish_rewrite(
        &self,
        mut lp: ALogicalPlan,
        expr_arena: &mut Arena<AExpr>,
        lp_arena: &mut Arena<ALogicalPlan>,
        path: &Path,
        with_columns: Option<Arc<Vec<String>>>,
    ) -> ALogicalPlan {
        // if the original projection is less than the new one. Also project locally
        if let Some(mut with_columns) = with_columns {
            let agg = self.columns.get(path).unwrap();
            if with_columns.len() < agg.len() {
                let node = lp_arena.add(lp);

                let projections = std::mem::take(Arc::make_mut(&mut with_columns))
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

    fn extract_columns(&mut self, path: &PathBuf) -> Option<Arc<Vec<String>>> {
        self.columns.get(path).cloned()
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
                    let with_columns = self.extract_columns(&path);
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
                    let mut with_columns = self.extract_columns(&path);
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
                    let with_columns = self.extract_columns(&path);
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
