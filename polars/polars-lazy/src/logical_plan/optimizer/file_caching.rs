use crate::logical_plan::optimizer::stack_opt::OptimizationRule;
use crate::logical_plan::ALogicalPlanBuilder;
use crate::prelude::*;
use polars_core::datatypes::PlHashMap;
use polars_core::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Hash, Eq, PartialEq, Clone, Debug)]
pub(crate) struct FileFingerPrint {
    pub path: PathBuf,
    pub predicate: Option<Expr>,
    pub slice: (usize, Option<usize>),
}

#[allow(clippy::type_complexity)]
fn process_with_columns(
    path: &Path,
    with_columns: &Option<Arc<Vec<String>>>,
    predicate: Option<Expr>,
    slice: (usize, Option<usize>),
    file_count_and_column_union: &mut PlHashMap<FileFingerPrint, (FileCount, PlIndexSet<String>)>,
    schema: &Schema,
) {
    let cols = file_count_and_column_union
        .entry(FileFingerPrint {
            path: path.into(),
            predicate,
            slice,
        })
        .or_insert_with(|| {
            (
                0,
                PlIndexSet::with_capacity_and_hasher(32, Default::default()),
            )
        });

    // increment file count
    cols.0 += 1;

    match with_columns {
        // add only the projected columns
        Some(with_columns) => cols.1.extend(with_columns.iter().cloned()),
        // no projection, so we must take all columns
        None => {
            cols.1.extend(schema.iter_names().map(|t| t.to_string()));
        }
    }
}

#[allow(clippy::type_complexity)]
pub(crate) fn collect_fingerprints(
    root: Node,
    fps: &mut Vec<FileFingerPrint>,
    lp_arena: &Arena<ALogicalPlan>,
    expr_arena: &Arena<AExpr>,
) {
    use ALogicalPlan::*;
    match lp_arena.get(root) {
        #[cfg(feature = "csv-file")]
        CsvScan {
            path,
            options,
            predicate,
            ..
        } => {
            let slice = (options.skip_rows, options.n_rows);
            let predicate = predicate.map(|node| node_to_expr(node, expr_arena));
            let fp = FileFingerPrint {
                path: path.clone(),
                predicate,
                slice,
            };
            fps.push(fp);
        }
        #[cfg(feature = "parquet")]
        ParquetScan {
            path,
            options,
            predicate,
            ..
        } => {
            let slice = (0, options.n_rows);
            let predicate = predicate.map(|node| node_to_expr(node, expr_arena));
            let fp = FileFingerPrint {
                path: path.clone(),
                predicate,
                slice,
            };
            fps.push(fp);
        }
        #[cfg(feature = "ipc")]
        IpcScan {
            path,
            options,
            predicate,
            ..
        } => {
            let slice = (0, options.n_rows);
            let predicate = predicate.map(|node| node_to_expr(node, expr_arena));
            let fp = FileFingerPrint {
                path: path.clone(),
                predicate,
                slice,
            };
            fps.push(fp);
        }
        DataFrameScan { .. } => (),
        lp => {
            for input in lp.get_inputs() {
                collect_fingerprints(input, fps, lp_arena, expr_arena)
            }
        }
    }
}

/// Find the union between the columns per unique IO operation.
/// A unique IO operation is the file + the predicates pushed down to that file
#[allow(clippy::type_complexity)]
pub(crate) fn find_column_union_and_fingerprints(
    root: Node,
    // The hashmap maps files to a hashset over column names.
    // we also keep track of how often a needs file needs to be read so we can cache until last read
    columns: &mut PlHashMap<FileFingerPrint, (FileCount, PlIndexSet<String>)>,
    lp_arena: &Arena<ALogicalPlan>,
    expr_arena: &Arena<AExpr>,
) {
    use ALogicalPlan::*;
    match lp_arena.get(root) {
        #[cfg(feature = "csv-file")]
        CsvScan {
            path,
            options,
            predicate,
            schema,
            ..
        } => {
            let slice = (options.skip_rows, options.n_rows);
            let predicate = predicate.map(|node| node_to_expr(node, expr_arena));
            process_with_columns(
                path,
                &options.with_columns,
                predicate,
                slice,
                columns,
                schema,
            );
        }
        #[cfg(feature = "parquet")]
        ParquetScan {
            path,
            options,
            schema,
            predicate,
            ..
        } => {
            let slice = (0, options.n_rows);
            let predicate = predicate.map(|node| node_to_expr(node, expr_arena));
            process_with_columns(
                path,
                &options.with_columns,
                predicate,
                slice,
                columns,
                schema,
            );
        }
        #[cfg(feature = "ipc")]
        IpcScan {
            path,
            options,
            schema,
            predicate,
            ..
        } => {
            let slice = (0, options.n_rows);
            let predicate = predicate.map(|node| node_to_expr(node, expr_arena));
            process_with_columns(
                path,
                &options.with_columns,
                predicate,
                slice,
                columns,
                schema,
            );
        }
        DataFrameScan { .. } => (),
        lp => {
            for input in lp.get_inputs() {
                find_column_union_and_fingerprints(input, columns, lp_arena, expr_arena)
            }
        }
    }
}

/// Aggregate all the columns used in csv scans and make sure that all columns are scanned in one go.
/// Due to self joins there can be multiple Scans of the same file in a LP. We already cache the scans
/// in the PhysicalPlan, but we need to make sure that the first scan has all the columns needed.
pub struct FileCacher {
    file_count_and_column_union: PlHashMap<FileFingerPrint, (FileCount, Arc<Vec<String>>)>,
}

impl FileCacher {
    pub(crate) fn new(
        columns: PlHashMap<FileFingerPrint, (FileCount, PlIndexSet<String>)>,
    ) -> Self {
        let new_columns_mapping = columns
            .into_iter()
            .map(|(k, agg)| {
                let file_count = agg.0;
                let columns = agg.1.iter().cloned().collect::<Vec<_>>();
                (k, (file_count, Arc::new(columns)))
            })
            .collect();
        Self {
            file_count_and_column_union: new_columns_mapping,
        }
    }

    fn finish_rewrite(
        &self,
        mut lp: ALogicalPlan,
        expr_arena: &mut Arena<AExpr>,
        lp_arena: &mut Arena<ALogicalPlan>,
        finger_print: &FileFingerPrint,
        with_columns: Option<Arc<Vec<String>>>,
    ) -> ALogicalPlan {
        // if the original projection is less than the new one. Also project locally
        if let Some(mut with_columns) = with_columns {
            // we cannot always find the predicates, because some have `SpecialEq` functions so for those
            // cases we may read the file twice and/or do an extra projection
            let do_projection = match self.file_count_and_column_union.get(finger_print) {
                Some((_file_count, agg_columns)) => with_columns.len() < agg_columns.len(),
                None => true,
            };
            if do_projection {
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

    fn extract_columns_and_count(
        &mut self,
        finger_print: &FileFingerPrint,
    ) -> Option<(FileCount, Arc<Vec<String>>)> {
        self.file_count_and_column_union.get(finger_print).cloned()
    }
}

impl OptimizationRule for FileCacher {
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
                    let predicate_expr = predicate.map(|node| node_to_expr(node, expr_arena));
                    let finger_print = FileFingerPrint {
                        path,
                        predicate: predicate_expr,
                        slice: (0, options.n_rows),
                    };

                    let with_columns = self.extract_columns_and_count(&finger_print);
                    options.file_counter = with_columns.as_ref().map(|t| t.0).unwrap_or(0);
                    let with_columns = with_columns.map(|t| t.1);
                    // prevent infinite loop
                    if options.with_columns == with_columns {
                        let lp = ALogicalPlan::IpcScan {
                            path: finger_print.path,
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
                        path: finger_print.path.clone(),
                        schema,
                        output_schema,
                        predicate,
                        aggregate,
                        options: options.clone(),
                    };
                    Some(self.finish_rewrite(
                        lp,
                        expr_arena,
                        lp_arena,
                        &finger_print,
                        options.with_columns,
                    ))
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
                    let predicate_expr = predicate.map(|node| node_to_expr(node, expr_arena));
                    let finger_print = FileFingerPrint {
                        path,
                        predicate: predicate_expr,
                        slice: (0, options.n_rows),
                    };
                    let with_columns = self.extract_columns_and_count(&finger_print);
                    options.file_counter = with_columns.as_ref().map(|t| t.0).unwrap_or(0);
                    let mut with_columns = with_columns.map(|t| t.1);
                    // prevent infinite loop
                    if options.with_columns == with_columns {
                        let lp = ALogicalPlan::ParquetScan {
                            path: finger_print.path,
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
                        path: finger_print.path.clone(),
                        schema,
                        output_schema,
                        predicate,
                        aggregate,
                        options,
                    };
                    Some(self.finish_rewrite(lp, expr_arena, lp_arena, &finger_print, with_columns))
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
                    let predicate_expr = predicate.map(|node| node_to_expr(node, expr_arena));
                    let finger_print = FileFingerPrint {
                        path,
                        predicate: predicate_expr,
                        slice: (options.skip_rows, options.n_rows),
                    };
                    let with_columns = self.extract_columns_and_count(&finger_print);
                    options.file_counter = with_columns.as_ref().map(|t| t.0).unwrap_or(0);
                    let with_columns = with_columns.map(|t| t.1);
                    if options.with_columns == with_columns {
                        let lp = ALogicalPlan::CsvScan {
                            path: finger_print.path,
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
                        path: finger_print.path.clone(),
                        schema,
                        output_schema,
                        options: options.clone(),
                        predicate,
                        aggregate,
                    };
                    Some(self.finish_rewrite(
                        lp,
                        expr_arena,
                        lp_arena,
                        &finger_print,
                        options.with_columns,
                    ))
                } else {
                    unreachable!()
                }
            }
            _ => None,
        }
    }
}
