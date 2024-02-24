use std::path::PathBuf;

use super::*;

pub(super) struct CountStar;

impl CountStar {
    pub(super) fn new() -> Self {
        Self
    }
}

impl OptimizationRule for CountStar {
    // Replace select count(*) from datasource with specialized map function.
    fn optimize_plan(
        &mut self,
        lp_arena: &mut Arena<ALogicalPlan>,
        _expr_arena: &mut Arena<AExpr>,
        node: Node,
    ) -> Option<ALogicalPlan> {
        let mut paths = Vec::new();
        visit_logical_plan_for_scan_paths(&mut paths, node, lp_arena).map(|scan_type| {
            // MapFunction needs a leaf node, hence we create a dummy placeholder node
            let placeholder = ALogicalPlan::DataFrameScan {
                df: Arc::new(Default::default()),
                schema: Arc::new(Default::default()),
                output_schema: None,
                projection: None,
                selection: None,
            };
            let placeholder_node = lp_arena.add(placeholder);

            let sliced_paths: Arc<[PathBuf]> = paths.into();

            let alp = ALogicalPlan::MapFunction {
                input: placeholder_node,
                function: FunctionNode::Count {
                    paths: sliced_paths,
                    scan_type,
                },
            };
            lp_arena.replace(node, alp.clone());
            alp
        })
    }
}

// Visit the logical plan and return the file paths / scan type
// Return None if query is not a simple COUNT(*) FROM SOURCE
fn visit_logical_plan_for_scan_paths(
    all_paths: &mut Vec<PathBuf>,
    node: Node,
    lp_arena: &Arena<ALogicalPlan>,
) -> Option<FileScan> {
    match lp_arena.get(node) {
        ALogicalPlan::Union { inputs, .. } => {
            // Preallocate right amount in case of globbing
            if all_paths.is_empty() {
                let _ = std::mem::replace(all_paths, Vec::with_capacity(inputs.len()));
            }
            let mut scan_type = None;
            for input in inputs {
                match visit_logical_plan_for_scan_paths(all_paths, *input, lp_arena) {
                    Some(leaf_scan_type) => {
                        match &scan_type {
                            None => scan_type = Some(leaf_scan_type),
                            Some(scan_type) => {
                                // All scans must be of the same type (e.g. csv / parquet)
                                if std::mem::discriminant(scan_type)
                                    != std::mem::discriminant(&leaf_scan_type)
                                {
                                    return None;
                                }
                            },
                        };
                    },
                    None => return None,
                }
            }
            scan_type
        },
        ALogicalPlan::Scan {
            scan_type, paths, ..
        } if !matches!(scan_type, FileScan::Anonymous { .. }) => {
            all_paths.extend(paths.iter().cloned());
            Some(scan_type.clone())
        },
        ALogicalPlan::Projection { input, .. } => {
            visit_logical_plan_for_scan_paths(all_paths, *input, lp_arena)
        },
        _ => None,
    }
}
