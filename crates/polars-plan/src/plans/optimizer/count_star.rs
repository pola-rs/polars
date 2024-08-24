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
        lp_arena: &mut Arena<IR>,
        expr_arena: &mut Arena<AExpr>,
        node: Node,
    ) -> Option<IR> {
        visit_logical_plan_for_scan_paths(node, lp_arena, expr_arena, false).map(
            |count_star_expr| {
                // MapFunction needs a leaf node, hence we create a dummy placeholder node
                let placeholder = IR::DataFrameScan {
                    df: Arc::new(Default::default()),
                    schema: Arc::new(Default::default()),
                    output_schema: None,
                    filter: None,
                };
                let placeholder_node = lp_arena.add(placeholder);

                let alp = IR::MapFunction {
                    input: placeholder_node,
                    function: FunctionIR::FastCount {
                        paths: count_star_expr.paths,
                        scan_type: count_star_expr.scan_type,
                        alias: count_star_expr.alias,
                    },
                };

                lp_arena.replace(count_star_expr.node, alp.clone());
                alp
            },
        )
    }
}

struct CountStarExpr {
    // Top node of the projection to replace
    node: Node,
    // Paths to the input files
    paths: Arc<Vec<PathBuf>>,
    // File Type
    scan_type: FileScan,
    // Column Alias
    alias: Option<Arc<str>>,
}

// Visit the logical plan and return CountStarExpr with the expr information gathered
// Return None if query is not a simple COUNT(*) FROM SOURCE
fn visit_logical_plan_for_scan_paths(
    node: Node,
    lp_arena: &Arena<IR>,
    expr_arena: &Arena<AExpr>,
    inside_union: bool, // Inside union's we do not check for COUNT(*) expression
) -> Option<CountStarExpr> {
    match lp_arena.get(node) {
        IR::Union { inputs, .. } => {
            let mut scan_type: Option<FileScan> = None;
            let mut paths = Vec::with_capacity(inputs.len());
            for input in inputs {
                match visit_logical_plan_for_scan_paths(*input, lp_arena, expr_arena, true) {
                    Some(expr) => {
                        paths.extend(expr.paths.iter().cloned());
                        match &scan_type {
                            None => scan_type = Some(expr.scan_type),
                            Some(scan_type) => {
                                // All scans must be of the same type (e.g. csv / parquet)
                                if std::mem::discriminant(scan_type)
                                    != std::mem::discriminant(&expr.scan_type)
                                {
                                    return None;
                                }
                            },
                        };
                    },
                    None => return None,
                }
            }
            Some(CountStarExpr {
                paths: paths.into(),
                scan_type: scan_type.unwrap(),
                node,
                alias: None,
            })
        },
        IR::Scan {
            scan_type, paths, ..
        } if !matches!(scan_type, FileScan::Anonymous { .. }) => Some(CountStarExpr {
            paths: paths.clone(),
            scan_type: scan_type.clone(),
            node,
            alias: None,
        }),
        // A union can insert a simple projection to ensure all projections align.
        // We can ignore that if we are inside a count star.
        IR::SimpleProjection { input, .. } if inside_union => {
            visit_logical_plan_for_scan_paths(*input, lp_arena, expr_arena, false)
        },
        IR::Select { input, expr, .. } => {
            if expr.len() == 1 {
                let (valid, alias) = is_valid_count_expr(&expr[0], expr_arena);
                if valid || inside_union {
                    return visit_logical_plan_for_scan_paths(*input, lp_arena, expr_arena, false)
                        .map(|mut expr| {
                            expr.alias = alias;
                            expr.node = node;
                            expr
                        });
                }
            }
            None
        },
        _ => None,
    }
}

fn is_valid_count_expr(e: &ExprIR, expr_arena: &Arena<AExpr>) -> (bool, Option<Arc<str>>) {
    match expr_arena.get(e.node()) {
        AExpr::Len => (true, e.get_alias().cloned()),
        _ => (false, None),
    }
}
