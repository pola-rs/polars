use super::scans::SourcesToFileInfo;
use super::*;

pub(super) struct DslConversionContext<'a> {
    pub(super) expr_arena: &'a mut Arena<AExpr>,
    pub(super) lp_arena: &'a mut Arena<IR>,
    pub(super) conversion_optimizer: ConversionOptimizer,
    pub(super) opt_flags: &'a mut OptFlags,
    pub(super) nodes_scratch: &'a mut UnitVec<Node>,
    pub(super) cache_file_info: SourcesToFileInfo,
    pub(super) pushdown_maintain_errors: bool,
    pub(super) verbose: bool,
    pub(super) seen_caches: PlHashMap<UniqueId, Node>,
}

pub(super) fn expand_expressions(
    input: Node,
    exprs: Vec<Expr>,
    lp_arena: &Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
    opt_flags: &mut OptFlags,
) -> PolarsResult<Vec<ExprIR>> {
    let schema = lp_arena.get(input).schema(lp_arena);
    let exprs = rewrite_projections(exprs, &Default::default(), &schema, opt_flags)?;
    to_expr_irs(
        exprs,
        &mut ExprToIRContext::new_with_opt_eager(expr_arena, &schema, opt_flags),
    )
}

pub(super) fn empty_df() -> IR {
    IR::DataFrameScan {
        df: Arc::new(Default::default()),
        schema: Arc::new(Default::default()),
        output_schema: None,
    }
}

pub(super) fn validate_expression(
    node: Node,
    expr_arena: &Arena<AExpr>,
    input_schema: &Schema,
    operation_name: &str,
) -> PolarsResult<()> {
    let iter = aexpr_to_leaf_names_iter(node, expr_arena);
    validate_columns_in_input(iter, input_schema, operation_name)
}

pub(super) fn validate_expressions<N: Into<Node>, I: IntoIterator<Item = N>>(
    nodes: I,
    expr_arena: &Arena<AExpr>,
    input_schema: &Schema,
    operation_name: &str,
) -> PolarsResult<()> {
    let nodes = nodes.into_iter();

    for node in nodes {
        validate_expression(node.into(), expr_arena, input_schema, operation_name)?
    }
    Ok(())
}
