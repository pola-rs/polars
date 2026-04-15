use super::*;

/// Optimization rule that removes literal keys from `group_by` operations.
///
/// Literal keys don't contribute to grouping (they're constant across all rows),
/// so including them is wasteful.
///
/// Two cases:
/// 1. All keys are literals: Rewrite to a Select (whole table is one group)
/// 2. Some keys are literals: Remove literals from grouping, add them back via Select
pub(super) struct SimplifyGroupBy;

impl SimplifyGroupBy {
    pub(super) fn new() -> Self {
        Self
    }
}

impl OptimizationRule for SimplifyGroupBy {
    fn optimize_plan(
        &mut self,
        lp_arena: &mut Arena<IR>,
        expr_arena: &mut Arena<AExpr>,
        node: Node,
    ) -> PolarsResult<Option<IR>> {
        let lp = lp_arena.get(node);

        let IR::GroupBy {
            input,
            keys,
            aggs,
            schema,
            maintain_order,
            options,
            apply,
        } = lp
        else {
            return Ok(None);
        };

        // Don't optimize if there's a custom apply function
        if apply.is_some() {
            return Ok(None);
        }

        // Don't optimize dynamic or rolling group_by
        #[cfg(feature = "dynamic_group_by")]
        if options.dynamic.is_some() || options.rolling.is_some() {
            return Ok(None);
        }

        // Partition keys into literals and non-literals
        let mut literal_keys: Vec<ExprIR> = Vec::new();
        let mut non_literal_keys: Vec<ExprIR> = Vec::new();

        for key in keys {
            if is_literal_expr(key.node(), expr_arena) {
                literal_keys.push(key.clone());
            } else {
                non_literal_keys.push(key.clone());
            }
        }

        // No literal keys, nothing to optimize
        if literal_keys.is_empty() {
            return Ok(None);
        }

        let input = *input;
        let aggs = aggs.clone();
        let maintain_order = *maintain_order;
        let options = options.clone();
        let original_schema = schema.clone();

        if non_literal_keys.is_empty() {
            // All keys are literals: rewrite to Select
            rewrite_to_select(
                input,
                literal_keys,
                aggs,
                original_schema,
                lp_arena,
                expr_arena,
            )
        } else {
            // Some keys are literals: remove them from grouping, add Select wrapper
            rewrite_with_literal_projection(
                input,
                literal_keys,
                non_literal_keys,
                aggs,
                original_schema,
                maintain_order,
                options,
                lp_arena,
                expr_arena,
            )
        }
    }
}

/// Check if an expression is a literal (constant value)
fn is_literal_expr(node: Node, expr_arena: &Arena<AExpr>) -> bool {
    matches!(expr_arena.get(node), AExpr::Literal(_))
}

/// Rewrite `group_by(literal_keys).agg(...)` to `select(literal_keys + aggs)`
fn rewrite_to_select(
    input: Node,
    literal_keys: Vec<ExprIR>,
    aggs: Vec<ExprIR>,
    original_schema: SchemaRef,
    lp_arena: &mut Arena<IR>,
    _expr_arena: &mut Arena<AExpr>,
) -> PolarsResult<Option<IR>> {
    // Combine literal keys and aggregations into a single select
    let mut exprs = literal_keys;
    exprs.extend(aggs);

    let select = IR::Select {
        input,
        expr: exprs,
        schema: original_schema,
        options: ProjectionOptions::default(),
    };

    Ok(Some(select))
}

/// Rewrite `group_by(literals + non_literals).agg(...)` to
/// `group_by(non_literals).agg(...).select(literals + keys + aggs)`
fn rewrite_with_literal_projection(
    input: Node,
    literal_keys: Vec<ExprIR>,
    non_literal_keys: Vec<ExprIR>,
    aggs: Vec<ExprIR>,
    original_schema: SchemaRef,
    maintain_order: bool,
    options: Arc<GroupbyOptions>,
    lp_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
) -> PolarsResult<Option<IR>> {
    // Build schema for the reduced group_by by filtering out literal keys from original
    let group_by_schema: Schema = original_schema
        .iter()
        .filter(|(name, _)| !literal_keys.iter().any(|k| k.output_name() == *name))
        .map(|(name, dtype)| Field::new(name.clone(), dtype.clone()))
        .collect();

    // Create the reduced group_by (without literal keys)
    let group_by = IR::GroupBy {
        input,
        keys: non_literal_keys.clone(),
        aggs: aggs.clone(),
        schema: Arc::new(group_by_schema),
        maintain_order,
        options,
        apply: None,
    };
    let group_by_node = lp_arena.add(group_by);

    // Build the final select that adds literal keys back in the correct position
    // The original schema tells us the expected column order
    let mut final_exprs: Vec<ExprIR> = Vec::with_capacity(original_schema.len());

    for (name, _dtype) in original_schema.iter() {
        if let Some(lit_key) = literal_keys.iter().find(|k| k.output_name() == name) {
            // Literal key: use the literal expression directly
            final_exprs.push(lit_key.clone());
        } else {
            // Non-literal key or aggregation: reference column from group_by output
            let col_node = expr_arena.add(AExpr::Column(name.clone()));
            final_exprs.push(ExprIR::new(col_node, OutputName::Alias(name.clone())));
        }
    }

    let select = IR::Select {
        input: group_by_node,
        expr: final_exprs,
        schema: original_schema,
        options: ProjectionOptions::default(),
    };

    Ok(Some(select))
}
