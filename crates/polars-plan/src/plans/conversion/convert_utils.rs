use super::*;

pub(super) fn convert_st_union(
    inputs: &mut [Node],
    lp_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
) -> PolarsResult<()> {
    let mut schema = (**lp_arena.get(inputs[0]).schema(lp_arena)).clone();

    let mut changed = false;
    for input in inputs[1..].iter() {
        let schema_other = lp_arena.get(*input).schema(lp_arena);
        changed |= schema.to_supertype(schema_other.as_ref())?;
    }

    if changed {
        for input in inputs {
            let mut exprs = vec![];
            let input_schema = lp_arena.get(*input).schema(lp_arena);

            let to_cast = input_schema.iter().zip(schema.iter_values()).flat_map(
                |((left_name, left_type), st)| {
                    if left_type != st {
                        Some(col(left_name.clone()).cast(st.clone()))
                    } else {
                        None
                    }
                },
            );
            exprs.extend(to_cast);

            if !exprs.is_empty() {
                let expr = to_expr_irs(exprs, expr_arena)?;
                let lp = IRBuilder::new(*input, expr_arena, lp_arena)
                    .with_columns(expr, Default::default())
                    .build();

                let node = lp_arena.add(lp);
                *input = node
            }
        }
    }
    Ok(())
}

fn nodes_to_schemas(inputs: &[Node], lp_arena: &mut Arena<IR>) -> Vec<SchemaRef> {
    inputs
        .iter()
        .map(|n| lp_arena.get(*n).schema(lp_arena).into_owned())
        .collect()
}

pub(super) fn convert_diagonal_concat(
    mut inputs: Vec<Node>,
    lp_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
) -> PolarsResult<Vec<Node>> {
    let schemas = nodes_to_schemas(&inputs, lp_arena);

    let upper_bound_width = schemas.iter().map(|sch| sch.len()).sum();

    let mut total_schema = Schema::with_capacity(upper_bound_width);

    for sch in schemas.iter() {
        sch.iter().for_each(|(name, dtype)| {
            if !total_schema.contains(name) {
                total_schema.with_column(name.as_str().into(), dtype.clone());
            }
        });
    }
    if total_schema.is_empty() {
        return Ok(inputs);
    }

    let mut has_empty = false;

    for (node, lf_schema) in inputs.iter_mut().zip(schemas.iter()) {
        // Discard, this works physically
        if lf_schema.is_empty() {
            has_empty = true;
        }
        let mut columns_to_add = vec![];

        for (name, dtype) in total_schema.iter() {
            // If a name from Total Schema is not present - append
            if lf_schema.get_field(name).is_none() {
                columns_to_add.push(NULL.lit().cast(dtype.clone()).alias(name.clone()))
            }
        }
        let expr = to_expr_irs(columns_to_add, expr_arena)?;
        *node = IRBuilder::new(*node, expr_arena, lp_arena)
            // Add the missing columns
            .with_columns(expr, Default::default())
            // Now, reorder to match schema.
            .project_simple(total_schema.iter_names().map(|v| v.as_str()))
            .unwrap()
            .node();
    }

    if has_empty {
        Ok(inputs
            .into_iter()
            .zip(schemas)
            .filter_map(|(input, schema)| if schema.is_empty() { None } else { Some(input) })
            .collect())
    } else {
        Ok(inputs)
    }
}

pub(super) fn h_concat_schema(
    inputs: &[Node],
    lp_arena: &mut Arena<IR>,
) -> PolarsResult<SchemaRef> {
    let schemas = nodes_to_schemas(inputs, lp_arena);
    let combined_schema = merge_schemas(&schemas)?;
    Ok(Arc::new(combined_schema))
}

/// Split expression that are ANDed into multiple Filter nodes as the optimizer can then
/// push them down independently. Especially if they refer columns from different tables
/// this will be more performant.
///
/// So:
/// * `filter[foo == bar & ham == spam]`
///
/// Becomes:
/// * `filter [foo == bar]`
/// * `filter [ham == spam]`
pub(super) struct SplitPredicates {
    pub(super) pushable: Vec<Node>,
    pub(super) fallible: Option<Node>,
}

impl SplitPredicates {
    /// Returns None if a barrier expression is encountered
    pub(super) fn new(
        predicate: Node,
        expr_arena: &mut Arena<AExpr>,
        scratch: Option<&mut UnitVec<Node>>,
        maintain_errors: bool,
    ) -> Option<Self> {
        let mut local_scratch = unitvec![];
        let scratch = scratch.unwrap_or(&mut local_scratch);

        let mut pushable = vec![];
        let mut acc_fallible = unitvec![];

        for predicate in MintermIter::new(predicate, expr_arena) {
            use ExprPushdownGroup::*;

            let ae = expr_arena.get(predicate);

            match ExprPushdownGroup::Pushable.update_with_expr_rec(ae, expr_arena, Some(scratch)) {
                Pushable => pushable.push(predicate),

                Fallible => {
                    if maintain_errors {
                        return None;
                    }

                    acc_fallible.push(predicate);
                },

                Barrier => return None,
            }
        }

        let fallible = (!acc_fallible.is_empty()).then(|| {
            let mut node = acc_fallible.pop().unwrap();

            for next_node in acc_fallible.iter() {
                node = expr_arena.add(AExpr::BinaryExpr {
                    left: node,
                    op: Operator::And,
                    right: *next_node,
                })
            }

            node
        });

        let out = Self { pushable, fallible };

        Some(out)
    }
}
