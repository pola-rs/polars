use polars_core::utils::split_df_as_ref;

use super::*;
use crate::plans::hive::HivePartitionsDf;
use crate::plans::inputs::Inputs;
use crate::utils::deep_clone_ir;

fn is_hive_partitioned(node: Node, ir_arena: &Arena<IR>) -> Option<HivePartitionsDf> {
    for (_, ir) in ir_arena.iter(node) {
        match ir {
            IR::Scan { hive_parts, .. } => return hive_parts.clone(),
            // We only want to return hive partitions for the first joins
            // Any node in between with more than one input (join, union, etc) will not return a
            // match.
            ir if matches!(ir.inputs(), Inputs::Single { .. }) => continue,
            _ => return None,
        }
    }

    None
}

#[cfg(feature = "is_in")]
fn hive_rewrite_supports_join_type(how: &JoinType) -> bool {
    match how {
        JoinType::Inner | JoinType::Left | JoinType::Right => true,
        #[cfg(feature = "semi_anti_join")]
        JoinType::Semi => true,
        _ => false,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn rewrite_hive(
    input_left: Node,
    input_right: Node,
    left_on: Vec<ExprIR>,
    right_on: Vec<ExprIR>,
    schema: SchemaRef,
    options: Arc<JoinOptionsIR>,
    opt: &mut PredicatePushDown,
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
) -> PolarsResult<IR> {
    #[cfg(not(feature = "is_in"))]
    return Ok(IR::Join {
        input_left,
        input_right,
        left_on,
        right_on,
        schema,
        options,
    });

    // This replaces a join on a hive partitioned key
    // by a union on hive partitioned joins.
    // We do that by pushing down an is_in predicate
    // Later in the optimizer we prune the hive paths
    // based on all the predicates.
    #[cfg(feature = "is_in")]
    if !opt.hive_rewrite_active
        && let (MaintainOrderJoin::None, true, Some(hive_left), Some(hive_right)) = (
            &options.args.maintain_order,
            hive_rewrite_supports_join_type(&options.args.how),
            is_hive_partitioned(input_left, ir_arena),
            is_hive_partitioned(input_right, ir_arena),
        )
    {
        let mut hive_cols = None;
        let hive_left_schema = hive_left.schema();
        let hive_right_schema = hive_right.schema();
        for (l, r) in left_on.iter().zip(right_on.iter()) {
            let l = expr_arena.get(l.node());
            let r = expr_arena.get(r.node());
            if let (AExpr::Column(l), AExpr::Column(r)) = (l, r) {
                if hive_left_schema.index_of(l) == Some(0)
                    && hive_right_schema.index_of(r) == Some(0)
                {
                    hive_cols = Some((l.clone(), r.clone()));
                    break;
                }
            }
        }

        if let Some((l, r)) = hive_cols {
            let hive_l = hive_left
                .df()
                .select_at_idx(0)
                .unwrap()
                .clone()
                .into_frame();
            let hive_r = hive_right
                .df()
                .select_at_idx(0)
                .unwrap()
                .clone()
                .into_frame();

            let partitions = hive_l
                .join(
                    &hive_r,
                    [l.as_str()],
                    [r.as_str()],
                    JoinArgs {
                        how: options.args.how.clone(),
                        ..Default::default()
                    },
                    None,
                )
                .unwrap();

            let l_key_name = if partitions.schema().contains(l.as_str()) {
                l.clone()
            } else {
                r.clone()
            };
            let r_key_name = if partitions.schema().contains(r.as_str()) {
                r.clone()
            } else {
                l.clone()
            };

            if !opt.partition_hive {
                let (l_pred, r_pred) = make_predicates(
                    &partitions,
                    l_key_name.clone(),
                    l.clone(),
                    r_key_name.clone(),
                    r.clone(),
                    expr_arena,
                );

                // TODO: this goes into the branch twice. We could optimize for a single pass.
                let mut acc_left = init_indexmap(Some(1));
                insert_predicate_dedup(&mut acc_left, &l_pred, expr_arena);
                opt.pushdown_and_assign(input_left, acc_left, ir_arena, expr_arena)?;

                let mut acc_right = init_indexmap(Some(1));
                insert_predicate_dedup(&mut acc_right, &r_pred, expr_arena);
                opt.pushdown_and_assign(input_right, acc_right, ir_arena, expr_arena)?;
            } else {
                let n_parts = std::env::var("POLARS_HIVE_PARTITIONS")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(64);

                let chunks = split_df_as_ref(
                    &partitions,
                    std::cmp::min(n_parts, partitions.height()),
                    false,
                );

                let mut branches = Vec::with_capacity(chunks.len());

                for chunk in chunks {
                    if chunk.height() == 0 {
                        continue;
                    }

                    let (l_pred, r_pred) = make_predicates(
                        &chunk,
                        l_key_name.clone(),
                        l.clone(),
                        r_key_name.clone(),
                        r.clone(),
                        expr_arena,
                    );

                    // We need to deep clone as each branch hits different predicate pd passes.
                    let branch_left = deep_clone_ir(input_left, ir_arena);
                    let branch_right = deep_clone_ir(input_right, ir_arena);

                    let mut acc_left = init_indexmap(Some(1));
                    insert_predicate_dedup(&mut acc_left, &l_pred, expr_arena);
                    opt.pushdown_and_assign(branch_left, acc_left, ir_arena, expr_arena)?;

                    let mut acc_right = init_indexmap(Some(1));
                    insert_predicate_dedup(&mut acc_right, &r_pred, expr_arena);
                    opt.pushdown_and_assign(branch_right, acc_right, ir_arena, expr_arena)?;

                    branches.push(ir_arena.add(IR::Join {
                        input_left: branch_left,
                        input_right: branch_right,
                        left_on: left_on.clone(),
                        right_on: right_on.clone(),
                        schema: schema.clone(),
                        options: options.clone(),
                    }));
                }

                return Ok(IR::Union {
                    inputs: branches,
                    options: UnionOptions {
                        maintain_order: false,
                        ..Default::default()
                    },
                });
            }
        }
    }
    Ok(IR::Join {
        input_left,
        input_right,
        left_on,
        right_on,
        schema,
        options,
    })
}

#[cfg(feature = "is_in")]
fn make_predicates(
    partitions: &DataFrame,
    extract_name_left: PlSmallStr,
    predicate_name_left: PlSmallStr,
    extract_name_right: PlSmallStr,
    predicate_name_right: PlSmallStr,
    expr_arena: &mut Arena<AExpr>,
) -> (ExprIR, ExprIR) {
    let l_values = partitions
        .column(&extract_name_left)
        .unwrap()
        .as_materialized_series()
        .implode()
        .unwrap()
        .into_series();
    let r_values = partitions
        .column(&extract_name_right)
        .unwrap()
        .as_materialized_series()
        .implode()
        .unwrap()
        .into_series();

    let l_pred = AExprBuilder::col(predicate_name_left, expr_arena)
        .is_in(
            AExprBuilder::lit(LiteralValue::Series(SpecialEq::new(l_values)), expr_arena),
            false, // nulls_equal
            expr_arena,
        )
        .expr_ir_unnamed();

    let r_pred = AExprBuilder::col(predicate_name_right, expr_arena)
        .is_in(
            AExprBuilder::lit(LiteralValue::Series(SpecialEq::new(r_values)), expr_arena),
            false,
            expr_arena,
        )
        .expr_ir_unnamed();
    (l_pred, r_pred)
}
