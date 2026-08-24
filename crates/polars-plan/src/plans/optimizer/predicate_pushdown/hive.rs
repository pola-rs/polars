use polars_core::utils::split_df_as_ref;
use polars_ops::frame::DataFrameJoinOps;

use super::*;
use crate::plans::hive::HivePartitionsDf;
use crate::plans::inputs::Inputs;
use crate::utils::deep_clone_ir;

fn is_hive_partitioned(node: Node, ir_arena: &Arena<IR>) -> Option<HivePartitionsDf> {
    for (_, ir) in ir_arena.iter(node) {
        match ir {
            IR::Scan { hive_parts, .. } => return hive_parts.clone(),
            // Only partition the first group-by.
            IR::GroupBy { .. } => return None,
            // They can modify hive names.
            // Be conservative for now.
            IR::Select { .. } | IR::HStack { .. } => return None,
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

fn get_partitions(hive_df: &DataFrame) -> Vec<DataFrame> {
    let n_parts = std::env::var("POLARS_HIVE_PARTITIONS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(64);

    split_df_as_ref(hive_df, std::cmp::min(n_parts, hive_df.height()), false)
}

// We must deduplicate so that we get unique files (keys) per partition.
#[cfg(feature = "is_in")]
fn unique_key_frame(hive_df: &DataFrame, key_name: &PlSmallStr) -> PolarsResult<DataFrame> {
    hive_df
        .column(key_name)?
        .clone()
        .into_frame()
        .unique_stable(None, UniqueKeepStrategy::First, None)
}

#[allow(clippy::too_many_arguments)]
pub fn rewrite_hive(
    ir: IR,
    opt: &mut PredicatePushDown,
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
) -> PolarsResult<IR> {
    #[cfg(not(feature = "is_in"))]
    return Ok(ir);

    #[cfg(feature = "is_in")]
    {
        if opt.hive_rewrite_active {
            return Ok(ir);
        }

        match ir {
            IR::GroupBy {
                input,
                keys,
                aggs,
                schema,
                maintain_order,
                options,
                apply,
            } if opt.partition_hive
                && !maintain_order
                && let Some(hive) = is_hive_partitioned(input, ir_arena) =>
            {
                // This replaces a group-by on a hive partitioned key
                // by a union on hive partitioned group-by's.
                // We do that by pushing down an is_in predicate
                // Later in the optimizer we prune the hive paths
                // based on all the predicates.
                // Any hive column is valid to split on, prefer the leftmost as it is the coarsest.
                let mut hive_col: Option<(usize, PlSmallStr)> = None;
                let hive_schema = hive.schema();
                for e in keys.iter() {
                    let key = expr_arena.get(e.node());
                    if let AExpr::Column(name) = key {
                        if let Some(idx) = hive_schema.index_of(name)
                            && hive_col.as_ref().is_none_or(|(best, _)| idx < *best)
                        {
                            hive_col = Some((idx, name.clone()));
                        }
                    }
                }

                if let Some((_, key_name)) = hive_col {
                    let hive_df = unique_key_frame(hive.df(), &key_name)?;

                    let chunks = get_partitions(&hive_df);

                    let mut branches = Vec::with_capacity(chunks.len());

                    for chunk in chunks {
                        if chunk.height() == 0 {
                            continue;
                        }

                        let pred =
                            make_predicate(&chunk, key_name.clone(), key_name.clone(), expr_arena);

                        // We need to deep clone as each branch hits different predicate pd passes.
                        let branch = deep_clone_ir(input, ir_arena);

                        let mut acc_new = init_indexmap(Some(1));
                        insert_predicate_dedup(
                            &mut acc_new,
                            &pred,
                            expr_arena,
                            &mut opt.dedup_state,
                        );
                        opt.pushdown_and_assign(branch, acc_new, ir_arena, expr_arena)?;

                        branches.push(ir_arena.add(IR::GroupBy {
                            input: branch,
                            keys: keys.clone(),
                            aggs: aggs.clone(),
                            schema: schema.clone(),
                            maintain_order,
                            options: options.clone(),
                            apply: apply.clone(),
                        }));
                    }

                    Ok(IR::Union {
                        inputs: branches,
                        options: UnionOptions {
                            slice: options.slice,
                            maintain_order: false,
                            ..Default::default()
                        },
                    })
                } else {
                    Ok(IR::GroupBy {
                        input,
                        keys,
                        aggs,
                        schema,
                        maintain_order,
                        options,
                        apply,
                    })
                }
            },
            IR::Join {
                input_left,
                input_right,
                schema,
                options,
            } if let (MaintainOrderJoin::None, true, Some(hive_left), Some(hive_right)) = (
                &options.args.maintain_order,
                hive_rewrite_supports_join_type(&options.args.how),
                is_hive_partitioned(input_left, ir_arena),
                is_hive_partitioned(input_right, ir_arena),
            ) =>
            {
                // This replaces a join on a hive partitioned key
                // by a union on hive partitioned joins.
                // We do that by pushing down an is_in predicate
                // Later in the optimizer we prune the hive paths
                // based on all the predicates.
                // Any hive column is valid to split on, prefer the leftmost as it is the coarsest.
                let mut hive_cols: Option<(usize, PlSmallStr, PlSmallStr)> = None;
                let hive_left_schema = hive_left.schema();
                let hive_right_schema = hive_right.schema();
                for (l, r) in options.options.left_on().zip(options.options.right_on()) {
                    let l = expr_arena.get(l.node());
                    let r = expr_arena.get(r.node());
                    if let (AExpr::Column(l), AExpr::Column(r)) = (l, r) {
                        if let Some(idx) = hive_left_schema.index_of(l)
                            && hive_right_schema.contains(r)
                            && hive_cols.as_ref().is_none_or(|(best, _, _)| idx < *best)
                        {
                            hive_cols = Some((idx, l.clone(), r.clone()));
                        }
                    }
                }

                if let Some((_, l, r)) = hive_cols {
                    let hive_l = unique_key_frame(hive_left.df(), &l)?;
                    let hive_r = unique_key_frame(hive_right.df(), &r)?;

                    let partitions = hive_l
                        .join(
                            &hive_r,
                            [l.as_str()],
                            [r.as_str()],
                            JoinArgs {
                                how: options.args.how.clone(),
                                nulls_equal: options.args.nulls_equal,
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
                        insert_predicate_dedup(
                            &mut acc_left,
                            &l_pred,
                            expr_arena,
                            &mut opt.dedup_state,
                        );
                        opt.pushdown_and_assign(input_left, acc_left, ir_arena, expr_arena)?;

                        let mut acc_right = init_indexmap(Some(1));
                        insert_predicate_dedup(
                            &mut acc_right,
                            &r_pred,
                            expr_arena,
                            &mut opt.dedup_state,
                        );
                        opt.pushdown_and_assign(input_right, acc_right, ir_arena, expr_arena)?;
                    } else {
                        let chunks = get_partitions(&partitions);

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
                            insert_predicate_dedup(
                                &mut acc_left,
                                &l_pred,
                                expr_arena,
                                &mut opt.dedup_state,
                            );
                            opt.pushdown_and_assign(branch_left, acc_left, ir_arena, expr_arena)?;

                            let mut acc_right = init_indexmap(Some(1));
                            insert_predicate_dedup(
                                &mut acc_right,
                                &r_pred,
                                expr_arena,
                                &mut opt.dedup_state,
                            );
                            opt.pushdown_and_assign(branch_right, acc_right, ir_arena, expr_arena)?;

                            branches.push(ir_arena.add(IR::Join {
                                input_left: branch_left,
                                input_right: branch_right,
                                schema: schema.clone(),
                                options: options.clone(),
                            }));
                        }

                        return Ok(IR::Union {
                            inputs: branches,
                            options: UnionOptions {
                                maintain_order: false,
                                slice: options.args.slice,
                                ..Default::default()
                            },
                        });
                    }
                }

                Ok(IR::Join {
                    input_left,
                    input_right,
                    schema,
                    options,
                })
            },
            _ => Ok(ir),
        }
    }
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
    (
        make_predicate(
            partitions,
            extract_name_left,
            predicate_name_left,
            expr_arena,
        ),
        make_predicate(
            partitions,
            extract_name_right,
            predicate_name_right,
            expr_arena,
        ),
    )
}

#[cfg(feature = "is_in")]
fn make_predicate(
    partitions: &DataFrame,
    extract_name: PlSmallStr,
    predicate_name: PlSmallStr,
    expr_arena: &mut Arena<AExpr>,
) -> ExprIR {
    let values = partitions
        .column(&extract_name)
        .unwrap()
        .as_materialized_series()
        .implode()
        .unwrap()
        .into_series();

    AExprBuilder::col(predicate_name, expr_arena)
        .is_in(
            AExprBuilder::lit(LiteralValue::Series(SpecialEq::new(values)), expr_arena),
            // Hive __HIVE_DEFAULT_PARTITION__ can produce nulls
            true,
            expr_arena,
        )
        .expr_ir_unnamed()
}
