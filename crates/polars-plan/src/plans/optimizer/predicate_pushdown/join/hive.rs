use polars_core::utils::split_df_as_ref;

use super::*;
use crate::plans::hive::HivePartitionsDf;
use crate::plans::inputs::Inputs;

fn is_hive_partitioned(node: Node, ir_arena: &Arena<IR>) -> Option<HivePartitionsDf> {
    let ir = ir_arena.get(node);

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

pub fn rewrite_hive(
    input_left: Node,
    input_right: Node,
    left_on: Vec<ExprIR>,
    right_on: Vec<ExprIR>,
    schema: SchemaRef,
    options: Arc<JoinOptionsIR>,
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
) -> IR {
    if let (MaintainOrderJoin::None, JoinType::Inner, Some(hive_left), Some(hive_right)) = (
        &options.args.maintain_order,
        &options.args.how,
        is_hive_partitioned(input_left, ir_arena),
        is_hive_partitioned(input_right, ir_arena),
    ) {
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

            let r_key_name = if partitions.schema().contains(r.as_str()) {
                r.clone()
            } else {
                l.clone()
            };

            let chunks =
                split_df_as_ref(&partitions, std::cmp::min(64, partitions.height()), false);

            let mut branches = Vec::with_capacity(chunks.len());

            for chunk in chunks {
                if chunk.height() == 0 {
                    continue;
                }

                let l_values = chunk.column(&l).unwrap().as_materialized_series().clone();
                let r_values = chunk
                    .column(&r_key_name)
                    .unwrap()
                    .as_materialized_series()
                    .clone();

                let l_pred = AExprBuilder::col(l.clone(), expr_arena)
                    .is_in(
                        AExprBuilder::lit(
                            LiteralValue::Series(SpecialEq::new(l_values)),
                            expr_arena,
                        ),
                        false, // nulls_equal
                        expr_arena,
                    )
                    .expr_ir_unnamed();

                let r_pred = AExprBuilder::col(r.clone(), expr_arena)
                    .is_in(
                        AExprBuilder::lit(
                            LiteralValue::Series(SpecialEq::new(r_values)),
                            expr_arena,
                        ),
                        false,
                        expr_arena,
                    )
                    .expr_ir_unnamed();

                let filtered_left = ir_arena.add(IR::Filter {
                    input: input_left,
                    predicate: l_pred,
                });
                let filtered_right = ir_arena.add(IR::Filter {
                    input: input_right,
                    predicate: r_pred,
                });

                branches.push(ir_arena.add(IR::Join {
                    input_left: filtered_left,
                    input_right: filtered_right,
                    left_on: left_on.clone(),
                    right_on: right_on.clone(),
                    schema: schema.clone(),
                    options: options.clone(),
                }));
            }

            return IR::Union {
                inputs: branches,
                options: UnionOptions {
                    ..Default::default()
                },
            };
        }
    }
    IR::Join {
        input_left,
        input_right,
        left_on,
        right_on,
        schema,
        options,
    }
}
