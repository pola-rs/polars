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
    mut input_left: Node,
    mut input_right: Node,
    mut left_on: Vec<ExprIR>,
    mut right_on: Vec<ExprIR>,
    mut schema: SchemaRef,
    mut options: Arc<JoinOptionsIR>,
    ir_arena: &Arena<IR>,
    expr_arena: &Arena<AExpr>,
) -> IR {
    if let (JoinType::Inner, Some(hive_left), Some(hive_right)) = (
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
                    hive_cols = Some((l, r));
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
                    [l],
                    [r],
                    JoinArgs {
                        how: options.args.how.clone(),
                        ..Default::default()
                    },
                    None,
                )
                .unwrap();

            let partitions =
                split_df_as_ref(&partitions, std::cmp::min(64, partitions.height()), false);
            dbg!(hive_cols);
            dbg!(partitions);

            // `left_on` to names
            dbg!(hive_left, hive_right);
            // hive_left.df().join(hive_right.df(), left_on, right_on, args, options)
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
