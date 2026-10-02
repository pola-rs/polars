use std::sync::Arc;

use polars_core::prelude::*;
use polars_error::PolarsResult;
use polars_expr::reduce::into_reduction;
use polars_plan::plans::expr_ir::ExprIR;
use polars_plan::plans::{AExpr, IRAggExpr};
use polars_plan::prelude::WindowMapping;
use polars_utils::arena::Arena;
use polars_utils::pl_str::PlSmallStr;

use crate::nodes::scalar_window::{ScalarWindow, ScalarWindowParams};

/// Whether `exprs` are all windows without an order that `ScalarWindowNode` can evaluate.
pub fn is_reducible_windows(
    exprs: &[ExprIR],
    input_schema: &Schema,
    expr_arena: &Arena<AExpr>,
) -> bool {
    let is_numeric_column = |node| match expr_arena.get(node) {
        AExpr::Column(name) => input_schema
            .get(name)
            .is_some_and(|dtype| dtype.is_primitive_numeric()),
        _ => false,
    };
    !exprs.is_empty()
        && exprs.iter().all(|e| {
            let AExpr::Over {
                function,
                order_by: None,
                mapping: WindowMapping::GroupsToRows,
                ..
            } = expr_arena.get(e.node())
            else {
                return false;
            };
            match expr_arena.get(*function) {
                AExpr::Len => true,
                AExpr::Agg(
                    IRAggExpr::Sum { input, .. }
                    | IRAggExpr::Min { input, .. }
                    | IRAggExpr::Max { input, .. }
                    | IRAggExpr::Count { input, .. }
                    | IRAggExpr::Mean(input),
                ) => is_numeric_column(*input),
                _ => false,
            }
        })
}

/// The parameters of a `ScalarWindowNode` for windows accepted by [`is_reducible_windows`].
pub fn scalar_window_params(
    partition_by: &[PlSmallStr],
    exprs: &[ExprIR],
    input_schema: &Schema,
    output_schema: Arc<Schema>,
    expr_arena: &mut Arena<AExpr>,
) -> PolarsResult<ScalarWindowParams> {
    let mut read_schema = Schema::default();
    let mut key_schema = Schema::default();
    for name in partition_by {
        let dtype = input_schema.try_get(name)?;
        read_schema.insert(name.clone(), dtype.clone());
        key_schema.insert(name.clone(), dtype.clone());
    }

    let mut windows = Vec::with_capacity(exprs.len());
    for e in exprs {
        let AExpr::Over { function, .. } = expr_arena.get(e.node()) else {
            unreachable!()
        };
        // `len` reads the first input column.
        let (reduction, inputs) = into_reduction(*function, expr_arena, input_schema, false)?;
        let [input] = inputs.as_slice() else {
            unreachable!()
        };
        let AExpr::Column(input) = expr_arena.get(*input) else {
            unreachable!()
        };
        let input = input.clone();
        if !read_schema.contains(&input) {
            read_schema.insert(input.clone(), input_schema.try_get(&input)?.clone());
        }
        windows.push(ScalarWindow {
            name: e.output_name().clone(),
            input,
            reduction,
        });
    }

    Ok(ScalarWindowParams {
        partition_by: partition_by.to_vec(),
        windows,
        read_schema: Arc::new(read_schema),
        key_schema: Arc::new(key_schema),
        output_schema,
    })
}
