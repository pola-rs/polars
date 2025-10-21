use std::sync::Arc;

use polars_core::error::PolarsResult;
use polars_core::prelude::Column;
use polars_plan::dsl::{ColumnsUdf, SpecialEq};
use polars_plan::plans::IRBitwiseFunction;

pub fn function_expr_to_udf(func: IRBitwiseFunction) -> SpecialEq<Arc<dyn ColumnsUdf>> {
    use IRBitwiseFunction as B;

    match func {
        B::CountOnes => map!(count_ones),
        B::CountZeros => map!(count_zeros),
        B::LeadingOnes => map!(leading_ones),
        B::LeadingZeros => map!(leading_zeros),
        B::TrailingOnes => map!(trailing_ones),
        B::TrailingZeros => map!(trailing_zeros),

        B::And => map!(reduce_and),
        B::Or => map!(reduce_or),
        B::Xor => map!(reduce_xor),
    }
}

fn count_ones(c: &Column) -> PolarsResult<Column> {
    c.try_apply_unary_elementwise(polars_ops::series::count_ones)
}

fn count_zeros(c: &Column) -> PolarsResult<Column> {
    c.try_apply_unary_elementwise(polars_ops::series::count_zeros)
}

fn leading_ones(c: &Column) -> PolarsResult<Column> {
    c.try_apply_unary_elementwise(polars_ops::series::leading_ones)
}

fn leading_zeros(c: &Column) -> PolarsResult<Column> {
    c.try_apply_unary_elementwise(polars_ops::series::leading_zeros)
}

fn trailing_ones(c: &Column) -> PolarsResult<Column> {
    c.try_apply_unary_elementwise(polars_ops::series::trailing_ones)
}

fn trailing_zeros(c: &Column) -> PolarsResult<Column> {
    c.try_apply_unary_elementwise(polars_ops::series::trailing_zeros)
}

fn reduce_and(c: &Column) -> PolarsResult<Column> {
    c.and_reduce().map(|v| v.into_column(c.name().clone()))
}

fn reduce_or(c: &Column) -> PolarsResult<Column> {
    c.or_reduce().map(|v| v.into_column(c.name().clone()))
}

fn reduce_xor(c: &Column) -> PolarsResult<Column> {
    c.xor_reduce().map(|v| v.into_column(c.name().clone()))
}
