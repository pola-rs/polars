use std::sync::Arc;

use polars_core::error::PolarsResult;
use polars_core::prelude::*;
use polars_plan::dsl::{ColumnsUdf, SpecialEq};
use polars_plan::plans::IRUnionFunction;

pub fn function_expr_to_udf(func: IRUnionFunction) -> SpecialEq<Arc<dyn ColumnsUdf>> {
    match func {
        IRUnionFunction::Tag => map!(tag),
        IRUnionFunction::Field(name) => map!(field, &name),
        IRUnionFunction::Value { dtype, variant } => map!(value, &dtype, &variant),
    }
}

fn tag(c: &Column) -> PolarsResult<Column> {
    c.try_apply_unary_elementwise(|s| Ok(s.union()?.tag()))
}

fn field(c: &Column, name: &str) -> PolarsResult<Column> {
    c.try_apply_unary_elementwise(|s| {
        let ca = s.union()?;
        Ok(ca.variant(ca.union_fields().try_index_of(name)?))
    })
}

fn value(c: &Column, dtype: &DataType, variant: &str) -> PolarsResult<Column> {
    let DataType::Union(fields) = dtype else {
        unreachable!()
    };
    let variant = fields.try_index_of(variant)?;
    c.try_apply_unary_elementwise(|s| {
        Ok(UnionChunked::from_variant(dtype, variant, s)?.into_series())
    })
}
