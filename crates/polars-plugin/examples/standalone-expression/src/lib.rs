#![deny(unsafe_op_in_unsafe_fn)]
//! A Polars expression plugin that depends only on `polars-plugin`
//! and not PyO3, eliminating the Python build dependency.
use polars_plugin::polars_expr;
use polars_plugin::prelude::*;
use serde::Deserialize;

// output_type + no args
#[polars_expr(output_type = Int64)]
fn double(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].i64()?;
    Ok(ca.apply_values(|v| v.wrapping_mul(2)).into_series())
}

#[derive(Deserialize)]
struct ScaleKwargs {
    scale: i64,
    enabled: bool,
    #[allow(dead_code)]
    label: String,
}

// output_type + kwargs
#[polars_expr(output_type = Int64)]
fn scale(inputs: &[Series], kwargs: ScaleKwargs) -> PolarsResult<Series> {
    let ca = inputs[0].i64()?;
    let factor = if kwargs.enabled { kwargs.scale } else { 1 };
    Ok(ca.apply_values(|v| v.wrapping_mul(factor)).into_series())
}

// output_type + context
#[polars_expr(output_type = Int64)]
fn identity(inputs: &[Series], context: CallerContext) -> PolarsResult<Series> {
    let _ = context;
    Ok(inputs[0].clone())
}

// output_type + context + kwargs
#[polars_expr(output_type = Int64)]
fn scale_ctx(
    inputs: &[Series],
    context: CallerContext,
    kwargs: ScaleKwargs,
) -> PolarsResult<Series> {
    let _ = context;
    let ca = inputs[0].i64()?;
    Ok(ca
        .apply_values(|v| v.wrapping_mul(kwargs.scale))
        .into_series())
}

fn same_type(fields: &[Field]) -> PolarsResult<Field> {
    Ok(fields[0].clone())
}

// output_type_func
#[polars_expr(output_type_func = same_type)]
fn passthrough(inputs: &[Series]) -> PolarsResult<Series> {
    Ok(inputs[0].clone())
}

fn same_type_kw(fields: &[Field], _kwargs: ScaleKwargs) -> PolarsResult<Field> {
    Ok(fields[0].clone())
}

// output_type_func_with_kwargs
#[polars_expr(output_type_func_with_kwargs = same_type_kw)]
fn passthrough_kw(inputs: &[Series], kwargs: ScaleKwargs) -> PolarsResult<Series> {
    let _ = kwargs;
    Ok(inputs[0].clone())
}
