use polars_core::prelude::*;
use pyo3_polars::rewrite::*;
use serde::Deserialize;

#[derive(Deserialize)]
struct ScaleKwargs {
    factor: f64,
}

#[polars_rewrite]
fn no_args(inputs: &[Field]) -> PolarsResult<Expr> {
    polars_ensure!(inputs.len() == 1, InvalidOperation: "expected one input");
    Ok(rewrite_input(0))
}

#[polars_rewrite]
fn with_kwargs(_inputs: &[Field], kwargs: ScaleKwargs) -> PolarsResult<Expr> {
    Ok(rewrite_input(0) * lit(kwargs.factor))
}

#[polars_rewrite]
fn with_context(inputs: &[Field], context: RewriteContext) -> PolarsResult<Expr> {
    Ok(match inputs[0].extension_metadata().as_deref() {
        Some("planar") => rewrite_input(0).ext().storage().median(),
        _ => context.plugin_function(
            "spherical_median",
            vec![rewrite_input(0)],
            FunctionOptions::aggregation(),
        ),
    })
}

#[polars_rewrite]
fn with_context_and_kwargs(
    _inputs: &[Field],
    context: RewriteContext,
    kwargs: ScaleKwargs,
) -> PolarsResult<Expr> {
    context.plugin_function_with_kwargs(
        "scale",
        vec![rewrite_input(0)],
        FunctionOptions::elementwise(),
        &kwargs.factor,
    )
}

fn main() {
    // The functions stay callable, e.g. to test them from Rust.
    let field = Field::new("a".into(), DataType::Float64);
    assert!(no_args(&[field]).is_ok());
}
