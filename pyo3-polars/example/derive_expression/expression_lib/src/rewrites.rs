//! Plugin rewrites
//! Decide during query planning what an expression becomes,
//! based on the input types.
use polars::prelude::*;
use pyo3_polars::export::polars_core::datatypes::extension::get_extension_type_or_generic;
use pyo3_polars::rewrite::*;
use serde::Deserialize;

const LENGTH: &str = "expression_lib.length";

/// `expression_lib.length`: lengths stored as floats, the metadata holds the unit.
fn length(unit: &str) -> DataType {
    let ext = get_extension_type_or_generic(LENGTH, &DataType::Float64, Some(unit));
    DataType::Extension(ext, Box::new(DataType::Float64))
}

fn meters_per_unit(unit: &str) -> PolarsResult<f64> {
    Ok(match unit {
        "mm" => 0.001,
        "cm" => 0.01,
        "m" => 1.0,
        "km" => 1000.0,
        _ => polars_bail!(InvalidOperation: "unknown length unit '{unit}'"),
    })
}

#[derive(Deserialize)]
struct ToUnitKwargs {
    unit: String,
}

/// Converts a `length` column to another unit. The conversion factor is picked from the unit in the
/// input's metadata.
#[polars_rewrite]
fn to_unit(inputs: &[Field], kwargs: ToUnitKwargs) -> PolarsResult<Expr> {
    polars_ensure!(
        inputs[0].extension_name().as_deref() == Some(LENGTH),
        InvalidOperation: "to_unit expects an '{LENGTH}' column, got {}", inputs[0].dtype()
    );
    let from = inputs[0].extension_metadata().unwrap_or_default();
    let factor = meters_per_unit(&from)? / meters_per_unit(&kwargs.unit)?;
    // Polars doesn't check the output type, so pin it to the new unit.
    Ok((rewrite_input(0).ext().storage() * lit(factor))
        .ext()
        .to(length(&kwargs.unit)))
}

/// Per-field median of a struct, restructured into a struct again; a plain median otherwise.
#[polars_rewrite]
fn struct_median(inputs: &[Field]) -> PolarsResult<Expr> {
    Ok(match inputs[0].dtype() {
        DataType::Struct(fields) => as_struct(
            fields
                .iter()
                .map(|f| rewrite_input(0).struct_().field_by_name(f.name()).median())
                .collect(),
        ),
        _ => rewrite_input(0).median(),
    })
}

/// Calls the `is_leap_year` function of this plugin, casting datetimes to dates first.
#[polars_rewrite]
fn is_leap_year_any(inputs: &[Field], context: RewriteContext) -> PolarsResult<Expr> {
    let input = match inputs[0].dtype() {
        DataType::Date => rewrite_input(0),
        DataType::Datetime(_, _) => rewrite_input(0).cast(DataType::Date),
        dt => {
            polars_bail!(InvalidOperation: "is_leap_year_any expects a date or datetime, got {dt}")
        },
    };
    Ok(context.plugin_function("is_leap_year", vec![input], FunctionOptions::elementwise()))
}

// The rewrites below exist to test error handling.

#[polars_rewrite]
fn returns_rewrite(_inputs: &[Field], context: RewriteContext) -> PolarsResult<Expr> {
    let inner = DslRewriteSource::Ffi {
        lib: context.plugin_path().into(),
        symbol: "struct_median".into(),
        kwargs: Arc::from([]),
    };
    Ok(Expr::Function {
        input: vec![rewrite_input(0)],
        function: FunctionExpr::DslRewrite(inner),
    })
}

#[polars_rewrite]
fn bad_input_index(_inputs: &[Field]) -> PolarsResult<Expr> {
    Ok(rewrite_input(5))
}

#[polars_rewrite]
fn input_in_eval(_inputs: &[Field]) -> PolarsResult<Expr> {
    Ok(col("l").list().eval(rewrite_input(0)))
}

#[polars_rewrite]
fn multiple_outputs(_inputs: &[Field]) -> PolarsResult<Expr> {
    Ok(all().as_expr())
}

#[polars_rewrite]
fn always_fails(_inputs: &[Field]) -> PolarsResult<Expr> {
    polars_bail!(ComputeError: "this rewrite always fails")
}

#[polars_rewrite]
fn panics(_inputs: &[Field]) -> PolarsResult<Expr> {
    panic!("this rewrite panics")
}

/// Returns bytes with a DSL header from an incompatible Polars version, so it is written against
/// the raw C ABI instead of with `#[polars_rewrite]`.
#[no_mangle]
pub unsafe extern "C" fn _polars_plugin_rewrite_bad_version(
    _fields: *const polars_arrow::ffi::ArrowSchema,
    _n_fields: usize,
    _kwargs: *const u8,
    _kwargs_len: usize,
    _lib: *const u8,
    _lib_len: usize,
    out: *mut pyo3_polars::export::polars_ffi::dsl_rewrite::BytesExport,
) {
    let mut buf = b"DSL_VERSION".to_vec();
    buf.extend_from_slice(&u16::MAX.to_le_bytes());
    buf.extend_from_slice(&0u16.to_le_bytes());
    *out = buf.into();
}
