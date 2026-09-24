//! Plugin rewrites
//! Decide during query planning what an expression becomes,
//! based on the input types.
//! These are written against the raw C ABI; `#[polars_rewrite]` will generate this.
use std::panic::UnwindSafe;

use polars::prelude::*;
use polars_arrow::ffi::{import_field_from_c, ArrowSchema};
use polars_ffi::dsl_rewrite::BytesExport;
use polars_plan::prelude::*;
use serde::Deserialize;

/// Shared body of the exported `_polars_plugin_rewrite_{name}` functions.
#[allow(clippy::too_many_arguments)]
unsafe fn export_rewrite(
    fields: *const ArrowSchema,
    n_fields: usize,
    kwargs: *const u8,
    kwargs_len: usize,
    lib: *const u8,
    lib_len: usize,
    out: *mut BytesExport,
    f: impl FnOnce(&[Field], &[u8], &str) -> PolarsResult<Expr> + UnwindSafe,
) {
    let panic_result = std::panic::catch_unwind(move || {
        let fields = std::slice::from_raw_parts(fields, n_fields)
            .iter()
            .map(|f| Field::from(&import_field_from_c(f).unwrap()))
            .collect::<Vec<_>>();
        let kwargs = std::slice::from_raw_parts(kwargs, kwargs_len);
        let lib = std::str::from_utf8(std::slice::from_raw_parts(lib, lib_len)).unwrap();

        let result = f(&fields, kwargs, lib).and_then(|expr| {
            let mut buf = Vec::new();
            expr.serialize_versioned(&mut buf)?;
            Ok(buf)
        });
        match result {
            Ok(buf) => *out = BytesExport::from(buf),
            Err(err) => pyo3_polars::derive::_update_last_error(err),
        }
    });

    if panic_result.is_err() {
        pyo3_polars::derive::_set_panic();
    }
}

macro_rules! rewrite {
    ($symbol:ident, $f:expr) => {
        #[no_mangle]
        pub unsafe extern "C" fn $symbol(
            fields: *const ArrowSchema,
            n_fields: usize,
            kwargs: *const u8,
            kwargs_len: usize,
            lib: *const u8,
            lib_len: usize,
            out: *mut BytesExport,
        ) {
            export_rewrite(fields, n_fields, kwargs, kwargs_len, lib, lib_len, out, $f)
        }
    };
}

/// Call a kernel of this plugin, like `register_plugin_function(is_elementwise=True)`.
fn plugin_kernel(lib: &str, symbol: &str, inputs: Vec<Expr>) -> Expr {
    let mut flags = FunctionOptions::default();
    flags.set_elementwise();
    Expr::Function {
        input: inputs,
        function: FunctionExpr::FfiPlugin {
            flags,
            lib: lib.into(),
            symbol: symbol.into(),
            kwargs: Arc::from([]),
        },
    }
}

#[derive(Deserialize)]
struct ToUnitKwargs {
    unit: String,
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

// Converts an `expression_lib.length` extension column, whose metadata is its unit, to another
// unit. The conversion factor is picked from the metadata.
rewrite!(_polars_plugin_rewrite_to_unit, |fields, kwargs, _lib| {
    let kwargs: ToUnitKwargs = pyo3_polars::derive::_parse_kwargs(kwargs)?;
    let DataType::Extension(ext, _) = fields[0].dtype() else {
        polars_bail!(InvalidOperation: "to_unit expects an 'expression_lib.length' column, got {}", fields[0].dtype());
    };
    polars_ensure!(ext.name() == "expression_lib.length", InvalidOperation: "to_unit expects an 'expression_lib.length' column, got {}", ext.name());
    let from = ext.serialize_metadata().unwrap_or_default();
    let factor = meters_per_unit(&from)? / meters_per_unit(&kwargs.unit)?;
    Ok(rewrite_input(0).ext().storage() * lit(factor))
});

// Per-field median of a struct, restructured into a struct again; a plain median otherwise.
rewrite!(
    _polars_plugin_rewrite_struct_median,
    |fields, _kwargs, _lib| {
        Ok(match fields[0].dtype() {
            DataType::Struct(struct_fields) => as_struct(
                struct_fields
                    .iter()
                    .map(|f| rewrite_input(0).struct_().field_by_name(f.name()).median())
                    .collect(),
            ),
            _ => rewrite_input(0).median(),
        })
    }
);

// Calls the `is_leap_year` kernel of this plugin, casting datetimes to dates first.
rewrite!(
    _polars_plugin_rewrite_is_leap_year_any,
    |fields, _kwargs, lib| {
        let input = match fields[0].dtype() {
            DataType::Date => rewrite_input(0),
            DataType::Datetime(_, _) => rewrite_input(0).cast(DataType::Date),
            dt => {
                polars_bail!(InvalidOperation: "is_leap_year_any expects a date or datetime, got {dt}")
            },
        };
        Ok(plugin_kernel(lib, "is_leap_year", vec![input]))
    }
);

// The rewrites below exist to test error handling.

rewrite!(
    _polars_plugin_rewrite_returns_rewrite,
    |_fields, _kwargs, lib| {
        let inner = DslRewriteSource::Ffi {
            lib: lib.into(),
            symbol: "struct_median".into(),
            kwargs: Arc::from([]),
        };
        Ok(Expr::Function {
            input: vec![rewrite_input(0)],
            function: FunctionExpr::DslRewrite(inner),
        })
    }
);

rewrite!(
    _polars_plugin_rewrite_bad_input_index,
    |_fields, _kwargs, _lib| { Ok(rewrite_input(5)) }
);

rewrite!(
    _polars_plugin_rewrite_input_in_eval,
    |_fields, _kwargs, _lib| { Ok(col("l").list().eval(rewrite_input(0))) }
);

rewrite!(
    _polars_plugin_rewrite_multiple_outputs,
    |_fields, _kwargs, _lib| { Ok(all().as_expr()) }
);

rewrite!(
    _polars_plugin_rewrite_always_fails,
    |_fields, _kwargs, _lib| { polars_bail!(ComputeError: "this rewrite always fails") }
);

rewrite!(_polars_plugin_rewrite_panics, |_fields, _kwargs, _lib| {
    panic!("this rewrite panics")
});

/// Returns bytes with a DSL header from an incompatible Polars version.
#[no_mangle]
pub unsafe extern "C" fn _polars_plugin_rewrite_bad_version(
    _fields: *const ArrowSchema,
    _n_fields: usize,
    _kwargs: *const u8,
    _kwargs_len: usize,
    _lib: *const u8,
    _lib_len: usize,
    out: *mut BytesExport,
) {
    let mut buf = b"DSL_VERSION".to_vec();
    buf.extend_from_slice(&u16::MAX.to_le_bytes());
    buf.extend_from_slice(&0u16.to_le_bytes());
    *out = BytesExport::from(buf);
}
