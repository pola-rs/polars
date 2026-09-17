mod count;
#[cfg(feature = "python")]
mod python_udf;

use polars_core::frame::DataFrame;
use polars_core::series::IsSorted;
use polars_error::{PolarsResult, feature_gated};
#[cfg(feature = "python")]
use polars_plan::plans::OpaquePythonUdf;
use polars_plan::plans::{FunctionIR, HintIR};

/// Execute a [`FunctionIR`] node on a materialized [`DataFrame`].
pub fn evaluate_function_ir(function: &FunctionIR, mut df: DataFrame) -> PolarsResult<DataFrame> {
    use FunctionIR::*;
    match function {
        Opaque { function, .. } => function.call_udf(df),
        #[cfg(feature = "python")]
        OpaquePython(OpaquePythonUdf {
            function,
            validate_output,
            schema,
            ..
        }) => python_udf::call_python_udf(function, df, *validate_output, schema.clone()),
        FastCount {
            sources,
            scan_type,
            alias,
            cloud_options,
        } => {
            debug_assert_eq!(df.shape(), (0, 0));
            count::count_rows(
                sources,
                scan_type,
                alias.clone(),
                cloud_options.as_ref().as_ref(),
            )
        },
        Rechunk => {
            df.rechunk_mut_par();
            Ok(df)
        },
        Unnest { columns, separator } => {
            feature_gated!(
                "dtype-struct",
                df.unnest(columns.iter().cloned(), separator.as_deref())
            )
        },
        Explode {
            columns, options, ..
        } => df.explode(columns.iter().cloned(), *options),
        #[cfg(feature = "pivot")]
        Unpivot { args, .. } => {
            use polars_ops::unpivot::UnpivotDF;
            let args = (**args).clone();
            df.unpivot2(args)
        },
        RowIndex { name, offset, .. } => df.with_row_index(name.clone(), *offset),
        Hint(hint) => {
            let HintIR::Sorted(s) = &hint;
            if let Some(s) = s.first() {
                let idx = df.try_get_column_index(&s.column)?;
                let col = &mut unsafe { df.columns_mut_retain_schema() }[idx];
                if let Some(d) = s.descending {
                    let flag = if d {
                        IsSorted::Descending
                    } else {
                        IsSorted::Ascending
                    };
                    col.set_sorted_flag(flag);
                }
            }

            Ok(df)
        },
    }
}
