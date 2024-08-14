use polars_core::prelude::*;
use polars_expr::{create_physical_expr, ExpressionConversionState};
use rayon::prelude::*;

use super::*;
use crate::prelude::*;

pub(crate) fn eval_field_to_dtype(f: &Field, expr: &Expr, list: bool) -> Field {
    // Dummy df to determine output dtype.
    let dtype = f
        .data_type()
        .inner_dtype()
        .cloned()
        .unwrap_or_else(|| f.data_type().clone());

    let df = Series::new_empty("", &dtype).into_frame();

    #[cfg(feature = "python")]
    let out = {
        use pyo3::Python;
        Python::with_gil(|py| py.allow_threads(|| df.lazy().select([expr.clone()]).collect()))
    };
    #[cfg(not(feature = "python"))]
    let out = { df.lazy().select([expr.clone()]).collect() };

    match out {
        Ok(out) => {
            let dtype = out.get_columns()[0].dtype();
            if list {
                Field::new(f.name(), DataType::List(Box::new(dtype.clone())))
            } else {
                Field::new(f.name(), dtype.clone())
            }
        },
        Err(_) => Field::new(f.name(), DataType::Null),
    }
}

pub trait ExprEvalExtension: IntoExpr + Sized {
    /// Run an expression over a sliding window that increases `1` slot every iteration.
    ///
    /// # Warning
    /// This can be really slow as it can have `O(n^2)` complexity. Don't use this for operations
    /// that visit all elements.
    fn cumulative_eval(self, expr: Expr, min_periods: usize, parallel: bool) -> Expr {
        let this = self.into_expr();
        let expr2 = expr.clone();
        let func = move |mut s: Series| {
            let name = s.name().to_string();
            s.rename("");

            // Ensure we get the new schema.
            let output_field = eval_field_to_dtype(s.field().as_ref(), &expr, false);

            let expr = expr.clone();
            let mut arena = Arena::with_capacity(10);
            let aexpr = to_expr_ir(expr, &mut arena)?;
            let phys_expr = create_physical_expr(
                &aexpr,
                Context::Default,
                &arena,
                None,
                &mut ExpressionConversionState::new(true, 0),
            )?;

            let state = ExecutionState::new();

            let finish = |out: Series| {
                polars_ensure!(
                    out.len() <= 1,
                    ComputeError:
                    "expected single value, got a result with length {}, {:?}",
                    out.len(), out,
                );
                Ok(out.get(0).unwrap().into_static().unwrap())
            };

            let avs = if parallel {
                (1..s.len() + 1)
                    .into_par_iter()
                    .map(|len| {
                        let s = s.slice(0, len);
                        if (len - s.null_count()) >= min_periods {
                            let df = s.into_frame();
                            let out = phys_expr.evaluate(&df, &state)?;
                            finish(out)
                        } else {
                            Ok(AnyValue::Null)
                        }
                    })
                    .collect::<PolarsResult<Vec<_>>>()?
            } else {
                let mut df_container = DataFrame::empty();
                (1..s.len() + 1)
                    .map(|len| {
                        let s = s.slice(0, len);
                        if (len - s.null_count()) >= min_periods {
                            unsafe {
                                df_container.get_columns_mut().push(s);
                                let out = phys_expr.evaluate(&df_container, &state)?;
                                df_container.get_columns_mut().clear();
                                finish(out)
                            }
                        } else {
                            Ok(AnyValue::Null)
                        }
                    })
                    .collect::<PolarsResult<Vec<_>>>()?
            };
            let s = Series::new(&name, avs);

            if s.dtype() != output_field.data_type() {
                s.cast(output_field.data_type()).map(Some)
            } else {
                Ok(Some(s))
            }
        };

        this.apply(
            func,
            GetOutput::map_field(move |f| Ok(eval_field_to_dtype(f, &expr2, false))),
        )
        .with_fmt("expanding_eval")
    }
}

impl ExprEvalExtension for Expr {}
