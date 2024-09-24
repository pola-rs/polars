use polars_core::prelude::*;
use polars_core::POOL;
use polars_expr::{create_physical_expr, ExpressionConversionState};
use rayon::prelude::*;

use super::*;
use crate::prelude::*;

pub(crate) fn eval_field_to_dtype(f: &Field, expr: &Expr, list: bool) -> Field {
    // Dummy df to determine output dtype.
    let dtype = f
        .dtype()
        .inner_dtype()
        .cloned()
        .unwrap_or_else(|| f.dtype().clone());

    let df = Series::new_empty(PlSmallStr::EMPTY, &dtype).into_frame();

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
                Field::new(f.name().clone(), DataType::List(Box::new(dtype.clone())))
            } else {
                Field::new(f.name().clone(), dtype.clone())
            }
        },
        Err(_) => Field::new(f.name().clone(), DataType::Null),
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
        let func = move |mut c: Column| {
            let name = c.name().clone();
            c.rename(PlSmallStr::EMPTY);

            // Ensure we get the new schema.
            let output_field = eval_field_to_dtype(c.field().as_ref(), &expr, false);

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

            let finish = |out: Column| {
                polars_ensure!(
                    out.len() <= 1,
                    ComputeError:
                    "expected single value, got a result with length {}, {:?}",
                    out.len(), out,
                );
                Ok(out.get(0).unwrap().into_static())
            };

            let avs = if parallel {
                POOL.install(|| {
                    (1..c.len() + 1)
                        .into_par_iter()
                        .map(|len| {
                            let s = c.slice(0, len);
                            if (len - s.null_count()) >= min_periods {
                                let df = c.clone().into_frame();
                                let out = phys_expr.evaluate(&df, &state)?.into_column();
                                finish(out)
                            } else {
                                Ok(AnyValue::Null)
                            }
                        })
                        .collect::<PolarsResult<Vec<_>>>()
                })?
            } else {
                let mut df_container = DataFrame::empty();
                (1..c.len() + 1)
                    .map(|len| {
                        let c = c.slice(0, len);
                        if (len - c.null_count()) >= min_periods {
                            unsafe {
                                df_container.get_columns_mut().push(c.into_column());
                                let out = phys_expr.evaluate(&df_container, &state)?.into_column();
                                df_container.get_columns_mut().clear();
                                finish(out)
                            }
                        } else {
                            Ok(AnyValue::Null)
                        }
                    })
                    .collect::<PolarsResult<Vec<_>>>()?
            };
            let c = Column::new(name, avs);

            if c.dtype() != output_field.dtype() {
                c.cast(output_field.dtype()).map(Some)
            } else {
                Ok(Some(c))
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
