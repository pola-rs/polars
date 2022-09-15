use rayon::prelude::*;

use super::*;
use crate::physical_plan::state::ExecutionState;

impl Expr {
    /// Run an expression over a sliding window that increases `1` slot every iteration.
    ///
    /// # Warning
    /// this can be really slow as it can have `O(n^2)` complexity. Don't use this for operations
    /// that visit all elements.
    pub fn cumulative_eval(self, expr: Expr, min_periods: usize, parallel: bool) -> Self {
        let expr2 = expr.clone();
        let func = move |mut s: Series| {
            let name = s.name().to_string();
            s.rename("");
            let expr = expr.clone();
            let mut arena = Arena::with_capacity(10);
            let aexpr = to_aexpr(expr, &mut arena);
            let planner = PhysicalPlanner::default();
            let phys_expr = planner.create_physical_expr(aexpr, Context::Default, &mut arena)?;

            let state = ExecutionState::new();

            let finish = |out: Series| {
                if out.len() > 1 {
                    Err(PolarsError::ComputeError(
                        format!(
                            "expected single value, got a result with length: {}, {:?}",
                            out.len(),
                            out
                        )
                        .into(),
                    ))
                } else {
                    Ok(out.get(0).into_static().unwrap())
                }
            };

            let avs = if parallel {
                (1..s.len() + 1)
                    .into_par_iter()
                    .map(|len| {
                        let s = s.slice(0, len);
                        if (len - s.null_count()) >= min_periods {
                            let df = DataFrame::new_no_checks(vec![s]);
                            let out = phys_expr.evaluate(&df, &state)?;
                            finish(out)
                        } else {
                            Ok(AnyValue::Null)
                        }
                    })
                    .collect::<PolarsResult<Vec<_>>>()?
            } else {
                let mut df_container = DataFrame::new_no_checks(vec![]);
                (1..s.len() + 1)
                    .map(|len| {
                        let s = s.slice(0, len);
                        if (len - s.null_count()) >= min_periods {
                            df_container.get_columns_mut().push(s);
                            let out = phys_expr.evaluate(&df_container, &state)?;
                            df_container.get_columns_mut().clear();
                            finish(out)
                        } else {
                            Ok(AnyValue::Null)
                        }
                    })
                    .collect::<PolarsResult<Vec<_>>>()?
            };
            Ok(Series::new(&name, avs))
        };

        self.apply(
            func,
            GetOutput::map_field(move |f| {
                // dummy df to determine output dtype
                let dtype = f
                    .data_type()
                    .inner_dtype()
                    .cloned()
                    .unwrap_or_else(|| f.data_type().clone());

                let df = Series::new_empty("", &dtype).into_frame();

                #[cfg(feature = "python")]
                let out = {
                    use pyo3::Python;
                    Python::with_gil(|py| {
                        py.allow_threads(|| df.lazy().select([expr2.clone()]).collect())
                    })
                };
                #[cfg(not(feature = "python"))]
                let out = { df.lazy().select([expr2.clone()]).collect() };

                match out {
                    Ok(out) => {
                        let dtype = out.get_columns()[0].dtype();
                        Field::new(f.name(), dtype.clone())
                    }
                    Err(_) => Field::new(f.name(), DataType::Null),
                }
            }),
        )
        .with_fmt("expanding_eval")
    }
}
