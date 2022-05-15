use super::*;
use crate::physical_plan::state::ExecutionState;
use parking_lot::Mutex;
use rayon::prelude::*;

pub(super) fn prepare_eval_expr(mut expr: Expr) -> Expr {
    expr.mutate().apply(|e| match e {
        Expr::Column(name) => {
            *name = Arc::from("");
            true
        }
        Expr::Nth(_) => {
            *e = Expr::Column(Arc::from(""));
            true
        }
        _ => true,
    });
    expr
}

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
            let planner = DefaultPlanner::default();
            let phys_expr = planner.create_physical_expr(aexpr, Context::Default, &mut arena)?;

            let state = ExecutionState::new();

            let mut err = None;

            let avs = if parallel {
                let m_err = Mutex::new(None);
                let avs = (1..s.len() + 1)
                    .into_par_iter()
                    .map(|len| {
                        let s = s.slice(0, len);
                        if (len - s.null_count()) >= min_periods {
                            let df = DataFrame::new_no_checks(vec![s]);
                            let out = phys_expr.evaluate(&df, &state);
                            match out {
                                Ok(s) => s.get(0).into_static().unwrap(),
                                Err(e) => {
                                    *m_err.lock() = Some(e);
                                    AnyValue::Null
                                }
                            }
                        } else {
                            AnyValue::Null
                        }
                    })
                    .collect::<Vec<_>>();
                err = m_err.lock().take();
                avs
            } else {
                let mut df_container = DataFrame::new_no_checks(vec![]);
                (1..s.len() + 1)
                    .map(|len| {
                        let s = s.slice(0, len);
                        if (len - s.null_count()) >= min_periods {
                            df_container.get_columns_mut().push(s);
                            let out = phys_expr.evaluate(&df_container, &state);
                            df_container.get_columns_mut().clear();
                            match out {
                                Ok(s) => s.get(0).into_static().unwrap(),
                                Err(e) => {
                                    err = Some(e);
                                    AnyValue::Null
                                }
                            }
                        } else {
                            AnyValue::Null
                        }
                    })
                    .collect::<Vec<_>>()
            };

            match err {
                None => Ok(Series::new(&name, avs)),
                Some(e) => Err(e),
            }
        };

        self.map(
            func,
            GetOutput::map_field(move |f| {
                // dummy df to determine output dtype
                let dtype = f
                    .data_type()
                    .inner_dtype()
                    .cloned()
                    .unwrap_or_else(|| f.data_type().clone());

                let df = Series::new_empty("", &dtype).into_frame();
                match df.lazy().select([expr2.clone()]).collect() {
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
