#[cfg(feature = "list_eval")]
use std::sync::Mutex;

#[cfg(feature = "list_eval")]
use polars_arrow::utils::CustomIterTools;
#[cfg(feature = "list_eval")]
use polars_core::prelude::*;
#[cfg(feature = "list_eval")]
use polars_plan::dsl::*;
#[cfg(feature = "list_eval")]
use rayon::prelude::*;

use crate::prelude::*;

pub trait IntoListNameSpace {
    fn into_list_name_space(self) -> ListNameSpace;
}

impl IntoListNameSpace for ListNameSpace {
    fn into_list_name_space(self) -> ListNameSpace {
        self
    }
}

pub trait ListNameSpaceExtension: IntoListNameSpace + Sized {
    /// Run any [`Expr`] on these lists elements
    #[cfg(feature = "list_eval")]
    fn eval(self, expr: Expr, parallel: bool) -> Expr {
        let this = self.into_list_name_space();

        use crate::physical_plan::exotic::prepare_expression_for_context;
        use crate::physical_plan::state::ExecutionState;

        let expr2 = expr.clone();
        let func = move |s: Series| {
            for e in expr.into_iter() {
                match e {
                    #[cfg(feature = "dtype-categorical")]
                    Expr::Cast {
                        data_type: DataType::Categorical(_),
                        ..
                    } => {
                        return Err(PolarsError::ComputeError(
                            "Casting to 'Categorical' not allowed in 'arr.eval'".into(),
                        ))
                    }
                    Expr::Column(name) => {
                        if !name.is_empty() {
                            return Err(PolarsError::ComputeError(r#"Named columns not allowed in 'arr.eval'. Consider using 'element' or 'col("")'."#.into()));
                        }
                    }
                    _ => {}
                }
            }

            let lst = s.list()?;
            // ensure we get the new schema
            let output_field = eval_field_to_dtype(lst.ref_field(), &expr, true);
            if lst.is_empty() {
                return Ok(Some(Series::new_empty(s.name(), output_field.data_type())));
            }
            if lst.null_count() == lst.len() {
                return Ok(Some(s));
            }

            let phys_expr =
                prepare_expression_for_context("", &expr, &lst.inner_dtype(), Context::Default)?;

            let state = ExecutionState::new();

            let mut err = None;
            let mut ca: ListChunked = if parallel {
                let m_err = Mutex::new(None);
                let ca: ListChunked = lst
                    .par_iter()
                    .map(|opt_s| {
                        opt_s.and_then(|s| {
                            let df = DataFrame::new_no_checks(vec![s]);
                            let out = phys_expr.evaluate(&df, &state);
                            match out {
                                Ok(s) => Some(s),
                                Err(e) => {
                                    *m_err.lock().unwrap() = Some(e);
                                    None
                                }
                            }
                        })
                    })
                    .collect();
                err = m_err.lock().unwrap().take();
                ca
            } else {
                let mut df_container = DataFrame::new_no_checks(vec![]);

                lst.into_iter()
                    .map(|s| {
                        s.and_then(|s| {
                            df_container.get_columns_mut().push(s);
                            let out = phys_expr.evaluate(&df_container, &state);
                            df_container.get_columns_mut().clear();
                            match out {
                                Ok(s) => Some(s),
                                Err(e) => {
                                    err = Some(e);
                                    None
                                }
                            }
                        })
                    })
                    .collect_trusted()
            };

            ca.rename(s.name());

            if ca.dtype() != output_field.data_type() {
                ca.cast(output_field.data_type()).map(Some)
            } else {
                match err {
                    None => Ok(Some(ca.into_series())),
                    Some(e) => Err(e),
                }
            }
        };

        this.0
            .map(
                func,
                GetOutput::map_field(move |f| eval_field_to_dtype(f, &expr2, true)),
            )
            .with_fmt("eval")
    }
}

impl ListNameSpaceExtension for ListNameSpace {}
