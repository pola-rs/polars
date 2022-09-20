use parking_lot::Mutex;
use polars_arrow::utils::CustomIterTools;
use polars_core::prelude::*;
use polars_core::series::ops::NullBehavior;
use polars_ops::prelude::*;
use rayon::prelude::*;

use crate::dsl::function_expr::FunctionExpr;
#[cfg(feature = "list_eval")]
use crate::physical_plan::exotic::prepare_eval_expr;
#[cfg(feature = "list_eval")]
use crate::physical_plan::exotic::prepare_expression_for_context;
use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;

/// Specialized expressions for [`Series`] of [`DataType::List`].
pub struct ListNameSpace(pub(crate) Expr);

impl ListNameSpace {
    /// Get lengths of the arrays in the List type.
    pub fn lengths(self) -> Expr {
        let function = |s: Series| {
            let ca = s.list()?;
            Ok(ca.lst_lengths().into_series())
        };
        self.0
            .map(function, GetOutput::from_type(IDX_DTYPE))
            .with_fmt("arr.len")
    }

    /// Compute the maximum of the items in every sublist.
    pub fn max(self) -> Expr {
        self.0
            .map(
                |s| Ok(s.list()?.lst_max()),
                GetOutput::map_field(|f| {
                    if let DataType::List(adt) = f.data_type() {
                        Field::new(f.name(), *adt.clone())
                    } else {
                        // inner type
                        f.clone()
                    }
                }),
            )
            .with_fmt("arr.max")
    }

    /// Compute the minimum of the items in every sublist.
    pub fn min(self) -> Expr {
        self.0
            .map(
                |s| Ok(s.list()?.lst_min()),
                GetOutput::map_field(|f| {
                    if let DataType::List(adt) = f.data_type() {
                        Field::new(f.name(), *adt.clone())
                    } else {
                        // inner type
                        f.clone()
                    }
                }),
            )
            .with_fmt("arr.min")
    }

    /// Compute the sum the items in every sublist.
    pub fn sum(self) -> Expr {
        self.0
            .map(
                |s| Ok(s.list()?.lst_sum()),
                GetOutput::map_field(|f| {
                    if let DataType::List(adt) = f.data_type() {
                        Field::new(f.name(), *adt.clone())
                    } else {
                        // inner type
                        f.clone()
                    }
                }),
            )
            .with_fmt("arr.sum")
    }

    /// Compute the mean of every sublist and return a `Series` of dtype `Float64`
    pub fn mean(self) -> Expr {
        self.0
            .map(
                |s| Ok(s.list()?.lst_mean().into_series()),
                GetOutput::from_type(DataType::Float64),
            )
            .with_fmt("arr.mean")
    }

    /// Sort every sublist.
    pub fn sort(self, reverse: bool) -> Expr {
        self.0
            .map(
                move |s| Ok(s.list()?.lst_sort(reverse).into_series()),
                GetOutput::same_type(),
            )
            .with_fmt("arr.sort")
    }

    /// Reverse every sublist
    pub fn reverse(self) -> Expr {
        self.0
            .map(
                move |s| Ok(s.list()?.lst_reverse().into_series()),
                GetOutput::same_type(),
            )
            .with_fmt("arr.reverse")
    }

    /// Keep only the unique values in every sublist.
    pub fn unique(self) -> Expr {
        self.0
            .map(
                move |s| Ok(s.list()?.lst_unique()?.into_series()),
                GetOutput::same_type(),
            )
            .with_fmt("arr.unique")
    }

    /// Get items in every sublist by index.
    pub fn get(self, index: i64) -> Expr {
        self.0.map(
            move |s| s.list()?.lst_get(index),
            GetOutput::map_field(|field| match field.data_type() {
                DataType::List(inner) => Field::new(field.name(), *inner.clone()),
                dt => panic!("should be a list type, got {:?}", dt),
            }),
        )
    }

    /// Get first item of every sublist.
    pub fn first(self) -> Expr {
        self.get(0)
    }

    /// Get last item of every sublist.
    pub fn last(self) -> Expr {
        self.get(-1)
    }

    /// Join all string items in a sublist and place a separator between them.
    /// # Error
    /// This errors if inner type of list `!= DataType::Utf8`.
    pub fn join(self, separator: &str) -> Expr {
        let separator = separator.to_string();
        self.0
            .map(
                move |s| s.list()?.lst_join(&separator).map(|ca| ca.into_series()),
                GetOutput::from_type(DataType::Utf8),
            )
            .with_fmt("arr.join")
    }

    /// Return the index of the minimal value of every sublist
    pub fn arg_min(self) -> Expr {
        self.0
            .map(
                |s| Ok(s.list()?.lst_arg_min().into_series()),
                GetOutput::from_type(IDX_DTYPE),
            )
            .with_fmt("arr.arg_min")
    }

    /// Return the index of the maximum value of every sublist
    pub fn arg_max(self) -> Expr {
        self.0
            .map(
                |s| Ok(s.list()?.lst_arg_max().into_series()),
                GetOutput::from_type(IDX_DTYPE),
            )
            .with_fmt("arr.arg_max")
    }

    /// Diff every sublist.
    #[cfg(feature = "diff")]
    #[cfg_attr(docsrs, doc(cfg(feature = "diff")))]
    pub fn diff(self, n: usize, null_behavior: NullBehavior) -> Expr {
        self.0
            .map(
                move |s| Ok(s.list()?.lst_diff(n, null_behavior).into_series()),
                GetOutput::same_type(),
            )
            .with_fmt("arr.diff")
    }

    /// Shift every sublist.
    pub fn shift(self, periods: i64) -> Expr {
        self.0
            .map(
                move |s| Ok(s.list()?.lst_shift(periods).into_series()),
                GetOutput::same_type(),
            )
            .with_fmt("arr.shift")
    }

    /// Slice every sublist.
    pub fn slice(self, offset: i64, length: usize) -> Expr {
        self.0
            .map(
                move |s| Ok(s.list()?.lst_slice(offset, length).into_series()),
                GetOutput::same_type(),
            )
            .with_fmt("arr.slice")
    }

    /// Get the head of every sublist
    pub fn head(self, n: usize) -> Expr {
        self.slice(0, n)
    }

    /// Get the tail of every sublist
    pub fn tail(self, n: usize) -> Expr {
        self.slice(-(n as i64), n)
    }

    /// Run any [`Expr`] on these lists elements
    #[cfg(feature = "list_eval")]
    #[cfg_attr(docsrs, doc(cfg(feature = "list_eval")))]
    pub fn eval(self, expr: Expr, parallel: bool) -> Expr {
        let expr = prepare_eval_expr(expr);

        let expr2 = expr.clone();
        let func = move |s: Series| {
            let lst = s.list()?;

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
                                    *m_err.lock() = Some(e);
                                    None
                                }
                            }
                        })
                    })
                    .collect();
                err = m_err.lock().take();
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

            match err {
                None => Ok(ca.into_series()),
                Some(e) => Err(e),
            }
        };

        self.0
            .map(
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
                            Field::new(f.name(), DataType::List(Box::new(dtype.clone())))
                        }
                        Err(_) => Field::new(f.name(), DataType::Null),
                    }
                }),
            )
            .with_fmt("eval")
    }

    #[cfg(feature = "list_to_struct")]
    #[cfg_attr(docsrs, doc(cfg(feature = "list_to_struct")))]
    #[allow(clippy::wrong_self_convention)]
    /// Convert this `List` to a `Series` of type `Struct`. The width will be determined according to
    /// `ListToStructWidthStrategy` and the names of the fields determined by the given `name_generator`.
    pub fn to_struct(
        self,
        n_fields: ListToStructWidthStrategy,
        name_generator: Option<NameGenerator>,
    ) -> Expr {
        self.0
            .map(
                move |s| {
                    s.list()?
                        .to_struct(n_fields, name_generator.clone())
                        .map(|s| s.into_series())
                },
                // we don't yet know the fields
                GetOutput::from_type(DataType::Struct(vec![])),
            )
            .with_fmt("arr.to_struct")
    }

    #[cfg(feature = "is_in")]
    #[cfg_attr(docsrs, doc(cfg(feature = "is_in")))]
    /// Check if the list array contain an element
    pub fn contains<E: Into<Expr>>(self, other: E) -> Expr {
        let other = other.into();

        Expr::Function {
            input: vec![self.0, other],
            function: FunctionExpr::ListContains,
            options: FunctionOptions {
                collect_groups: ApplyOptions::ApplyFlat,
                input_wildcard_expansion: true,
                auto_explode: true,
                fmt_str: "arr.contains",
                ..Default::default()
            },
        }
    }
}
