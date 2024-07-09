use std::sync::Mutex;

use arrow::array::ValueSize;
use arrow::legacy::utils::CustomIterTools;
use polars_core::chunked_array::from_iterator_par::ChunkedCollectParIterExt;
use polars_core::prelude::*;
use polars_plan::constants::MAP_LIST_NAME;
use polars_plan::dsl::*;
use rayon::prelude::*;

use crate::physical_plan::exotic::prepare_expression_for_context;
use crate::prelude::*;

pub trait IntoListNameSpace {
    fn into_list_name_space(self) -> ListNameSpace;
}

impl IntoListNameSpace for ListNameSpace {
    fn into_list_name_space(self) -> ListNameSpace {
        self
    }
}

fn offsets_to_groups(offsets: &[i64]) -> Option<GroupsProxy> {
    let mut start = offsets[0];
    let end = *offsets.last().unwrap();
    if IdxSize::try_from(end - start).is_err() {
        return None;
    }
    let groups = offsets
        .iter()
        .skip(1)
        .map(|end| {
            let offset = start as IdxSize;
            let len = (*end - start) as IdxSize;
            start = *end;
            [offset, len]
        })
        .collect();
    Some(GroupsProxy::Slice {
        groups,
        rolling: false,
    })
}

fn run_per_sublist(
    s: Series,
    lst: &ListChunked,
    expr: &Expr,
    parallel: bool,
    output_field: Field,
) -> PolarsResult<Option<Series>> {
    let phys_expr = prepare_expression_for_context("", expr, lst.inner_dtype(), Context::Default)?;

    let state = ExecutionState::new();

    let mut err = None;
    let mut ca: ListChunked = if parallel {
        let m_err = Mutex::new(None);
        let ca: ListChunked = lst
            .par_iter()
            .map(|opt_s| {
                opt_s.and_then(|s| {
                    let df = s.into_frame();
                    let out = phys_expr.evaluate(&df, &state);
                    match out {
                        Ok(s) => Some(s),
                        Err(e) => {
                            *m_err.lock().unwrap() = Some(e);
                            None
                        },
                    }
                })
            })
            .collect_ca_with_dtype("", output_field.dtype.clone());
        err = m_err.into_inner().unwrap();
        ca
    } else {
        let mut df_container = DataFrame::empty();

        lst.into_iter()
            .map(|s| {
                s.and_then(|s| unsafe {
                    df_container.get_columns_mut().push(s);
                    let out = phys_expr.evaluate(&df_container, &state);
                    df_container.get_columns_mut().clear();
                    match out {
                        Ok(s) => Some(s),
                        Err(e) => {
                            err = Some(e);
                            None
                        },
                    }
                })
            })
            .collect_trusted()
    };
    if let Some(err) = err {
        return Err(err);
    }

    ca.rename(s.name());

    if ca.dtype() != output_field.data_type() {
        ca.cast(output_field.data_type()).map(Some)
    } else {
        Ok(Some(ca.into_series()))
    }
}

fn run_on_group_by_engine(
    name: &str,
    lst: &ListChunked,
    expr: &Expr,
) -> PolarsResult<Option<Series>> {
    let lst = lst.rechunk();
    let arr = lst.downcast_iter().next().unwrap();
    let groups = offsets_to_groups(arr.offsets()).unwrap();

    // List elements in a series.
    let values = Series::try_from(("", arr.values().clone())).unwrap();
    let inner_dtype = lst.inner_dtype();
    // SAFETY:
    // Invariant in List means values physicals can be cast to inner dtype
    let values = unsafe { values.cast_unchecked(inner_dtype).unwrap() };

    let df_context = values.into_frame();
    let phys_expr = prepare_expression_for_context("", expr, inner_dtype, Context::Aggregation)?;

    let state = ExecutionState::new();
    let mut ac = phys_expr.evaluate_on_groups(&df_context, &groups, &state)?;
    let out = match ac.agg_state() {
        AggState::AggregatedScalar(_) | AggState::Literal(_) => {
            let out = ac.aggregated();
            out.as_list().into_series()
        },
        _ => ac.aggregated(),
    };
    Ok(Some(out.with_name(name)))
}

pub trait ListNameSpaceExtension: IntoListNameSpace + Sized {
    /// Run any [`Expr`] on these lists elements
    fn eval(self, expr: Expr, parallel: bool) -> Expr {
        let this = self.into_list_name_space();

        let expr2 = expr.clone();
        let func = move |s: Series| {
            for e in expr.into_iter() {
                match e {
                    #[cfg(feature = "dtype-categorical")]
                    Expr::Cast {
                        data_type: DataType::Categorical(_, _) | DataType::Enum(_, _),
                        ..
                    } => {
                        polars_bail!(
                            ComputeError: "casting to categorical not allowed in `list.eval`"
                        )
                    },
                    Expr::Column(name) => {
                        polars_ensure!(
                            name.is_empty(),
                            ComputeError:
                            "named columns are not allowed in `list.eval`; consider using `element` or `col(\"\")`"
                        );
                    },
                    _ => {},
                }
            }
            let lst = s.list()?.clone();

            // # fast returns
            // ensure we get the new schema
            let output_field = eval_field_to_dtype(lst.ref_field(), &expr, true);
            if lst.is_empty() {
                return Ok(Some(Series::new_empty(s.name(), output_field.data_type())));
            }
            if lst.null_count() == lst.len() {
                return Ok(Some(s.cast(output_field.data_type())?));
            }

            let fits_idx_size = lst.get_values_size() <= (IdxSize::MAX as usize);
            // If a users passes a return type to `apply`, e.g. `return_dtype=pl.Int64`,
            // this fails as the list builder expects `List<Int64>`, so let's skip that for now.
            let is_user_apply = || {
                expr.into_iter().any(|e| matches!(e, Expr::AnonymousFunction { options, .. } if options.fmt_str == MAP_LIST_NAME))
            };

            if fits_idx_size && s.null_count() == 0 && !is_user_apply() {
                run_on_group_by_engine(s.name(), &lst, &expr)
            } else {
                run_per_sublist(s, &lst, &expr, parallel, output_field)
            }
        };

        this.0
            .map(
                func,
                GetOutput::map_field(move |f| Ok(eval_field_to_dtype(f, &expr2, true))),
            )
            .with_fmt("eval")
    }
}

impl ListNameSpaceExtension for ListNameSpace {}
