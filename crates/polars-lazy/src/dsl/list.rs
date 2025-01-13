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

fn offsets_to_groups(offsets: &[i64]) -> Option<GroupPositions> {
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
    Some(
        GroupsType::Slice {
            groups,
            rolling: false,
        }
        .into_sliceable(),
    )
}

fn run_per_sublist(
    s: Column,
    lst: &ListChunked,
    expr: &Expr,
    parallel: bool,
    output_field: Field,
) -> PolarsResult<Option<Column>> {
    let phys_expr = prepare_expression_for_context(
        PlSmallStr::EMPTY,
        expr,
        lst.inner_dtype(),
        Context::Default,
    )?;

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
                        Ok(s) => Some(s.take_materialized_series()),
                        Err(e) => {
                            *m_err.lock().unwrap() = Some(e);
                            None
                        },
                    }
                })
            })
            .collect_ca_with_dtype(PlSmallStr::EMPTY, output_field.dtype.clone());
        err = m_err.into_inner().unwrap();
        ca
    } else {
        let mut df_container = DataFrame::empty();

        lst.into_iter()
            .map(|s| {
                s.and_then(|s| unsafe {
                    df_container.with_column_unchecked(s.into_column());
                    let out = phys_expr.evaluate(&df_container, &state);
                    df_container.clear_columns();
                    match out {
                        Ok(s) => Some(s.take_materialized_series()),
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

    ca.rename(s.name().clone());

    if ca.dtype() != output_field.dtype() {
        ca.cast(output_field.dtype()).map(Column::from).map(Some)
    } else {
        Ok(Some(ca.into_column()))
    }
}

fn run_on_group_by_engine(
    name: PlSmallStr,
    lst: &ListChunked,
    expr: &Expr,
) -> PolarsResult<Option<Column>> {
    let lst = lst.rechunk();
    let arr = lst.downcast_iter().next().unwrap();
    let groups = offsets_to_groups(arr.offsets()).unwrap();

    // List elements in a series.
    let values = Series::try_from((PlSmallStr::EMPTY, arr.values().clone())).unwrap();
    let inner_dtype = lst.inner_dtype();
    // SAFETY:
    // Invariant in List means values physicals can be cast to inner dtype
    let values = unsafe { values.from_physical_unchecked(inner_dtype).unwrap() };

    let df_context = values.into_frame();
    let phys_expr =
        prepare_expression_for_context(PlSmallStr::EMPTY, expr, inner_dtype, Context::Aggregation)?;

    let state = ExecutionState::new();
    let mut ac = phys_expr.evaluate_on_groups(&df_context, &groups, &state)?;
    let out = match ac.agg_state() {
        AggState::AggregatedScalar(_) => {
            let out = ac.aggregated();
            out.as_list().into_column()
        },
        _ => ac.aggregated(),
    };
    Ok(Some(out.with_name(name).into_column()))
}

pub trait ListNameSpaceExtension: IntoListNameSpace + Sized {
    /// Run any [`Expr`] on these lists elements
    fn eval(self, expr: Expr, parallel: bool) -> Expr {
        let this = self.into_list_name_space();

        let expr2 = expr.clone();
        let func = move |c: Column| {
            for e in expr.into_iter() {
                match e {
                    #[cfg(feature = "dtype-categorical")]
                    Expr::Cast {
                        dtype: DataType::Categorical(_, _) | DataType::Enum(_, _),
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
            let lst = c.list()?.clone();

            // # fast returns
            // ensure we get the new schema
            let output_field = eval_field_to_dtype(lst.ref_field(), &expr, true);
            if lst.is_empty() {
                return Ok(Some(Column::new_empty(
                    c.name().clone(),
                    output_field.dtype(),
                )));
            }
            if lst.null_count() == lst.len() {
                return Ok(Some(c.cast(output_field.dtype())?.into_column()));
            }

            let fits_idx_size = lst.get_values_size() <= (IdxSize::MAX as usize);
            // If a users passes a return type to `apply`, e.g. `return_dtype=pl.Int64`,
            // this fails as the list builder expects `List<Int64>`, so let's skip that for now.
            let is_user_apply = || {
                expr.into_iter().any(|e| matches!(e, Expr::AnonymousFunction { options, .. } if options.fmt_str == MAP_LIST_NAME))
            };

            if fits_idx_size && c.null_count() == 0 && !is_user_apply() {
                run_on_group_by_engine(c.name().clone(), &lst, &expr)
            } else {
                run_per_sublist(c, &lst, &expr, parallel, output_field)
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
