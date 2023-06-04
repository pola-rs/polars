use super::*;

#[cfg(feature = "arange")]
fn arange_impl<T>(start: T::Native, end: T::Native, step: i64) -> PolarsResult<Option<Series>>
where
    T: PolarsNumericType,
    ChunkedArray<T>: IntoSeries,
    std::ops::Range<T::Native>: Iterator<Item = T::Native>,
    std::ops::RangeInclusive<T::Native>: DoubleEndedIterator<Item = T::Native>,
{
    let mut ca = match step {
        1 => ChunkedArray::<T>::from_iter_values("arange", start..end),
        2.. => ChunkedArray::<T>::from_iter_values("arange", (start..end).step_by(step as usize)),
        _ => {
            polars_ensure!(start > end, InvalidOperation: "range must be decreasing if 'step' is negative");
            ChunkedArray::<T>::from_iter_values(
                "arange",
                (end..=start).rev().step_by(step.unsigned_abs() as usize),
            )
        }
    };
    let is_sorted = if end < start {
        IsSorted::Descending
    } else {
        IsSorted::Ascending
    };
    ca.set_sorted_flag(is_sorted);
    Ok(Some(ca.into_series()))
}

// TODO! rewrite this with the apply_private architecture
/// Create list entries that are range arrays
/// - if `start` and `end` are a column, every element will expand into an array in a list column.
/// - if `start` and `end` are literals the output will be of `Int64`.
#[cfg(feature = "arange")]
pub fn arange(start: Expr, end: Expr, step: i64) -> Expr {
    let output_name = "arange";

    let has_col_without_agg = |e: &Expr| {
        has_expr(e, |ae| matches!(ae, Expr::Column(_)))
            &&
            // check if there is no aggregation
            !has_expr(e, |ae| {
                matches!(
                    ae,
                    Expr::Agg(_)
                        | Expr::Count
                        | Expr::AnonymousFunction {
                            options: FunctionOptions {
                                collect_groups: ApplyOptions::ApplyGroups,
                                ..
                            },
                            ..
                        }
                        | Expr::Function {
                            options: FunctionOptions {
                                collect_groups: ApplyOptions::ApplyGroups,
                                ..
                            },
                            ..
                        },
                )
            })
    };
    let has_lit = |e: &Expr| {
        (matches!(e, Expr::Literal(_)) && !matches!(e, Expr::Literal(LiteralValue::Series(_))))
    };

    let any_column_no_agg = has_col_without_agg(&start) || has_col_without_agg(&end);
    let literal_start = has_lit(&start);
    let literal_end = has_lit(&end);

    if (literal_start || literal_end) && !any_column_no_agg {
        let f = move |sa: Series, sb: Series| {
            polars_ensure!(step != 0, InvalidOperation: "step must not be zero");

            match sa.dtype() {
                dt if dt == &IDX_DTYPE => {
                    let start = sa
                        .idx()?
                        .get(0)
                        .ok_or_else(|| polars_err!(NoData: "no data in `start` evaluation"))?;
                    let sb = sb.cast(&IDX_DTYPE)?;
                    let end = sb
                        .idx()?
                        .get(0)
                        .ok_or_else(|| polars_err!(NoData: "no data in `end` evaluation"))?;
                    #[cfg(feature = "bigidx")]
                    {
                        arange_impl::<UInt64Type>(start, end, step)
                    }
                    #[cfg(not(feature = "bigidx"))]
                    {
                        arange_impl::<UInt32Type>(start, end, step)
                    }
                }
                _ => {
                    let sa = sa.cast(&DataType::Int64)?;
                    let sb = sb.cast(&DataType::Int64)?;
                    let start = sa
                        .i64()?
                        .get(0)
                        .ok_or_else(|| polars_err!(NoData: "no data in `start` evaluation"))?;
                    let end = sb
                        .i64()?
                        .get(0)
                        .ok_or_else(|| polars_err!(NoData: "no data in `end` evaluation"))?;
                    arange_impl::<Int64Type>(start, end, step)
                }
            }
        };
        apply_binary(
            start,
            end,
            f,
            GetOutput::map_field(|input| {
                let dtype = if input.data_type() == &IDX_DTYPE {
                    IDX_DTYPE
                } else {
                    DataType::Int64
                };
                Field::new(output_name, dtype)
            }),
        )
        .alias(output_name)
    } else {
        let f = move |sa: Series, sb: Series| {
            polars_ensure!(step != 0, InvalidOperation: "step must not be zero");
            let mut sa = sa.cast(&DataType::Int64)?;
            let mut sb = sb.cast(&DataType::Int64)?;

            if sa.len() != sb.len() {
                if sa.len() == 1 {
                    sa = sa.new_from_index(0, sb.len())
                } else if sb.len() == 1 {
                    sb = sb.new_from_index(0, sa.len())
                } else {
                    polars_bail!(
                        ComputeError:
                        "lengths of `start`: {} and `end`: {} arguments `\
                        cannot be matched in the `arange` expression",
                        sa.len(), sb.len()
                    );
                }
            }

            let start = sa.i64()?;
            let end = sb.i64()?;
            let mut builder = ListPrimitiveChunkedBuilder::<Int64Type>::new(
                output_name,
                start.len(),
                start.len() * 3,
                DataType::Int64,
            );

            for (opt_start, opt_end) in start.into_iter().zip(end.into_iter()) {
                match (opt_start, opt_end) {
                    (Some(start_v), Some(end_v)) => match step {
                        1 => {
                            builder.append_iter_values(start_v..end_v);
                        }
                        2.. => {
                            builder.append_iter_values((start_v..end_v).step_by(step as usize));
                        }
                        _ => {
                            polars_ensure!(start_v > end_v, InvalidOperation: "range must be decreasing if 'step' is negative");
                            builder.append_iter_values(
                                (end_v..=start_v)
                                    .rev()
                                    .step_by(step.unsigned_abs() as usize),
                            )
                        }
                    },
                    _ => builder.append_null(),
                }
            }

            Ok(Some(builder.finish().into_series()))
        };
        apply_binary(
            start,
            end,
            f,
            GetOutput::map_field(|_| {
                Field::new(output_name, DataType::List(DataType::Int64.into()))
            }),
        )
        .alias(output_name)
    }
}

pub trait Range<T> {
    fn into_range(self, high: T) -> Expr;
}

macro_rules! impl_into_range {
    ($dt: ty) => {
        impl Range<$dt> for $dt {
            fn into_range(self, high: $dt) -> Expr {
                Expr::Literal(LiteralValue::Range {
                    low: self as i64,
                    high: high as i64,
                    data_type: DataType::Int32,
                })
            }
        }
    };
}

impl_into_range!(i32);
impl_into_range!(i64);
impl_into_range!(u32);

/// Create a range literal.
pub fn range<T: Range<T>>(low: T, high: T) -> Expr {
    low.into_range(high)
}

/// Create a date range from a `start` and `stop` expression.
#[cfg(feature = "temporal")]
pub fn date_range(
    start: Expr,
    end: Expr,
    every: Duration,
    closed: ClosedWindow,
    tz: Option<TimeZone>,
) -> Expr {
    let input = vec![start, end];

    Expr::Function {
        input,
        function: FunctionExpr::TemporalExpr(TemporalFunction::DateRange { every, closed, tz }),
        options: FunctionOptions {
            collect_groups: ApplyOptions::ApplyGroups,
            cast_to_supertypes: true,
            allow_rename: true,
            ..Default::default()
        },
    }
}

/// Create a time range from a `start` and `stop` expression.
#[cfg(feature = "temporal")]
pub fn time_range(start: Expr, end: Expr, every: Duration, closed: ClosedWindow) -> Expr {
    let input = vec![start, end];

    Expr::Function {
        input,
        function: FunctionExpr::TemporalExpr(TemporalFunction::TimeRange { every, closed }),
        options: FunctionOptions {
            collect_groups: ApplyOptions::ApplyGroups,
            cast_to_supertypes: false,
            allow_rename: true,
            ..Default::default()
        },
    }
}

/// Create a column of length `n` containing `n` copies of the literal `value`. Generally you won't need this function,
/// as `lit(value)` already represents a column containing only `value` whose length is automatically set to the correct
/// number of rows.
pub fn repeat<L: Literal>(value: L, n: Expr) -> Expr {
    let function = |s: Series, n: Series| {
        polars_ensure!(
            n.dtype().is_integer(),
            SchemaMismatch: "expected expression of dtype 'integer', got '{}'", n.dtype()
        );
        let first_value = n.get(0)?;
        let n = first_value.extract::<usize>().ok_or_else(
            || polars_err!(ComputeError: "could not parse value '{}' as a size.", first_value),
        )?;
        Ok(Some(s.new_from_index(0, n)))
    };
    apply_binary(lit(value), n, function, GetOutput::same_type()).alias("repeat")
}
