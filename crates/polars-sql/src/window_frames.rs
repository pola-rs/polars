//! SQL aggregates (SUM, COUNT, MIN, MAX, AVG, TOTAL) over the window frame of an OVER clause.

use polars_core::chunked_array::ops::FillNullStrategy;
#[cfg(feature = "rolling_window")]
use polars_core::prelude::RollingOptionsFixedWindow;
use polars_core::prelude::{DataType, PolarsResult, polars_bail, polars_err};
use polars_lazy::dsl::Expr;
use polars_plan::dsl::functions::{len, lit, repeat, when};
use polars_plan::plans::{DynLiteralValue, LiteralValue};
use sqlparser::ast::{Expr as SQLExpr, FunctionArgExpr, WindowFrameBound, WindowFrameUnits};

use crate::functions::{
    PolarsSQLFunctions, SQLFunctionVisitor, extract_args_distinct, is_constant_key, is_first_peer,
    is_last_peer, is_non_null_literal, sql_sum, window_row_index,
};
use crate::sql_expr::parse_sql_expr;

/// The rows of a window frame, in the sorted window partition.
#[derive(Clone, Copy)]
enum FrameShape {
    /// All rows of the partition.
    Partition,
    /// From the first row to the current row, or to its last peer if `peers`.
    ToCurrentRow { peers: bool },
    /// From the current row, or from its first peer if `peers`, to the last row.
    FromCurrentRow { peers: bool },
    /// The current row and its peers.
    Peers,
    /// The current row and the given number of rows before it.
    Preceding(usize),
}

enum Aggregate {
    Avg(Expr),
    Count(Expr),
    /// `COUNT(*)`
    CountRows,
    Max(Expr),
    Min(Expr),
    Sum(Expr),
}

/// Whether the function is an aggregate that is computed over its window frame.
pub(crate) fn is_frame_aggregate(function: &PolarsSQLFunctions, is_distinct: bool) -> bool {
    use PolarsSQLFunctions::*;
    match function {
        Avg | Max | Min | Sum | Total => true,
        Count => !is_distinct,
        _ => false,
    }
}

impl SQLFunctionVisitor<'_> {
    /// Lower an aggregate with an OVER clause (see [`is_frame_aggregate`]).
    pub(crate) fn visit_window_aggregate(
        &mut self,
        function: &PolarsSQLFunctions,
    ) -> PolarsResult<Expr> {
        use PolarsSQLFunctions::*;
        let order_keys = self.parse_window_order_keys()?;
        let frame = self.window_frame_shape(!order_keys.is_empty())?;

        let (args, is_distinct) = extract_args_distinct(self.func)?;
        let value = match args.as_slice() {
            [] | [FunctionArgExpr::Wildcard] if matches!(function, Count) => None,
            [FunctionArgExpr::Expr(e)] if matches!(function, Count) && is_non_null_literal(e) => {
                None
            },
            [FunctionArgExpr::Expr(e)] => Some(parse_sql_expr(e, self.ctx, self.active_schema)?),
            _ => return self.not_supported_error(),
        };
        // Rows that don't pass the FILTER become NULL, which these aggregates skip.
        let value = match (&self.filter, value) {
            (Some(predicate), value) => Some(
                when(predicate.clone())
                    .then(value.unwrap_or(lit(true)))
                    .otherwise(Expr::Literal(LiteralValue::untyped_null())),
            ),
            (None, value) => value,
        };
        // A scalar counts once per row, as in `SUM(1)`.
        let is_scalar = value.as_ref().is_some_and(is_constant_key);
        let value = value.map(|value| match frame {
            FrameShape::Partition | FrameShape::Peers => value,
            _ if is_scalar => repeat(value, len()),
            _ => value,
        });

        // DISTINCT doesn't change MIN and MAX.
        let distinct_supported = match function {
            Max | Min => true,
            Avg => matches!(frame, FrameShape::Partition),
            _ => false,
        };
        if is_distinct && !distinct_supported {
            polars_bail!(
                SQLInterface: "{}(DISTINCT ...) is not supported with this OVER clause",
                self.func.name
            );
        }
        let aggregate = match (function, value) {
            (Count, None) => Aggregate::CountRows,
            (Count, Some(value)) => Aggregate::Count(value),
            (Avg, Some(value)) if is_distinct => Aggregate::Avg(value.unique()),
            (Avg, Some(value)) => Aggregate::Avg(value),
            (Max, Some(value)) => Aggregate::Max(value),
            (Min, Some(value)) => Aggregate::Min(value),
            (Sum | Total, Some(value)) => Aggregate::Sum(value),
            _ => return self.not_supported_error(),
        };

        let keys: Vec<Expr> = order_keys.iter().map(|(key, _)| key.clone()).collect();
        let (expr, peer_keys, order_keys) = match frame {
            FrameShape::Partition => (aggregate.whole(is_scalar), Vec::new(), Vec::new()),
            FrameShape::Peers => (aggregate.whole(is_scalar), keys, Vec::new()),
            FrameShape::ToCurrentRow { peers } => {
                let running = aggregate.running(false);
                let expr = if peers {
                    at_last_peer(running, &keys)
                } else {
                    running
                };
                (expr, Vec::new(), order_keys)
            },
            FrameShape::FromCurrentRow { peers } => {
                let running = aggregate.running(true);
                let expr = if peers {
                    at_first_peer(running, &keys)
                } else {
                    running
                };
                (expr, Vec::new(), order_keys)
            },
            FrameShape::Preceding(n) => (aggregate.rolling(n)?, Vec::new(), order_keys),
        };
        let expr = match function {
            Total => expr.cast(DataType::Float64).fill_null(lit(0.0)),
            _ => expr,
        };
        self.apply_over(expr, peer_keys, order_keys)
    }

    /// The rows of the window frame. Without ORDER BY keys, all rows are peers.
    fn window_frame_shape(&mut self, has_order_keys: bool) -> PolarsResult<FrameShape> {
        use WindowFrameBound::{CurrentRow, Following, Preceding};
        let spec = self.window.as_ref().unwrap();
        let has_order_by = !spec.order_by.is_empty();
        let Some(frame) = spec.window_frame.clone() else {
            return Ok(if has_order_keys {
                FrameShape::ToCurrentRow { peers: true }
            } else {
                FrameShape::Partition
            });
        };
        let end_bound = frame.end_bound.clone().unwrap_or(CurrentRow);
        let shape = match (&frame.units, &frame.start_bound, &end_bound) {
            (_, Preceding(None), Following(None)) => Some(FrameShape::Partition),
            (WindowFrameUnits::Rows, _, _) if !has_order_by => polars_bail!(
                SQLInterface: "{} with a ROWS window frame but no ORDER BY is not supported",
                self.func.name
            ),
            (WindowFrameUnits::Rows, Preceding(None), CurrentRow) => {
                Some(FrameShape::ToCurrentRow { peers: false })
            },
            (WindowFrameUnits::Rows, CurrentRow, Following(None)) => {
                Some(FrameShape::FromCurrentRow { peers: false })
            },
            (WindowFrameUnits::Rows, CurrentRow, CurrentRow) => Some(FrameShape::Preceding(0)),
            (WindowFrameUnits::Rows, Preceding(Some(n)), CurrentRow) => {
                Some(FrameShape::Preceding(self.window_frame_offset(n)?))
            },
            (WindowFrameUnits::Rows, _, _) => None,
            // In RANGE and GROUPS frames, the current row includes its peers.
            (_, Preceding(None) | CurrentRow, CurrentRow | Following(None)) if !has_order_keys => {
                Some(FrameShape::Partition)
            },
            (_, Preceding(None), CurrentRow) => Some(FrameShape::ToCurrentRow { peers: true }),
            (_, CurrentRow, Following(None)) => Some(FrameShape::FromCurrentRow { peers: true }),
            (_, CurrentRow, CurrentRow) => Some(FrameShape::Peers),
            _ => None,
        };
        shape.ok_or_else(|| {
            polars_err!(
                SQLInterface: "{} with window frame '{} BETWEEN {} AND {}' is not supported",
                self.func.name, frame.units, frame.start_bound, end_bound
            )
        })
    }

    fn window_frame_offset(&mut self, offset: &SQLExpr) -> PolarsResult<usize> {
        let n = match parse_sql_expr(offset, self.ctx, self.active_schema)? {
            Expr::Literal(LiteralValue::Dyn(DynLiteralValue::Int(n))) => usize::try_from(n).ok(),
            _ => None,
        };
        n.ok_or_else(|| {
            polars_err!(
                SQLSyntax: "window frame offset must be a non-negative integer; found {}", offset
            )
        })
    }
}

impl Aggregate {
    /// The aggregate over all rows. A `scalar` value counts once per row.
    fn whole(self, scalar: bool) -> Expr {
        match self {
            // Scaled by the number of rows, not repeated: a reduction inside a reduction reads
            // as a nested aggregate (`SUM(SUM(x))`), which gives one output row.
            Aggregate::Count(value) if scalar => (value.count() * len()).cast(DataType::Int64),
            Aggregate::Sum(value) if scalar => value * len().cast(DataType::Int64),
            Aggregate::Avg(value) => value.mean(),
            Aggregate::Count(value) => value.count().cast(DataType::Int64),
            Aggregate::CountRows => len().cast(DataType::Int64),
            Aggregate::Max(value) => value.max(),
            Aggregate::Min(value) => value.min(),
            Aggregate::Sum(value) => sql_sum(value),
        }
    }

    /// The aggregate over the rows up to each row, or from each row if `reverse`.
    fn running(&self, reverse: bool) -> Expr {
        // The running value is NULL where the value is NULL; take it from the row before (or
        // after, if `reverse`).
        let fill = if reverse {
            FillNullStrategy::Backward(None)
        } else {
            FillNullStrategy::Forward(None)
        };
        match self {
            Aggregate::Avg(value) => {
                // Sum in Float64, so that an integer sum can't overflow.
                let sum = Aggregate::Sum(value.clone().cast(DataType::Float64)).running(reverse);
                let count = Aggregate::Count(value.clone()).running(reverse);
                sum / count
            },
            Aggregate::Count(value) => value.clone().cum_count(reverse).cast(DataType::Int64),
            Aggregate::CountRows => {
                let (row_index, n) = window_row_index();
                if reverse {
                    n - row_index
                } else {
                    row_index + lit(1i64)
                }
            },
            Aggregate::Max(value) => value.clone().cum_max(reverse).fill_null_with_strategy(fill),
            Aggregate::Min(value) => value.clone().cum_min(reverse).fill_null_with_strategy(fill),
            Aggregate::Sum(value) => value.clone().cum_sum(reverse).fill_null_with_strategy(fill),
        }
    }

    /// The aggregate over each row and the `n` rows before it.
    fn rolling(&self, n: usize) -> PolarsResult<Expr> {
        #[cfg(feature = "rolling_window")]
        {
            let window_size = n.saturating_add(1);
            let options = RollingOptionsFixedWindow {
                window_size,
                min_periods: 1,
                ..Default::default()
            };
            Ok(match self {
                Aggregate::Avg(value) => value.clone().rolling_mean(options),
                Aggregate::Count(value) => value
                    .clone()
                    .is_not_null()
                    .cast(DataType::Int64)
                    .rolling_sum(options),
                Aggregate::CountRows => {
                    let (row_index, _) = window_row_index();
                    let window_size = lit(i64::try_from(window_size).unwrap_or(i64::MAX));
                    when(row_index.clone().lt(window_size.clone()))
                        .then(row_index + lit(1i64))
                        .otherwise(window_size)
                },
                Aggregate::Max(value) => value.clone().rolling_max(options),
                Aggregate::Min(value) => value.clone().rolling_min(options),
                Aggregate::Sum(value) => value.clone().rolling_sum(options),
            })
        }
        #[cfg(not(feature = "rolling_window"))]
        {
            let _ = n;
            polars_bail!(SQLInterface: "'ROWS <n> PRECEDING' window frames require the 'rolling_window' feature")
        }
    }
}

/// `expr` at the last peer of each row, in the sorted window.
fn at_last_peer(expr: Expr, keys: &[Expr]) -> Expr {
    let (row_index, n) = window_row_index();
    let last_peer = when(is_last_peer(keys, &row_index, &n))
        .then(row_index)
        .otherwise(n - lit(1i64))
        .cum_min(true);
    expr.gather(last_peer, false)
}

/// `expr` at the first peer of each row, in the sorted window.
fn at_first_peer(expr: Expr, keys: &[Expr]) -> Expr {
    let (row_index, _) = window_row_index();
    let first_peer = when(is_first_peer(keys, &row_index))
        .then(row_index)
        .otherwise(lit(0i64))
        .cum_max(false);
    expr.gather(first_peer, false)
}
