use polars_core::chunked_array::cast::CastOptions;
use polars_core::prelude::*;
use polars_time::Duration;
use polars_utils::arena::Arena;

use crate::plans::{AExpr, ExprIR, IRFunctionExpr, IRTemporalFunction, LiteralValue};

#[macro_export]
macro_rules! ensure_datetime {
    ($dtype:ident) => {
        polars_ensure!(
            matches!($dtype, DataType::Datetime(_, _) | DataType::Date),
            ComputeError: "'{}' must be Date or Datetime, got {:?}", stringify!($dtype), $dtype
        )
    }
}
#[macro_export]
macro_rules! ensure_int {
    ($dtype:ident) => {
        polars_ensure!(
            $dtype.is_integer(),
            ComputeError: "'{}' must be Date or Datetime, got {:?}", stringify!($dtype), $dtype
        )
    }
}
pub use {ensure_datetime, ensure_int};

/// Cast a date or datetime node to a supertype.
///
/// If the target datetime type has a timezone, then:
///   * If the source is a Date, we first cast to naive datetime, then replace the time zone.
///   * If the source is a Datetime and is naive, we replace the time zone.
///   * If the source is a Datetime with a different time zone, we convert time zone.
///
#[cfg(any(feature = "dtype-date", feature = "dtype-datetime"))]
pub fn coerce_temporal_dt(
    from_type: &DataType,
    to_type: &DataType,
    expr: &mut ExprIR,
    expr_arena: &mut Arena<AExpr>,
) -> PolarsResult<()> {
    if from_type != to_type {
        #[cfg(feature = "dtype-datetime")]
        if let &DataType::Datetime(tu, Some(ref from_tz)) = to_type {
            match from_type {
                #[cfg(feature = "dtype-date")]
                DataType::Date => {
                    expr.set_node(expr_arena.add(AExpr::Cast {
                        expr: expr.node(),
                        dtype: DataType::Datetime(tu, None),
                        options: CastOptions::Strict,
                    }));
                    replace_tz(expr, to_type, expr_arena)?;
                    return Ok(());
                },
                #[cfg(feature = "timezones")]
                DataType::Datetime(_, None) => {
                    replace_tz(expr, to_type, expr_arena)?;
                },
                #[cfg(feature = "timezones")]
                DataType::Datetime(_, Some(to_tz)) if from_tz != to_tz => {
                    convert_tz(expr, to_type, expr_arena)?;
                },
                _ => (),
            }
        }
        expr.set_node(expr_arena.add(AExpr::Cast {
            expr: expr.node(),
            dtype: to_type.clone(),
            options: CastOptions::Strict,
        }));
        expr.set_dtype(to_type.clone());
    }
    Ok(())
}

#[cfg(all(feature = "dtype-datetime", feature = "timezones"))]
pub(super) fn replace_tz(
    e: &mut ExprIR,
    dtype: &DataType,
    expr_arena: &mut Arena<AExpr>,
) -> PolarsResult<()> {
    let replacement_expr = match &dtype {
        &DataType::Datetime(_, tz @ Some(_)) => {
            // Wrap our node in a ReplaceTimezone node
            let function = IRFunctionExpr::TemporalExpr(IRTemporalFunction::ReplaceTimeZone(
                tz.clone(),
                NonExistent::Raise,
            ));
            let options = function.function_options();
            AExpr::Function {
                input: {
                    // We must add the ambiguous input argument.
                    let scalar = Scalar::new(DataType::String, AnyValue::from("raise"));
                    let node = expr_arena.add(AExpr::Literal(LiteralValue::Scalar(scalar)));
                    let ambiguous = ExprIR::from_node(node, expr_arena);
                    vec![e.clone(), ambiguous]
                },
                function,
                options,
            }
        },
        dt => polars_bail!(ComputeError: "cannot replace time zone of dtype {:?}", dt),
    };

    e.set_node(expr_arena.add(replacement_expr));
    e.set_dtype(dtype.clone());
    Ok(())
}

#[cfg(all(feature = "dtype-datetime", feature = "timezones"))]
pub(super) fn convert_tz(
    e: &mut ExprIR,
    dtype: &DataType,
    expr_arena: &mut Arena<AExpr>,
) -> PolarsResult<()> {
    let replacement_expr = match &dtype {
        // Wrap our node in a ReplaceTimezone node
        &DataType::Datetime(_, Some(tz)) => {
            let function =
                IRFunctionExpr::TemporalExpr(IRTemporalFunction::ConvertTimeZone(tz.clone()));
            let options = function.function_options();
            AExpr::Function {
                input: vec![e.clone()],
                function,
                options,
            }
        },
        dt => polars_bail!(ComputeError: "cannot convert time zone of dtype {:?}", dt),
    };

    e.set_node(expr_arena.add(replacement_expr));
    e.set_dtype(dtype.clone());
    Ok(())
}

#[doc(hidden)]
#[cfg(feature = "dtype-datetime")]
// Determine the output dtype, given a `Date`/`Datetime` dtype and optional time unit, time zone, and
// interval string.
//
// If an explicit time unit is provided, it is used regardless of the interval's temporal
// granularity.
pub fn temporal_range_output_type(
    dt: DataType,
    tu: &Option<TimeUnit>,
    tz: &Option<TimeZone>,
    interval: &Option<Duration>,
) -> PolarsResult<DataType> {
    let mut dtype_out = match (&dt, tu) {
        (DataType::Date, time_unit) => {
            if let Some(tu) = time_unit {
                DataType::Datetime(*tu, None)
            } else if interval.is_some_and(|i| i.nanoseconds() % 1_000 != 0) {
                DataType::Datetime(TimeUnit::Nanoseconds, None)
            } else {
                // No datatype, use microseconds
                DataType::Datetime(TimeUnit::Microseconds, None)
            }
        },
        // overwrite nothing, keep as-is
        (DataType::Datetime(_, _), None) => dt,
        // overwrite time unit, keep timezone
        (DataType::Datetime(_, tz), Some(tu)) => DataType::Datetime(*tu, tz.clone()),
        (dt, _) => {
            polars_bail!(InvalidOperation: "expected a temporal datatype, got {}", dt)
        },
    };

    // Overwrite time zone, if specified
    #[cfg(feature = "timezones")]
    if let (DataType::Datetime(tu, _), Some(tz)) = (&dtype_out, tz) {
        dtype_out = DataType::Datetime(*tu, Some(tz.clone()));
    };
    Ok(dtype_out)
}
