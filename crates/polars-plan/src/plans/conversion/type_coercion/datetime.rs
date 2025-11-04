use polars_core::chunked_array::cast::CastOptions;
use polars_core::prelude::*;
use polars_core::utils::try_get_supertype;
use polars_time::Duration;
use polars_utils::arena::Arena;

use super::try_get_dtype;
use crate::plans::{AExpr, ExprIR, IRFunctionExpr, IRTemporalFunction, LiteralValue};
use crate::prelude::FunctionOptions;

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

// Determines the output data type, including user-specified t tz, and interval.
#[cfg(feature = "dtype-datetime")]
pub(super) fn temporal_range_output_type(
    start_end_supertype: DataType,
    tu: &Option<TimeUnit>,
    tz: &Option<TimeZone>,
    interval: &Duration,
) -> PolarsResult<DataType> {
    let mut dtype_out = match (&start_end_supertype, tu) {
        (DataType::Date, time_unit) => {
            if let Some(tu) = time_unit {
                DataType::Datetime(*tu, None)
            } else if interval.nanoseconds() % 1_000 != 0 {
                DataType::Datetime(TimeUnit::Nanoseconds, None)
            } else {
                // No datatype, use microseconds
                DataType::Datetime(TimeUnit::Microseconds, None)
            }
        },
        // overwrite nothing, keep as-is
        (DataType::Datetime(_, _), None) => start_end_supertype,
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

#[cfg(all(feature = "dtype-date", feature = "range"))]
pub(super) fn update_date_range_types(
    input: &mut [ExprIR],
    expr_arena: &Arena<AExpr>,
    schema: &Schema,
) -> PolarsResult<(Vec<DataType>, Vec<DataType>)> {
    let type_start = try_get_dtype(expr_arena, input[0].node(), schema)?;
    let type_end = try_get_dtype(expr_arena, input[1].node(), schema)?;
    let from_types = vec![type_start, type_end];
    let to_types = vec![DataType::Date, DataType::Date];
    Ok((from_types, to_types))
}

#[cfg(all(feature = "dtype-datetime", feature = "range"))]
pub(super) fn update_datetime_range_types(
    input: &mut [ExprIR],
    expr_arena: &Arena<AExpr>,
    schema: &Schema,
    interval: &Duration,
    tu: &Option<TimeUnit>,
    tz: &Option<TimeZone>,
) -> PolarsResult<(Vec<DataType>, Vec<DataType>)> {
    let type_start = try_get_dtype(expr_arena, input[0].node(), schema)?;
    let type_end = try_get_dtype(expr_arena, input[1].node(), schema)?;
    let default = try_get_supertype(&type_start, &type_end)?;
    let supertype = temporal_range_output_type(default, tu, tz, interval)?;
    let from_types = vec![type_start, type_end];
    let to_types = vec![supertype.clone(), supertype];
    Ok((from_types, to_types))
}
