use polars_core::prelude::*;
use polars_core::utils::get_supertype;
use polars_time::Duration;
use polars_utils::arena::Arena;

use super::get_aexpr_and_type;
use crate::dsl::DateRangeArgs;
use crate::plans::{AExpr, ExprIR, IRFunctionExpr, IRTemporalFunction, LiteralValue};
use crate::prelude::FunctionOptions;

macro_rules! unpack {
    ($packed:expr) => {
        match $packed {
            Some(payload) => payload,
            None => return Ok(None),
        }
    };
}

macro_rules! extract_date {
    ($input:expr, $expr_arena:expr, $schema:expr, $idx:literal, $arg:literal) => {{
        let (_, dtype) =
            unpack!(get_aexpr_and_type($expr_arena, $input[$idx].node(), $schema));
        polars_ensure!(
            matches!(dtype, DataType::Datetime(_, _) | DataType::Date),
            ComputeError: "'{}' must be Date or Datetime, got {:?}", $arg, dtype
        );
        dtype
    }}
}

macro_rules! extract_samples {
    ($input:expr, $expr_arena:expr, $schema:expr, $idx:literal) => {{
        let (_, dtype) =
        unpack!(get_aexpr_and_type($expr_arena, $input[$idx].node(), $schema));
        polars_ensure!(
            dtype.is_integer(),
            ComputeError: "'num_samples' must be integer, got {:?}", dtype
        );
        dtype
    }}
}

// Determines the output data type, including user-specified t tz, and interval.
pub(super) fn build_datetime_supertype(
    default: DataType,
    tu: &Option<TimeUnit>,
    tz: &Option<TimeZone>,
    interval: &Option<Duration>,
) -> PolarsResult<DataType> {
    let mut dtype_out = match (&default, tu) {
        (DataType::Date, time_unit) => {
            if let Some(tu) = time_unit {
                DataType::Datetime(*tu, None)
            } else if interval.is_some_and(|i| i.nanoseconds() % 1_000 != 0) {
                DataType::Datetime(TimeUnit::Nanoseconds, None)
            } else {
                DataType::Datetime(TimeUnit::Microseconds, None)
            }
        },
        // overwrite nothing, keep as-is
        (DataType::Datetime(_, _), None) => default,
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

pub(super) fn replace_tz(
    e: &mut ExprIR,
    dtype: &DataType,
    expr_arena: &mut Arena<AExpr>,
) -> PolarsResult<()> {
    let replacement_expr = match &dtype {
        // Wrap our node in a ReplaceTimezone node
        &DataType::Datetime(_, tz @ Some(_)) => AExpr::Function {
            input: {
                // We must add the ambiguous input argument.
                let scalar = Scalar::new(DataType::String, AnyValue::from("raise"));
                let node = expr_arena.add(AExpr::Literal(LiteralValue::Scalar(scalar)));
                let ambiguous = ExprIR::from_node(node, expr_arena);
                vec![e.clone(), ambiguous]
            },
            function: IRFunctionExpr::TemporalExpr(IRTemporalFunction::ReplaceTimeZone(
                tz.clone(),
                NonExistent::Raise,
            )),
            options: FunctionOptions::elementwise(),
        },
        dt => polars_bail!(ComputeError: "cannot replace time zone of dtype {:?}", dt),
    };

    e.set_node(expr_arena.add(replacement_expr));
    e.set_dtype(dtype.clone());
    Ok(())
}

pub(super) fn update_date_range_types(
    input: &mut Vec<ExprIR>,
    expr_arena: &Arena<AExpr>,
    schema: &Schema,
    arg_type: DateRangeArgs,
) -> PolarsResult<Option<(Vec<DataType>, Vec<DataType>)>> {
    let dt_date = DataType::Date;
    Ok(Some(match arg_type {
        DateRangeArgs::StartEndInterval => {
            let type_start = extract_date!(input, expr_arena, schema, 0, "start");
            let type_end = extract_date!(input, expr_arena, schema, 1, "end");
            let from_types = vec![type_start, type_end];
            let to_types = vec![dt_date.clone(), dt_date];
            (from_types, to_types)
        },
        DateRangeArgs::StartEndSamples => {
            let type_start = extract_date!(input, expr_arena, schema, 0, "start");
            let type_end = extract_date!(input, expr_arena, schema, 1, "end");
            let type_samples = extract_samples!(input, expr_arena, schema, 2);
            let from_types = vec![type_start, type_end, type_samples];
            let to_types = vec![dt_date.clone(), dt_date, DataType::UInt64];
            (from_types, to_types)
        },
        DateRangeArgs::StartIntervalSamples => {
            let type_start = extract_date!(input, expr_arena, schema, 0, "start");
            let type_samples = extract_samples!(input, expr_arena, schema, 1);
            let from_types = vec![type_start.clone(), type_samples];
            let to_types = vec![type_start, DataType::UInt64];
            (from_types, to_types)
        },
        DateRangeArgs::EndIntervalSamples => {
            let type_end = extract_date!(input, expr_arena, schema, 0, "end");
            let type_samples = extract_samples!(input, expr_arena, schema, 1);
            let from_types = vec![type_end.clone(), type_samples];
            let to_types = vec![dt_date, DataType::UInt64];
            (from_types, to_types)
        },
    }))
}

pub(super) fn update_datetime_range_types(
    input: &mut Vec<ExprIR>,
    expr_arena: &Arena<AExpr>,
    schema: &Schema,
    interval: &Option<Duration>,
    tu: &Option<TimeUnit>,
    tz: &Option<TimeZone>,
    arg_type: DateRangeArgs,
) -> PolarsResult<Option<(Vec<DataType>, Vec<DataType>)>> {
    Ok(Some(match arg_type {
        DateRangeArgs::StartEndInterval => {
            // Determine supertype of input types.
            let type_start = extract_date!(input, expr_arena, schema, 0, "start");
            let type_end = extract_date!(input, expr_arena, schema, 1, "end");
            let default = unpack!(get_supertype(&type_start, &type_end));
            let supertype = build_datetime_supertype(default, tu, tz, interval)?;
            let from_types = vec![type_start, type_end];
            let to_types = vec![supertype.clone(), supertype];
            (from_types, to_types)
        },
        DateRangeArgs::StartEndSamples => {
            let type_start = extract_date!(input, expr_arena, schema, 0, "start");
            let type_end = extract_date!(input, expr_arena, schema, 1, "end");
            let type_samples = extract_samples!(input, expr_arena, schema, 2);
            let default = unpack!(get_supertype(&type_start, &type_end));
            let supertype = build_datetime_supertype(default, tu, tz, interval)?;
            let from_types = vec![type_start, type_end, type_samples];
            let to_types = vec![supertype.clone(), supertype, DataType::UInt64];
            (from_types, to_types)
        },
        DateRangeArgs::StartIntervalSamples => {
            let type_start = extract_date!(input, expr_arena, schema, 0, "start");
            let type_samples = extract_samples!(input, expr_arena, schema, 1);
            let supertype = build_datetime_supertype(type_start.clone(), tu, tz, interval)?;
            let from_types = vec![supertype.clone(), type_samples];
            let to_types = vec![supertype, DataType::UInt64];
            (from_types, to_types)
        },
        DateRangeArgs::EndIntervalSamples => {
            let type_end = extract_date!(input, expr_arena, schema, 0, "end");
            let type_samples = extract_samples!(input, expr_arena, schema, 1);
            let supertype = build_datetime_supertype(type_end.clone(), tu, tz, interval)?;
            let from_types = vec![type_end, type_samples];
            let to_types = vec![supertype, DataType::UInt64];
            (from_types, to_types)
        },
    }))
}
