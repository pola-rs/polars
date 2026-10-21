use polars_arrow::temporal_conversions::{
    MICROSECONDS_IN_DAY, MILLISECONDS_IN_DAY, NANOSECONDS_IN_DAY, date32_to_date_opt,
    timestamp_ms_to_datetime_opt, timestamp_ns_to_datetime_opt, timestamp_us_to_datetime_opt,
};
use polars_defs::join::{AsOfOptions, AsofStrategy};

use super::*;
use crate::plans::aexpr::filter_constraint::{and_chain_bounds, comparison_node};
use crate::plans::aexpr::or_factoring::collect_or_branches;

/// The right keys an as-of join can match to a left key `t` lie in
/// `[t - below, t + above]`. `None` means that side has no limit.
struct MatchWindow {
    below: Option<u64>,
    above: Option<u64>,
}

impl MatchWindow {
    fn new(asof: &AsOfOptions) -> Option<Self> {
        // The planner stores the tolerance of integer keys as an unsigned distance.
        let tolerance = match &asof.tolerance {
            None => None,
            Some(tolerance) if tolerance.dtype() == &DataType::UInt64 => {
                Some(tolerance.value().extract::<u64>()?)
            },
            Some(_) => return None,
        };
        let (below, above) = match asof.strategy {
            AsofStrategy::Backward => (tolerance, Some(0)),
            AsofStrategy::Forward => (Some(0), tolerance),
            AsofStrategy::Nearest => (tolerance, tolerance),
        };
        (below.is_some() || above.is_some()).then_some(Self { below, above })
    }
}

/// Filters on `right_key` that keep every right row the as-of join can match to a left row
/// passing `predicate`, which only reads the join output (`left_key` is the left key there).
///
/// `predicate` must be row-wise; the pushdown checks before the join ensure that.
pub(super) fn asof_right_predicates(
    predicate: Node,
    asof: &AsOfOptions,
    left_key: &PlSmallStr,
    right_key: &PlSmallStr,
    key_dtype: &DataType,
    schema_left: &Schema,
    expr_arena: &mut Arena<AExpr>,
) -> Vec<Node> {
    if physical_range(key_dtype).is_none() {
        return Vec::new();
    }
    let Some(window) = MatchWindow::new(asof) else {
        return Vec::new();
    };

    let conjuncts = MintermIter::new(predicate, expr_arena).collect::<Vec<_>>();
    let mut branches = Vec::new();
    let mut out = Vec::new();
    for conjunct in conjuncts {
        branches.clear();
        collect_or_branches(conjunct, expr_arena, &mut branches);
        let Some(intervals) = key_intervals(&branches, left_key, &window, schema_left, expr_arena)
        else {
            continue;
        };
        if let Some(node) = intervals_predicate(intervals, right_key, key_dtype, expr_arena) {
            out.push(node);
        }
    }
    out
}

/// The ranges of right keys the OR of `branches` can match, as physical values.
/// `i128::MIN` / `i128::MAX` mean no bound.
fn key_intervals(
    branches: &[Node],
    left_key: &PlSmallStr,
    window: &MatchWindow,
    schema_left: &Schema,
    expr_arena: &Arena<AExpr>,
) -> Option<Vec<(i128, i128)>> {
    let physical = |s: &Scalar| s.value().extract::<i128>();

    let mut intervals = Vec::with_capacity(branches.len());
    for &branch in branches {
        let bounds = and_chain_bounds(branch, schema_left, expr_arena);
        if bounds.unsat {
            continue;
        }
        let key_bounds = bounds.columns.iter().find(|c| c.name == *left_key)?;

        let mut lower = key_bounds.lower.as_ref().and_then(|(v, _)| physical(v));
        let mut upper = key_bounds.upper.as_ref().and_then(|(v, _)| physical(v));
        if let Some(allowed) = &key_bounds.allowed {
            let (min, max) = allowed
                .iter()
                .try_fold((i128::MAX, i128::MIN), |(min, max), v| {
                    let v = physical(v)?;
                    Some((min.min(v), max.max(v)))
                })?;
            lower = Some(lower.map_or(min, |l| l.max(min)));
            upper = Some(upper.map_or(max, |u| u.min(max)));
        }

        let lower = lower
            .zip(window.below)
            .map_or(i128::MIN, |(v, d)| v - d as i128);
        let upper = upper
            .zip(window.above)
            .map_or(i128::MAX, |(v, d)| v + d as i128);
        if lower == i128::MIN && upper == i128::MAX {
            return None;
        }
        intervals.push((lower, upper));
    }

    (!intervals.is_empty()).then_some(intervals)
}

/// `OR` over the intervals of `col(right_key) >= lower AND col(right_key) <= upper`, with
/// overlapping intervals merged and bounds that every value of `key_dtype` meets dropped.
fn intervals_predicate(
    mut intervals: Vec<(i128, i128)>,
    right_key: &PlSmallStr,
    key_dtype: &DataType,
    expr_arena: &mut Arena<AExpr>,
) -> Option<Node> {
    intervals.sort_unstable();
    let mut n_merged = 0;
    for i in 0..intervals.len() {
        let (lower, upper) = intervals[i];
        if n_merged > 0 && lower <= intervals[n_merged - 1].1.saturating_add(1) {
            let last = &mut intervals[n_merged - 1].1;
            *last = (*last).max(upper);
        } else {
            intervals[n_merged] = (lower, upper);
            n_merged += 1;
        }
    }
    intervals.truncate(n_merged);

    let (min, max) = physical_range(key_dtype)?;
    let mut out: Option<AExprBuilder> = None;
    for (lower, upper) in intervals {
        let lower = (lower > min)
            .then(|| key_scalar(lower, key_dtype))
            .flatten()
            .map(|v| comparison_node(right_key, Operator::GtEq, v, expr_arena));
        let upper = (upper < max)
            .then(|| key_scalar(upper, key_dtype))
            .flatten()
            .map(|v| comparison_node(right_key, Operator::LtEq, v, expr_arena));
        let interval = match (lower, upper) {
            (Some(l), Some(u)) => AExprBuilder::new_from_node(l).and(u, expr_arena),
            (Some(n), None) | (None, Some(n)) => AExprBuilder::new_from_node(n),
            (None, None) => return None,
        };
        out = Some(match out {
            None => interval,
            Some(acc) => acc.or(interval, expr_arena),
        });
    }
    out.map(AExprBuilder::node)
}

/// The smallest and largest valid physical value of `dtype`. `None` for key dtypes this rule
/// does not handle: floats, 128-bit integers and non-numeric dtypes.
fn physical_range(dtype: &DataType) -> Option<(i128, i128)> {
    Some(match dtype.to_physical() {
        _ if matches!(dtype, DataType::Time) => (0, NANOSECONDS_IN_DAY as i128 - 1),
        DataType::Int8 => (i8::MIN as i128, i8::MAX as i128),
        DataType::Int16 => (i16::MIN as i128, i16::MAX as i128),
        DataType::Int32 => (i32::MIN as i128, i32::MAX as i128),
        DataType::Int64 => (i64::MIN as i128, i64::MAX as i128),
        DataType::UInt8 => (0, u8::MAX as i128),
        DataType::UInt16 => (0, u16::MAX as i128),
        DataType::UInt32 => (0, u32::MAX as i128),
        DataType::UInt64 => (0, u64::MAX as i128),
        _ => return None,
    })
}

/// `value`, a physical value inside `physical_range(key_dtype)`, as a value of `key_dtype`.
fn key_scalar(value: i128, key_dtype: &DataType) -> Option<Scalar> {
    if !is_calendar_value(value, key_dtype) {
        return None;
    }
    let value = AnyValue::Int128(value)
        .strict_cast(key_dtype)?
        .into_static();
    Some(Scalar::new(key_dtype.clone(), value))
}

/// Whether a physical Date or Datetime `value` is a calendar date, with a day to spare for
/// time zone offsets. Printing a plan turns such values into dates.
fn is_calendar_value(value: i128, dtype: &DataType) -> bool {
    let (day, is_valid): (i64, fn(i64) -> bool) = match dtype {
        DataType::Date => (1, |v| {
            i32::try_from(v).ok().and_then(date32_to_date_opt).is_some()
        }),
        DataType::Datetime(TimeUnit::Milliseconds, _) => (MILLISECONDS_IN_DAY, |v| {
            timestamp_ms_to_datetime_opt(v).is_some()
        }),
        DataType::Datetime(TimeUnit::Microseconds, _) => (MICROSECONDS_IN_DAY, |v| {
            timestamp_us_to_datetime_opt(v).is_some()
        }),
        DataType::Datetime(TimeUnit::Nanoseconds, _) => (NANOSECONDS_IN_DAY, |v| {
            timestamp_ns_to_datetime_opt(v).is_some()
        }),
        _ => return true,
    };
    [value - day as i128, value + day as i128]
        .into_iter()
        .all(|v| i64::try_from(v).is_ok_and(is_valid))
}
