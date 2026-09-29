//! A dynamic or rolling group-by whose windows are placed on the IR, the way a distributed
//! planner does per partition.

use polars_arrow::legacy::time_zone::Tz;
use polars_core::prelude::*;
use polars_defs::time::duration::Duration;
use polars_defs::time::group_by::{
    ClosedWindow, DynamicGroupOptions, DynamicWindowPlacement, IndexRange, Label,
    RollingGroupOptions, RollingWindowPlacement, StartBy,
};
use polars_plan::plans::IR;
use polars_plan::plans::options::GroupbyOptionsIR;
use polars_time::Window;
use polars_utils::arena::Node;

use crate::frame::run_in_memory_query;
use crate::prelude::*;

const ENGINES: [Engine; 2] = [Engine::InMemory, Engine::Streaming];

/// Runs `lf` on `engine` with `placement` written into its single `IR::GroupBy` node.
fn collect_placed(
    lf: LazyFrame,
    placement: Option<DynamicWindowPlacement>,
    engine: Engine,
) -> PolarsResult<DataFrame> {
    collect_with_options(lf, engine, |options| {
        options.dynamic.as_mut().unwrap().placement = placement
    })
}

/// Runs `lf` on `engine` with `placement` written into its single rolling `IR::GroupBy` node.
fn collect_rolling_placed(
    lf: LazyFrame,
    placement: Option<RollingWindowPlacement>,
    engine: Engine,
) -> PolarsResult<DataFrame> {
    collect_with_options(lf, engine, |options| {
        options.rolling.as_mut().unwrap().placement = placement
    })
}

fn collect_with_options(
    lf: LazyFrame,
    engine: Engine,
    patch: impl Fn(&mut GroupbyOptionsIR),
) -> PolarsResult<DataFrame> {
    let lf = match engine {
        Engine::Streaming => lf.with_streaming(true),
        _ => lf,
    };
    let mut plan = lf.to_alp_optimized()?;
    let mut group_by_nodes = 0;
    for i in 0..plan.lp_arena.len() {
        if let IR::GroupBy { options, .. } = plan.lp_arena.get_mut(Node(i)) {
            patch(Arc::make_mut(options));
            group_by_nodes += 1;
        }
    }
    assert_eq!(group_by_nodes, 1);
    plan.ensure_root_node_is_sink();
    let out = match engine {
        Engine::Streaming => {
            polars_stream::run_query(plan.lp_top, &mut plan.lp_arena, &mut plan.expr_arena, None)?
        },
        _ => run_in_memory_query(
            plan.lp_top,
            &mut plan.lp_arena,
            &mut plan.expr_arena,
            Engine::InMemory,
            None,
        )?,
    };
    Ok(out.unwrap_single())
}

/// The index column's values in the space the window grid is computed in, with that space's
/// time unit and zone.
fn physical_index(df: &DataFrame, index: &str) -> (Vec<i64>, TimeUnit, Option<TimeZone>) {
    let col = df.column(index).unwrap();
    let (col, tu, tz) = match col.dtype() {
        DataType::Datetime(tu, tz) => (col.clone(), *tu, tz.clone()),
        DataType::Date => (
            col.cast(&DataType::Datetime(TimeUnit::Microseconds, None))
                .unwrap(),
            TimeUnit::Microseconds,
            None,
        ),
        DataType::Int64 | DataType::Int32 => (
            col.cast(&DataType::Int64).unwrap(),
            TimeUnit::Nanoseconds,
            None,
        ),
        dt => panic!("unexpected index dtype {dt}"),
    };
    let values = col
        .to_physical_repr()
        .cast(&DataType::Int64)
        .unwrap()
        .i64()
        .unwrap()
        .into_no_null_iter()
        .collect();
    (values, tu, tz)
}

fn add(duration: &Duration, t: i64, tu: TimeUnit, tz: Option<&Tz>) -> i64 {
    match tu {
        TimeUnit::Nanoseconds => duration.add_ns(t, tz),
        TimeUnit::Microseconds => duration.add_us(t, tz),
        TimeUnit::Milliseconds => duration.add_ms(t, tz),
    }
    .unwrap()
}

/// `df` as a union of five-row frames, so the streaming engine sees several morsels.
fn source(df: &DataFrame) -> LazyFrame {
    if df.height() == 0 {
        return df.clone().lazy();
    }
    let chunks: Vec<LazyFrame> = (0..df.height())
        .step_by(5)
        .map(|o| df.slice(o as i64, 5).lazy())
        .collect();
    let args = UnionArgs {
        rechunk: false,
        ..Default::default()
    };
    concat(chunks, args).unwrap()
}

fn query(df: &DataFrame, index: &str, options: &DynamicGroupOptions, aggs: &[Expr]) -> LazyFrame {
    source(df)
        .group_by_dynamic(col(index), [] as [Expr; 0], options.clone())
        .agg(aggs)
}

/// The placements and inputs a planner would hand `parts` consecutive partitions of `df`.
fn partitions(
    df: &DataFrame,
    index: &str,
    options: &DynamicGroupOptions,
    parts: usize,
) -> Vec<(DataFrame, DynamicWindowPlacement)> {
    let (values, tu, tz) = physical_index(df, index);
    let tz = tz.map(|tz| tz.to_chrono().unwrap());
    let window = Window::new(options.every, options.period, options.offset);
    let origin = window
        .first_window_start(
            values[0],
            options.closed_window,
            tu,
            tz.as_ref(),
            options.start_by,
        )
        .unwrap();

    // Partition `i` owns the values in `firsts[i]..firsts[i + 1]`.
    let len = values.len();
    let part_len = len.div_ceil(parts);
    let firsts: Vec<i64> = std::iter::once(i64::MIN)
        .chain((part_len..len).step_by(part_len).map(|i| values[i]))
        .collect();
    let mut out = vec![];
    for (i, &first) in firsts.iter().enumerate() {
        let end = firsts.get(i + 1).copied();
        // The forward read: every row that can be a member of a window starting before `end`.
        let read_start = values.partition_point(|v| *v < first);
        let read_end = match end {
            Some(end) => {
                let limit = add(&options.period, end, tu, tz.as_ref());
                values.partition_point(|v| *v < limit)
            },
            None => len,
        };
        let placement = DynamicWindowPlacement {
            origin,
            start_range: IndexRange::new(first, end),
        };
        out.push((
            df.slice(read_start as i64, read_end - read_start),
            placement,
        ));
    }
    out
}

/// Concatenating the partitions' placed results equals the plain result, on both engines.
fn check_partition_invariant(
    df: &DataFrame,
    index: &str,
    options: DynamicGroupOptions,
    aggs: &[Expr],
) {
    let expected = query(df, index, &options, aggs).collect().unwrap();
    for parts in [1, 2, 5] {
        for engine in ENGINES {
            let mut got: Option<DataFrame> = None;
            for (input, placement) in partitions(df, index, &options, parts) {
                let part = collect_placed(
                    query(&input, index, &options, aggs),
                    Some(placement),
                    engine,
                )
                .unwrap();
                got = Some(match got {
                    None => part,
                    Some(acc) => acc.vstack(&part).unwrap(),
                });
            }
            let got = got.unwrap();
            assert!(
                got.equals_missing(&expected),
                "{engine:?}, {parts} parts, {options:?}\n{got}\n{expected}"
            );
        }
    }
}

fn options(every: &str, period: &str, offset: &str) -> DynamicGroupOptions {
    DynamicGroupOptions {
        every: Duration::parse(every),
        period: Duration::parse(period),
        offset: Duration::parse(offset),
        ..Default::default()
    }
}

fn aggs() -> Vec<Expr> {
    vec![
        col("x").sum().alias("sum"),
        col("x").first().alias("first"),
        col("x").last().alias("last"),
        len().alias("n"),
    ]
}

fn datetime_frame(values: Vec<i64>, tu: TimeUnit, tz: Option<&str>) -> DataFrame {
    let x: Vec<i64> = (0..values.len() as i64).collect();
    let tz = TimeZone::opt_try_new(tz).unwrap();
    df!(
        "t" => Int64Chunked::from_vec("t".into(), values).into_datetime(tu, tz).into_series(),
        "x" => x,
    )
    .unwrap()
}

/// Dense hourly data, a gap of several days, repeated values, then dense data again.
fn hourly_with_gaps(hour: i64) -> Vec<i64> {
    let mut values: Vec<i64> = (0..40).map(|i| i * hour + 17 * hour / 60).collect();
    values.extend((0..12).map(|i| 200 * hour + i * 3 * hour));
    values.extend([200 * hour + 36 * hour; 4]);
    values.extend((0..20).map(|i| 300 * hour + i * hour));
    values
}

#[test]
fn partition_invariant_datetime() {
    let hour = 3_600_000_000i64;
    let df = datetime_frame(hourly_with_gaps(hour), TimeUnit::Microseconds, None);
    let cases = [
        ("4h", "4h", "0h", ClosedWindow::Left, StartBy::WindowBound),
        ("4h", "4h", "0h", ClosedWindow::Right, StartBy::DataPoint),
        ("2h", "5h", "0h", ClosedWindow::None, StartBy::Tuesday),
        ("6h", "2h", "1h", ClosedWindow::Both, StartBy::DataPoint),
        ("1d", "3d", "-5h", ClosedWindow::Left, StartBy::Tuesday),
        ("3h", "3h", "7h", ClosedWindow::Right, StartBy::WindowBound),
    ];
    for (every, period, offset, closed_window, start_by) in cases {
        let options = DynamicGroupOptions {
            closed_window,
            start_by,
            ..options(every, period, offset)
        };
        check_partition_invariant(&df, "t", options, &aggs());
    }
}

#[test]
fn partition_invariant_labels_and_boundaries() {
    let hour = 3_600_000_000i64;
    let df = datetime_frame(hourly_with_gaps(hour), TimeUnit::Microseconds, None);
    for label in [Label::Left, Label::Right, Label::DataPoint] {
        for include_boundaries in [false, true] {
            let options = DynamicGroupOptions {
                label,
                include_boundaries,
                ..options("5h", "7h", "2h")
            };
            check_partition_invariant(&df, "t", options, &aggs());
        }
    }
}

#[test]
fn partition_invariant_time_zone_with_dst() {
    // Amsterdam ends daylight saving on 2024-10-27 01:00 UTC.
    let hour = 3_600_000_000i64;
    let start = 1_729_900_800_000_000i64; // 2024-10-26 00:00 UTC
    let values: Vec<i64> = (0..96).map(|i| start + i * hour / 2).collect();
    let df = datetime_frame(values, TimeUnit::Microseconds, Some("Europe/Amsterdam"));
    for opts in [
        options("2h", "2h", "0h"),
        options("1d", "1d", "0h"),
        options("1d", "2d", "3h"),
        options("90m", "3h", "0m"),
    ] {
        for closed_window in [ClosedWindow::Left, ClosedWindow::Right] {
            let options = DynamicGroupOptions {
                closed_window,
                ..opts.clone()
            };
            check_partition_invariant(&df, "t", options, &aggs());
        }
    }
}

#[test]
fn partition_invariant_calendar_months() {
    // Daily data from mid January to May, over a February and month ends.
    let day = 86_400_000i64;
    let start = 1_705_276_800_000i64; // 2024-01-15
    let values: Vec<i64> = (0..120).map(|i| start + i * day).collect();
    let df = datetime_frame(values, TimeUnit::Milliseconds, None);
    for (opts, start_by) in [
        (options("1mo", "1mo", "0d"), StartBy::WindowBound),
        (options("1mo", "1mo", "0d"), StartBy::DataPoint),
        (options("1mo", "1mo", "15d"), StartBy::DataPoint),
        (options("1mo", "45d", "-3d"), StartBy::WindowBound),
        (options("1w", "1w", "0d"), StartBy::Monday),
        (options("2w", "1mo", "1d"), StartBy::Monday),
        (options("2w", "1mo", "1d"), StartBy::DataPoint),
    ] {
        let options = DynamicGroupOptions { start_by, ..opts };
        check_partition_invariant(&df, "t", options, &aggs());
    }
}

#[test]
fn partition_invariant_date_and_integer_index() {
    let days: Vec<i32> = (0..60).chain((100..130).step_by(3)).collect();
    let x: Vec<i64> = (0..days.len() as i64).collect();
    let df = df!(
        "t" => Int32Chunked::from_vec("t".into(), days).into_date().into_series(),
        "x" => x.clone(),
    )
    .unwrap();
    for opts in [
        options("1w", "1w", "0d"),
        options("3d", "10d", "1d"),
        options("1mo", "1mo", "0d"),
    ] {
        check_partition_invariant(&df, "t", opts, &aggs());
    }

    let ints: Vec<i64> = (0..50)
        .chain((200..260).step_by(2))
        .chain([260; 5])
        .collect();
    let x: Vec<i64> = (0..ints.len() as i64).collect();
    let df = df!("t" => ints, "x" => x).unwrap();
    for opts in [
        options("5i", "5i", "0i"),
        options("2i", "5i", "1i"),
        options("7i", "3i", "0i"),
    ] {
        for closed_window in [ClosedWindow::Left, ClosedWindow::Both] {
            let options = DynamicGroupOptions {
                closed_window,
                ..opts.clone()
            };
            check_partition_invariant(&df, "t", options, &aggs());
        }
    }
}

#[test]
fn windows_outside_the_range_are_not_evaluated() {
    // Exactly four rows per tumbling window, so `get(3)` only fails on a partial window.
    let hour = 3_600_000_000i64;
    let df = datetime_frame(
        (0..48).map(|i| i * hour).collect(),
        TimeUnit::Microseconds,
        None,
    );
    let options = options("4h", "4h", "0h");
    let aggs = [col("x").get(lit(3), false).alias("fourth")];
    let expected = query(&df, "t", &options, &aggs).collect().unwrap();
    assert_eq!(expected.height(), 12);

    // The second of five partitions owns rows 10 to 19 and reads on to row 23: it starts inside
    // the window [8h, 12h) and its read ends exactly on a window end, so the windows it owns
    // start at 12h and 16h.
    let parts = partitions(&df, "t", &options, 5);
    let (input, placement) = &parts[1];
    assert_eq!(input.height(), 14);
    for engine in ENGINES {
        let placed =
            collect_placed(query(input, "t", &options, &aggs), Some(*placement), engine).unwrap();
        assert!(
            placed.equals_missing(&expected.slice(3, 2)),
            "{engine:?}\n{placed}"
        );
        assert!(collect_placed(query(input, "t", &options, &aggs), None, engine).is_err());
    }
}

/// A slice after the placed group-by slices the placed result, on both engines.
fn check_slice(
    query: impl Fn() -> LazyFrame,
    collect: impl Fn(LazyFrame, Engine) -> PolarsResult<DataFrame>,
) {
    for engine in ENGINES {
        let full = collect(query(), engine).unwrap();
        assert!(full.height() > 4);
        let sliced = collect(query().slice(1, 3), engine).unwrap();
        assert!(
            sliced.equals_missing(&full.slice(1, 3)),
            "{engine:?}\n{sliced}\n{full}"
        );
    }
}

#[test]
fn placement_with_slice_and_keys() {
    let hour = 3_600_000_000i64;
    let df = datetime_frame(hourly_with_gaps(hour), TimeUnit::Microseconds, None);
    let options = options("4h", "6h", "0h");
    let placement = DynamicWindowPlacement {
        origin: 0,
        start_range: IndexRange::new(8 * hour + 1, Some(210 * hour)),
    };
    check_slice(
        || query(&df, "t", &options, &aggs()),
        |lf, engine| collect_placed(lf, Some(placement), engine),
    );

    let keys: Vec<&str> = (0..df.height())
        .map(|i| if i % 3 == 0 { "a" } else { "b" })
        .collect();
    let mut df = df;
    df.with_column(Series::new("k".into(), keys).into())
        .unwrap();
    // A placement is only supported without keys.
    let lf = df
        .clone()
        .lazy()
        .group_by_dynamic(col("t"), [col("k")], options.clone())
        .agg(aggs());
    assert!(collect_placed(lf, Some(placement), Engine::InMemory).is_err());

    let options = rolling_options("4h", "-4h", ClosedWindow::Right);
    let placement = RollingWindowPlacement {
        owned_range: IndexRange::new(8 * hour + 1, Some(210 * hour)),
    };
    check_slice(
        || rolling_query(&df, "t", &options, &aggs()),
        |lf, engine| collect_rolling_placed(lf, Some(placement), engine),
    );
    let lf = df
        .clone()
        .lazy()
        .rolling(col("t"), [col("k")], options.clone())
        .agg(aggs());
    assert!(collect_rolling_placed(lf, Some(placement), Engine::InMemory).is_err());
}

fn rolling_query(
    df: &DataFrame,
    index: &str,
    options: &RollingGroupOptions,
    aggs: &[Expr],
) -> LazyFrame {
    source(df)
        .rolling(col(index), [] as [Expr; 0], options.clone())
        .agg(aggs)
}

fn rolling_options(period: &str, offset: &str, closed_window: ClosedWindow) -> RollingGroupOptions {
    RollingGroupOptions {
        index_column: "".into(),
        period: Duration::parse(period),
        offset: Duration::parse(offset),
        closed_window,
    }
}

/// The placements and inputs a planner would hand `parts` consecutive partitions of `df` for a
/// rolling group-by: a partition owns the rows with values in `first..end` and reads every
/// row that can fall in one of their windows.
fn rolling_partitions(
    df: &DataFrame,
    index: &str,
    options: &RollingGroupOptions,
    parts: usize,
) -> Vec<(DataFrame, RollingWindowPlacement)> {
    let (values, tu, tz) = physical_index(df, index);
    let tz = tz.map(|tz| tz.to_chrono().unwrap());
    let len = values.len();
    let part_len = len.div_ceil(parts);
    let firsts: Vec<i64> = std::iter::once(i64::MIN)
        .chain((part_len..len).step_by(part_len).map(|i| values[i]))
        .collect();
    let mut out = vec![];
    for (i, &first) in firsts.iter().enumerate() {
        let end = firsts.get(i + 1).copied();
        let owned = IndexRange::new(first, end).row_range(&values);
        let (first_row, last_row) = (values[owned.start], values[owned.end - 1]);
        let lo = add(&options.offset, first_row, tu, tz.as_ref()).min(first_row);
        let hi = add(
            &options.period,
            add(&options.offset, last_row, tu, tz.as_ref()),
            tu,
            tz.as_ref(),
        )
        .max(last_row);
        let read_start = values.partition_point(|v| *v < lo);
        let read_end = values.partition_point(|v| *v <= hi);
        let placement = RollingWindowPlacement {
            owned_range: IndexRange::new(first, end),
        };
        out.push((
            df.slice(read_start as i64, read_end - read_start),
            placement,
        ));
    }
    out
}

fn check_rolling_partition_invariant(
    df: &DataFrame,
    index: &str,
    options: RollingGroupOptions,
    aggs: &[Expr],
) {
    let expected = rolling_query(df, index, &options, aggs).collect().unwrap();
    for parts in [1, 2, 5] {
        for engine in ENGINES {
            let mut got: Option<DataFrame> = None;
            for (input, placement) in rolling_partitions(df, index, &options, parts) {
                let part = collect_rolling_placed(
                    rolling_query(&input, index, &options, aggs),
                    Some(placement),
                    engine,
                )
                .unwrap();
                got = Some(match got {
                    None => part,
                    Some(acc) => acc.vstack(&part).unwrap(),
                });
            }
            let got = got.unwrap();
            assert!(
                got.equals_missing(&expected),
                "{engine:?}, {parts} parts, {options:?}\n{got}\n{expected}"
            );
        }
    }
}

#[test]
fn rolling_partition_invariant() {
    let hour = 3_600_000_000i64;
    let df = datetime_frame(hourly_with_gaps(hour), TimeUnit::Microseconds, None);
    for (period, offset) in [
        ("3h", "-3h"),
        ("5h", "0h"),
        ("4h", "-7h"),
        ("2h", "3h"),
        ("1d", "-12h"),
    ] {
        for closed_window in [
            ClosedWindow::Left,
            ClosedWindow::Right,
            ClosedWindow::Both,
            ClosedWindow::None,
        ] {
            let options = rolling_options(period, offset, closed_window);
            check_rolling_partition_invariant(&df, "t", options, &aggs());
        }
    }

    // Amsterdam ends daylight saving on 2024-10-27 01:00 UTC.
    let start = 1_729_900_800_000_000i64;
    let values: Vec<i64> = (0..96).map(|i| start + i * hour / 2).collect();
    let df = datetime_frame(values, TimeUnit::Microseconds, Some("Europe/Amsterdam"));
    for (period, offset) in [("2h", "-2h"), ("1d", "-1d"), ("1d", "-12h")] {
        let options = rolling_options(period, offset, ClosedWindow::Right);
        check_rolling_partition_invariant(&df, "t", options, &aggs());
    }

    let ints: Vec<i64> = (0..50)
        .chain((200..260).step_by(2))
        .chain([260; 5])
        .collect();
    let x: Vec<i64> = (0..ints.len() as i64).collect();
    let df = df!("t" => ints, "x" => x).unwrap();
    for (period, offset) in [("5i", "-5i"), ("3i", "0i"), ("4i", "-2i")] {
        let options = rolling_options(period, offset, ClosedWindow::Both);
        check_rolling_partition_invariant(&df, "t", options, &aggs());
    }
}
