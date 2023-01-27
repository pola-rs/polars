use polars_core::error::PolarsResult as Result;
use polars_core::prelude::*;
use polars_lazy::prelude::*;
use polars_ops::prelude::*;

pub fn hist(
    s: &Series,
    bins: Option<&Series>,
    bin_count: Option<usize>,
    start: Option<f64>,
    stop: Option<f64>,
) -> Result<DataFrame> {
    let breakpoint_str = &"break_point";

    // if the bins are provided, then we can just use them
    let bins = if let Some(bins_in) = bins {
        Series::new(breakpoint_str, bins_in)
    } else {
        // If start and stop is provided, we don't need to scan the series
        let start = if let Some(start_in) = start {
            start_in
        } else {
            s.cast(&DataType::Float64)?
                .min::<f64>()
                .expect("Cannot find minimum value of series")
                .floor()
                - 1.0
        };

        let stop = if let Some(stop_in) = stop {
            stop_in
        } else {
            s.cast(&DataType::Float64)?
                .max::<f64>()
                .expect("Cannot find maximum value of series")
                .ceil()
                + 1.0
        };

        // If bin_count is omitted, default to the difference between start and stop (unit bins)
        let bin_count = if let Some(bin_count) = bin_count {
            bin_count
        } else {
            (stop - start).round() as usize
        };

        // Calculate the breakpoints and make the array
        let interval = (stop - start) / (bin_count as f64);
        let breaks: Vec<f64> = (0..(bin_count))
            .map(|b| start as f64 + (b as f64) * interval)
            .collect();

        Series::new(breakpoint_str, breaks)
    };

    let category_str = "category";

    let (min_value, max_value): (Expr, AnyValue) = match s.dtype() {
        // Floating point values have a notion of infinity
        DataType::Float64 => Ok((lit(f64::NEG_INFINITY), AnyValue::Float64(f64::INFINITY))),
        DataType::Float32 => Ok((lit(f32::NEG_INFINITY), AnyValue::Float32(f32::INFINITY))),
        // However, integers don't.  So, the best we can do is use the maximum for the type
        DataType::Int64 => Ok((lit(i64::MIN), AnyValue::Int64(i64::MAX))),
        DataType::Int32 => Ok((lit(i32::MIN), AnyValue::Int32(i32::MAX))),
        DataType::Int16 => Ok((lit(i32::MIN), AnyValue::Int16(i16::MAX))),
        DataType::UInt64 => Ok((lit(u64::MIN), AnyValue::UInt64(u64::MAX))),
        DataType::UInt32 => Ok((lit(u32::MIN), AnyValue::UInt32(u32::MAX))),
        DataType::UInt16 => Ok((lit(u32::MIN), AnyValue::UInt16(u16::MAX))),
        _ => Err(PolarsError::InvalidOperation(
            "Cannot take histogram of non-numeric types; consider a groupby and count.".into(),
        )),
    }?;

    let cuts_df = df![
        breakpoint_str => bins.extend_constant(max_value, 1)?
    ]?;

    let cuts_df = cuts_df
        .lazy()
        .with_column(
            format_str(
                "({}, {}]",
                [
                    col(breakpoint_str).shift_and_fill(1, min_value),
                    col(breakpoint_str),
                ],
            )?
            .alias(category_str),
        )
        .collect()?;

    let cuts = cuts_df
        .lazy()
        .with_columns([
            col(category_str).cast(DataType::Categorical(None)),
            col(breakpoint_str).cast(s.dtype().to_owned()),
        ])
        .collect()?;

    let out = s.sort(false).into_frame().join_asof(
        &cuts,
        s.name(),
        breakpoint_str,
        AsofStrategy::Forward,
        None,
        None,
    )?;

    let out = out
        .select(["category", s.name()])?
        .groupby(["category"])?
        .count()?;

    cuts.left_join(&out, [category_str], [category_str])?
        .fill_null(FillNullStrategy::Zero)?
        .sort(["category"], false)
}

#[test]
fn test_hist_integer() -> Result<()> {
    let df = df!(
        "value" => [3, 3, 5, 5, 6]
    )?;

    let series = &df[0];
    let out = hist(series, None, Some(6), Some(1.0), Some(7.0))?;

    let expected = df!(
        "break_point" => [1, 2, 3, 4, 5, 6, i32::MAX],
        "category"    => [
            "(-2147483648.0, 1.0]",
            "(1.0, 2.0]",
            "(2.0, 3.0]",
            "(3.0, 4.0]",
            "(4.0, 5.0]",
            "(5.0, 6.0]",
            "(6.0, 2147483647.0]"
        ],
        "value_count" => [0, 0, 2, 0, 2, 1, 0]
    )?;

    assert!(out.frame_equal_missing(&expected));

    Ok(())
}

#[test]
fn test_hist_float() -> Result<()> {
    let df = df!(
        "value" => [1.0, 3.4, 3.2, 6.3, 7.0]
    )?;

    let series = &df[0];
    let bins = Series::new("bins", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    let out = hist(series, Some(&bins), None, None, None)?;

    let expected = df!(
        "break_point" => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, f64::INFINITY],
        "category"    => [
            "(-inf, 1.0]",
            "(1.0, 2.0]",
            "(2.0, 3.0]",
            "(3.0, 4.0]",
            "(4.0, 5.0]",
            "(5.0, 6.0]",
            "(6.0, 7.0]",
            "(7.0, inf]"
        ],
        "value_count" => [1, 0, 0, 2, 0, 0, 2, 0]
    )?;

    assert!(out.frame_equal_missing(&expected));

    Ok(())
}

pub fn cut(
    s: &Series,
    bins: Vec<f64>,
    labels: Option<Vec<&str>>,
    break_point_label: Option<&str>,
    category_label: Option<&str>,
) -> PolarsResult<DataFrame> {
    let var_name = s.name();

    let breakpoint_str = if let Some(label) = break_point_label {
        label
    } else {
        &"break_point"
    };

    let category_str = if let Some(label) = category_label {
        label
    } else {
        &"category"
    };

    let cuts_df = df![
        breakpoint_str => Series::new(breakpoint_str, &bins)
            .extend_constant(AnyValue::Float64(f64::INFINITY), 1)?
    ]?;

    let cuts_df = if let Some(labels) = labels {
        if labels.len() != (bins.len() + 1) {
            return Err(PolarsError::ShapeMisMatch(
                "Labels count must equal bins count".into(),
            ));
        }

        cuts_df
            .lazy()
            .with_column(lit(Series::new(category_str, labels)))
    } else {
        cuts_df.lazy().with_column(
            format_str(
                "({}, {}]",
                [
                    col(breakpoint_str).shift_and_fill(1, lit(f64::NEG_INFINITY)),
                    col(breakpoint_str),
                ],
            )?
            .alias(category_str),
        )
    }
    .collect()?;

    let cuts = cuts_df
        .lazy()
        .with_columns([
            col(category_str).cast(DataType::Categorical(None)),
            col(breakpoint_str).cast(s.dtype().to_owned()),
        ])
        .collect()?;

    s.cast(&DataType::Float64)?
        .sort(false)
        .into_frame()
        .join_asof(
            &cuts,
            var_name,
            breakpoint_str,
            AsofStrategy::Forward,
            None,
            None,
        )
}

#[test]
fn test_cut_f32() -> Result<()> {
    let samples: Vec<f32> = (0..12).map(|i| -3.0 + i as f32 * 0.5).collect();
    let series = Series::new("a", samples);

    let out = cut(&series, vec![-1.0, 1.0], None, None, None)?;

    let expected = df!(
        "a"           => [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
        "break_point" => [-1.0, -1.0, -1.0, -1.0, -1.0,  1.0, 1.0, 1.0, 1.0, f64::INFINITY, f64::INFINITY, f64::INFINITY],
        "category"    => [
            "(-inf, -1.0]",
            "(-inf, -1.0]",
            "(-inf, -1.0]",
            "(-inf, -1.0]",
            "(-inf, -1.0]",
            "(-1.0, 1.0]",
            "(-1.0, 1.0]",
            "(-1.0, 1.0]",
            "(-1.0, 1.0]",
            "(1.0, inf]",
            "(1.0, inf]",
            "(1.0, inf]"
        ]
    )?;

    assert!(out.frame_equal_missing(&expected));

    Ok(())
}

#[test]
fn test_cut_f64() -> Result<()> {
    let samples: Vec<f64> = (0..12).map(|i| -3.0 + i as f64 * 0.5).collect();
    let series = Series::new("a", samples);

    let out = cut(&series, vec![-1.0, 1.0], None, None, None)?;

    let expected = df!(
        "a"           => [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
        "break_point" => [-1.0, -1.0, -1.0, -1.0, -1.0,  1.0, 1.0, 1.0, 1.0, f64::INFINITY, f64::INFINITY, f64::INFINITY],
        "category"    => [
            "(-inf, -1.0]",
            "(-inf, -1.0]",
            "(-inf, -1.0]",
            "(-inf, -1.0]",
            "(-inf, -1.0]",
            "(-1.0, 1.0]",
            "(-1.0, 1.0]",
            "(-1.0, 1.0]",
            "(-1.0, 1.0]",
            "(1.0, inf]",
            "(1.0, inf]",
            "(1.0, inf]"
        ]
    )?;

    assert!(out.frame_equal_missing(&expected));

    Ok(())
}
