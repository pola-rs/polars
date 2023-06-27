use polars_core::error::PolarsResult as Result;
use polars_core::prelude::*;
use polars_core::series::IsSorted;
use polars_lazy::prelude::*;
use polars_ops::prelude::*;

pub fn hist(s: &Series, bins: Option<&Series>, bin_count: Option<usize>) -> Result<DataFrame> {
    let breakpoint_str = &"break_point";
    let s = s.cast(&DataType::Float64)?.sort(false);

    // if the bins are provided, then we can just use them
    let bins = if let Some(bins_in) = bins {
        Series::new(breakpoint_str, bins_in).sort(false)
    } else {
        // data is sorted, so this is O(1)
        let start = s.min::<f64>().unwrap().floor() - 1.0;
        let stop = s.max::<f64>().unwrap().ceil() + 1.0;

        // If bin_count is omitted, default to the difference between start and stop (unit bins)
        let bin_count = if let Some(bin_count) = bin_count {
            bin_count
        } else {
            (stop - start).round() as usize
        };

        // Calculate the breakpoints and make the array
        let interval = (stop - start) / (bin_count as f64);
        let breaks: Vec<f64> = (0..(bin_count))
            .map(|b| start + (b as f64) * interval)
            .collect();

        Series::new(breakpoint_str, breaks)
    };

    let category_str = "category";

    let (min_value, max_value): (Expr, AnyValue) = match s.dtype() {
        // Floating point values have a notion of infinity
        DataType::Float64 => (lit(f64::NEG_INFINITY), AnyValue::Float64(f64::INFINITY)),
        DataType::Float32 => (lit(f32::NEG_INFINITY), AnyValue::Float32(f32::INFINITY)),
        // However, integers don't.  So, the best we can do is use the maximum for the type
        DataType::Int64 => (lit(i64::MIN), AnyValue::Int64(i64::MAX)),
        DataType::Int32 => (lit(i32::MIN), AnyValue::Int32(i32::MAX)),
        DataType::Int16 => (lit(i32::MIN), AnyValue::Int16(i16::MAX)),
        DataType::UInt64 => (lit(u64::MIN), AnyValue::UInt64(u64::MAX)),
        DataType::UInt32 => (lit(u32::MIN), AnyValue::UInt32(u32::MAX)),
        DataType::UInt16 => (lit(u32::MIN), AnyValue::UInt16(u16::MAX)),
        _ => polars_bail!(
            InvalidOperation:
            "cannot take histogram of non-numeric types; consider a groupby and count"
        ),
    };
    let mut bins = bins.extend_constant(max_value, 1)?;
    bins.set_sorted_flag(IsSorted::Ascending);

    let cuts_df = df![
        breakpoint_str => bins
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
            col(breakpoint_str)
                .cast(s.dtype().to_owned())
                .set_sorted_flag(IsSorted::Ascending),
        ])
        .collect()?;

    let out = s.clone().into_frame().join_asof(
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

pub fn qcut(
    s: &Series,
    quantiles: &[f64],
    labels: Option<Vec<&str>>,
    break_point_label: Option<&str>,
    category_label: Option<&str>,
    maintain_order: bool,
) -> PolarsResult<DataFrame> {
    let s = s.cast(&DataType::Float64)?;

    // amortize quantile computation
    let s_sorted = s.sort(false);
    let ca = s_sorted.f64().unwrap();

    let mut bins = Vec::with_capacity(quantiles.len());
    for quantile_level in quantiles {
        if let Some(quantile) = ca.quantile(*quantile_level, QuantileInterpolOptions::Linear)? {
            bins.push(quantile)
        }
    }

    let bins = Series::new("", bins);
    if maintain_order {
        cut(
            &s,
            bins,
            labels,
            break_point_label,
            category_label,
            maintain_order,
        )
    } else {
        // already sorted, saves an extra sort
        cut(
            &s_sorted,
            bins,
            labels,
            break_point_label,
            category_label,
            maintain_order,
        )
    }
}

pub fn cut(
    s: &Series,
    mut bins: Series,
    labels: Option<Vec<&str>>,
    break_point_label: Option<&str>,
    category_label: Option<&str>,
    maintain_order: bool,
) -> PolarsResult<DataFrame> {
    let var_name = s.name();
    let breakpoint_str = break_point_label.unwrap_or("break_point");
    let category_str = category_label.unwrap_or("category");

    let bins_len = bins.len();

    bins.rename(breakpoint_str);

    let mut s_bins = bins
        .cast(&DataType::Float64)
        .map_err(|_| PolarsError::ComputeError("expected numeric bins".into()))?
        .extend_constant(AnyValue::Float64(f64::INFINITY), 1)?;
    s_bins.set_sorted_flag(IsSorted::Ascending);
    let cuts_df = df![
        breakpoint_str => s_bins
    ]?;

    let cuts_df = if let Some(labels) = labels {
        polars_ensure!(
            labels.len() == (bins_len + 1),
            ShapeMismatch: "labels count must equal bins count",
        );
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

    const ROW_COUNT: &str = "__POLARS_IDX";

    let cuts = cuts_df
        .lazy()
        .with_columns([col(category_str).cast(DataType::Categorical(None))])
        .collect()?;

    let mut s = s.cast(&DataType::Float64)?;
    let valids = if s.null_count() > 0 {
        let valids = Some(s.is_not_null());
        s = s.fill_null(FillNullStrategy::MaxBound).unwrap();
        valids
    } else {
        None
    };
    let mut frame = s.clone().into_frame();

    if maintain_order {
        frame = frame.with_row_count(ROW_COUNT, None)?;
    }

    let mut out = frame.sort(vec![var_name], vec![false])?.join_asof(
        &cuts,
        var_name,
        breakpoint_str,
        AsofStrategy::Forward,
        None,
        None,
    )?;

    if maintain_order {
        out = out.sort([ROW_COUNT], false)?.drop(ROW_COUNT).unwrap()
    };

    if let Some(mut valids) = valids {
        if !maintain_order {
            let idx = s.arg_sort(SortOptions {
                nulls_last: true,
                ..Default::default()
            });
            valids = unsafe { valids.take_unchecked((&idx).into()) };
        }

        let arr = valids.downcast_iter().next().unwrap();
        let validity = arr.values().clone();

        // Safety: we don't change the length/dtype
        unsafe {
            for col in out.get_columns_mut() {
                let mut s = col.rechunk();
                let chunks = s.chunks_mut();
                chunks[0] = chunks[0].with_validity(Some(validity.clone()));
                *col = s;
            }
        }
    }
    Ok(out)
}
