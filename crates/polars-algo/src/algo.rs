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
            "cannot take histogram of non-numeric types; consider a `group_by` and `count`"
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
        .group_by(["category"])?
        .count()?;

    cuts.left_join(&out, [category_str], [category_str])?
        .fill_null(FillNullStrategy::Zero)?
        .sort(["category"], false, false)
}
