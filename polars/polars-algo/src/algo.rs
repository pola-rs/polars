use polars_core::prelude::*;
use polars_lazy::prelude::*;

pub fn cut(
    s: Series,
    bins: Vec<f32>,
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
        .with_columns([col(category_str).cast(DataType::Categorical(None))])
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
fn test_cut() -> PolarsResult<()> {
    let samples: Vec<f32> = (0..12).map(|i| -3.0 + i as f32 * 0.5).collect();
    let series = Series::new("a", samples);

    let out = cut(series, vec![-1.0, 1.0], None, None, None)?;

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
