use polars_core::prelude::*;
use polars_lazy::prelude::*;

pub fn cut(
    s: Series,
    bins: Vec<f32>,
    labels: Option<Vec<&str>>,
    break_point_label: Option<&str>,
    category_label: Option<&str>
) -> Result<DataFrame> {
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
        breakpoint_str => Series::new(
            breakpoint_str, &bins)
            .extend_constant(AnyValue::Float64(f64::INFINITY), 1)?
    ]?;

    let cuts_df = if let Some(labels) = labels {
        if labels.len() != (bins.len() + 1) {
            panic!("Expected more labels");
        }

        cuts_df.lazy().with_column(
            lit(Series::new(category_str, labels))
        )
    } else {
        let labels = vec!["labels"; bins.len() + 1];
        cuts_df.lazy().with_column(
            // TODO: Fix!
            // format(
        lit(Series::new(category_str, labels))

            //     "({}, {}]",
                // col(breakpoint_str).shift_and_fill(1, lit(f64::NEG_INFINITY)),
                // col(breakpoint_str),
            // ).alias(category_label)
        )
    }.collect()?;

    let cuts = cuts_df.lazy().with_columns([
        col(category_str).cast(DataType::Categorical(None))
    ]).collect()?;

    s.sort(false)
        .into_frame()
        .join_asof(
            &cuts,
            var_name,
            breakpoint_str,
            AsofStrategy::Forward,
            None,
            None
        )
}