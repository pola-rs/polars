use polars::df;
use polars::prelude::*;

/// Replaces NaN with proper missing values.
fn fill_nan_with_nulls() -> Result<DataFrame> {
    let nan = f64::NAN;

    let mut df = df! {
       "a" => [nan, 1.0, 2.0],
       "b" => [nan, 1.0, 2.0]
    }
    .unwrap();

    for idx in 0..df.width() {
        df.may_apply_at_idx(idx, |series| {
            let mask = series.is_nan()?;
            let ca = series.f64()?;
            ca.set(&mask, None)
        })?;
    }
    Ok(df)
}

fn main() {
    dbg!(fill_nan_with_nulls());
}
