use super::*;

pub(super) fn fill_null(s: &[Series]) -> PolarsResult<Series> {
    let series = s[0].clone();
    let fill_value = s[1].clone();

    // Nothing to fill, so return early
    // this is done after casting as the output type must be correct
    if series.null_count() == 0 {
        return Ok(series);
    }

    // default branch
    fn default(series: Series, fill_value: Series) -> PolarsResult<Series> {
        let mask = series.is_not_null();
        series.zip_with_same_type(&mask, &fill_value)
    }

    match series.dtype() {
        #[cfg(feature = "dtype-categorical")]
        // for Categoricals we first need to check if the category already exist
        DataType::Categorical(Some(rev_map), _) => {
            if rev_map.is_local() && fill_value.len() == 1 && fill_value.null_count() == 0 {
                let fill_av = fill_value.get(0).unwrap();
                let fill_str = fill_av.get_str().unwrap();

                if let Some(idx) = rev_map.find(fill_str) {
                    let cats = series.to_physical_repr();
                    let mask = cats.is_not_null();
                    let out = cats
                        .zip_with_same_type(&mask, &Series::new("", &[idx]))
                        .unwrap();
                    unsafe { return out.cast_unchecked(series.dtype()) }
                }
            }
            let fill_value = if fill_value.dtype().is_string() {
                fill_value
                    .cast(&DataType::Categorical(None, Default::default()))
                    .unwrap()
            } else {
                fill_value
            };
            default(series, fill_value)
        },
        _ => default(series, fill_value),
    }
}

pub(super) fn coalesce(s: &mut [Series]) -> PolarsResult<Series> {
    coalesce_series(s)
}
