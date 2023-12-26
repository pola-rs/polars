use super::*;

pub(super) fn fill_null(s: &[Series], super_type: &DataType) -> PolarsResult<Series> {
    let series = &s[0];
    let fill_value = &s[1];

    let (series, fill_value) = if matches!(super_type, DataType::Unknown) {
        let fill_value = fill_value.cast(series.dtype()).map_err(|_| {
            polars_err!(
                SchemaMismatch:
                "`fill_null` supertype could not be determined; set correct literal value or \
                ensure the type of the expression is known"
            )
        })?;
        (series.clone(), fill_value)
    } else {
        (series.cast(super_type)?, fill_value.cast(super_type)?)
    };
    // nothing to fill, so return early
    // this is done after casting as the output type must be correct
    if series.null_count() == 0 {
        return Ok(series);
    }

    // default branch
    fn default(series: Series, mut fill_value: Series) -> PolarsResult<Series> {
        // broadcast to the proper length for zip_with
        if fill_value.len() == 1 && series.len() != 1 {
            fill_value = fill_value.new_from_index(0, series.len());
        }
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
            default(series, fill_value)
        },
        _ => default(series, fill_value),
    }
}

pub(super) fn coalesce(s: &mut [Series]) -> PolarsResult<Series> {
    coalesce_series(s)
}
