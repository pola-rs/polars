use super::*;

pub(super) fn fill_null(s: &[Series], super_type: &DataType) -> PolarsResult<Series> {
    let array = &s[0];
    let fill_value = &s[1];

    let (array, fill_value) = if matches!(super_type, DataType::Unknown) {
        let fill_value = fill_value.cast(array.dtype()).map_err(|_| {
            polars_err!(
                SchemaMismatch:
                "`fill_null` supertype could not be determined; set correct literal value or \
                ensure the type of the expression is known"
            )
        })?;
        (array.clone(), fill_value)
    } else {
        (array.cast(super_type)?, fill_value.cast(super_type)?)
    };

    if !array.null_count() == 0 {
        Ok(array)
    } else {
        let mask = array.is_not_null();
        array.zip_with_same_type(&mask, &fill_value)
    }
}

pub(super) fn coalesce(s: &mut [Series]) -> PolarsResult<Series> {
    polars_ensure!(!s.is_empty(), NoData: "cannot coalesce empty list");
    let mut out = s[0].clone();
    for s in s {
        if !out.null_count() == 0 {
            return Ok(out);
        } else {
            let mask = out.is_not_null();
            out = out.zip_with_same_type(&mask, s)?;
        }
    }
    Ok(out)
}
