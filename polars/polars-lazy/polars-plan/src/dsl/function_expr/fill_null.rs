use super::*;

pub(super) fn fill_null(s: &[Series], super_type: &DataType) -> PolarsResult<Series> {
    let array = &s[0];
    let fill_value = &s[1];

    if matches!(super_type, DataType::Unknown) {
        return Err(PolarsError::SchemaMisMatch(
            format!(
                "Cannot 'fill_null' a 'Series' of dtype: '{}' with an argument of dtype: '{}'",
                array.dtype(),
                fill_value.dtype()
            )
            .into(),
        ));
    };

    let array = array.cast(super_type)?;
    let fill_value = fill_value.cast(super_type)?;

    if !array.null_count() == 0 {
        Ok(array)
    } else {
        let mask = array.is_not_null();
        array.zip_with_same_type(&mask, &fill_value)
    }
}

pub(super) fn coalesce(s: &mut [Series]) -> PolarsResult<Series> {
    if s.is_empty() {
        Err(PolarsError::ComputeError(
            "cannot coalesce empty list".into(),
        ))
    } else {
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
}
