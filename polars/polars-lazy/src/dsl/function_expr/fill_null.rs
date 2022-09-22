use super::*;

pub(super) fn fill_null(s: &[Series], super_type: &DataType) -> PolarsResult<Series> {
    let array = s[0].cast(super_type)?;
    let fill_value = s[1].cast(super_type)?;

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
