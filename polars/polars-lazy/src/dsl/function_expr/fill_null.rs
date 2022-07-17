use super::*;

pub(super) fn fill_null(s: &[Series], super_type: &DataType) -> Result<Series> {
    let array = s[0].cast(super_type)?;
    let fill_value = s[1].cast(super_type)?;

    if !array.null_count() == 0 {
        Ok(array)
    } else {
        let mask = array.is_not_null();
        array.zip_with_same_type(&mask, &fill_value)
    }
}
