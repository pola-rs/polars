use super::*;

pub(super) fn drop_nans(s: Series) -> PolarsResult<Series> {
    match s.dtype() {
        DataType::Float32 => {
            let ca = s.f32()?;
            let mask = ca.is_not_nan() | ca.is_null();
            ca.filter(&mask).map(|ca| ca.into_series())
        },
        DataType::Float64 => {
            let ca = s.f64()?;
            let mask = ca.is_not_nan() | ca.is_null();
            ca.filter(&mask).map(|ca| ca.into_series())
        },
        _ => Ok(s),
    }
}
