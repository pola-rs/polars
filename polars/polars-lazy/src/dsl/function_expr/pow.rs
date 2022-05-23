use super::*;
use polars_arrow::utils::CustomIterTools;
use polars_core::export::num;

fn pow_on_floats<T>(base: &ChunkedArray<T>, exponent: &Series) -> Result<Series>
where
    T: PolarsFloatType,
    T::Native: num::pow::Pow<T::Native, Output = T::Native>,
    ChunkedArray<T>: IntoSeries,
{
    let dtype = T::get_dtype();
    let exponent = exponent.cast(&dtype)?;
    let exponent = base.unpack_series_matching_type(&exponent).unwrap();

    Ok(base
        .into_iter()
        .zip(exponent.into_iter())
        .map(|(opt_base, opt_exponent)| match (opt_base, opt_exponent) {
            (Some(base), Some(exponent)) => Some(num::pow::Pow::pow(base, exponent)),
            _ => None,
        })
        .collect_trusted::<ChunkedArray<T>>()
        .into_series())
}

fn pow_on_series(base: &Series, exponent: &Series) -> Result<Series> {
    use DataType::*;
    match base.dtype() {
        Float32 => {
            let ca = base.f32().unwrap();
            pow_on_floats(ca, exponent)
        }
        Float64 => {
            let ca = base.f64().unwrap();
            pow_on_floats(ca, exponent)
        }
        _ => {
            let base = base.cast(&DataType::Float64)?;
            pow_on_series(&base, exponent)
        }
    }
}

pub(super) fn pow(s: &mut [Series]) -> Result<Series> {
    let base = &s[0];
    let exponent = &s[1];

    match exponent.len() {
        1 => {
            let av = exponent.get(0);
            let exponent = av.extract::<f64>().ok_or_else(|| {
                PolarsError::ComputeError(
                    format!(
                        "expected a numerical exponent in the pow expression, but got dtype: {}",
                        exponent.dtype()
                    )
                    .into(),
                )
            })?;
            base.pow(exponent)
        }
        len => {
            let base_len = base.len();
            if len != base_len {
                Err(PolarsError::ComputeError(
                    format!("pow expression: the exponents length: {len} does not match that of the base: {base_len}. Please ensure the lengths match or consider a literal exponent.").into()))
            } else {
                pow_on_series(base, exponent)
            }
        }
    }
}
