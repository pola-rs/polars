use num::pow::Pow;
use polars_arrow::utils::CustomIterTools;
use polars_core::export::num;
use polars_core::export::num::{Float, ToPrimitive};

use super::*;

fn pow_on_floats<T>(base: &ChunkedArray<T>, exponent: &Series) -> PolarsResult<Option<Series>>
where
    T: PolarsFloatType,
    T::Native: num::pow::Pow<T::Native, Output = T::Native> + ToPrimitive + Float,
    ChunkedArray<T>: IntoSeries,
{
    let dtype = T::get_dtype();
    let exponent = exponent.cast(&dtype)?;
    let exponent = base.unpack_series_matching_type(&exponent).unwrap();

    if exponent.len() == 1 {
        let exponent_value = exponent
            .get(0)
            .ok_or_else(|| PolarsError::ComputeError("exponent is null".into()))?;
        let s = match exponent_value.to_f64().unwrap() {
            a if a == 1.0 => base.clone().into_series(),
            // specialized sqrt will ensure (-inf)^0.5 = NaN
            // and will likely be faster as well.
            a if a == 0.5 => base.apply(|v| v.sqrt()).into_series(),
            a if a.fract() == 0.0 && a < 10.0 && a > 1.0 => {
                let mut out = base.clone();

                for _ in 1..exponent_value.to_u8().unwrap() {
                    out = out * base.clone()
                }
                out.into_series()
            }
            _ => base.apply(|v| Pow::pow(v, exponent_value)).into_series(),
        };
        Ok(Some(s))
    } else if (base.len() == 1) && (exponent.len() != 1) {
        let base = base
            .get(0)
            .ok_or_else(|| PolarsError::ComputeError("base is null".into()))?;

        Ok(Some(
            exponent.apply(|exp| Pow::pow(base, exp)).into_series(),
        ))
    } else {
        Ok(Some(
            base.into_iter()
                .zip(exponent.into_iter())
                .map(|(opt_base, opt_exponent)| match (opt_base, opt_exponent) {
                    (Some(base), Some(exponent)) => Some(num::pow::Pow::pow(base, exponent)),
                    _ => None,
                })
                .collect_trusted::<ChunkedArray<T>>()
                .into_series(),
        ))
    }
}

fn pow_on_series(base: &Series, exponent: &Series) -> PolarsResult<Option<Series>> {
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

pub(super) fn pow(s: &mut [Series]) -> PolarsResult<Option<Series>> {
    let base = &s[0];
    let exponent = &s[1];

    let base_len = base.len();
    let exp_len = exponent.len();
    match (base_len, exp_len) {
        (1, _) | (_, 1) => pow_on_series(base, exponent),
        (len_a, len_b) if len_a == len_b => {
            pow_on_series(base, exponent)
        }
        _ => {
            Err(PolarsError::ComputeError(
                format!("pow expression: the exponents length: {exp_len} does not match that of the base: {base_len}. Please ensure the lengths match or consider a literal exponent.").into()))
        }

    }
}
