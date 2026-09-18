use num_traits::Zero;
use polars_arrow::bitmap::BitmapBuilder;
pub use polars_compute::ewm::EWMOptions;
use polars_compute::ewm::mean::ewm_mean as kernel_ewm_mean;
use polars_compute::ewm::sum::ewm_sum as kernel_ewm_sum;
use polars_compute::ewm::{ewm_std as kernel_ewm_std, ewm_var as kernel_ewm_var};
use polars_core::prelude::*;

fn check_alpha(alpha: f64) -> PolarsResult<()> {
    polars_ensure!((0.0..=1.0).contains(&alpha), ComputeError: "alpha must be in [0; 1]");
    Ok(())
}

/// What the scalar fast paths below ask of a float native.
trait EwmFloat: Copy {
    /// Whether this value is neither infinite nor `NaN`.
    fn is_finite(self) -> bool;
}

impl EwmFloat for f32 {
    #[inline]
    fn is_finite(self) -> bool {
        f32::is_finite(self)
    }
}

impl EwmFloat for f64 {
    #[inline]
    fn is_finite(self) -> bool {
        f64::is_finite(self)
    }
}

#[cfg(feature = "dtype-f16")]
impl EwmFloat for polars_utils::float16::pf16 {
    #[inline]
    fn is_finite(self) -> bool {
        polars_utils::float16::pf16::is_finite(self)
    }
}

/// The finite value every element of `ca` is, if it repeats that one value with no null among them.
fn one_finite_value<T>(ca: &ChunkedArray<T>) -> Option<T::Native>
where
    T: PolarsNumericType,
    T::Native: EwmFloat,
{
    if ca.len() <= 1 {
        return None;
    }

    let value = match ca.chunks().as_slice() {
        [_] => ca.scalar_value()?,
        _ => ca
            .repeats_one_element()
            // SAFETY: the column was just seen to hold more than one element.
            .then(|| unsafe { ca.get_unchecked(0) })?,
    };

    value.filter(|value| value.is_finite())
}

/// `length` elements of `value`, the first `nulls` of them null instead.
fn repeat_from<T: PolarsNumericType>(
    name: PlSmallStr,
    value: T::Native,
    nulls: usize,
    length: usize,
) -> ChunkedArray<T> {
    if nulls >= length {
        return ChunkedArray::full_null(name, length);
    }

    let values = PlPrimitiveArray::new_scalar(value, length);
    let arr = if nulls == 0 {
        values
    } else {
        let mut mask = BitmapBuilder::with_capacity(length);
        mask.extend_constant(nulls, false);
        mask.extend_constant(length - nulls, true);
        values.with_validity(Some(PlBitmap::from_bitmap(mask.freeze())))
    };

    ChunkedArray::with_chunk(name, arr)
}

/// The `ewm_mean` of a chunk that repeats one finite value, which is that value itself.
fn ewm_mean_scalar<T>(ca: &ChunkedArray<T>, options: &EWMOptions) -> Option<ChunkedArray<T>>
where
    T: PolarsNumericType,
    T::Native: EwmFloat,
{
    let value = one_finite_value(ca)?;
    let nulls = options.min_periods.saturating_sub(1);
    Some(repeat_from(ca.name().clone(), value, nulls, ca.len()))
}

/// The `ewm_std`/`ewm_var` of a chunk that repeats one finite value, which is zero throughout.
fn ewm_dispersion_scalar<T>(ca: &ChunkedArray<T>, options: &EWMOptions) -> Option<ChunkedArray<T>>
where
    T: PolarsNumericType,
    T::Native: EwmFloat,
{
    one_finite_value(ca)?;

    let nulls = if !options.bias && options.alpha == 1.0 {
        ca.len()
    } else {
        options
            .min_periods
            .saturating_sub(1)
            .max(usize::from(!options.bias))
    };
    Some(repeat_from(
        ca.name().clone(),
        T::Native::zero(),
        nulls,
        ca.len(),
    ))
}

/// The answer `$answer` has for a chunk that repeats one element, if `$s` is such a float chunk.
macro_rules! ewm_scalar_answer {
    ($s:expr, $options:expr, $answer:ident) => {{
        match $s.dtype() {
            #[cfg(feature = "dtype-f16")]
            DataType::Float16 => $answer($s.f16().unwrap(), &$options).map(|ca| ca.into_series()),
            DataType::Float32 => $answer($s.f32().unwrap(), &$options).map(|ca| ca.into_series()),
            DataType::Float64 => $answer($s.f64().unwrap(), &$options).map(|ca| ca.into_series()),
            _ => None,
        }
    }};
}

macro_rules! dispatch_ewm_kernel {
    ($s:expr, $options:expr, $fallback:ident, |$xs:ident, $alpha:ident| $kernel:expr) => {{
        check_alpha($options.alpha).inspect_err(|_| {
            if cfg!(debug_assertions) {
                panic!()
            }
        })?;
        match $s.dtype() {
            #[cfg(feature = "dtype-f16")]
            DataType::Float16 => {
                use num_traits::AsPrimitive;

                let $xs = $s.f16().unwrap();
                let $alpha = $options.alpha.as_();
                let result = $kernel;
                Ok(Float16Chunked::with_chunk($s.name().clone(), result).into_series())
            },
            DataType::Float32 => {
                let $xs = $s.f32().unwrap();
                let $alpha = $options.alpha as f32;
                let result = $kernel;
                Ok(Float32Chunked::with_chunk($s.name().clone(), result).into_series())
            },
            DataType::Float64 => {
                let $xs = $s.f64().unwrap();
                let $alpha = $options.alpha;
                let result = $kernel;
                Ok(Float64Chunked::with_chunk($s.name().clone(), result).into_series())
            },
            dt => polars_bail!(opq = $fallback, dt),
        }
    }};
}

pub fn ewm_mean(s: &Series, options: EWMOptions) -> PolarsResult<Series> {
    if check_alpha(options.alpha).is_ok()
        && let Some(out) = ewm_scalar_answer!(s, options, ewm_mean_scalar)
    {
        return Ok(out);
    }

    dispatch_ewm_kernel!(s, options, ewm_mean, |xs, alpha| kernel_ewm_mean(
        xs.iter(),
        alpha,
        options.adjust,
        options.min_periods,
        options.ignore_nulls,
    ))
}

pub fn ewm_sum(s: &Series, options: EWMOptions) -> PolarsResult<Series> {
    dispatch_ewm_kernel!(s, options, ewm_sum, |xs, alpha| kernel_ewm_sum(
        xs.iter(),
        alpha,
        options.min_periods,
        options.ignore_nulls,
    ))
}

pub fn ewm_std(s: &Series, options: EWMOptions) -> PolarsResult<Series> {
    check_alpha(options.alpha)?;

    if let Some(out) = ewm_scalar_answer!(s, options, ewm_dispersion_scalar) {
        return Ok(out);
    }

    match s.dtype() {
        #[cfg(feature = "dtype-f16")]
        DataType::Float16 => {
            use num_traits::AsPrimitive;

            let xs = s.f16().unwrap();
            let result = kernel_ewm_std(
                xs.iter(),
                options.alpha.as_(),
                options.adjust,
                options.bias,
                options.min_periods,
                options.ignore_nulls,
            );
            Ok(Float16Chunked::with_chunk(s.name().clone(), result).into_series())
        },
        DataType::Float32 => {
            let xs = s.f32().unwrap();
            let result = kernel_ewm_std(
                xs.iter(),
                options.alpha as f32,
                options.adjust,
                options.bias,
                options.min_periods,
                options.ignore_nulls,
            );
            Ok(Float32Chunked::with_chunk(s.name().clone(), result).into_series())
        },
        DataType::Float64 => {
            let xs = s.f64().unwrap();
            let result = kernel_ewm_std(
                xs.iter(),
                options.alpha,
                options.adjust,
                options.bias,
                options.min_periods,
                options.ignore_nulls,
            );
            Ok(Float64Chunked::with_chunk(s.name().clone(), result).into_series())
        },
        dt => {
            let casted = s.cast(&DataType::Float64)?;
            polars_ensure!(casted.dtype() == &DataType::Float64, opq = ewm_std, dt);
            ewm_std(&casted, options)
        },
    }
}

pub fn ewm_var(s: &Series, options: EWMOptions) -> PolarsResult<Series> {
    check_alpha(options.alpha)?;

    if let Some(out) = ewm_scalar_answer!(s, options, ewm_dispersion_scalar) {
        return Ok(out);
    }

    match s.dtype() {
        #[cfg(feature = "dtype-f16")]
        DataType::Float16 => {
            use num_traits::AsPrimitive;

            let xs = s.f16().unwrap();
            let result = kernel_ewm_var(
                xs.iter(),
                options.alpha.as_(),
                options.adjust,
                options.bias,
                options.min_periods,
                options.ignore_nulls,
            );
            Ok(Float16Chunked::with_chunk(s.name().clone(), result).into_series())
        },
        DataType::Float32 => {
            let xs = s.f32().unwrap();
            let result = kernel_ewm_var(
                xs.iter(),
                options.alpha as f32,
                options.adjust,
                options.bias,
                options.min_periods,
                options.ignore_nulls,
            );
            Ok(Float32Chunked::with_chunk(s.name().clone(), result).into_series())
        },
        DataType::Float64 => {
            let xs = s.f64().unwrap();
            let result = kernel_ewm_var(
                xs.iter(),
                options.alpha,
                options.adjust,
                options.bias,
                options.min_periods,
                options.ignore_nulls,
            );
            Ok(Float64Chunked::with_chunk(s.name().clone(), result).into_series())
        },
        dt => {
            let casted = s.cast(&DataType::Float64)?;
            polars_ensure!(casted.dtype() == &DataType::Float64, opq = ewm_var, dt);
            ewm_var(&casted, options)
        },
    }
}
