use std::ops::{AddAssign, Mul};

use arity::unary_elementwise_values;
use arrow::array::{Array, BooleanArray};
use arrow::bitmap::{Bitmap, BitmapBuilder};
use num_traits::{Bounded, One, Zero};
use polars_core::prelude::*;
use polars_core::series::IsSorted;
use polars_core::utils::{CustomIterTools, NoNull};
use polars_core::with_match_physical_numeric_polars_type;
use polars_utils::float::IsFloat;
use polars_utils::kahan_sum::KahanSum;
use polars_utils::min_max::MinMax;

fn det_max<T>(state: &mut T, v: Option<T>) -> Option<Option<T>>
where
    T: Copy + MinMax,
{
    match v {
        Some(v) => {
            *state = MinMax::max_ignore_nan(*state, v);
            Some(Some(*state))
        },
        None => Some(None),
    }
}

fn det_min<T>(state: &mut T, v: Option<T>) -> Option<Option<T>>
where
    T: Copy + MinMax,
{
    match v {
        Some(v) => {
            *state = MinMax::min_ignore_nan(*state, v);
            Some(Some(*state))
        },
        None => Some(None),
    }
}

fn det_sum<T>(state: &mut T, v: Option<T>) -> Option<Option<T>>
where
    T: Copy + AddAssign,
{
    match v {
        Some(v) => {
            *state += v;
            Some(Some(*state))
        },
        None => Some(None),
    }
}

fn det_prod<T>(state: &mut T, v: Option<T>) -> Option<Option<T>>
where
    T: Copy + Mul<Output = T>,
{
    match v {
        Some(v) => {
            *state = *state * v;
            Some(Some(*state))
        },
        None => Some(None),
    }
}

fn cum_scan_numeric<T, F>(
    ca: &ChunkedArray<T>,
    reverse: bool,
    init: T::Native,
    update: F,
) -> ChunkedArray<T>
where
    T: PolarsNumericType,
    ChunkedArray<T>: FromIterator<Option<T::Native>>,
    F: Fn(&mut T::Native, Option<T::Native>) -> Option<Option<T::Native>>,
{
    let out: ChunkedArray<T> = match reverse {
        false => ca.iter().scan(init, update).collect_trusted(),
        true => ca.iter().rev().scan(init, update).collect_reversed(),
    };
    out.with_name(ca.name().clone())
}

fn cum_max_numeric<T>(
    ca: &ChunkedArray<T>,
    reverse: bool,
    init: Option<T::Native>,
) -> ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: MinMax + Bounded,
    ChunkedArray<T>: FromIterator<Option<T::Native>>,
{
    let init = init.unwrap_or(if T::Native::is_float() {
        T::Native::nan_value()
    } else {
        Bounded::min_value()
    });
    cum_scan_numeric(ca, reverse, init, det_max)
}

fn cum_min_numeric<T>(
    ca: &ChunkedArray<T>,
    reverse: bool,
    init: Option<T::Native>,
) -> ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: MinMax + Bounded,
    ChunkedArray<T>: FromIterator<Option<T::Native>>,
{
    let init = init.unwrap_or(if T::Native::is_float() {
        T::Native::nan_value()
    } else {
        Bounded::max_value()
    });
    cum_scan_numeric(ca, reverse, init, det_min)
}

fn cum_max_bool(ca: &BooleanChunked, reverse: bool, init: Option<bool>) -> BooleanChunked {
    if ca.len() == ca.null_count() {
        return ca.clone();
    }

    if init == Some(true) {
        return unsafe {
            BooleanChunked::from_chunks(
                ca.name().clone(),
                ca.downcast_iter()
                    .map(|arr| {
                        arr.with_values(Bitmap::new_with_value(true, arr.len()))
                            .to_boxed()
                    })
                    .collect(),
            )
        };
    }

    let mut out;
    if !reverse {
        // TODO: efficient bitscan.
        let Some(first_true_idx) = ca.iter().position(|x| x == Some(true)) else {
            return ca.clone();
        };
        out = BitmapBuilder::with_capacity(ca.len());
        out.extend_constant(first_true_idx, false);
        out.extend_constant(ca.len() - first_true_idx, true);
    } else {
        // TODO: efficient bitscan.
        let Some(last_true_idx) = ca.iter().rposition(|x| x == Some(true)) else {
            return ca.clone();
        };
        out = BitmapBuilder::with_capacity(ca.len());
        out.extend_constant(last_true_idx + 1, true);
        out.extend_constant(ca.len() - 1 - last_true_idx, false);
    }

    let arr: BooleanArray = out.freeze().into();
    BooleanChunked::with_chunk_like(ca, arr.with_validity(ca.rechunk_validity()))
}

fn cum_min_bool(ca: &BooleanChunked, reverse: bool, init: Option<bool>) -> BooleanChunked {
    if ca.len() == ca.null_count() {
        return ca.clone();
    }

    if init == Some(false) {
        return unsafe {
            BooleanChunked::from_chunks(
                ca.name().clone(),
                ca.downcast_iter()
                    .map(|arr| {
                        arr.with_values(Bitmap::new_with_value(false, arr.len()))
                            .to_boxed()
                    })
                    .collect(),
            )
        };
    }

    let mut out;
    if !reverse {
        // TODO: efficient bitscan.
        let Some(first_false_idx) = ca.iter().position(|x| x == Some(false)) else {
            return ca.clone();
        };
        out = BitmapBuilder::with_capacity(ca.len());
        out.extend_constant(first_false_idx, true);
        out.extend_constant(ca.len() - first_false_idx, false);
    } else {
        // TODO: efficient bitscan.
        let Some(last_false_idx) = ca.iter().rposition(|x| x == Some(false)) else {
            return ca.clone();
        };
        out = BitmapBuilder::with_capacity(ca.len());
        out.extend_constant(last_false_idx + 1, false);
        out.extend_constant(ca.len() - 1 - last_false_idx, true);
    }

    let arr: BooleanArray = out.freeze().into();
    BooleanChunked::with_chunk_like(ca, arr.with_validity(ca.rechunk_validity()))
}

fn cum_sum_numeric<T>(
    ca: &ChunkedArray<T>,
    reverse: bool,
    init: Option<T::Native>,
) -> ChunkedArray<T>
where
    T: PolarsNumericType,
    ChunkedArray<T>: FromIterator<Option<T::Native>>,
{
    let init = init.unwrap_or(T::Native::zero());
    cum_scan_numeric(ca, reverse, init, det_sum)
}

#[cfg(feature = "dtype-decimal")]
fn cum_sum_decimal(
    ca: &Int128Chunked,
    reverse: bool,
    init: Option<i128>,
) -> PolarsResult<Int128Chunked> {
    use polars_compute::decimal::{DEC128_MAX_PREC, dec128_add};

    let mut value = init.unwrap_or(0);
    let update = |opt_v| {
        if let Some(v) = opt_v {
            value = dec128_add(value, v, DEC128_MAX_PREC).ok_or_else(
                || polars_err!(ComputeError: "overflow in decimal addition in cum_sum"),
            )?;
            Ok(Some(value))
        } else {
            Ok(None)
        }
    };
    if reverse {
        ca.iter().rev().map(update).try_collect_ca_trusted_like(ca)
    } else {
        ca.iter().map(update).try_collect_ca_trusted_like(ca)
    }
}

#[cfg(feature = "dtype-decimal")]
fn cum_mean_decimal(
    ca: &DecimalChunked,
    reverse: bool,
    init_sum: Option<i128>,
    init_count: Option<u64>,
) -> PolarsResult<(DecimalChunked, i128, u64)> {
    use polars_compute::decimal::{DEC128_MAX_PREC, dec128_add, dec128_div};

    let mut sum = init_sum.unwrap_or(0);
    let mut count = init_count.unwrap_or(0);
    let update = |opt_v| -> PolarsResult<Option<i128>> {
        if let Some(v) = opt_v {
            sum = dec128_add(sum, v, DEC128_MAX_PREC).ok_or_else(
                || polars_err!(ComputeError: "overflow in decimal addition in cum_mean"),
            )?;
            count += 1;
            let mean = dec128_div(sum, count as i128, DEC128_MAX_PREC, 0).ok_or_else(
                || polars_err!(ComputeError: "overflow in decimal division in cum_mean"),
            )?;
            Ok(Some(mean))
        } else {
            Ok(None)
        }
    };
    let precision = ca.precision();
    let scale = ca.scale();
    let phys = ca.physical();
    let out = if reverse {
        phys.iter()
            .rev()
            .map(update)
            .try_collect_ca_trusted_like(phys)?
    } else {
        phys.iter().map(update).try_collect_ca_trusted_like(phys)?
    };
    Ok((out.into_decimal_unchecked(precision, scale), sum, count))
}

/// Cumulative mean for Decimal with i128 state (for streaming).
/// Returns (output_series, final_sum_i128, final_count).
#[cfg(feature = "dtype-decimal")]
pub fn cum_mean_decimal_with_init(
    s: &Series,
    reverse: bool,
    init_sum: Option<i128>,
    init_count: Option<u64>,
) -> PolarsResult<(Series, i128, u64)> {
    let ca = s.decimal()?;
    let (out, sum, count) = cum_mean_decimal(ca, reverse, init_sum, init_count)?;
    Ok((out.into_series(), sum, count))
}

fn cum_prod_numeric<T>(
    ca: &ChunkedArray<T>,
    reverse: bool,
    init: Option<T::Native>,
) -> ChunkedArray<T>
where
    T: PolarsNumericType,
    ChunkedArray<T>: FromIterator<Option<T::Native>>,
{
    let init = init.unwrap_or(T::Native::one());
    cum_scan_numeric(ca, reverse, init, det_prod)
}

pub fn cum_prod_with_init(
    s: &Series,
    reverse: bool,
    init: &AnyValue<'static>,
) -> PolarsResult<Series> {
    use DataType::*;
    let out = match s.dtype() {
        Boolean | Int8 | UInt8 | Int16 | UInt16 | Int32 | UInt32 => {
            let s = s.cast(&Int64)?;
            cum_prod_numeric(s.i64()?, reverse, init.extract()).into_series()
        },
        Int64 => cum_prod_numeric(s.i64()?, reverse, init.extract()).into_series(),
        UInt64 => cum_prod_numeric(s.u64()?, reverse, init.extract()).into_series(),
        #[cfg(feature = "dtype-i128")]
        Int128 => cum_prod_numeric(s.i128()?, reverse, init.extract()).into_series(),
        #[cfg(feature = "dtype-u128")]
        UInt128 => cum_prod_numeric(s.u128()?, reverse, init.extract()).into_series(),
        #[cfg(feature = "dtype-f16")]
        Float16 => cum_prod_numeric(s.f16()?, reverse, init.extract()).into_series(),
        Float32 => cum_prod_numeric(s.f32()?, reverse, init.extract()).into_series(),
        Float64 => cum_prod_numeric(s.f64()?, reverse, init.extract()).into_series(),
        dt => polars_bail!(opq = cum_prod, dt),
    };
    Ok(out)
}

/// Get an array with the cumulative product computed at every element.
///
/// If the [`DataType`] is one of `{Int8, UInt8, Int16, UInt16, Int32, UInt32}` the `Series` is
/// first cast to `Int64` to prevent overflow issues.
pub fn cum_prod(s: &Series, reverse: bool) -> PolarsResult<Series> {
    cum_prod_with_init(s, reverse, &AnyValue::Null)
}

pub fn cum_sum_with_init(
    s: &Series,
    reverse: bool,
    init: &AnyValue<'static>,
) -> PolarsResult<Series> {
    use DataType::*;
    let out = match s.dtype() {
        Boolean => {
            let s = s.cast(&UInt32)?;
            cum_sum_numeric(s.u32()?, reverse, init.extract()).into_series()
        },
        Int8 | UInt8 | Int16 | UInt16 => {
            let s = s.cast(&Int64)?;
            cum_sum_numeric(s.i64()?, reverse, init.extract()).into_series()
        },
        Int32 => cum_sum_numeric(s.i32()?, reverse, init.extract()).into_series(),
        UInt32 => cum_sum_numeric(s.u32()?, reverse, init.extract()).into_series(),
        Int64 => cum_sum_numeric(s.i64()?, reverse, init.extract()).into_series(),
        UInt64 => cum_sum_numeric(s.u64()?, reverse, init.extract()).into_series(),
        #[cfg(feature = "dtype-u128")]
        UInt128 => cum_sum_numeric(s.u128()?, reverse, init.extract()).into_series(),
        #[cfg(feature = "dtype-i128")]
        Int128 => cum_sum_numeric(s.i128()?, reverse, init.extract()).into_series(),
        #[cfg(feature = "dtype-f16")]
        Float16 => cum_sum_numeric(s.f16()?, reverse, init.extract()).into_series(),
        Float32 => cum_sum_numeric(s.f32()?, reverse, init.extract()).into_series(),
        Float64 => cum_sum_numeric(s.f64()?, reverse, init.extract()).into_series(),
        #[cfg(feature = "dtype-decimal")]
        Decimal(_precision, scale) => {
            use polars_compute::decimal::DEC128_MAX_PREC;
            let ca = s.decimal().unwrap().physical();
            cum_sum_decimal(ca, reverse, init.clone().to_physical().extract())?
                .into_decimal_unchecked(DEC128_MAX_PREC, *scale)
                .into_series()
        },
        #[cfg(feature = "dtype-duration")]
        Duration(tu) => {
            let s = s.to_physical_repr();
            let ca = s.i64()?;
            cum_sum_numeric(ca, reverse, init.extract()).cast(&Duration(*tu))?
        },
        dt => polars_bail!(opq = cum_sum, dt),
    };
    Ok(out)
}

/// Get an array with the cumulative sum computed at every element
///
/// If the [`DataType`] is one of `{Int8, UInt8, Int16, UInt16}` the `Series` is
/// first cast to `Int64` to prevent overflow issues.
pub fn cum_sum(s: &Series, reverse: bool) -> PolarsResult<Series> {
    cum_sum_with_init(s, reverse, &AnyValue::Null)
}

pub fn cum_min_with_init(
    s: &Series,
    reverse: bool,
    init: &AnyValue<'static>,
) -> PolarsResult<Series> {
    match s.dtype() {
        DataType::Boolean => {
            Ok(cum_min_bool(s.bool()?, reverse, init.extract_bool()).into_series())
        },
        #[cfg(feature = "dtype-decimal")]
        DataType::Decimal(precision, scale) => {
            let ca = s.decimal().unwrap().physical();
            let out = cum_min_numeric(ca, reverse, init.clone().to_physical().extract())
                .into_decimal_unchecked(*precision, *scale)
                .into_series();
            Ok(out)
        },
        dt if dt.to_physical().is_primitive_numeric() => {
            let s = s.to_physical_repr();
            with_match_physical_numeric_polars_type!(s.dtype(), |$T| {
                let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
                let out = cum_min_numeric(ca, reverse, init.extract()).into_series();
                if dt.is_logical() {
                    out.cast(dt)
                } else {
                    Ok(out)
                }
            })
        },
        dt => polars_bail!(opq = cum_min, dt),
    }
}

/// Get an array with the cumulative min computed at every element.
pub fn cum_min(s: &Series, reverse: bool) -> PolarsResult<Series> {
    cum_min_with_init(s, reverse, &AnyValue::Null)
}

pub fn cum_max_with_init(
    s: &Series,
    reverse: bool,
    init: &AnyValue<'static>,
) -> PolarsResult<Series> {
    match s.dtype() {
        DataType::Boolean => {
            Ok(cum_max_bool(s.bool()?, reverse, init.extract_bool()).into_series())
        },
        #[cfg(feature = "dtype-decimal")]
        DataType::Decimal(precision, scale) => {
            let ca = s.decimal().unwrap().physical();
            let out = cum_max_numeric(ca, reverse, init.clone().to_physical().extract())
                .into_decimal_unchecked(*precision, *scale)
                .into_series();
            Ok(out)
        },
        dt if dt.to_physical().is_primitive_numeric() => {
            let s = s.to_physical_repr();
            with_match_physical_numeric_polars_type!(s.dtype(), |$T| {
                let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
                let out = cum_max_numeric(ca, reverse, init.extract()).into_series();
                if dt.is_logical() {
                    out.cast(dt)
                } else {
                    Ok(out)
                }
            })
        },
        dt => polars_bail!(opq = cum_max, dt),
    }
}

/// Get an array with the cumulative max computed at every element.
pub fn cum_max(s: &Series, reverse: bool) -> PolarsResult<Series> {
    cum_max_with_init(s, reverse, &AnyValue::Null)
}

/// Get an array with the cumulative count computed at every element.
pub fn cum_count(s: &Series, reverse: bool) -> PolarsResult<Series> {
    cum_count_with_init(s, reverse, 0)
}

pub fn cum_count_with_init(s: &Series, reverse: bool, init: IdxSize) -> PolarsResult<Series> {
    let mut out = if s.null_count() == 0 {
        // Fast paths for no nulls
        cum_count_no_nulls(s.name().clone(), s.len(), reverse, init)
    } else {
        let ca = s.is_not_null();
        let out: IdxCa = if reverse {
            let mut count = init + (s.len() - s.null_count()) as IdxSize;
            let mut prev = false;
            unary_elementwise_values(&ca, |v: bool| {
                if prev {
                    count -= 1;
                }
                prev = v;
                count
            })
        } else {
            let mut count = init;
            unary_elementwise_values(&ca, |v: bool| {
                if v {
                    count += 1;
                }
                count
            })
        };

        out.into()
    };

    out.set_sorted_flag([IsSorted::Ascending, IsSorted::Descending][reverse as usize]);

    Ok(out)
}

fn cum_count_no_nulls(name: PlSmallStr, len: usize, reverse: bool, init: IdxSize) -> Series {
    let start = 1 as IdxSize;
    let end = len as IdxSize + 1;
    let ca: NoNull<IdxCa> = if reverse {
        (start..end).rev().map(|v| v + init).collect()
    } else {
        (start..end).map(|v| v + init).collect()
    };
    let mut ca = ca.into_inner();
    ca.rename(name);
    ca.into_series()
}

/// Get an array with the cumulative mean computed at every element.
pub fn cum_mean(s: &Series, reverse: bool) -> PolarsResult<Series> {
    #[cfg(feature = "dtype-decimal")]
    if matches!(s.dtype(), DataType::Decimal(_, _)) {
        return cum_mean_decimal_with_init(s, reverse, None, None).map(|(s, _, _)| s);
    }
    cum_mean_with_init(s, reverse, None, None, None).map(|(s, _, _, _)| s)
}

fn cum_mean_numeric<T>(
    ca: &ChunkedArray<T>,
    reverse: bool,
    init_sum: Option<f64>,
    init_count: Option<u64>,
    init_err: Option<f64>,
) -> (Float64Chunked, f64, u64, f64)
where
    T: PolarsNumericType,
{
    let mut ksum = KahanSum::new(init_sum.unwrap_or(0.0));
    if let Some(err) = init_err {
        ksum.set_err(err);
    }
    let mut count = init_count.unwrap_or(0);
    let update = |opt_v: Option<T::Native>| -> Option<f64> {
        opt_v.map(|v| {
            ksum += num_traits::ToPrimitive::to_f64(&v).unwrap();
            count += 1;
            ksum.sum() / count as f64
        })
    };
    let out: Float64Chunked = if reverse {
        ca.iter().rev().map(update).collect_reversed()
    } else {
        ca.iter().map(update).collect_trusted()
    };
    (
        out.with_name(ca.name().clone()),
        ksum.sum(),
        count,
        ksum.err(),
    )
}

fn cum_mean_temporal(
    s: &Series,
    reverse: bool,
    init_sum: Option<f64>,
    init_count: Option<u64>,
    init_err: Option<f64>,
    target_dtype: &DataType,
) -> PolarsResult<(Series, f64, u64, f64)> {
    use DataType::*;

    let phys = s.to_physical_repr();
    let (out, sum, count, err) =
        cum_mean_numeric(phys.i64()?, reverse, init_sum, init_count, init_err);
    let out = out.cast(&Int64)?.cast(target_dtype)?;
    Ok((out, sum, count, err))
}

/// Get an array with the cumulative mean computed at every element.
/// Also returns the final (sum, count, kahan_err) state for streaming.
pub fn cum_mean_with_init(
    s: &Series,
    reverse: bool,
    init_sum: Option<f64>,
    init_count: Option<u64>,
    init_err: Option<f64>,
) -> PolarsResult<(Series, f64, u64, f64)> {
    use DataType::*;

    match s.dtype() {
        Boolean => {
            let s = s.cast(&UInt8)?;
            let (out, sum, count, err) =
                cum_mean_numeric(s.u8()?, reverse, init_sum, init_count, init_err);
            Ok((out.into_series(), sum, count, err))
        },
        #[cfg(feature = "dtype-duration")]
        dt @ Duration(_) => cum_mean_temporal(s, reverse, init_sum, init_count, init_err, dt),
        #[cfg(feature = "dtype-datetime")]
        dt @ Datetime(_, _) => cum_mean_temporal(s, reverse, init_sum, init_count, init_err, dt),
        #[cfg(feature = "dtype-date")]
        Date => {
            let target = Datetime(polars_core::prelude::TimeUnit::Microseconds, None);
            cum_mean_temporal(
                &s.cast(&target)?,
                reverse,
                init_sum,
                init_count,
                init_err,
                &target,
            )
        },
        Float32 => {
            let (out, sum, count, err) =
                cum_mean_numeric(s.f32()?, reverse, init_sum, init_count, init_err);
            Ok((out.cast(&Float32)?, sum, count, err))
        },
        #[cfg(feature = "dtype-f16")]
        Float16 => {
            let (out, sum, count, err) =
                cum_mean_numeric(s.f16()?, reverse, init_sum, init_count, init_err);
            Ok((out.cast(&Float16)?, sum, count, err))
        },
        dt if dt.is_primitive_numeric() => {
            with_match_physical_numeric_polars_type!(s.dtype(), |$T| {
                let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
                let (out, sum, count, err) =
                    cum_mean_numeric(ca, reverse, init_sum, init_count, init_err);
                Ok((out.into_series(), sum, count, err))
            })
        },
        dt => polars_bail!(opq = cum_mean, dt),
    }
}
