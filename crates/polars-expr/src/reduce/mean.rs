use std::marker::PhantomData;

use num_traits::{AsPrimitive, Zero};
use polars_arrow::temporal_conversions::MICROSECONDS_IN_DAY;
use polars_core::with_match_physical_numeric_polars_type;

use super::*;

pub fn new_mean_reduction(dtype: DataType) -> PolarsResult<Box<dyn GroupedReduction>> {
    // TODO: Move the error checks up and make this function infallible
    use DataType::*;
    use VecGroupedReduction as VGR;
    Ok(match dtype {
        Boolean => Box::new(VGR::new(dtype, BoolMeanReducer)),
        _ if dtype.is_primitive_numeric() || dtype.is_temporal() => {
            with_match_physical_numeric_polars_type!(dtype.to_physical(), |$T| {
                Box::new(VGR::new(dtype, NumMeanReducer::<$T>(PhantomData)))
            })
        },
        #[cfg(feature = "dtype-decimal")]
        Decimal(_, _) => Box::new(VGR::new(dtype, DecimalMeanReducer)),
        Null => Box::new(super::NullGroupedReduction::new(Scalar::null(
            DataType::Null,
        ))),
        _ => polars_bail!(InvalidOperation: "`mean` operation not supported for dtype `{dtype}`"),
    })
}

fn finish_output(values: Vec<(f64, usize)>, dtype: &DataType) -> Series {
    match dtype {
        #[cfg(feature = "dtype-f16")]
        DataType::Float16 => {
            let ca: Float16Chunked = values
                .into_iter()
                .map(|(s, c)| (c != 0).then(|| (s / c as f64).as_()))
                .collect_ca(PlSmallStr::EMPTY);
            ca.into_series()
        },
        DataType::Float32 => {
            let ca: Float32Chunked = values
                .into_iter()
                .map(|(s, c)| (c != 0).then(|| (s / c as f64) as f32))
                .collect_ca(PlSmallStr::EMPTY);
            ca.into_series()
        },
        dt if dt.is_primitive_numeric() => {
            let ca: Float64Chunked = values
                .into_iter()
                .map(|(s, c)| (c != 0).then(|| s / c as f64))
                .collect_ca(PlSmallStr::EMPTY);
            ca.into_series()
        },
        #[cfg(feature = "dtype-datetime")]
        DataType::Date => {
            const US_IN_DAY: f64 = MICROSECONDS_IN_DAY as f64;
            let ca: Int64Chunked = values
                .into_iter()
                .map(|(s, c)| (c != 0).then(|| (s / c as f64 * US_IN_DAY) as i64))
                .collect_ca(PlSmallStr::EMPTY);
            ca.into_datetime(TimeUnit::Microseconds, None).into_series()
        },
        DataType::Datetime(_, _) | DataType::Duration(_) | DataType::Time => {
            let ca: Int64Chunked = values
                .into_iter()
                .map(|(s, c)| (c != 0).then(|| (s / c as f64) as i64))
                .collect_ca(PlSmallStr::EMPTY);
            ca.into_series().cast(dtype).unwrap()
        },
        _ => unimplemented!(),
    }
}

struct NumMeanReducer<T>(PhantomData<T>);
impl<T> Clone for NumMeanReducer<T> {
    fn clone(&self) -> Self {
        Self(PhantomData)
    }
}

impl<T> Reducer for NumMeanReducer<T>
where
    T: PolarsNumericType,
    ChunkedArray<T>: ChunkAgg<T::Native>,
{
    type Dtype = T;
    type Value = (f64, usize);

    #[inline(always)]
    fn init(&self) -> Self::Value {
        (0.0, 0)
    }

    fn cast_series<'a>(&self, s: &'a Series) -> Cow<'a, Series> {
        s.to_physical_repr()
    }

    #[inline(always)]
    fn combine(&self, a: &mut Self::Value, b: &Self::Value) {
        a.0 += b.0;
        a.1 += b.1;
    }

    #[inline(always)]
    fn reduce_one(&self, a: &mut Self::Value, b: Option<T::Native>, _seq_id: u64) {
        a.0 += b.unwrap_or(T::Native::zero()).as_();
        a.1 += b.is_some() as usize;
    }

    fn reduce_ca(&self, v: &mut Self::Value, ca: &ChunkedArray<Self::Dtype>, _seq_id: u64) {
        v.0 += ChunkAgg::_sum_as_f64(ca);
        v.1 += ca.len() - ca.null_count();
    }

    fn finish(
        &self,
        v: Vec<Self::Value>,
        m: Option<Bitmap>,
        dtype: &DataType,
    ) -> PolarsResult<Series> {
        assert!(m.is_none());
        Ok(finish_output(v, dtype))
    }
}

/// Accumulates Decimal128 values in i128, spilling into an f64 when a group's
/// sum would overflow. A sum past i128 can still have a mean that f64 holds.
#[cfg(feature = "dtype-decimal")]
#[derive(Clone)]
struct DecimalMeanReducer;

#[cfg(feature = "dtype-decimal")]
impl Reducer for DecimalMeanReducer {
    type Dtype = Int128Type;
    type Value = (i128, f64, usize);

    #[inline(always)]
    fn init(&self) -> Self::Value {
        (0, 0.0, 0)
    }

    fn cast_series<'a>(&self, s: &'a Series) -> Cow<'a, Series> {
        s.to_physical_repr()
    }

    #[inline(always)]
    fn combine(&self, a: &mut Self::Value, b: &Self::Value) {
        a.1 += b.1;
        a.2 += b.2;
        match a.0.checked_add(b.0) {
            Some(v) => a.0 = v,
            None => {
                a.1 += a.0 as f64;
                a.0 = b.0;
            },
        }
    }

    #[inline(always)]
    fn reduce_one(&self, a: &mut Self::Value, b: Option<i128>, _seq_id: u64) {
        let x = b.unwrap_or(0);
        let (v, overflowed) = a.0.overflowing_add(x);
        if overflowed {
            a.1 += a.0 as f64;
            a.0 = x;
        } else {
            a.0 = v;
        }
        a.2 += b.is_some() as usize;
    }

    fn reduce_ca(&self, v: &mut Self::Value, ca: &ChunkedArray<Self::Dtype>, _seq_id: u64) {
        for x in ca.iter().flatten() {
            self.reduce_one(v, Some(x), 0);
        }
    }

    fn finish(
        &self,
        v: Vec<Self::Value>,
        m: Option<Bitmap>,
        dtype: &DataType,
    ) -> PolarsResult<Series> {
        assert!(m.is_none());
        let DataType::Decimal(_, scale) = dtype else {
            unreachable!()
        };
        let scale_factor = 10u128.pow(*scale as u32) as f64;
        let ca: Float64Chunked = v
            .into_iter()
            .map(|(acc, spilled, c)| {
                (c != 0).then(|| (spilled + acc as f64) / c as f64 / scale_factor)
            })
            .collect_ca(PlSmallStr::EMPTY);
        Ok(ca.into_series())
    }
}

#[derive(Clone)]
struct BoolMeanReducer;

impl Reducer for BoolMeanReducer {
    type Dtype = BooleanType;
    type Value = (usize, usize);

    #[inline(always)]
    fn init(&self) -> Self::Value {
        (0, 0)
    }

    #[inline(always)]
    fn combine(&self, a: &mut Self::Value, b: &Self::Value) {
        a.0 += b.0;
        a.1 += b.1;
    }

    #[inline(always)]
    fn reduce_one(&self, a: &mut Self::Value, b: Option<bool>, _seq_id: u64) {
        a.0 += b.unwrap_or(false) as usize;
        a.1 += b.is_some() as usize;
    }

    fn reduce_ca(&self, v: &mut Self::Value, ca: &ChunkedArray<Self::Dtype>, _seq_id: u64) {
        v.0 += ca.sum().unwrap_or(0) as usize;
        v.1 += ca.len() - ca.null_count();
    }

    fn finish(
        &self,
        v: Vec<Self::Value>,
        m: Option<Bitmap>,
        dtype: &DataType,
    ) -> PolarsResult<Series> {
        assert!(m.is_none());
        assert!(dtype == &DataType::Boolean);
        let ca: Float64Chunked = v
            .into_iter()
            .map(|(s, c)| (c != 0).then(|| s as f64 / c as f64))
            .collect_ca(PlSmallStr::EMPTY);
        Ok(ca.into_series())
    }
}
