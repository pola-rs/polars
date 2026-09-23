use std::marker::PhantomData;

use num_traits::AsPrimitive;
use polars_arrow::temporal_conversions::MICROSECONDS_IN_DAY;
use polars_compute::mean::{IntMeanRounding, MeanAcc, MeanSum, int_mean};
use polars_compute::sum::WrappingAdd;
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
        Decimal(_, _) => Box::new(VGR::new(dtype, NumMeanReducer::<Int128Type>(PhantomData))),
        Null => Box::new(super::NullGroupedReduction::new(Scalar::null(
            DataType::Null,
        ))),
        _ => polars_bail!(InvalidOperation: "`mean` operation not supported for dtype `{dtype}`"),
    })
}

fn finish_output<A: MeanAcc>(values: Vec<(A, usize)>, dtype: &DataType) -> Series {
    let int_means = |scale: i64, rounding: IntMeanRounding| -> Int64Chunked {
        values
            .iter()
            .map(|&(s, c)| {
                (c != 0).then(|| int_mean(s.try_into_i128().unwrap(), c, scale, rounding))
            })
            .collect_ca(PlSmallStr::EMPTY)
    };
    match dtype {
        #[cfg(feature = "dtype-f16")]
        DataType::Float16 => {
            let ca: Float16Chunked = values
                .into_iter()
                .map(|(s, c)| (c != 0).then(|| (s.into_f64() / c as f64).as_()))
                .collect_ca(PlSmallStr::EMPTY);
            ca.into_series()
        },
        DataType::Float32 => {
            let ca: Float32Chunked = values
                .into_iter()
                .map(|(s, c)| (c != 0).then(|| (s.into_f64() / c as f64) as f32))
                .collect_ca(PlSmallStr::EMPTY);
            ca.into_series()
        },
        dt if dt.is_primitive_numeric() => {
            let ca: Float64Chunked = values
                .into_iter()
                .map(|(s, c)| (c != 0).then(|| s.into_f64() / c as f64))
                .collect_ca(PlSmallStr::EMPTY);
            ca.into_series()
        },
        #[cfg(feature = "dtype-decimal")]
        DataType::Decimal(_prec, scale) => {
            let scale_factor = 10u128.pow(*scale as u32) as f64;
            let ca: Float64Chunked = values
                .into_iter()
                .map(|(s, c)| (c != 0).then(|| s.into_f64() / c as f64 / scale_factor))
                .collect_ca(PlSmallStr::EMPTY);
            ca.into_series()
        },
        #[cfg(feature = "dtype-datetime")]
        DataType::Date => int_means(MICROSECONDS_IN_DAY, IntMeanRounding::Floor)
            .into_datetime(TimeUnit::Microseconds, None)
            .into_series(),
        DataType::Datetime(_, _) | DataType::Time => int_means(1, IntMeanRounding::Floor)
            .into_series()
            .cast(dtype)
            .unwrap(),
        DataType::Duration(_) => int_means(1, IntMeanRounding::Trunc)
            .into_series()
            .cast(dtype)
            .unwrap(),
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
    type Value = (<T::Native as MeanSum>::Acc, usize);

    #[inline(always)]
    fn init(&self) -> Self::Value {
        (Default::default(), 0)
    }

    fn cast_series<'a>(&self, s: &'a Series) -> Cow<'a, Series> {
        s.to_physical_repr()
    }

    #[inline(always)]
    fn combine(&self, a: &mut Self::Value, b: &Self::Value) {
        a.0 = a.0.wrapping_add(&b.0);
        a.1 += b.1;
    }

    #[inline(always)]
    fn reduce_one(&self, a: &mut Self::Value, b: Option<T::Native>, _seq_id: u64) {
        if let Some(b) = b {
            a.0 = a.0.wrapping_add(&b.to_mean_acc());
            a.1 += 1;
        }
    }

    fn reduce_ca(&self, v: &mut Self::Value, ca: &ChunkedArray<Self::Dtype>, _seq_id: u64) {
        v.0 = v.0.wrapping_add(&ca.mean_sum());
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
