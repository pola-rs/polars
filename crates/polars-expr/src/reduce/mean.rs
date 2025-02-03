use std::marker::PhantomData;

use num_traits::{AsPrimitive, Zero};
use polars_core::with_match_physical_numeric_polars_type;

use super::*;

pub fn new_mean_reduction(dtype: DataType) -> Box<dyn GroupedReduction> {
    use DataType::*;
    use VecGroupedReduction as VGR;
    match dtype {
        Boolean => Box::new(VGR::new(dtype, BoolMeanReducer)),
        _ if dtype.is_primitive_numeric() || dtype.is_temporal() => {
            with_match_physical_numeric_polars_type!(dtype.to_physical(), |$T| {
                Box::new(VGR::new(dtype, NumMeanReducer::<$T>(PhantomData)))
            })
        },
        #[cfg(feature = "dtype-decimal")]
        Decimal(_, _) => Box::new(VGR::new(dtype, NumMeanReducer::<Int128Type>(PhantomData))),

        // For compatibility with the current engine, should probably be an error.
        String | Binary => Box::new(super::NullGroupedReduction::new(dtype)),

        _ => unimplemented!("{dtype:?} is not supported by mean reduction"),
    }
}

fn finish_output(values: Vec<(f64, usize)>, dtype: &DataType) -> Series {
    match dtype {
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
        #[cfg(feature = "dtype-decimal")]
        DataType::Decimal(_prec, scale) => {
            let inv_scale_factor = 1.0 / 10u128.pow(scale.unwrap() as u32) as f64;
            let ca: Float64Chunked = values
                .into_iter()
                .map(|(s, c)| (c != 0).then(|| s / c as f64 * inv_scale_factor))
                .collect_ca(PlSmallStr::EMPTY);
            ca.into_series()
        },
        #[cfg(feature = "dtype-datetime")]
        DataType::Date => {
            const MS_IN_DAY: i64 = 86_400_000;
            let ca: Int64Chunked = values
                .into_iter()
                .map(|(s, c)| (c != 0).then(|| (s / c as f64 * MS_IN_DAY as f64) as i64))
                .collect_ca(PlSmallStr::EMPTY);
            ca.into_datetime(TimeUnit::Milliseconds, None).into_series()
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
    ChunkedArray<T>: ChunkAgg<T::Native> + IntoSeries,
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
