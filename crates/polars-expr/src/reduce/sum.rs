use std::borrow::Cow;

use arrow::array::PrimitiveArray;
use num_traits::Zero;
use polars_core::with_match_physical_numeric_polars_type;
use polars_utils::float::IsFloat;

use super::*;

pub trait SumCast: Sized {
    type Sum: NumericNative + From<Self>;
}

macro_rules! impl_sum_cast {
    ($($x:ty),*) => {
        $(impl SumCast for $x { type Sum = $x; })*
    };
    ($($from:ty as $to:ty),*) => {
        $(impl SumCast for $from { type Sum = $to; })*
    };
}

impl_sum_cast!(
    bool as IdxSize,
    u8 as i64,
    u16 as i64,
    i8 as i64,
    i16 as i64
);
impl_sum_cast!(u32, u64, i32, i64, f32, f64);
#[cfg(feature = "dtype-i128")]
impl_sum_cast!(i128);

fn out_dtype(in_dtype: &DataType) -> DataType {
    use DataType::*;
    match in_dtype {
        Boolean => IDX_DTYPE,
        Int8 | UInt8 | Int16 | UInt16 => Int64,
        dt => dt.clone(),
    }
}

pub fn new_sum_reduction(dtype: DataType) -> Box<dyn GroupedReduction> {
    use DataType::*;
    use VecGroupedReduction as VGR;
    match dtype {
        Boolean => Box::new(VGR::new(dtype, BoolSumReducer)),
        _ if dtype.is_primitive_numeric() => {
            with_match_physical_numeric_polars_type!(dtype.to_physical(), |$T| {
                Box::new(VGR::new(dtype, NumSumReducer::<$T>(PhantomData)))
            })
        },
        #[cfg(feature = "dtype-decimal")]
        Decimal(_, _) => Box::new(VGR::new(dtype, NumSumReducer::<Int128Type>(PhantomData))),
        Duration(_) => Box::new(VGR::new(dtype, NumSumReducer::<Int64Type>(PhantomData))),
        // For compatibility with the current engine, should probably be an error.
        String | Binary => Box::new(super::NullGroupedReduction::new(dtype)),
        _ => unimplemented!("{dtype:?} is not supported by sum reduction"),
    }
}

struct NumSumReducer<T>(PhantomData<T>);
impl<T> Clone for NumSumReducer<T> {
    fn clone(&self) -> Self {
        Self(PhantomData)
    }
}

impl<T> Reducer for NumSumReducer<T>
where
    T: PolarsNumericType,
    <T as PolarsNumericType>::Native: SumCast,
    ChunkedArray<T>: ChunkAgg<T::Native> + IntoSeries,
{
    type Dtype = T;
    type Value = <T::Native as SumCast>::Sum;

    #[inline(always)]
    fn init(&self) -> Self::Value {
        Zero::zero()
    }

    fn cast_series<'a>(&self, s: &'a Series) -> Cow<'a, Series> {
        s.to_physical_repr()
    }

    #[inline(always)]
    fn combine(&self, a: &mut Self::Value, b: &Self::Value) {
        *a += *b;
    }

    #[inline(always)]
    fn reduce_one(&self, a: &mut Self::Value, b: Option<T::Native>, _seq_id: u64) {
        *a += b.map(Into::into).unwrap_or(Zero::zero());
    }

    fn reduce_ca(&self, v: &mut Self::Value, ca: &ChunkedArray<Self::Dtype>, _seq_id: u64) {
        if T::Native::is_float() {
            *v += ChunkAgg::sum(ca).map(Into::into).unwrap_or(Zero::zero());
        } else {
            for arr in ca.downcast_iter() {
                if arr.has_nulls() {
                    for x in arr.iter() {
                        *v += x.copied().map(Into::into).unwrap_or(Zero::zero());
                    }
                } else {
                    for x in arr.values_iter().copied() {
                        *v += x.into();
                    }
                }
            }
        }
    }

    fn finish(
        &self,
        v: Vec<Self::Value>,
        m: Option<Bitmap>,
        dtype: &DataType,
    ) -> PolarsResult<Series> {
        assert!(m.is_none());
        let arr = Box::new(PrimitiveArray::from_vec(v));
        Ok(unsafe {
            Series::from_chunks_and_dtype_unchecked(PlSmallStr::EMPTY, vec![arr], &out_dtype(dtype))
        })
    }
}

#[derive(Clone)]
struct BoolSumReducer;

impl Reducer for BoolSumReducer {
    type Dtype = BooleanType;
    type Value = IdxSize;

    #[inline(always)]
    fn init(&self) -> Self::Value {
        0
    }

    #[inline(always)]
    fn combine(&self, a: &mut Self::Value, b: &Self::Value) {
        *a += *b;
    }

    #[inline(always)]
    fn reduce_one(&self, a: &mut Self::Value, b: Option<bool>, _seq_id: u64) {
        *a += b.unwrap_or(false) as IdxSize;
    }

    fn reduce_ca(&self, v: &mut Self::Value, ca: &ChunkedArray<Self::Dtype>, _seq_id: u64) {
        *v += ca.sum().unwrap_or(0) as IdxSize;
    }

    fn finish(
        &self,
        v: Vec<Self::Value>,
        m: Option<Bitmap>,
        dtype: &DataType,
    ) -> PolarsResult<Series> {
        assert!(m.is_none());
        assert!(dtype == &DataType::Boolean);
        Ok(IdxCa::from_vec(PlSmallStr::EMPTY, v).into_series())
    }
}
