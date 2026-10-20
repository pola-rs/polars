use std::marker::PhantomData;

use num_traits::{AsPrimitive, Zero};
use polars_arrow::temporal_conversions::MICROSECONDS_IN_DAY;
use polars_core::with_match_physical_numeric_polars_type;

use super::split::{SplitStage, merge_reduction, null_reduction, split_reduction};
use super::var_std::BoolCountState;
use super::*;

pub fn new_mean_reduction(dtype: DataType) -> PolarsResult<Box<dyn GroupedReduction>> {
    mean_reduction(dtype, SplitStage::Whole)
}

/// Like [`new_mean_reduction`], but outputs the state of each group, to be merged by
/// [`new_mean_merge_reduction`].
#[cfg(feature = "serde")]
pub fn new_mean_state_reduction(dtype: DataType) -> PolarsResult<Box<dyn GroupedReduction>> {
    mean_reduction(dtype, SplitStage::State)
}

/// Merges the states of [`new_mean_state_reduction`] over values of `values_dtype`, and outputs
/// what [`new_mean_reduction`] does.
#[cfg(feature = "serde")]
pub fn new_mean_merge_reduction(values_dtype: DataType) -> PolarsResult<Box<dyn GroupedReduction>> {
    mean_reduction(values_dtype, SplitStage::Merge)
}

fn mean_reduction(dtype: DataType, stage: SplitStage) -> PolarsResult<Box<dyn GroupedReduction>> {
    // TODO: Move the error checks up and make this function infallible
    use DataType::*;
    let is_merge = matches!(stage, SplitStage::Merge);
    Ok(match dtype {
        Boolean if is_merge => merge_reduction(dtype, BoolMeanReducer),
        Boolean => split_reduction(dtype, BoolMeanReducer, stage),
        // `finish` depends only on `values_dtype`, so one reducer merges the states of every
        // numeric input.
        _ if is_merge && (dtype.is_primitive_numeric() || dtype.is_temporal()) => {
            merge_reduction(dtype, NumMeanReducer::<Float64Type>(PhantomData))
        },
        _ if dtype.is_primitive_numeric() || dtype.is_temporal() => {
            with_match_physical_numeric_polars_type!(dtype.to_physical(), |$T| {
                split_reduction(dtype, NumMeanReducer::<$T>(PhantomData), stage)
            })
        },
        #[cfg(feature = "dtype-decimal")]
        Decimal(_, _) if is_merge => merge_reduction(dtype, DecimalMeanReducer),
        #[cfg(feature = "dtype-decimal")]
        Decimal(_, _) => split_reduction(dtype, DecimalMeanReducer, stage),
        Null => null_reduction(stage, Scalar::null(Null)),
        _ => polars_bail!(InvalidOperation: "`mean` operation not supported for dtype `{dtype}`"),
    })
}

#[derive(Clone, Default)]
pub(super) struct MeanState {
    pub(super) sum: f64,
    pub(super) count: usize,
}

fn finish_output(values: Vec<MeanState>, dtype: &DataType) -> Series {
    let values = values.into_iter().map(|s| (s.sum, s.count));
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
    type Value = MeanState;
    const ORDER_INDEPENDENT: bool = true;

    #[inline(always)]
    fn init(&self) -> Self::Value {
        MeanState::default()
    }

    fn cast_series<'a>(&self, s: &'a Series) -> Cow<'a, Series> {
        s.to_physical_repr()
    }

    #[inline(always)]
    fn combine(&self, a: &mut Self::Value, b: &Self::Value) {
        a.sum += b.sum;
        a.count += b.count;
    }

    #[inline(always)]
    fn reduce_one(&self, a: &mut Self::Value, b: Option<T::Native>, _seq_id: u64) {
        a.sum += b.unwrap_or(T::Native::zero()).as_();
        a.count += b.is_some() as usize;
    }

    fn reduce_ca(&self, v: &mut Self::Value, ca: &ChunkedArray<Self::Dtype>, _seq_id: u64) {
        v.sum += ChunkAgg::_sum_as_f64(ca);
        v.count += ca.len() - ca.null_count();
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
#[derive(Clone, Default)]
pub(super) struct DecimalMeanState {
    pub(super) sum: i128,
    pub(super) spilled: f64,
    pub(super) count: usize,
}

#[cfg(feature = "dtype-decimal")]
#[inline(always)]
fn add_to_sum(a: &mut DecimalMeanState, x: i128) {
    match a.sum.checked_add(x) {
        Some(v) => a.sum = v,
        None => {
            a.spilled += a.sum as f64;
            a.sum = x;
        },
    }
}

#[cfg(feature = "dtype-decimal")]
impl Reducer for DecimalMeanReducer {
    type Dtype = Int128Type;
    type Value = DecimalMeanState;
    const ORDER_INDEPENDENT: bool = true;

    #[inline(always)]
    fn init(&self) -> Self::Value {
        DecimalMeanState::default()
    }

    fn cast_series<'a>(&self, s: &'a Series) -> Cow<'a, Series> {
        s.to_physical_repr()
    }

    #[inline(always)]
    fn combine(&self, a: &mut Self::Value, b: &Self::Value) {
        add_to_sum(a, b.sum);
        a.spilled += b.spilled;
        a.count += b.count;
    }

    #[inline(always)]
    fn reduce_one(&self, a: &mut Self::Value, b: Option<i128>, _seq_id: u64) {
        add_to_sum(a, b.unwrap_or(0));
        a.count += b.is_some() as usize;
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
            .map(|s| {
                (s.count != 0).then(|| (s.spilled + s.sum as f64) / s.count as f64 / scale_factor)
            })
            .collect_ca(PlSmallStr::EMPTY);
        Ok(ca.into_series())
    }
}

#[derive(Clone)]
struct BoolMeanReducer;

impl Reducer for BoolMeanReducer {
    type Dtype = BooleanType;
    type Value = BoolCountState;

    #[inline(always)]
    fn init(&self) -> Self::Value {
        BoolCountState::default()
    }

    #[inline(always)]
    fn combine(&self, a: &mut Self::Value, b: &Self::Value) {
        a.true_count += b.true_count;
        a.count += b.count;
    }

    #[inline(always)]
    fn reduce_one(&self, a: &mut Self::Value, b: Option<bool>, _seq_id: u64) {
        a.true_count += b.unwrap_or(false) as usize;
        a.count += b.is_some() as usize;
    }

    fn reduce_ca(&self, v: &mut Self::Value, ca: &ChunkedArray<Self::Dtype>, _seq_id: u64) {
        v.true_count += ca.sum().unwrap_or(0) as usize;
        v.count += ca.len() - ca.null_count();
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
            .map(|s| (s.count != 0).then(|| s.true_count as f64 / s.count as f64))
            .collect_ca(PlSmallStr::EMPTY);
        Ok(ca.into_series())
    }
}
