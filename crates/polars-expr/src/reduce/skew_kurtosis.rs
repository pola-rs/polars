use std::marker::PhantomData;

use num_traits::AsPrimitive;
use polars_compute::moment::{KurtosisState, SkewState};
use polars_core::with_match_physical_numeric_polars_type;

use super::split::{SplitStage, split_reduction};
use super::*;

pub fn new_skew_reduction(dtype: DataType, bias: bool) -> PolarsResult<Box<dyn GroupedReduction>> {
    skew_reduction(dtype, bias, SplitStage::Whole)
}

/// Like [`new_skew_reduction`], but outputs the serialized state of each group, to be merged by
/// [`new_skew_merge_reduction`].
#[cfg(feature = "serde")]
pub fn new_skew_state_reduction(dtype: DataType) -> PolarsResult<Box<dyn GroupedReduction>> {
    // `bias` only matters when finalizing.
    skew_reduction(dtype, false, SplitStage::State)
}

/// Merges the states of [`new_skew_state_reduction`] over values of `values_dtype`, and outputs
/// what [`new_skew_reduction`] does.
#[cfg(feature = "serde")]
pub fn new_skew_merge_reduction(
    values_dtype: DataType,
    bias: bool,
) -> PolarsResult<Box<dyn GroupedReduction>> {
    skew_reduction(values_dtype, bias, SplitStage::Merge)
}

fn skew_reduction(
    dtype: DataType,
    bias: bool,
    stage: SplitStage,
) -> PolarsResult<Box<dyn GroupedReduction>> {
    use DataType::*;
    Ok(match dtype {
        // `finish` ignores the input type, so one reducer merges the states of every numeric input.
        #[cfg(feature = "serde")]
        _ if matches!(stage, SplitStage::Merge)
            && (dtype.is_primitive_numeric() || dtype.is_decimal()) =>
        {
            split_reduction(
                dtype,
                SkewReducer::<Float64Type> {
                    bias,
                    needs_cast: false,
                    _phantom: PhantomData,
                },
                stage,
            )
        },
        _ if dtype.is_primitive_numeric() => {
            with_match_physical_numeric_polars_type!(dtype.to_physical(), |$T| {
                split_reduction(dtype, SkewReducer::<$T> {
                    bias,
                    needs_cast: false,
                    _phantom: PhantomData,
                }, stage)
            })
        },
        #[cfg(feature = "dtype-decimal")]
        Decimal(_, _) => split_reduction(
            dtype,
            SkewReducer::<Float64Type> {
                bias,
                needs_cast: true,
                _phantom: PhantomData,
            },
            stage,
        ),
        Null => Box::new(super::NullGroupedReduction::new(Scalar::null(
            DataType::Null,
        ))),
        _ => {
            polars_bail!(InvalidOperation: "`skew` operation not supported for dtype `{dtype}`")
        },
    })
}

pub fn new_kurtosis_reduction(
    dtype: DataType,
    fisher: bool,
    bias: bool,
) -> PolarsResult<Box<dyn GroupedReduction>> {
    kurtosis_reduction(dtype, fisher, bias, SplitStage::Whole)
}

/// Like [`new_kurtosis_reduction`], but outputs the serialized state of each group, to be merged
/// by [`new_kurtosis_merge_reduction`].
#[cfg(feature = "serde")]
pub fn new_kurtosis_state_reduction(dtype: DataType) -> PolarsResult<Box<dyn GroupedReduction>> {
    // `fisher` and `bias` only matter when finalizing.
    kurtosis_reduction(dtype, false, false, SplitStage::State)
}

/// Merges the states of [`new_kurtosis_state_reduction`] over values of `values_dtype`, and
/// outputs what [`new_kurtosis_reduction`] does.
#[cfg(feature = "serde")]
pub fn new_kurtosis_merge_reduction(
    values_dtype: DataType,
    fisher: bool,
    bias: bool,
) -> PolarsResult<Box<dyn GroupedReduction>> {
    kurtosis_reduction(values_dtype, fisher, bias, SplitStage::Merge)
}

fn kurtosis_reduction(
    dtype: DataType,
    fisher: bool,
    bias: bool,
    stage: SplitStage,
) -> PolarsResult<Box<dyn GroupedReduction>> {
    use DataType::*;
    Ok(match dtype {
        // `finish` ignores the input type, so one reducer merges the states of every numeric input.
        #[cfg(feature = "serde")]
        _ if matches!(stage, SplitStage::Merge)
            && (dtype.is_primitive_numeric() || dtype.is_decimal()) =>
        {
            split_reduction(
                dtype,
                KurtosisReducer::<Float64Type> {
                    fisher,
                    bias,
                    needs_cast: false,
                    _phantom: PhantomData,
                },
                stage,
            )
        },
        _ if dtype.is_primitive_numeric() => {
            with_match_physical_numeric_polars_type!(dtype.to_physical(), |$T| {
                split_reduction(dtype, KurtosisReducer::<$T> {
                    fisher,
                    bias,
                    needs_cast: false,
                    _phantom: PhantomData,
                }, stage)
            })
        },
        #[cfg(feature = "dtype-decimal")]
        Decimal(_, _) => split_reduction(
            dtype,
            KurtosisReducer::<Float64Type> {
                fisher,
                bias,
                needs_cast: true,
                _phantom: PhantomData,
            },
            stage,
        ),
        Null => Box::new(super::NullGroupedReduction::new(Scalar::null(
            DataType::Null,
        ))),
        _ => {
            polars_bail!(InvalidOperation: "`kurtosis` operation not supported for dtype `{dtype}`")
        },
    })
}

struct SkewReducer<T> {
    bias: bool,
    needs_cast: bool,
    _phantom: PhantomData<T>,
}

impl<T> Clone for SkewReducer<T> {
    fn clone(&self) -> Self {
        Self {
            bias: self.bias,
            needs_cast: self.needs_cast,
            _phantom: PhantomData,
        }
    }
}

impl<T: PolarsNumericType> Reducer for SkewReducer<T> {
    type Dtype = T;
    type Value = SkewState;

    fn init(&self) -> Self::Value {
        SkewState::default()
    }

    fn cast_series<'a>(&self, s: &'a Series) -> Cow<'a, Series> {
        if self.needs_cast {
            Cow::Owned(s.cast(&DataType::Float64).unwrap())
        } else {
            Cow::Borrowed(s)
        }
    }

    fn combine(&self, a: &mut Self::Value, b: &Self::Value) {
        a.combine(b)
    }

    #[inline(always)]
    fn reduce_one(&self, a: &mut Self::Value, b: Option<T::Native>, _seq_id: u64) {
        if let Some(x) = b {
            a.insert_one(x.as_());
        }
    }

    fn reduce_ca(&self, v: &mut Self::Value, ca: &ChunkedArray<Self::Dtype>, _seq_id: u64) {
        for arr in ca.downcast_iter() {
            v.combine(&polars_compute::moment::skew(arr))
        }
    }

    fn finish(
        &self,
        v: Vec<Self::Value>,
        m: Option<Bitmap>,
        _dtype: &DataType,
    ) -> PolarsResult<Series> {
        assert!(m.is_none());
        let bias = self.bias;
        let ca: Float64Chunked = v
            .into_iter()
            .map(|s| s.finalize(bias))
            .collect_ca(PlSmallStr::EMPTY);
        Ok(ca.into_series())
    }
}

struct KurtosisReducer<T> {
    fisher: bool,
    bias: bool,
    needs_cast: bool,
    _phantom: PhantomData<T>,
}

impl<T> Clone for KurtosisReducer<T> {
    fn clone(&self) -> Self {
        Self {
            fisher: self.fisher,
            bias: self.bias,
            needs_cast: self.needs_cast,
            _phantom: PhantomData,
        }
    }
}

impl<T: PolarsNumericType> Reducer for KurtosisReducer<T> {
    type Dtype = T;
    type Value = KurtosisState;

    fn init(&self) -> Self::Value {
        KurtosisState::default()
    }

    fn cast_series<'a>(&self, s: &'a Series) -> Cow<'a, Series> {
        if self.needs_cast {
            Cow::Owned(s.cast(&DataType::Float64).unwrap())
        } else {
            Cow::Borrowed(s)
        }
    }

    fn combine(&self, a: &mut Self::Value, b: &Self::Value) {
        a.combine(b)
    }

    #[inline(always)]
    fn reduce_one(&self, a: &mut Self::Value, b: Option<T::Native>, _seq_id: u64) {
        if let Some(x) = b {
            a.insert_one(x.as_());
        }
    }

    fn reduce_ca(&self, v: &mut Self::Value, ca: &ChunkedArray<Self::Dtype>, _seq_id: u64) {
        for arr in ca.downcast_iter() {
            v.combine(&polars_compute::moment::kurtosis(arr))
        }
    }

    fn finish(
        &self,
        v: Vec<Self::Value>,
        m: Option<Bitmap>,
        _dtype: &DataType,
    ) -> PolarsResult<Series> {
        assert!(m.is_none());
        let (fisher, bias) = (self.fisher, self.bias);
        let ca: Float64Chunked = v
            .into_iter()
            .map(|s| s.finalize(fisher, bias))
            .collect_ca(PlSmallStr::EMPTY);
        Ok(ca.into_series())
    }
}
