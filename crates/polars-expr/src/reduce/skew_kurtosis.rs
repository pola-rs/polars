use std::marker::PhantomData;

use num_traits::AsPrimitive;
use polars_compute::moment::{KurtosisState, SkewState};
use polars_core::with_match_physical_numeric_polars_type;

use super::*;

pub fn new_skew_reduction(dtype: DataType, bias: bool) -> PolarsResult<Box<dyn GroupedReduction>> {
    use DataType::*;
    use VecGroupedReduction as VGR;
    Ok(match dtype {
        _ if dtype.is_primitive_numeric() => {
            with_match_physical_numeric_polars_type!(dtype.to_physical(), |$T| {
                Box::new(VGR::new(dtype, SkewReducer::<$T> {
                    bias,
                    needs_cast: false,
                    _phantom: PhantomData,
                }))
            })
        },
        #[cfg(feature = "dtype-decimal")]
        Decimal(_, _) => Box::new(VGR::new(
            dtype,
            SkewReducer::<Float64Type> {
                bias,
                needs_cast: true,
                _phantom: PhantomData,
            },
        )),
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
    use DataType::*;
    use VecGroupedReduction as VGR;
    Ok(match dtype {
        _ if dtype.is_primitive_numeric() => {
            with_match_physical_numeric_polars_type!(dtype.to_physical(), |$T| {
                Box::new(VGR::new(dtype, KurtosisReducer::<$T> {
                    fisher,
                    bias,
                    needs_cast: false,
                    _phantom: PhantomData,
                }))
            })
        },
        #[cfg(feature = "dtype-decimal")]
        Decimal(_, _) => Box::new(VGR::new(
            dtype,
            KurtosisReducer::<Float64Type> {
                fisher,
                bias,
                needs_cast: true,
                _phantom: PhantomData,
            },
        )),
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
