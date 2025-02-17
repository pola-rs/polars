use std::marker::PhantomData;

use num_traits::AsPrimitive;
use polars_compute::var_cov::VarState;
use polars_core::with_match_physical_numeric_polars_type;

use super::*;

pub fn new_var_std_reduction(dtype: DataType, is_std: bool, ddof: u8) -> Box<dyn GroupedReduction> {
    use DataType::*;
    use VecGroupedReduction as VGR;
    match dtype {
        Boolean => Box::new(VGR::new(dtype, BoolVarStdReducer { is_std, ddof })),
        _ if dtype.is_primitive_numeric() => {
            with_match_physical_numeric_polars_type!(dtype.to_physical(), |$T| {
                Box::new(VGR::new(dtype, VarStdReducer::<$T> {
                    is_std,
                    ddof,
                    needs_cast: false,
                    _phantom: PhantomData,
                }))
            })
        },
        #[cfg(feature = "dtype-decimal")]
        Decimal(_, _) => Box::new(VGR::new(
            dtype,
            VarStdReducer::<Float64Type> {
                is_std,
                ddof,
                needs_cast: true,
                _phantom: PhantomData,
            },
        )),
        Duration(..) => todo!(),
        _ => unimplemented!(),
    }
}

struct VarStdReducer<T> {
    is_std: bool,
    ddof: u8,
    needs_cast: bool,
    _phantom: PhantomData<T>,
}

impl<T> Clone for VarStdReducer<T> {
    fn clone(&self) -> Self {
        Self {
            is_std: self.is_std,
            ddof: self.ddof,
            needs_cast: self.needs_cast,
            _phantom: PhantomData,
        }
    }
}

impl<T: PolarsNumericType> Reducer for VarStdReducer<T> {
    type Dtype = T;
    type Value = VarState;

    fn init(&self) -> Self::Value {
        VarState::default()
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
            a.add_one(x.as_());
        }
    }

    fn reduce_ca(&self, v: &mut Self::Value, ca: &ChunkedArray<Self::Dtype>, _seq_id: u64) {
        for arr in ca.downcast_iter() {
            v.combine(&polars_compute::var_cov::var(arr))
        }
    }

    fn finish(
        &self,
        v: Vec<Self::Value>,
        m: Option<Bitmap>,
        _dtype: &DataType,
    ) -> PolarsResult<Series> {
        assert!(m.is_none());
        let ca: Float64Chunked = v
            .into_iter()
            .map(|s| {
                let var = s.finalize(self.ddof);
                if self.is_std {
                    var.map(f64::sqrt)
                } else {
                    var
                }
            })
            .collect_ca(PlSmallStr::EMPTY);
        Ok(ca.into_series())
    }
}

#[derive(Clone)]
struct BoolVarStdReducer {
    is_std: bool,
    ddof: u8,
}

impl Reducer for BoolVarStdReducer {
    type Dtype = BooleanType;
    type Value = (usize, usize);

    fn init(&self) -> Self::Value {
        (0, 0)
    }

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
        _dtype: &DataType,
    ) -> PolarsResult<Series> {
        assert!(m.is_none());
        let ca: Float64Chunked = v
            .into_iter()
            .map(|v| {
                if v.1 <= self.ddof as usize {
                    return None;
                }

                let sum = v.0 as f64; // Both the sum and sum-of-squares, letting us simplify.
                let n = v.1;
                let var = sum * (1.0 - sum / n as f64) / ((n - self.ddof as usize) as f64);
                if self.is_std {
                    Some(var.sqrt())
                } else {
                    Some(var)
                }
            })
            .collect_ca(PlSmallStr::EMPTY);
        Ok(ca.into_series())
    }
}
