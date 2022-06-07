#[cfg(feature = "is_in")]
mod is_in;
mod pow;

use super::*;
use polars_core::prelude::*;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, PartialEq, Debug)]
pub enum FunctionExpr {
    NullCount,
    Pow,
    #[cfg(feature = "row_hash")]
    Hash(usize),
    #[cfg(feature = "is_in")]
    IsIn,
}

impl FunctionExpr {
    pub(crate) fn get_field(
        &self,
        _input_schema: &Schema,
        _cntxt: Context,
        fields: &[Field],
    ) -> Result<Field> {
        let with_dtype = |dtype: DataType| Ok(Field::new(fields[0].name(), dtype));
        let map_dtype = |func: &dyn Fn(&DataType) -> DataType| {
            let dtype = func(fields[0].data_type());
            Ok(Field::new(fields[0].name(), dtype))
        };

        let float_dtype = || {
            map_dtype(&|dtype| match dtype {
                DataType::Float32 => DataType::Float32,
                _ => DataType::Float64,
            })
        };

        use FunctionExpr::*;
        match self {
            NullCount => with_dtype(IDX_DTYPE),
            Pow => float_dtype(),
            #[cfg(feature = "row_hash")]
            Hash(_) => with_dtype(DataType::UInt64),
            #[cfg(feature = "is_in")]
            IsIn => with_dtype(DataType::Boolean),
        }
    }
}

macro_rules! wrap {
    ($e:expr) => {
        NoEq::new(Arc::new($e))
    };
}

impl From<FunctionExpr> for NoEq<Arc<dyn SeriesUdf>> {
    fn from(func: FunctionExpr) -> Self {
        use FunctionExpr::*;
        match func {
            NullCount => {
                let f = |s: &mut [Series]| {
                    let s = &s[0];
                    Ok(Series::new(s.name(), [s.null_count() as IdxSize]))
                };
                wrap!(f)
            }
            Pow => {
                wrap!(pow::pow)
            }
            #[cfg(feature = "row_hash")]
            Hash(seed) => {
                let f = move |s: &mut [Series]| {
                    let s = &s[0];
                    Ok(s.hash(ahash::RandomState::with_seed(seed)).into_series())
                };
                wrap!(f)
            }
            #[cfg(feature = "is_in")]
            IsIn => {
                wrap!(is_in::is_in)
            }
        }
    }
}
