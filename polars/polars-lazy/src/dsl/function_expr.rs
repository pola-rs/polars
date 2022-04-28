use super::*;
use polars_core::prelude::*;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, PartialEq, Debug)]
pub enum FunctionExpr {
    NullCount,
}

impl FunctionExpr {
    pub(crate) fn get_field(
        &self,
        _input_schema: &Schema,
        _cntxt: Context,
        fields: &[Field],
    ) -> Result<Field> {
        use FunctionExpr::*;
        match self {
            NullCount => Ok(Field::new(fields[0].name(), IDX_DTYPE)),
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
        }
    }
}
