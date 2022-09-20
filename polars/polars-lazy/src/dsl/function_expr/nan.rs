#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::*;
use crate::{map, map_owned};

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, PartialEq, Debug, Eq, Hash)]
pub enum NanFunction {
    IsNan,
    IsNotNan,
    DropNans,
}

pub(super) fn is_nan(s: &Series) -> PolarsResult<Series> {
    s.is_nan().map(|ca| ca.into_series())
}

pub(super) fn is_not_nan(s: &Series) -> PolarsResult<Series> {
    s.is_not_nan().map(|ca| ca.into_series())
}

pub(super) fn drop_nans(s: Series) -> PolarsResult<Series> {
    match s.dtype() {
        DataType::Float32 => {
            let ca = s.f32()?;
            let mask = ca.is_not_nan();
            ca.filter(&mask).map(|ca| ca.into_series())
        }
        DataType::Float64 => {
            let ca = s.f64()?;
            let mask = ca.is_not_nan();
            ca.filter(&mask).map(|ca| ca.into_series())
        }
        _ => Ok(s),
    }
}

impl NanFunction {
    pub(crate) fn get_field(&self, fields: &[Field]) -> PolarsResult<Field> {
        let with_dtype = |dtype: DataType| Ok(Field::new(fields[0].name(), dtype));
        let map_dtype = |func: &dyn Fn(&DataType) -> DataType| {
            let dtype = func(fields[0].data_type());
            Ok(Field::new(fields[0].name(), dtype))
        };
        let same_type = || map_dtype(&|dtype| dtype.clone());

        match self {
            NanFunction::IsNan => with_dtype(DataType::Boolean),
            NanFunction::IsNotNan => with_dtype(DataType::Boolean),
            NanFunction::DropNans => same_type(),
        }
    }
}

impl From<NanFunction> for SpecialEq<Arc<dyn SeriesUdf>> {
    fn from(nan_function: NanFunction) -> Self {
        match nan_function {
            NanFunction::IsNan => map!(is_nan),
            NanFunction::IsNotNan => map!(is_not_nan),
            NanFunction::DropNans => map_owned!(drop_nans),
        }
    }
}

impl From<NanFunction> for FunctionExpr {
    fn from(nan_function: NanFunction) -> Self {
        FunctionExpr::Nan(nan_function)
    }
}
