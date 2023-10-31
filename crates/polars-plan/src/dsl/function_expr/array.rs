use polars_ops::chunked_array::array::*;

use super::*;
use crate::map;

#[derive(Clone, Copy, Eq, PartialEq, Hash, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ArrayFunction {
    Min,
    Max,
    Sum,
    ToList,
    Unique(bool),
}

impl ArrayFunction {
    pub(super) fn get_field(&self, mapper: FieldsMapper) -> PolarsResult<Field> {
        use ArrayFunction::*;
        match self {
            Min | Max => mapper.map_to_list_and_array_inner_dtype(),
            Sum => mapper.nested_sum_type(),
            ToList => mapper.try_map_dtype(map_array_dtype_to_list_dtype),
            Unique(_) => mapper.try_map_dtype(map_array_dtype_to_list_dtype),
        }
    }
}

fn map_array_dtype_to_list_dtype(datatype: &DataType) -> PolarsResult<DataType> {
    if let DataType::Array(inner, _) = datatype {
        Ok(DataType::List(inner.clone()))
    } else {
        polars_bail!(ComputeError: "expected array dtype")
    }
}

impl Display for ArrayFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use ArrayFunction::*;
        let name = match self {
            Min => "min",
            Max => "max",
            Sum => "sum",
            ToList => "to_list",
            Unique(_) => "unique",
        };
        write!(f, "arr.{name}")
    }
}

impl From<ArrayFunction> for SpecialEq<Arc<dyn SeriesUdf>> {
    fn from(func: ArrayFunction) -> Self {
        use ArrayFunction::*;
        match func {
            Min => map!(min),
            Max => map!(max),
            Sum => map!(sum),
            ToList => map!(to_list),
            Unique(stable) => map!(unique, stable),
        }
    }
}

pub(super) fn max(s: &Series) -> PolarsResult<Series> {
    Ok(s.array()?.array_max())
}

pub(super) fn min(s: &Series) -> PolarsResult<Series> {
    Ok(s.array()?.array_min())
}

pub(super) fn sum(s: &Series) -> PolarsResult<Series> {
    s.array()?.array_sum()
}

pub(super) fn unique(s: &Series, stable: bool) -> PolarsResult<Series> {
    let ca = s.array()?;
    let out = if stable {
        ca.array_unique_stable()
    } else {
        ca.array_unique()
    };
    out.map(|ca| ca.into_series())
}

pub(super) fn to_list(s: &Series) -> PolarsResult<Series> {
    let list_dtype = map_array_dtype_to_list_dtype(s.dtype())?;
    s.cast(&list_dtype)
}
