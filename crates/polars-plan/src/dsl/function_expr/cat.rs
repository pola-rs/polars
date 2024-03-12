use super::*;
use crate::map;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, PartialEq, Debug, Eq, Hash)]
pub enum CategoricalFunction {
    GetCategories,
    ToEnum,
}

impl CategoricalFunction {
    pub(super) fn get_field(&self, mapper: FieldsMapper) -> PolarsResult<Field> {
        use CategoricalFunction::*;
        match self {
            GetCategories => mapper.with_dtype(DataType::String),
            ToEnum => mapper.map_cat_to_enum_dtype(),
        }
    }
}

impl Display for CategoricalFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use CategoricalFunction::*;
        let s = match self {
            GetCategories => "get_categories",
            ToEnum => "to_enum",
        };
        write!(f, "cat.{s}")
    }
}

impl From<CategoricalFunction> for SpecialEq<Arc<dyn SeriesUdf>> {
    fn from(func: CategoricalFunction) -> Self {
        use CategoricalFunction::*;
        match func {
            GetCategories => map!(get_categories),
            ToEnum => map!(to_enum),
        }
    }
}

impl From<CategoricalFunction> for FunctionExpr {
    fn from(func: CategoricalFunction) -> Self {
        FunctionExpr::Categorical(func)
    }
}

fn get_categories(s: &Series) -> PolarsResult<Series> {
    // categorical check
    let ca = s.categorical()?;
    let rev_map = ca.get_rev_map();
    let arr = rev_map.get_categories().clone().boxed();
    Series::try_from((ca.name(), arr))
}

fn to_enum(s: &Series) -> PolarsResult<Series> {
    let ca = s.categorical()?.to_local();
    Ok(ca.convert_to_enum().into_series())
}
