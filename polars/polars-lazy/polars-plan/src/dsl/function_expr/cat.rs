use super::*;
use crate::map;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, PartialEq, Debug, Eq, Hash)]
pub enum CategoricalFunction {
    SetOrdering { lexical: bool },
    GetCategories,
}

impl CategoricalFunction {
    pub(super) fn get_field(&self, mapper: FieldsMapper) -> PolarsResult<Field> {
        use CategoricalFunction::*;
        match self {
            SetOrdering { .. } => mapper.with_same_dtype(),
            GetCategories => mapper.with_dtype(DataType::Utf8),
        }
    }
}

impl Display for CategoricalFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use CategoricalFunction::*;
        let s = match self {
            SetOrdering { .. } => "set_ordering",
            GetCategories => "get_categories",
        };
        write!(f, "{s}")
    }
}

impl From<CategoricalFunction> for SpecialEq<Arc<dyn SeriesUdf>> {
    fn from(func: CategoricalFunction) -> Self {
        use CategoricalFunction::*;
        match func {
            SetOrdering { lexical } => map!(set_ordering, lexical),
            GetCategories => map!(get_categories),
        }
    }
}

impl From<CategoricalFunction> for FunctionExpr {
    fn from(func: CategoricalFunction) -> Self {
        FunctionExpr::Categorical(func)
    }
}

fn set_ordering(s: &Series, lexical: bool) -> PolarsResult<Series> {
    let mut ca = s.categorical()?.clone();
    ca.set_lexical_sorted(lexical);
    Ok(ca.into_series())
}

fn get_categories(s: &Series) -> PolarsResult<Series> {
    // categorical check
    let ca = s.categorical()?;
    let DataType::Categorical(Some(rev_map)) = ca.dtype() else { unreachable!()  };
    let arr = rev_map.get_categories().clone().boxed();
    Series::try_from((ca.name(), arr))
}
