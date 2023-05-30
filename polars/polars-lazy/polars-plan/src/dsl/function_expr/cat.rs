use super::*;
use crate::map;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, PartialEq, Debug, Eq, Hash)]
pub enum CategoricalFunction {
    SetOrdering { lexical: bool },
}

impl CategoricalFunction {
    pub(super) fn get_field(&self, mapper: FieldsMapper) -> PolarsResult<Field> {
        mapper.with_dtype(DataType::Boolean)
    }
}

impl Display for CategoricalFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use CategoricalFunction::*;
        let s = match self {
            SetOrdering { .. } => "set_ordering",
        };
        write!(f, "{s}")
    }
}

impl From<CategoricalFunction> for SpecialEq<Arc<dyn SeriesUdf>> {
    fn from(func: CategoricalFunction) -> Self {
        use CategoricalFunction::*;
        match func {
            SetOrdering { lexical } => map!(set_ordering, lexical),
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
