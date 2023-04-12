use super::*;
use crate::map;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, PartialEq, Debug, Eq, Hash)]
pub enum CategoricalFunction {
    SetOrdering(CategoricalOrdering),
}

impl CategoricalFunction {
    pub(super) fn dtype_out(&self) -> DataType {
        DataType::Categorical(None)
    }
}

impl Display for CategoricalFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use CategoricalFunction::*;
        let s = match self {
            SetOrdering(..) => "set_ordering",
        };
        write!(f, "{s}")
    }
}

impl From<CategoricalFunction> for SpecialEq<Arc<dyn SeriesUdf>> {
    fn from(func: CategoricalFunction) -> Self {
        use CategoricalFunction::*;
        match func {
            SetOrdering(ordering) => map!(set_ordering, ordering),
        }
    }
}

impl From<CategoricalFunction> for FunctionExpr {
    fn from(func: CategoricalFunction) -> Self {
        FunctionExpr::Categorical(func)
    }
}

fn set_ordering(s: &Series, ordering: CategoricalOrdering) -> PolarsResult<Series> {
    let mut ca = s.categorical()?.clone();
    let set_lexical = match ordering {
        CategoricalOrdering::Lexical => true,
        CategoricalOrdering::Physical => false,
    };
    ca.set_lexical_sorted(set_lexical);
    Ok(ca.into_series())
}
