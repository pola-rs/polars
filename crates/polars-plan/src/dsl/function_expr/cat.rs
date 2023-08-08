use super::*;
use crate::{map, map_owned};

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, PartialEq, Debug, Eq, Hash)]
pub enum CategoricalFunction {
    SetOrdering { lexical: bool },
    GetCategories,
    ToLocal,
}

impl CategoricalFunction {
    pub(super) fn get_field(&self, mapper: FieldsMapper) -> PolarsResult<Field> {
        use CategoricalFunction::*;
        match self {
            SetOrdering { .. } => mapper.with_same_dtype(),
            GetCategories => mapper.with_dtype(DataType::Utf8),
            ToLocal => mapper.with_same_dtype(), // TODO: Update revmap?
        }
    }
}

impl Display for CategoricalFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use CategoricalFunction::*;
        let s = match self {
            SetOrdering { .. } => "set_ordering",
            GetCategories => "get_categories",
            ToLocal => "to_local",
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
            ToLocal => map_owned!(to_local),
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
    ca.set_lexical_ordering(lexical);
    Ok(ca.into_series())
}

fn get_categories(s: &Series) -> PolarsResult<Series> {
    // categorical check
    let ca = s.categorical()?;
    let rev_map = ca.get_rev_map();
    let arr = rev_map.get_categories().clone().boxed();
    Series::try_from((ca.name(), arr))
}

fn to_local(s: Series) -> PolarsResult<Series> {
    let ca = s.categorical()?;
    let rev_map = ca.get_rev_map();

    let (physical_map, categories) = match rev_map.get_physical_map_and_categories() {
        Some(v) => v,
        // The categorical is already local
        None => return Ok(s),
    };
    let local_ca = ca.logical().apply(|v| *physical_map.get(&v).unwrap());
    let local_rev_map = RevMapping::Local(categories.clone());

    let mut out = unsafe {
        CategoricalChunked::from_cats_and_rev_map_unchecked(local_ca, local_rev_map.into())
    };
    // TODO: Set ORIGINAL flag here?
    out.set_lexical_ordering(ca.uses_lexical_ordering());

    Ok(out.into_series())
}
