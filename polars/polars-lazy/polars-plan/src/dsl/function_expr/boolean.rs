use std::ops::Not;

use super::*;
use crate::map;
#[cfg(feature = "is_in")]
use crate::wrap;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, PartialEq, Debug, Eq, Hash)]
pub enum BooleanFunction {
    All,
    Any,
    IsNot,
    IsNull,
    IsNotNull,
    IsFinite,
    IsInfinite,
    IsNan,
    IsNotNan,
    #[cfg(feature = "is_first")]
    IsFirst,
    #[cfg(feature = "is_unique")]
    IsUnique,
    #[cfg(feature = "is_unique")]
    IsDuplicated,
    #[cfg(feature = "is_in")]
    IsIn,
}

impl BooleanFunction {
    pub(super) fn get_field(&self, mapper: FieldsMapper) -> PolarsResult<Field> {
        mapper.with_dtype(DataType::Boolean)
    }
}

impl Display for BooleanFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use BooleanFunction::*;
        let s = match self {
            All => "all",
            Any => "any",
            IsNot => "is_not",
            IsNull => "is_null",
            IsNotNull => "is_not_null",
            IsFinite => "is_finite",
            IsInfinite => "is_infinite",
            IsNan => "is_nan",
            IsNotNan => "is_not_nan",
            #[cfg(feature = "is_first")]
            IsFirst => "is_first",
            #[cfg(feature = "is_unique")]
            IsUnique => "is_unique",
            #[cfg(feature = "is_unique")]
            IsDuplicated => "is_duplicated",
            #[cfg(feature = "is_in")]
            IsIn => "is_in",
        };
        write!(f, "{s}")
    }
}

impl From<BooleanFunction> for SpecialEq<Arc<dyn SeriesUdf>> {
    fn from(func: BooleanFunction) -> Self {
        use BooleanFunction::*;
        match func {
            All => map!(all),
            Any => map!(any),
            IsNot => map!(is_not),
            IsNull => map!(is_null),
            IsNotNull => map!(is_not_null),
            IsFinite => map!(is_finite),
            IsInfinite => map!(is_infinite),
            IsNan => map!(is_nan),
            IsNotNan => map!(is_not_nan),
            #[cfg(feature = "is_first")]
            IsFirst => map!(is_first),
            #[cfg(feature = "is_unique")]
            IsUnique => map!(is_unique),
            #[cfg(feature = "is_unique")]
            IsDuplicated => map!(is_duplicated),
            #[cfg(feature = "is_in")]
            IsIn => wrap!(is_in),
        }
    }
}

impl From<BooleanFunction> for FunctionExpr {
    fn from(func: BooleanFunction) -> Self {
        FunctionExpr::Boolean(func)
    }
}

fn all(s: &Series) -> PolarsResult<Series> {
    let boolean = s.bool()?;
    if boolean.all() {
        Ok(Series::new(s.name(), [true]))
    } else {
        Ok(Series::new(s.name(), [false]))
    }
}

fn any(s: &Series) -> PolarsResult<Series> {
    let boolean = s.bool()?;
    if boolean.any() {
        Ok(Series::new(s.name(), [true]))
    } else {
        Ok(Series::new(s.name(), [false]))
    }
}

fn is_not(s: &Series) -> PolarsResult<Series> {
    Ok(s.bool()?.not().into_series())
}

fn is_null(s: &Series) -> PolarsResult<Series> {
    Ok(s.is_null().into_series())
}

fn is_not_null(s: &Series) -> PolarsResult<Series> {
    Ok(s.is_not_null().into_series())
}

fn is_finite(s: &Series) -> PolarsResult<Series> {
    s.is_finite().map(|ca| ca.into_series())
}

fn is_infinite(s: &Series) -> PolarsResult<Series> {
    s.is_infinite().map(|ca| ca.into_series())
}

pub(super) fn is_nan(s: &Series) -> PolarsResult<Series> {
    s.is_nan().map(|ca| ca.into_series())
}

pub(super) fn is_not_nan(s: &Series) -> PolarsResult<Series> {
    s.is_not_nan().map(|ca| ca.into_series())
}

#[cfg(feature = "is_first")]
fn is_first(s: &Series) -> PolarsResult<Series> {
    polars_ops::prelude::is_first(s).map(|ca| ca.into_series())
}

#[cfg(feature = "is_unique")]
fn is_unique(s: &Series) -> PolarsResult<Series> {
    polars_ops::prelude::is_unique(s).map(|ca| ca.into_series())
}

#[cfg(feature = "is_unique")]
fn is_duplicated(s: &Series) -> PolarsResult<Series> {
    polars_ops::prelude::is_duplicated(s).map(|ca| ca.into_series())
}

#[cfg(feature = "is_in")]
fn is_in(s: &mut [Series]) -> PolarsResult<Option<Series>> {
    let left = &s[0];
    let other = &s[1];
    left.is_in(other).map(|ca| Some(ca.into_series()))
}
