use std::ops::{BitAnd, BitOr, Not};

use polars_core::POOL;
use rayon::prelude::*;

use super::*;
use crate::{map, wrap};

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, PartialEq, Debug, Eq, Hash)]
pub enum BooleanFunction {
    All {
        ignore_nulls: bool,
    },
    AllHorizontal,
    Any {
        ignore_nulls: bool,
    },
    AnyHorizontal,
    #[cfg(feature = "is_unique")]
    IsDuplicated,
    IsFinite,
    #[cfg(feature = "is_first")]
    IsFirst,
    #[cfg(feature = "is_in")]
    IsIn,
    IsInfinite,
    IsNan,
    IsNotNan,
<<<<<<< HEAD
    #[cfg(feature = "is_first")]
    IsFirst,
    #[cfg(feature = "is_last")]
    IsLast,
=======
    IsNotNull,
    IsNull,
>>>>>>> 20c4bc91f (Replace relevant occurrences of is_not and IsNot)
    #[cfg(feature = "is_unique")]
    IsUnique,
    Not,
}

impl BooleanFunction {
    pub(super) fn get_field(&self, mapper: FieldsMapper) -> PolarsResult<Field> {
        use BooleanFunction::*;
        match self {
            AllHorizontal => Ok(Field::new("all", DataType::Boolean)),
            AnyHorizontal => Ok(Field::new("any", DataType::Boolean)),
            _ => mapper.with_dtype(DataType::Boolean),
        }
    }
}

impl Display for BooleanFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use BooleanFunction::*;
        let s = match self {
            All { .. } => "all",
            AllHorizontal => "all_horizontal",
            Any { .. } => "any",
            AnyHorizontal => "any_horizontal",
            #[cfg(feature = "is_unique")]
            IsDuplicated => "is_duplicated",
            IsFinite => "is_finite",
            #[cfg(feature = "is_first")]
            IsFirst => "is_first",
            #[cfg(feature = "is_in")]
            IsIn => "is_in",
            IsInfinite => "is_infinite",
            #[cfg(feature = "is_last")]
            IsLast => "is_last",
            IsNan => "is_nan",
            IsNotNan => "is_not_nan",
            IsNotNull => "is_not_null",
            IsNull => "is_null",
            #[cfg(feature = "is_unique")]
            IsUnique => "is_unique",
            Not => "not_",
        };
        write!(f, "{s}")
    }
}

impl From<BooleanFunction> for SpecialEq<Arc<dyn SeriesUdf>> {
    fn from(func: BooleanFunction) -> Self {
        use BooleanFunction::*;
        match func {
            All { ignore_nulls } => map!(all, ignore_nulls),
            AllHorizontal => wrap!(all_horizontal),
            Any { ignore_nulls } => map!(any, ignore_nulls),
            AnyHorizontal => wrap!(any_horizontal),
            #[cfg(feature = "is_unique")]
            IsDuplicated => map!(is_duplicated),
            IsFinite => map!(is_finite),
            #[cfg(feature = "is_first")]
            IsFirst => map!(is_first),
            #[cfg(feature = "is_in")]
            IsIn => wrap!(is_in),
            IsInfinite => map!(is_infinite),
            #[cfg(feature = "is_last")]
            IsLast => map!(is_last),
            IsNan => map!(is_nan),
            IsNotNan => map!(is_not_nan),
            IsNotNull => map!(is_not_null),
            IsNull => map!(is_null),
            #[cfg(feature = "is_unique")]
            IsUnique => map!(is_unique),
            Not => map!(not_),
        }
    }
}

impl From<BooleanFunction> for FunctionExpr {
    fn from(func: BooleanFunction) -> Self {
        FunctionExpr::Boolean(func)
    }
}

fn any(s: &Series, ignore_nulls: bool) -> PolarsResult<Series> {
    let ca = s.bool()?;
    if ignore_nulls {
        Ok(Series::new(s.name(), [ca.any()]))
    } else {
        Ok(Series::new(s.name(), [ca.any_kleene()]))
    }
}

fn all(s: &Series, ignore_nulls: bool) -> PolarsResult<Series> {
    let ca = s.bool()?;
    if ignore_nulls {
        Ok(Series::new(s.name(), [ca.all()]))
    } else {
        Ok(Series::new(s.name(), [ca.all_kleene()]))
    }
}

fn not_(s: &Series) -> PolarsResult<Series> {
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

#[cfg(feature = "is_last")]
fn is_last(s: &Series) -> PolarsResult<Series> {
    polars_ops::prelude::is_last(s).map(|ca| ca.into_series())
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
    polars_ops::prelude::is_in(left, other).map(|ca| Some(ca.into_series()))
}

fn any_horizontal(s: &mut [Series]) -> PolarsResult<Option<Series>> {
    let mut out = POOL.install(|| {
        s.par_iter()
            .try_fold(
                || BooleanChunked::new("", &[false]),
                |acc, b| {
                    let b = b.cast(&DataType::Boolean)?;
                    let b = b.bool()?;
                    PolarsResult::Ok((&acc).bitor(b))
                },
            )
            .try_reduce(|| BooleanChunked::new("", [false]), |a, b| Ok(a.bitor(b)))
    })?;
    out.rename("any");
    Ok(Some(out.into_series()))
}

fn all_horizontal(s: &mut [Series]) -> PolarsResult<Option<Series>> {
    let mut out = POOL.install(|| {
        s.par_iter()
            .try_fold(
                || BooleanChunked::new("", &[true]),
                |acc, b| {
                    let b = b.cast(&DataType::Boolean)?;
                    let b = b.bool()?;
                    PolarsResult::Ok((&acc).bitand(b))
                },
            )
            .try_reduce(|| BooleanChunked::new("", [true]), |a, b| Ok(a.bitand(b)))
    })?;
    out.rename("all");
    Ok(Some(out.into_series()))
}
