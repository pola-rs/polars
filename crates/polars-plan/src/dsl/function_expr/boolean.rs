use std::ops::{BitAnd, BitOr};

use polars_core::POOL;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use super::*;
#[cfg(feature = "is_in")]
use crate::wrap;
use crate::{map, map_as_slice};

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, PartialEq, Debug, Eq, Hash)]
pub enum BooleanFunction {
    Any {
        ignore_nulls: bool,
    },
    All {
        ignore_nulls: bool,
    },
    IsNull,
    IsNotNull,
    IsFinite,
    IsInfinite,
    IsNan,
    IsNotNan,
    #[cfg(feature = "is_first_distinct")]
    IsFirstDistinct,
    #[cfg(feature = "is_last_distinct")]
    IsLastDistinct,
    #[cfg(feature = "is_unique")]
    IsUnique,
    #[cfg(feature = "is_unique")]
    IsDuplicated,
    #[cfg(feature = "is_between")]
    IsBetween {
        closed: ClosedInterval,
    },
    #[cfg(feature = "is_in")]
    IsIn,
    AllHorizontal,
    AnyHorizontal,
    // Also bitwise negate
    Not,
}

impl BooleanFunction {
    pub(super) fn get_field(&self, mapper: FieldsMapper) -> PolarsResult<Field> {
        match self {
            BooleanFunction::Not => {
                mapper.try_map_dtype(|dtype| {
                    match dtype {
                        DataType::Boolean => Ok(DataType::Boolean),
                        dt if dt.is_integer() => Ok(dt.clone()),
                        dt => polars_bail!(InvalidOperation: "dtype {:?} not supported in 'not' operation", dt) 
                    }
                })

            },
            _ => mapper.with_dtype(DataType::Boolean),
        }
    }
}

impl Display for BooleanFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use BooleanFunction::*;
        let s = match self {
            All { .. } => "all",
            Any { .. } => "any",
            IsNull => "is_null",
            IsNotNull => "is_not_null",
            IsFinite => "is_finite",
            IsInfinite => "is_infinite",
            IsNan => "is_nan",
            IsNotNan => "is_not_nan",
            #[cfg(feature = "is_first_distinct")]
            IsFirstDistinct => "is_first_distinct",
            #[cfg(feature = "is_last_distinct")]
            IsLastDistinct => "is_last_distinct",
            #[cfg(feature = "is_unique")]
            IsUnique => "is_unique",
            #[cfg(feature = "is_unique")]
            IsDuplicated => "is_duplicated",
            #[cfg(feature = "is_between")]
            IsBetween { .. } => "is_between",
            #[cfg(feature = "is_in")]
            IsIn => "is_in",
            AnyHorizontal => "any_horizontal",
            AllHorizontal => "all_horizontal",
            Not => "not",
        };
        write!(f, "{s}")
    }
}

impl From<BooleanFunction> for SpecialEq<Arc<dyn SeriesUdf>> {
    fn from(func: BooleanFunction) -> Self {
        use BooleanFunction::*;
        match func {
            Any { ignore_nulls } => map!(any, ignore_nulls),
            All { ignore_nulls } => map!(all, ignore_nulls),
            IsNull => map!(is_null),
            IsNotNull => map!(is_not_null),
            IsFinite => map!(is_finite),
            IsInfinite => map!(is_infinite),
            IsNan => map!(is_nan),
            IsNotNan => map!(is_not_nan),
            #[cfg(feature = "is_first_distinct")]
            IsFirstDistinct => map!(is_first_distinct),
            #[cfg(feature = "is_last_distinct")]
            IsLastDistinct => map!(is_last_distinct),
            #[cfg(feature = "is_unique")]
            IsUnique => map!(is_unique),
            #[cfg(feature = "is_unique")]
            IsDuplicated => map!(is_duplicated),
            #[cfg(feature = "is_between")]
            IsBetween { closed } => map_as_slice!(is_between, closed),
            #[cfg(feature = "is_in")]
            IsIn => wrap!(is_in),
            Not => map!(not),
            AllHorizontal => map_as_slice!(all_horizontal),
            AnyHorizontal => map_as_slice!(any_horizontal),
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

#[cfg(feature = "is_first_distinct")]
fn is_first_distinct(s: &Series) -> PolarsResult<Series> {
    polars_ops::prelude::is_first_distinct(s).map(|ca| ca.into_series())
}

#[cfg(feature = "is_last_distinct")]
fn is_last_distinct(s: &Series) -> PolarsResult<Series> {
    polars_ops::prelude::is_last_distinct(s).map(|ca| ca.into_series())
}

#[cfg(feature = "is_unique")]
fn is_unique(s: &Series) -> PolarsResult<Series> {
    polars_ops::prelude::is_unique(s).map(|ca| ca.into_series())
}

#[cfg(feature = "is_unique")]
fn is_duplicated(s: &Series) -> PolarsResult<Series> {
    polars_ops::prelude::is_duplicated(s).map(|ca| ca.into_series())
}

#[cfg(feature = "is_between")]
fn is_between(s: &[Series], closed: ClosedInterval) -> PolarsResult<Series> {
    let ser = &s[0];
    let lower = &s[1];
    let upper = &s[2];
    polars_ops::prelude::is_between(ser, lower, upper, closed).map(|ca| ca.into_series())
}

#[cfg(feature = "is_in")]
fn is_in(s: &mut [Series]) -> PolarsResult<Option<Series>> {
    let left = &s[0];
    let other = &s[1];
    polars_ops::prelude::is_in(left, other).map(|ca| Some(ca.into_series()))
}

fn not(s: &Series) -> PolarsResult<Series> {
    polars_ops::series::negate_bitwise(s)
}

// We shouldn't hit these often only on very wide dataframes where we don't reduce to & expressions.
fn any_horizontal(s: &[Series]) -> PolarsResult<Series> {
    let out = POOL
        .install(|| {
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
        })?
        .with_name(s[0].name());
    Ok(out.into_series())
}

// We shouldn't hit these often only on very wide dataframes where we don't reduce to & expressions.
fn all_horizontal(s: &[Series]) -> PolarsResult<Series> {
    let out = POOL
        .install(|| {
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
        })?
        .with_name(s[0].name());
    Ok(out.into_series())
}
