use std::ops::{BitAnd, BitOr};

use polars_core::POOL;
use polars_core::utils::SuperTypeFlags;
#[cfg(feature = "is_close")]
use polars_utils::total_ord::TotalOrdWrap;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use super::*;
use crate::{map, map_as_slice, wrap};

#[cfg_attr(feature = "ir_serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, PartialEq, Debug, Eq, Hash)]
pub enum IRBooleanFunction {
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
    IsIn {
        nulls_equal: bool,
    },
    #[cfg(feature = "is_close")]
    IsClose {
        abs_tol: TotalOrdWrap<f64>,
        rel_tol: TotalOrdWrap<f64>,
        nans_equal: bool,
    },
    AllHorizontal,
    AnyHorizontal,
    // Also bitwise negate
    Not,
}

impl IRBooleanFunction {
    pub(super) fn get_field(&self, mapper: FieldsMapper) -> PolarsResult<Field> {
        match self {
            IRBooleanFunction::Not => {
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

    pub fn function_options(&self) -> FunctionOptions {
        use IRBooleanFunction as B;
        match self {
            B::Any { .. } | B::All { .. } => FunctionOptions::aggregation(),
            B::IsNull | B::IsNotNull => FunctionOptions::elementwise(),
            B::IsFinite | B::IsInfinite | B::IsNan | B::IsNotNan => FunctionOptions::elementwise()
                .with_flags(|f| f | FunctionFlags::PRESERVES_NULL_FIRST_INPUT),
            #[cfg(feature = "is_first_distinct")]
            B::IsFirstDistinct => FunctionOptions::length_preserving(),
            #[cfg(feature = "is_last_distinct")]
            B::IsLastDistinct => FunctionOptions::length_preserving(),
            #[cfg(feature = "is_unique")]
            B::IsUnique => FunctionOptions::length_preserving(),
            #[cfg(feature = "is_unique")]
            B::IsDuplicated => FunctionOptions::length_preserving(),
            #[cfg(feature = "is_between")]
            B::IsBetween { .. } => FunctionOptions::elementwise()
                .with_supertyping(
                    (SuperTypeFlags::default() & !SuperTypeFlags::ALLOW_PRIMITIVE_TO_STRING).into(),
                )
                .with_flags(|f| f | FunctionFlags::PRESERVES_NULL_ALL_INPUTS),
            #[cfg(feature = "is_in")]
            B::IsIn { nulls_equal } => FunctionOptions::elementwise()
                .with_supertyping(Default::default())
                .with_flags(|f| {
                    if !*nulls_equal {
                        f | FunctionFlags::PRESERVES_NULL_FIRST_INPUT
                    } else {
                        f
                    }
                }),
            #[cfg(feature = "is_close")]
            B::IsClose { .. } => FunctionOptions::elementwise()
                .with_supertyping(
                    (SuperTypeFlags::default() & !SuperTypeFlags::ALLOW_PRIMITIVE_TO_STRING).into(),
                )
                .with_flags(|f| f | FunctionFlags::PRESERVES_NULL_ALL_INPUTS),
            B::AllHorizontal | B::AnyHorizontal => FunctionOptions::elementwise().with_flags(|f| {
                f | FunctionFlags::INPUT_WILDCARD_EXPANSION | FunctionFlags::ALLOW_EMPTY_INPUTS
            }),
            B::Not => FunctionOptions::elementwise()
                .with_flags(|f| f | FunctionFlags::PRESERVES_NULL_FIRST_INPUT),
        }
    }
}

impl Display for IRBooleanFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use IRBooleanFunction::*;
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
            IsIn { .. } => "is_in",
            #[cfg(feature = "is_close")]
            IsClose { .. } => "is_close",
            AnyHorizontal => "any_horizontal",
            AllHorizontal => "all_horizontal",
            Not => "not",
        };
        write!(f, "{s}")
    }
}

impl From<IRBooleanFunction> for SpecialEq<Arc<dyn ColumnsUdf>> {
    fn from(func: IRBooleanFunction) -> Self {
        use IRBooleanFunction::*;
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
            IsIn { nulls_equal } => wrap!(is_in, nulls_equal),
            #[cfg(feature = "is_close")]
            IsClose {
                abs_tol,
                rel_tol,
                nans_equal,
            } => wrap!(is_close, abs_tol, rel_tol, nans_equal),
            Not => map!(not),
            AllHorizontal => map_as_slice!(all_horizontal),
            AnyHorizontal => map_as_slice!(any_horizontal),
        }
    }
}

impl From<IRBooleanFunction> for IRFunctionExpr {
    fn from(func: IRBooleanFunction) -> Self {
        IRFunctionExpr::Boolean(func)
    }
}

fn any(s: &Column, ignore_nulls: bool) -> PolarsResult<Column> {
    let ca = s.bool()?;
    if ignore_nulls {
        Ok(Column::new(s.name().clone(), [ca.any()]))
    } else {
        Ok(Column::new(s.name().clone(), [ca.any_kleene()]))
    }
}

fn all(s: &Column, ignore_nulls: bool) -> PolarsResult<Column> {
    let ca = s.bool()?;
    if ignore_nulls {
        Ok(Column::new(s.name().clone(), [ca.all()]))
    } else {
        Ok(Column::new(s.name().clone(), [ca.all_kleene()]))
    }
}

fn is_null(s: &Column) -> PolarsResult<Column> {
    Ok(s.is_null().into_column())
}

fn is_not_null(s: &Column) -> PolarsResult<Column> {
    Ok(s.is_not_null().into_column())
}

fn is_finite(s: &Column) -> PolarsResult<Column> {
    s.is_finite().map(|ca| ca.into_column())
}

fn is_infinite(s: &Column) -> PolarsResult<Column> {
    s.is_infinite().map(|ca| ca.into_column())
}

pub(super) fn is_nan(s: &Column) -> PolarsResult<Column> {
    s.is_nan().map(|ca| ca.into_column())
}

pub(super) fn is_not_nan(s: &Column) -> PolarsResult<Column> {
    s.is_not_nan().map(|ca| ca.into_column())
}

#[cfg(feature = "is_first_distinct")]
fn is_first_distinct(s: &Column) -> PolarsResult<Column> {
    polars_ops::prelude::is_first_distinct(s.as_materialized_series()).map(|ca| ca.into_column())
}

#[cfg(feature = "is_last_distinct")]
fn is_last_distinct(s: &Column) -> PolarsResult<Column> {
    polars_ops::prelude::is_last_distinct(s.as_materialized_series()).map(|ca| ca.into_column())
}

#[cfg(feature = "is_unique")]
fn is_unique(s: &Column) -> PolarsResult<Column> {
    polars_ops::prelude::is_unique(s.as_materialized_series()).map(|ca| ca.into_column())
}

#[cfg(feature = "is_unique")]
fn is_duplicated(s: &Column) -> PolarsResult<Column> {
    polars_ops::prelude::is_duplicated(s.as_materialized_series()).map(|ca| ca.into_column())
}

#[cfg(feature = "is_between")]
fn is_between(s: &[Column], closed: ClosedInterval) -> PolarsResult<Column> {
    let ser = &s[0];
    let lower = &s[1];
    let upper = &s[2];
    polars_ops::prelude::is_between(
        ser.as_materialized_series(),
        lower.as_materialized_series(),
        upper.as_materialized_series(),
        closed,
    )
    .map(|ca| ca.into_column())
}

#[cfg(feature = "is_in")]
fn is_in(s: &mut [Column], nulls_equal: bool) -> PolarsResult<Column> {
    let left = &s[0];
    let other = &s[1];
    polars_ops::prelude::is_in(
        left.as_materialized_series(),
        other.as_materialized_series(),
        nulls_equal,
    )
    .map(IntoColumn::into_column)
}

#[cfg(feature = "is_close")]
fn is_close(
    s: &mut [Column],
    abs_tol: TotalOrdWrap<f64>,
    rel_tol: TotalOrdWrap<f64>,
    nans_equal: bool,
) -> PolarsResult<Column> {
    let left = &s[0];
    let right = &s[1];
    polars_ops::prelude::is_close(
        left.as_materialized_series(),
        right.as_materialized_series(),
        abs_tol.0,
        rel_tol.0,
        nans_equal,
    )
    .map(IntoColumn::into_column)
}

fn not(s: &Column) -> PolarsResult<Column> {
    polars_ops::series::negate_bitwise(s.as_materialized_series()).map(Column::from)
}

// We shouldn't hit these often only on very wide dataframes where we don't reduce to & expressions.
fn any_horizontal(s: &[Column]) -> PolarsResult<Column> {
    let out = POOL
        .install(|| {
            s.par_iter()
                .try_fold(
                    || BooleanChunked::new(PlSmallStr::EMPTY, &[false]),
                    |acc, b| {
                        let b = b.cast(&DataType::Boolean)?;
                        let b = b.bool()?;
                        PolarsResult::Ok((&acc).bitor(b))
                    },
                )
                .try_reduce(
                    || BooleanChunked::new(PlSmallStr::EMPTY, [false]),
                    |a, b| Ok(a.bitor(b)),
                )
        })?
        .with_name(s[0].name().clone());
    Ok(out.into_column())
}

// We shouldn't hit these often only on very wide dataframes where we don't reduce to & expressions.
fn all_horizontal(s: &[Column]) -> PolarsResult<Column> {
    let out = POOL
        .install(|| {
            s.par_iter()
                .try_fold(
                    || BooleanChunked::new(PlSmallStr::EMPTY, &[true]),
                    |acc, b| {
                        let b = b.cast(&DataType::Boolean)?;
                        let b = b.bool()?;
                        PolarsResult::Ok((&acc).bitand(b))
                    },
                )
                .try_reduce(
                    || BooleanChunked::new(PlSmallStr::EMPTY, [true]),
                    |a, b| Ok(a.bitand(b)),
                )
        })?
        .with_name(s[0].name().clone());
    Ok(out.into_column())
}
