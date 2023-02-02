use polars_core::prelude::*;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::*;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, PartialEq, Debug, Eq, Hash)]
pub enum NumericFunction {
    IsFinite,
    IsInfinite,
    /// Argument is the number of decimals
    #[cfg(feature = "round_series")]
    Round(u32),
    #[cfg(feature = "round_series")]
    Floor,
    #[cfg(feature = "round_series")]
    Ceil,
    Abs,
    UpperBound,
    LowerBound,
    /// Argument is the base
    #[cfg(feature = "log")]
    Log(HashF64),
    #[cfg(feature = "log")]
    Exp,
    CumSum {
        reverse: bool,
    },
    CumProd {
        reverse: bool,
    },
    CumMin {
        reverse: bool,
    },
    CumMax {
        reverse: bool,
    },
    CumCount {
        reverse: bool,
    },
    /// Row reduce using max()
    RowMax,
    /// Row reduce using min()
    RowMin,
    /// Row fold using +
    RowSum,
    /// Row fold using bitwise OR
    RowAny,
    /// Row fold using bitwise AND
    RowAll,
}

impl From<NumericFunction> for FunctionExpr {
    fn from(f: NumericFunction) -> Self {
        FunctionExpr::NumFunction(f)
    }
}

impl Display for NumericFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use NumericFunction::*;
        match self {
            IsFinite => write!(f, "is_finite"),
            IsInfinite => write!(f, "is_infinite"),
            #[cfg(feature = "round_series")]
            Round(_) => write!(f, "round"),
            #[cfg(feature = "round_series")]
            Floor => write!(f, "floor"),
            #[cfg(feature = "round_series")]
            Ceil => write!(f, "ceil"),
            Abs => write!(f, "abs"),
            UpperBound => write!(f, "upper_bound"),
            LowerBound => write!(f, "lower_bound"),
            #[cfg(feature = "log")]
            Log(_) => write!(f, "log"),
            #[cfg(feature = "log")]
            Exp => write!(f, "exp"),
            CumSum { .. } => write!(f, "cumsum"),
            CumProd { .. } => write!(f, "cumprod"),
            CumMin { .. } => write!(f, "cummin"),
            CumMax { .. } => write!(f, "cummax"),
            CumCount { .. } => write!(f, "cumcount"),
            RowMax => write!(f, "rowmax"),
            RowMin => write!(f, "rowmin"),
            RowSum => write!(f, "rowsum"),
            RowAny => write!(f, "rowany"),
            RowAll => write!(f, "rowall"),
        }
    }
}

impl NumericFunction {
    pub(crate) fn get_field(
        &self,
        _input_schema: &Schema,
        _cntxt: Context,
        fields: &[Field],
    ) -> PolarsResult<Field> {
        use get_output::*;
        use NumericFunction::*;

        match self {
            IsFinite | IsInfinite => with_dtype(DataType::Boolean)(fields),
            #[cfg(feature = "round_series")]
            Round(_) | Floor | Ceil => same_type()(fields),
            UpperBound | LowerBound | Abs => same_type()(fields),
            #[cfg(feature = "log")]
            Log(_) | Exp => map_dtype(|dt| {
                if matches!(dt, DataType::Float32) {
                    DataType::Float32
                } else {
                    DataType::Float64
                }
            })(fields),
            CumSum { .. } => map_dtype(|dt| {
                use DataType::*;
                if dt.is_logical() {
                    dt.clone()
                } else {
                    match dt {
                        Boolean => UInt32,
                        Int32 => Int32,
                        UInt32 => UInt32,
                        UInt64 => UInt64,
                        Float32 => Float32,
                        Float64 => Float64,
                        _ => Int64,
                    }
                }
            })(fields),
            CumProd { .. } => map_dtype(|dt| {
                use DataType::*;
                match dt {
                    Boolean => Int64,
                    UInt64 => UInt64,
                    Float32 => Float32,
                    Float64 => Float64,
                    _ => Int64,
                }
            })(fields),
            CumMin { .. } | CumMax { .. } => same_type()(fields),
            CumCount { .. } => with_dtype(IDX_DTYPE)(fields),
            RowMax | RowMin | RowSum | RowAny | RowAll => super_type()(fields),
        }
    }
}

impl From<NumericFunction> for SpecialEq<Arc<dyn SeriesEval>> {
    fn from(func: NumericFunction) -> Self {
        fn make_row_fold(
            f: impl Fn(Series, Series) -> PolarsResult<Series> + Send + Sync + Clone + 'static,
        ) -> SpecialEq<Arc<dyn SeriesEval>> {
            // Accumulator is implicitely supplied as the last input
            wrap!(move |series: &mut [Series]| {
                let mut series = series.to_vec();
                let mut acc = series.pop().unwrap();

                for s in series {
                    acc = f(acc, s)?;
                }
                Ok(acc)
            })
        }

        fn make_row_reduce(
            f: impl Fn(Series, Series) -> PolarsResult<Series> + Send + Sync + Clone + 'static,
        ) -> SpecialEq<Arc<dyn SeriesEval>> {
            wrap!(move |series: &mut [Series]| {
                let mut s_iter = series.iter();

                match s_iter.next() {
                    Some(acc) => {
                        let mut acc = acc.clone();

                        for s in s_iter {
                            acc = f(acc, s.clone())?;
                        }
                        Ok(acc)
                    }
                    None => Err(PolarsError::ComputeError(
                        "Reduce did not have any expressions to fold".into(),
                    )),
                }
            })
        }

        use NumericFunction::*;
        match func {
            RowMax => make_row_reduce(|s1, s2| {
                let df = DataFrame::new_no_checks(vec![s1, s2]);
                df.hmax().map(|s| s.unwrap())
            }),
            RowMin => make_row_reduce(|s1, s2| {
                let df = DataFrame::new_no_checks(vec![s1, s2]);
                df.hmin().map(|s| s.unwrap())
            }),
            RowSum => make_row_fold(|s1, s2| Ok(&s1 + &s2)),
            RowAny => make_row_fold(|s1, s2| Ok(s1.bool()?.bitor(s2.bool()?).into_series())),
            RowAll => make_row_fold(|s1, s2| Ok(s1.bool()?.bitand(s2.bool()?).into_series())),
            IsFinite => map!(|s: &Series| s.is_finite().map(|ca| ca.into_series())),
            IsInfinite => map!(|s: &Series| s.is_infinite().map(|ca| ca.into_series())),
            #[cfg(feature = "round_series")]
            Round(decimals) => map!(|s: &Series| s.round(decimals)),
            #[cfg(feature = "round_series")]
            Floor => map!(|s: &Series| s.floor()),
            #[cfg(feature = "round_series")]
            Ceil => map!(|s: &Series| s.ceil()),
            Abs => map!(|s: &Series| s.abs()),
            UpperBound => map!(upper_bound),
            LowerBound => map!(lower_bound),
            #[cfg(feature = "log")]
            Log(base) => map!(|s: &Series| Ok(s.log(base.into()))),
            #[cfg(feature = "log")]
            Exp => map!(|s: &Series| Ok(s.exp())),
            CumSum { reverse } => map!(|s: &Series| Ok(s.cumsum(reverse))),
            CumProd { reverse } => map!(|s: &Series| Ok(s.cumprod(reverse))),
            CumMin { reverse } => map!(|s: &Series| Ok(s.cummin(reverse))),
            CumMax { reverse } => map!(|s: &Series| Ok(s.cummax(reverse))),
            CumCount { reverse } => map!(cumcount, reverse),
        }
    }
}

fn upper_bound(s: &Series) -> PolarsResult<Series> {
    let name = s.name();
    use DataType::*;
    let s = match s.dtype().to_physical() {
        #[cfg(feature = "dtype-i8")]
        Int8 => Series::new(name, &[i8::MAX]),
        #[cfg(feature = "dtype-i16")]
        Int16 => Series::new(name, &[i16::MAX]),
        Int32 => Series::new(name, &[i32::MAX]),
        Int64 => Series::new(name, &[i64::MAX]),
        #[cfg(feature = "dtype-u8")]
        UInt8 => Series::new(name, &[u8::MAX]),
        #[cfg(feature = "dtype-u16")]
        UInt16 => Series::new(name, &[u16::MAX]),
        UInt32 => Series::new(name, &[u32::MAX]),
        UInt64 => Series::new(name, &[u64::MAX]),
        Float32 => Series::new(name, &[f32::INFINITY]),
        Float64 => Series::new(name, &[f64::INFINITY]),
        dt => {
            return Err(PolarsError::ComputeError(
                format!("cannot determine upper bound of dtype {dt}").into(),
            ))
        }
    };
    Ok(s)
}

fn lower_bound(s: &Series) -> PolarsResult<Series> {
    let name = s.name();
    use DataType::*;
    let s = match s.dtype().to_physical() {
        #[cfg(feature = "dtype-i8")]
        Int8 => Series::new(name, &[i8::MIN]),
        #[cfg(feature = "dtype-i16")]
        Int16 => Series::new(name, &[i16::MIN]),
        Int32 => Series::new(name, &[i32::MIN]),
        Int64 => Series::new(name, &[i64::MIN]),
        #[cfg(feature = "dtype-u8")]
        UInt8 => Series::new(name, &[u8::MIN]),
        #[cfg(feature = "dtype-u16")]
        UInt16 => Series::new(name, &[u16::MIN]),
        UInt32 => Series::new(name, &[u32::MIN]),
        UInt64 => Series::new(name, &[u64::MIN]),
        Float32 => Series::new(name, &[f32::NEG_INFINITY]),
        Float64 => Series::new(name, &[f64::NEG_INFINITY]),
        dt => {
            return Err(PolarsError::ComputeError(
                format!("cannot determine lower bound of dtype {dt}").into(),
            ))
        }
    };
    Ok(s)
}

fn cumcount(s: &Series, reverse: bool) -> PolarsResult<Series> {
    if reverse {
        let ca: NoNull<UInt32Chunked> = (0u32..s.len() as u32).rev().collect();
        let mut ca = ca.into_inner();
        ca.rename(s.name());
        Ok(ca.into_series())
    } else {
        let ca: NoNull<UInt32Chunked> = (0u32..s.len() as u32).collect();
        let mut ca = ca.into_inner();
        ca.rename(s.name());
        Ok(ca.into_series())
    }
}
