#[cfg(feature = "arg_where")]
mod arg_where;
#[cfg(feature = "is_in")]
mod is_in;
mod pow;
#[cfg(feature = "strings")]
mod strings;
#[cfg(any(feature = "temporal", feature = "date_offset"))]
mod temporal;
#[cfg(feature = "trigonometry")]
mod trigonometry;

use super::*;
use polars_core::prelude::*;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, PartialEq, Debug, Eq, Hash)]
pub enum FunctionExpr {
    NullCount,
    Pow,
    #[cfg(feature = "row_hash")]
    Hash(u64, u64, u64, u64),
    #[cfg(feature = "is_in")]
    IsIn,
    #[cfg(feature = "arg_where")]
    ArgWhere,
    #[cfg(feature = "strings")]
    StringContains {
        pat: String,
        literal: bool,
    },
    #[cfg(feature = "strings")]
    StringStartsWith(String),
    #[cfg(feature = "strings")]
    StringEndsWith(String),
    #[cfg(feature = "date_offset")]
    DateOffset(Duration),
    #[cfg(feature = "trigonometry")]
    Trigonometry(TrigonometricFunction),
}

#[cfg(feature = "trigonometry")]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, PartialEq, Debug, Eq, Hash)]
pub enum TrigonometricFunction {
    Sin,
    Cos,
    Tan,
    ArcSin,
    ArcCos,
    ArcTan,
    Sinh,
    Cosh,
    Tanh,
    ArcSinh,
    ArcCosh,
    ArcTanh,
}

impl FunctionExpr {
    pub(crate) fn get_field(
        &self,
        _input_schema: &Schema,
        _cntxt: Context,
        fields: &[Field],
    ) -> Result<Field> {
        let with_dtype = |dtype: DataType| Ok(Field::new(fields[0].name(), dtype));
        let map_dtype = |func: &dyn Fn(&DataType) -> DataType| {
            let dtype = func(fields[0].data_type());
            Ok(Field::new(fields[0].name(), dtype))
        };

        let float_dtype = || {
            map_dtype(&|dtype| match dtype {
                DataType::Float32 => DataType::Float32,
                _ => DataType::Float64,
            })
        };

        let same_type = || map_dtype(&|dtype| dtype.clone());

        use FunctionExpr::*;
        match self {
            NullCount => with_dtype(IDX_DTYPE),
            Pow => float_dtype(),
            #[cfg(feature = "row_hash")]
            Hash(..) => with_dtype(DataType::UInt64),
            #[cfg(feature = "is_in")]
            IsIn => with_dtype(DataType::Boolean),
            #[cfg(feature = "arg_where")]
            ArgWhere => with_dtype(IDX_DTYPE),
            #[cfg(feature = "strings")]
            StringContains { .. } | StringEndsWith(_) | StringStartsWith(_) => {
                with_dtype(DataType::Boolean)
            }
            #[cfg(feature = "date_offset")]
            DateOffset(_) => same_type(),
            #[cfg(feature = "trigonometry")]
            Trigonometry(_) => float_dtype(),
        }
    }
}

macro_rules! wrap {
    ($e:expr) => {
        SpecialEq::new(Arc::new($e))
    };
}

macro_rules! map_with_args {
    ($func:path, $($args:expr),*) => {{
        let f = move |s: &mut [Series]| {
            let s = &s[0];
            $func(s, $($args),*)
        };

        SpecialEq::new(Arc::new(f))
    }};
}

macro_rules! map_owned_with_args {
    ($func:path, $($args:expr),*) => {{
        let f = move |s: &mut [Series]| {
            let s = std::mem::take(&mut s[0]);
            $func(s, $($args),*)
        };

        SpecialEq::new(Arc::new(f))
    }};
}

impl From<FunctionExpr> for SpecialEq<Arc<dyn SeriesUdf>> {
    fn from(func: FunctionExpr) -> Self {
        use FunctionExpr::*;
        match func {
            NullCount => {
                let f = |s: &mut [Series]| {
                    let s = &s[0];
                    Ok(Series::new(s.name(), [s.null_count() as IdxSize]))
                };
                wrap!(f)
            }
            Pow => {
                wrap!(pow::pow)
            }
            #[cfg(feature = "row_hash")]
            Hash(k0, k1, k2, k3) => {
                let f = move |s: &mut [Series]| {
                    let s = &s[0];
                    Ok(s.hash(ahash::RandomState::with_seeds(k0, k1, k2, k3))
                        .into_series())
                };
                wrap!(f)
            }
            #[cfg(feature = "is_in")]
            IsIn => {
                wrap!(is_in::is_in)
            }
            #[cfg(feature = "arg_where")]
            ArgWhere => {
                wrap!(arg_where::arg_where)
            }
            #[cfg(feature = "strings")]
            StringContains { pat, literal } => {
                map_with_args!(strings::contains, &pat, literal)
            }
            #[cfg(feature = "strings")]
            StringEndsWith(sub) => {
                map_with_args!(strings::ends_with, &sub)
            }
            #[cfg(feature = "strings")]
            StringStartsWith(sub) => {
                map_with_args!(strings::starts_with, &sub)
            }
            #[cfg(feature = "date_offset")]
            DateOffset(offset) => {
                map_owned_with_args!(temporal::date_offset, offset)
            }
            #[cfg(feature = "trigonometry")]
            Trigonometry(trig_function) => {
                map_with_args!(trigonometry::apply_trigonometric_function, trig_function)
            }
        }
    }
}
