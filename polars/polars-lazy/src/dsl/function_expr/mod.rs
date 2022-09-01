#[cfg(feature = "arg_where")]
mod arg_where;
#[cfg(feature = "round_series")]
mod clip;
mod fill_null;
#[cfg(feature = "is_in")]
mod is_in;
#[cfg(any(feature = "is_in", feature = "list"))]
mod list;
mod nan;
mod pow;
#[cfg(all(feature = "rolling_window", feature = "moment"))]
mod rolling;
#[cfg(feature = "row_hash")]
mod row_hash;
mod schema;
#[cfg(feature = "search_sorted")]
mod search_sorted;
mod shift;
mod shift_and_fill;
#[cfg(feature = "sign")]
mod sign;
#[cfg(feature = "strings")]
mod strings;
#[cfg(feature = "dtype-struct")]
mod struct_;
#[cfg(any(feature = "temporal", feature = "date_offset"))]
mod temporal;
#[cfg(feature = "trigonometry")]
mod trigonometry;

#[cfg(feature = "list")]
pub(super) use list::ListFunction;
use polars_core::prelude::*;
use polars_core::utils::slice_offsets;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

pub(super) use self::nan::NanFunction;
#[cfg(feature = "strings")]
pub(super) use self::strings::StringFunction;
#[cfg(feature = "dtype-struct")]
pub(super) use self::struct_::StructFunction;
use super::*;

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
    #[cfg(feature = "search_sorted")]
    SearchSorted,
    #[cfg(feature = "strings")]
    StringExpr(StringFunction),
    #[cfg(feature = "date_offset")]
    DateOffset(Duration),
    #[cfg(feature = "trigonometry")]
    Trigonometry(TrigonometricFunction),
    #[cfg(feature = "sign")]
    Sign,
    FillNull {
        super_type: DataType,
    },
    #[cfg(feature = "is_in")]
    ListContains,
    #[cfg(all(feature = "rolling_window", feature = "moment"))]
    // if we add more, make a sub enum
    RollingSkew {
        window_size: usize,
        bias: bool,
    },
    ShiftAndFill {
        periods: i64,
    },
    Nan(NanFunction),
    #[cfg(feature = "round_series")]
    Clip {
        min: Option<AnyValue<'static>>,
        max: Option<AnyValue<'static>>,
    },
    #[cfg(feature = "list")]
    ListExpr(ListFunction),
    #[cfg(feature = "dtype-struct")]
    StructExpr(StructFunction),
    #[cfg(feature = "top_k")]
    TopK {
        k: usize,
        reverse: bool,
    },
    Shift(i64),
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

macro_rules! wrap {
    ($e:expr) => {
        SpecialEq::new(Arc::new($e))
    };
}

// Fn(&[Series], args)
// all expression arguments are in the slice.
// the first element is the root expression.
macro_rules! map_as_slice {
    ($func:path, $($args:expr),*) => {{
        let f = move |s: &mut [Series]| {
            $func(s, $($args),*)
        };

        SpecialEq::new(Arc::new(f))
    }};
}

// FnOnce(Series)
// FnOnce(Series, args)
#[macro_export(super)]
macro_rules! map_owned {
    ($func:path) => {{
        let f = move |s: &mut [Series]| {
            let s = std::mem::take(&mut s[0]);
            $func(s)
        };

        SpecialEq::new(Arc::new(f))
    }};

    ($func:path, $($args:expr),*) => {{
        let f = move |s: &mut [Series]| {
            let s = std::mem::take(&mut s[0]);
            $func(s, $($args),*)
        };

        SpecialEq::new(Arc::new(f))
    }};
}

// Fn(&Series, args)
#[macro_export(super)]
macro_rules! map {
    ($func:path) => {{
        let f = move |s: &mut [Series]| {
            let s = &s[0];
            $func(s)
        };

        SpecialEq::new(Arc::new(f))
    }};

    ($func:path, $($args:expr),*) => {{
        let f = move |s: &mut [Series]| {
            let s = &s[0];
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
                map!(row_hash::row_hash, k0, k1, k2, k3)
            }
            #[cfg(feature = "is_in")]
            IsIn => {
                wrap!(is_in::is_in)
            }
            #[cfg(feature = "arg_where")]
            ArgWhere => {
                wrap!(arg_where::arg_where)
            }
            #[cfg(feature = "search_sorted")]
            SearchSorted => {
                wrap!(search_sorted::search_sorted_impl)
            }
            #[cfg(feature = "strings")]
            StringExpr(s) => s.into(),

            #[cfg(feature = "date_offset")]
            DateOffset(offset) => {
                map_owned!(temporal::date_offset, offset)
            }
            #[cfg(feature = "trigonometry")]
            Trigonometry(trig_function) => {
                map!(trigonometry::apply_trigonometric_function, trig_function)
            }
            #[cfg(feature = "sign")]
            Sign => {
                map!(sign::sign)
            }
            FillNull { super_type } => {
                map_as_slice!(fill_null::fill_null, &super_type)
            }

            #[cfg(feature = "is_in")]
            ListContains => {
                wrap!(list::contains)
            }
            #[cfg(all(feature = "rolling_window", feature = "moment"))]
            RollingSkew { window_size, bias } => {
                map!(rolling::rolling_skew, window_size, bias)
            }
            ShiftAndFill { periods } => {
                map_as_slice!(shift_and_fill::shift_and_fill, periods)
            }
            Nan(n) => n.into(),
            #[cfg(feature = "round_series")]
            Clip { min, max } => {
                map_owned!(clip::clip, min.clone(), max.clone())
            }
            #[cfg(feature = "list")]
            ListExpr(lf) => {
                use ListFunction::*;
                match lf {
                    Concat => wrap!(list::concat),
                }
            }
            #[cfg(feature = "dtype-struct")]
            StructExpr(sf) => {
                use StructFunction::*;
                match sf {
                    FieldByIndex(index) => map!(struct_::get_by_index, index),
                    FieldByName(name) => map!(struct_::get_by_name, name.clone()),
                }
            }
            #[cfg(feature = "top_k")]
            TopK { k, reverse } => {
                map!(top_k, k, reverse)
            }
            Shift(periods) => map!(shift::shift, periods),
        }
    }
}

#[cfg(feature = "strings")]
impl From<StringFunction> for SpecialEq<Arc<dyn SeriesUdf>> {
    fn from(func: StringFunction) -> Self {
        use StringFunction::*;
        match func {
            Contains { pat, literal } => {
                map!(strings::contains, &pat, literal)
            }
            EndsWith(sub) => {
                map!(strings::ends_with, &sub)
            }
            StartsWith(sub) => {
                map!(strings::starts_with, &sub)
            }
            Extract { pat, group_index } => {
                map!(strings::extract, &pat, group_index)
            }
            ExtractAll(pat) => {
                map!(strings::extract_all, &pat)
            }
            CountMatch(pat) => {
                map!(strings::count_match, &pat)
            }
            #[cfg(feature = "string_justify")]
            Zfill(alignment) => {
                map!(strings::zfill, alignment)
            }
            #[cfg(feature = "string_justify")]
            LJust { width, fillchar } => {
                map!(strings::ljust, width, fillchar)
            }
            #[cfg(feature = "string_justify")]
            RJust { width, fillchar } => {
                map!(strings::rjust, width, fillchar)
            }
            #[cfg(feature = "temporal")]
            Strptime(options) => {
                map!(strings::strptime, &options)
            }
            #[cfg(feature = "concat_str")]
            Concat(delimiter) => map!(strings::concat, &delimiter),
            #[cfg(feature = "regex")]
            Replace { all, literal } => map_as_slice!(strings::replace, literal, all),
            Uppercase => map!(strings::uppercase),
            Lowercase => map!(strings::lowercase),
        }
    }
}
