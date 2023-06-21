#[cfg(feature = "abs")]
mod abs;
#[cfg(feature = "arg_where")]
mod arg_where;
#[cfg(feature = "dtype-array")]
mod array;
mod binary;
mod boolean;
mod bounds;
#[cfg(feature = "dtype-categorical")]
mod cat;
#[cfg(feature = "round_series")]
mod clip;
mod concat;
mod correlation;
mod cum;
#[cfg(feature = "temporal")]
mod datetime;
mod dispatch;
mod fill_null;
#[cfg(feature = "fused")]
mod fused;
mod list;
#[cfg(feature = "log")]
mod log;
mod nan;
mod pow;
#[cfg(all(feature = "rolling_window", feature = "moment"))]
mod rolling;
#[cfg(feature = "round_series")]
mod round;
#[cfg(feature = "row_hash")]
mod row_hash;
mod schema;
#[cfg(feature = "search_sorted")]
mod search_sorted;
mod shift_and_fill;
mod shrink_type;
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
mod unique;

use std::fmt::{Display, Formatter};

#[cfg(feature = "dtype-array")]
pub(super) use array::ArrayFunction;
pub(crate) use correlation::CorrelationMethod;
#[cfg(feature = "fused")]
pub(crate) use fused::FusedOperator;
pub(super) use list::ListFunction;
use polars_core::prelude::*;
use schema::FieldsMapper;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

pub(crate) use self::binary::BinaryFunction;
pub use self::boolean::BooleanFunction;
#[cfg(feature = "dtype-categorical")]
pub(crate) use self::cat::CategoricalFunction;
#[cfg(feature = "temporal")]
pub(super) use self::datetime::TemporalFunction;
#[cfg(feature = "strings")]
pub(crate) use self::strings::StringFunction;
#[cfg(feature = "dtype-struct")]
pub(super) use self::struct_::StructFunction;
#[cfg(feature = "trigonometry")]
pub(super) use self::trigonometry::TrigonometricFunction;
use super::*;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, PartialEq, Debug)]
pub enum FunctionExpr {
    #[cfg(feature = "abs")]
    Abs,
    NullCount,
    Pow,
    #[cfg(feature = "row_hash")]
    Hash(u64, u64, u64, u64),
    #[cfg(feature = "arg_where")]
    ArgWhere,
    #[cfg(feature = "search_sorted")]
    SearchSorted(SearchSortedSide),
    #[cfg(feature = "strings")]
    StringExpr(StringFunction),
    BinaryExpr(BinaryFunction),
    #[cfg(feature = "temporal")]
    TemporalExpr(TemporalFunction),
    #[cfg(feature = "date_offset")]
    DateOffset(polars_time::Duration),
    #[cfg(feature = "trigonometry")]
    Trigonometry(TrigonometricFunction),
    #[cfg(feature = "sign")]
    Sign,
    FillNull {
        super_type: DataType,
    },
    #[cfg(all(feature = "rolling_window", feature = "moment"))]
    // if we add more, make a sub enum
    RollingSkew {
        window_size: usize,
        bias: bool,
    },
    ShiftAndFill {
        periods: i64,
    },
    DropNans,
    #[cfg(feature = "round_series")]
    Clip {
        min: Option<AnyValue<'static>>,
        max: Option<AnyValue<'static>>,
    },
    ListExpr(ListFunction),
    #[cfg(feature = "dtype-array")]
    ArrayExpr(ArrayFunction),
    #[cfg(feature = "dtype-struct")]
    StructExpr(StructFunction),
    #[cfg(feature = "top_k")]
    TopK {
        k: usize,
        descending: bool,
    },
    Shift(i64),
    Cumcount {
        reverse: bool,
    },
    Cumsum {
        reverse: bool,
    },
    Cumprod {
        reverse: bool,
    },
    Cummin {
        reverse: bool,
    },
    Cummax {
        reverse: bool,
    },
    Reverse,
    Boolean(BooleanFunction),
    #[cfg(feature = "approx_unique")]
    ApproxUnique,
    #[cfg(feature = "dtype-categorical")]
    Categorical(CategoricalFunction),
    Coalesce,
    ShrinkType,
    #[cfg(feature = "diff")]
    Diff(i64, NullBehavior),
    #[cfg(feature = "interpolate")]
    Interpolate(InterpolationMethod),
    #[cfg(feature = "log")]
    Entropy {
        base: f64,
        normalize: bool,
    },
    #[cfg(feature = "log")]
    Log {
        base: f64,
    },
    #[cfg(feature = "log")]
    Log1p,
    #[cfg(feature = "log")]
    Exp,
    Unique(bool),
    #[cfg(feature = "round_series")]
    Round {
        decimals: u32,
    },
    #[cfg(feature = "round_series")]
    Floor,
    #[cfg(feature = "round_series")]
    Ceil,
    UpperBound,
    LowerBound,
    #[cfg(feature = "fused")]
    Fused(fused::FusedOperator),
    ConcatExpr(bool),
    Correlation {
        method: correlation::CorrelationMethod,
        ddof: u8,
    },
    ToPhysical,
}

impl Display for FunctionExpr {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use FunctionExpr::*;

        let s = match self {
            #[cfg(feature = "abs")]
            Abs => "abs",
            NullCount => "null_count",
            Pow => "pow",
            #[cfg(feature = "row_hash")]
            Hash(_, _, _, _) => "hash",
            #[cfg(feature = "arg_where")]
            ArgWhere => "arg_where",
            #[cfg(feature = "search_sorted")]
            SearchSorted(_) => "search_sorted",
            #[cfg(feature = "strings")]
            StringExpr(s) => return write!(f, "{s}"),
            BinaryExpr(b) => return write!(f, "{b}"),
            #[cfg(feature = "temporal")]
            TemporalExpr(fun) => return write!(f, "{fun}"),
            #[cfg(feature = "date_offset")]
            DateOffset(_) => "dt.offset_by",
            #[cfg(feature = "trigonometry")]
            Trigonometry(func) => return write!(f, "{func}"),
            #[cfg(feature = "sign")]
            Sign => "sign",
            FillNull { .. } => "fill_null",
            #[cfg(all(feature = "rolling_window", feature = "moment"))]
            RollingSkew { .. } => "rolling_skew",
            ShiftAndFill { .. } => "shift_and_fill",
            DropNans => "drop_nans",
            #[cfg(feature = "round_series")]
            Clip { min, max } => match (min, max) {
                (Some(_), Some(_)) => "clip",
                (None, Some(_)) => "clip_max",
                (Some(_), None) => "clip_min",
                _ => unreachable!(),
            },
            ListExpr(func) => return write!(f, "{func}"),
            #[cfg(feature = "dtype-struct")]
            StructExpr(func) => return write!(f, "{func}"),
            #[cfg(feature = "top_k")]
            TopK { .. } => "top_k",
            Shift(_) => "shift",
            Cumcount { .. } => "cumcount",
            Cumsum { .. } => "cumsum",
            Cumprod { .. } => "cumprod",
            Cummin { .. } => "cummin",
            Cummax { .. } => "cummax",
            Reverse => "reverse",
            Boolean(func) => return write!(f, "{func}"),
            #[cfg(feature = "approx_unique")]
            ApproxUnique => "approx_unique",
            #[cfg(feature = "dtype-categorical")]
            Categorical(func) => return write!(f, "{func}"),
            Coalesce => "coalesce",
            ShrinkType => "shrink_dtype",
            #[cfg(feature = "diff")]
            Diff(_, _) => "diff",
            #[cfg(feature = "interpolate")]
            Interpolate(_) => "interpolate",
            #[cfg(feature = "log")]
            Entropy { .. } => "entropy",
            #[cfg(feature = "log")]
            Log { .. } => "log",
            #[cfg(feature = "log")]
            Log1p => "log1p",
            #[cfg(feature = "log")]
            Exp => "exp",
            Unique(stable) => {
                if *stable {
                    "unique_stable"
                } else {
                    "unique"
                }
            }
            #[cfg(feature = "round_series")]
            Round { .. } => "round",
            #[cfg(feature = "round_series")]
            Floor => "floor",
            #[cfg(feature = "round_series")]
            Ceil => "ceil",
            UpperBound => "upper_bound",
            LowerBound => "lower_bound",
            #[cfg(feature = "fused")]
            Fused(fused) => return Display::fmt(fused, f),
            #[cfg(feature = "dtype-array")]
            ArrayExpr(af) => return Display::fmt(af, f),
            ConcatExpr(_) => "concat_expr",
            Correlation { method, .. } => return Display::fmt(method, f),
            ToPhysical => "to_physical",
        };
        write!(f, "{s}")
    }
}

#[macro_export]
macro_rules! wrap {
    ($e:expr) => {
        SpecialEq::new(Arc::new($e))
    };
}

// Fn(&[Series], args)
// all expression arguments are in the slice.
// the first element is the root expression.
macro_rules! map_as_slice {
    ($func:path) => {{
        let f = move |s: &mut [Series]| {
            $func(s).map(Some)
        };

        SpecialEq::new(Arc::new(f))
    }};

    ($func:path, $($args:expr),*) => {{
        let f = move |s: &mut [Series]| {
            $func(s, $($args),*).map(Some)
        };

        SpecialEq::new(Arc::new(f))
    }};
}

// FnOnce(Series)
// FnOnce(Series, args)
#[macro_export]
macro_rules! map_owned {
    ($func:path) => {{
        let f = move |s: &mut [Series]| {
            let s = std::mem::take(&mut s[0]);
            $func(s).map(Some)
        };

        SpecialEq::new(Arc::new(f))
    }};

    ($func:path, $($args:expr),*) => {{
        let f = move |s: &mut [Series]| {
            let s = std::mem::take(&mut s[0]);
            $func(s, $($args),*).map(Some)
        };

        SpecialEq::new(Arc::new(f))
    }};
}

// Fn(&Series, args)
#[macro_export]
macro_rules! map {
    ($func:path) => {{
        let f = move |s: &mut [Series]| {
            let s = &s[0];
            $func(s).map(Some)
        };

        SpecialEq::new(Arc::new(f))
    }};

    ($func:path, $($args:expr),*) => {{
        let f = move |s: &mut [Series]| {
            let s = &s[0];
            $func(s, $($args),*).map(Some)
        };

        SpecialEq::new(Arc::new(f))
    }};
}

impl From<FunctionExpr> for SpecialEq<Arc<dyn SeriesUdf>> {
    fn from(func: FunctionExpr) -> Self {
        use FunctionExpr::*;
        match func {
            #[cfg(feature = "abs")]
            Abs => map!(abs::abs),
            NullCount => {
                let f = |s: &mut [Series]| {
                    let s = &s[0];
                    Ok(Some(Series::new(s.name(), [s.null_count() as IdxSize])))
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
            #[cfg(feature = "arg_where")]
            ArgWhere => {
                wrap!(arg_where::arg_where)
            }
            #[cfg(feature = "search_sorted")]
            SearchSorted(side) => {
                map_as_slice!(search_sorted::search_sorted_impl, side)
            }
            #[cfg(feature = "strings")]
            StringExpr(s) => s.into(),
            BinaryExpr(s) => s.into(),
            #[cfg(feature = "temporal")]
            TemporalExpr(func) => func.into(),

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

            #[cfg(all(feature = "rolling_window", feature = "moment"))]
            RollingSkew { window_size, bias } => {
                map!(rolling::rolling_skew, window_size, bias)
            }
            ShiftAndFill { periods } => {
                map_as_slice!(shift_and_fill::shift_and_fill, periods)
            }
            DropNans => map_owned!(nan::drop_nans),
            #[cfg(feature = "round_series")]
            Clip { min, max } => {
                map_owned!(clip::clip, min.clone(), max.clone())
            }
            ListExpr(lf) => {
                use ListFunction::*;
                match lf {
                    Concat => wrap!(list::concat),
                    #[cfg(feature = "is_in")]
                    Contains => wrap!(list::contains),
                    Slice => wrap!(list::slice),
                    Get => wrap!(list::get),
                    #[cfg(feature = "list_take")]
                    Take(null_ob_oob) => map_as_slice!(list::take, null_ob_oob),
                    #[cfg(feature = "list_count")]
                    CountMatch => map_as_slice!(list::count_match),
                    Sum => map!(list::sum),
                }
            }
            #[cfg(feature = "dtype-array")]
            ArrayExpr(lf) => {
                use ArrayFunction::*;
                match lf {
                    Min => map!(array::min),
                    Max => map!(array::max),
                    Sum => map!(array::sum),
                    Unique(stable) => map!(array::unique, stable),
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
            TopK { k, descending } => {
                map!(top_k, k, descending)
            }
            Shift(periods) => map!(dispatch::shift, periods),
            Cumcount { reverse } => map!(cum::cumcount, reverse),
            Cumsum { reverse } => map!(cum::cumsum, reverse),
            Cumprod { reverse } => map!(cum::cumprod, reverse),
            Cummin { reverse } => map!(cum::cummin, reverse),
            Cummax { reverse } => map!(cum::cummax, reverse),
            Reverse => map!(dispatch::reverse),
            Boolean(func) => func.into(),
            #[cfg(feature = "approx_unique")]
            ApproxUnique => map!(dispatch::approx_unique),
            #[cfg(feature = "dtype-categorical")]
            Categorical(func) => func.into(),
            Coalesce => map_as_slice!(fill_null::coalesce),
            ShrinkType => map_owned!(shrink_type::shrink),
            #[cfg(feature = "diff")]
            Diff(n, null_behavior) => map!(dispatch::diff, n, null_behavior),
            #[cfg(feature = "interpolate")]
            Interpolate(method) => {
                map!(dispatch::interpolate, method)
            }
            #[cfg(feature = "log")]
            Entropy { base, normalize } => map!(log::entropy, base, normalize),
            #[cfg(feature = "log")]
            Log { base } => map!(log::log, base),
            #[cfg(feature = "log")]
            Log1p => map!(log::log1p),
            #[cfg(feature = "log")]
            Exp => map!(log::exp),
            Unique(stable) => map!(unique::unique, stable),
            #[cfg(feature = "round_series")]
            Round { decimals } => map!(round::round, decimals),
            #[cfg(feature = "round_series")]
            Floor => map!(round::floor),
            #[cfg(feature = "round_series")]
            Ceil => map!(round::ceil),
            UpperBound => map!(bounds::upper_bound),
            LowerBound => map!(bounds::lower_bound),
            #[cfg(feature = "fused")]
            Fused(op) => map_as_slice!(fused::fused, op),
            ConcatExpr(rechunk) => map_as_slice!(concat::concat_expr, rechunk),
            Correlation { method, ddof } => map_as_slice!(correlation::corr, ddof, method),
            ToPhysical => map!(dispatch::to_physical),
        }
    }
}

#[cfg(feature = "strings")]
impl From<StringFunction> for SpecialEq<Arc<dyn SeriesUdf>> {
    fn from(func: StringFunction) -> Self {
        use StringFunction::*;
        match func {
            #[cfg(feature = "regex")]
            Contains { literal, strict } => map_as_slice!(strings::contains, literal, strict),
            EndsWith { .. } => map_as_slice!(strings::ends_with),
            StartsWith { .. } => map_as_slice!(strings::starts_with),
            Extract { pat, group_index } => {
                map!(strings::extract, &pat, group_index)
            }
            ExtractAll => {
                map_as_slice!(strings::extract_all)
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
            Strptime(dtype, options) => {
                map!(strings::strptime, dtype.clone(), &options)
            }
            #[cfg(feature = "concat_str")]
            ConcatVertical(delimiter) => map!(strings::concat, &delimiter),
            #[cfg(feature = "concat_str")]
            ConcatHorizontal(delimiter) => map_as_slice!(strings::concat_hor, &delimiter),
            #[cfg(feature = "regex")]
            Replace { n, literal } => map_as_slice!(strings::replace, literal, n),
            Uppercase => map!(strings::uppercase),
            Lowercase => map!(strings::lowercase),
            #[cfg(feature = "nightly")]
            Titlecase => map!(strings::titlecase),
            Strip(matches) => map!(strings::strip, matches.as_deref()),
            LStrip(matches) => map!(strings::lstrip, matches.as_deref()),
            RStrip(matches) => map!(strings::rstrip, matches.as_deref()),
            #[cfg(feature = "string_from_radix")]
            FromRadix(radix, strict) => map!(strings::from_radix, radix, strict),
            Slice(start, length) => map!(strings::str_slice, start, length),
            Explode => map!(strings::explode),
            #[cfg(feature = "dtype-decimal")]
            ToDecimal(infer_len) => map!(strings::to_decimal, infer_len),
            #[cfg(feature = "extract_jsonpath")]
            JsonExtract {
                dtype,
                infer_schema_len,
            } => map!(strings::json_extract, dtype.clone(), infer_schema_len),
        }
    }
}

impl From<BinaryFunction> for SpecialEq<Arc<dyn SeriesUdf>> {
    fn from(func: BinaryFunction) -> Self {
        use BinaryFunction::*;
        match func {
            Contains { pat, literal } => {
                map!(binary::contains, &pat, literal)
            }
            EndsWith(sub) => {
                map!(binary::ends_with, &sub)
            }
            StartsWith(sub) => {
                map!(binary::starts_with, &sub)
            }
        }
    }
}

#[cfg(feature = "temporal")]
#[allow(deprecated)] // tz_localize
impl From<TemporalFunction> for SpecialEq<Arc<dyn SeriesUdf>> {
    fn from(func: TemporalFunction) -> Self {
        use TemporalFunction::*;
        match func {
            Year => map!(datetime::year),
            IsLeapYear => map!(datetime::is_leap_year),
            IsoYear => map!(datetime::iso_year),
            Month => map!(datetime::month),
            Quarter => map!(datetime::quarter),
            Week => map!(datetime::week),
            WeekDay => map!(datetime::weekday),
            Day => map!(datetime::day),
            OrdinalDay => map!(datetime::ordinal_day),
            Time => map!(datetime::time),
            Date => map!(datetime::date),
            Datetime => map!(datetime::datetime),
            Hour => map!(datetime::hour),
            Minute => map!(datetime::minute),
            Second => map!(datetime::second),
            Millisecond => map!(datetime::millisecond),
            Microsecond => map!(datetime::microsecond),
            Nanosecond => map!(datetime::nanosecond),
            TimeStamp(tu) => map!(datetime::timestamp, tu),
            Truncate(every, offset) => map!(datetime::truncate, &every, &offset),
            #[cfg(feature = "date_offset")]
            MonthStart => map!(datetime::month_start),
            #[cfg(feature = "date_offset")]
            MonthEnd => map!(datetime::month_end),
            Round(every, offset) => map!(datetime::round, &every, &offset),
            #[cfg(feature = "timezones")]
            CastTimezone(tz, use_earliest) => {
                map!(datetime::replace_timezone, tz.as_deref(), use_earliest)
            }
            #[cfg(feature = "timezones")]
            TzLocalize(tz) => map!(datetime::tz_localize, &tz),
            Combine(tu) => map_as_slice!(temporal::combine, tu),
            DateRange { every, closed, tz } => {
                map_as_slice!(
                    temporal::temporal_range_dispatch,
                    "date",
                    every,
                    closed,
                    tz.clone()
                )
            }
            TimeRange { every, closed } => {
                map_as_slice!(
                    temporal::temporal_range_dispatch,
                    "time",
                    every,
                    closed,
                    None
                )
            }
        }
    }
}
