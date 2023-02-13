#[cfg(feature = "arg_where")]
mod arg_where;
#[cfg(feature = "dtype-binary")]
mod binary;
#[cfg(feature = "round_series")]
mod clip;
#[cfg(feature = "temporal")]
mod datetime;
mod dispatch;
mod fill_null;
#[cfg(feature = "is_in")]
mod is_in;
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

use std::fmt::{Display, Formatter};

pub(super) use list::ListFunction;
use polars_core::prelude::*;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "dtype-binary")]
pub(crate) use self::binary::BinaryFunction;
#[cfg(feature = "temporal")]
pub(super) use self::datetime::TemporalFunction;
pub(super) use self::nan::NanFunction;
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
    NullCount,
    Pow,
    #[cfg(feature = "row_hash")]
    Hash(u64, u64, u64, u64),
    #[cfg(feature = "is_in")]
    IsIn,
    #[cfg(feature = "arg_where")]
    ArgWhere,
    #[cfg(feature = "search_sorted")]
    SearchSorted(SearchSortedSide),
    #[cfg(feature = "strings")]
    StringExpr(StringFunction),
    #[cfg(feature = "dtype-binary")]
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
    Nan(NanFunction),
    #[cfg(feature = "round_series")]
    Clip {
        min: Option<AnyValue<'static>>,
        max: Option<AnyValue<'static>>,
    },
    ListExpr(ListFunction),
    #[cfg(feature = "dtype-struct")]
    StructExpr(StructFunction),
    #[cfg(feature = "top_k")]
    TopK {
        k: usize,
        reverse: bool,
    },
    Shift(i64),
    Reverse,
    IsNull,
    IsNotNull,
    Not,
    IsUnique,
    IsDuplicated,
    Coalesce,
    ShrinkType,
    #[cfg(feature = "diff")]
    Diff(usize, NullBehavior),
    #[cfg(feature = "interpolate")]
    Interpolate(InterpolationMethod),
    #[cfg(feature = "dot_product")]
    Dot,
    #[cfg(feature = "log")]
    Entropy {
        base: f64,
        normalize: bool,
    },
}

impl Display for FunctionExpr {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use FunctionExpr::*;

        let s = match self {
            NullCount => "null_count",
            Pow => "pow",
            #[cfg(feature = "row_hash")]
            Hash(_, _, _, _) => "hash",
            #[cfg(feature = "is_in")]
            IsIn => "is_in",
            #[cfg(feature = "arg_where")]
            ArgWhere => "arg_where",
            #[cfg(feature = "search_sorted")]
            SearchSorted(_) => "search_sorted",
            #[cfg(feature = "strings")]
            StringExpr(s) => return write!(f, "{s}"),
            #[cfg(feature = "dtype-binary")]
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
            Nan(func) => match func {
                NanFunction::IsNan => "is_nan",
                NanFunction::IsNotNan => "is_not_nan",
                NanFunction::DropNans => "drop_nans",
            },
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
            Reverse => "reverse",
            Not => "is_not",
            IsNull => "is_null",
            IsNotNull => "is_not_null",
            IsUnique => "is_unique",
            IsDuplicated => "is_duplicated",
            Coalesce => "coalesce",
            ShrinkType => "shrink_dtype",
            #[cfg(feature = "diff")]
            Diff(_, _) => "diff",
            #[cfg(feature = "interpolate")]
            Interpolate(_) => "interpolate",
            #[cfg(feature = "dot_product")]
            Dot => "dot",
            #[cfg(feature = "log")]
            Entropy { .. } => "entropy",
        };
        write!(f, "{s}")
    }
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
#[macro_export(super)]
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
#[macro_export(super)]
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
            #[cfg(feature = "is_in")]
            IsIn => {
                wrap!(is_in::is_in)
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
            #[cfg(feature = "dtype-binary")]
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
            Nan(n) => n.into(),
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
            Shift(periods) => map!(dispatch::shift, periods),
            Reverse => map!(dispatch::reverse),
            IsNull => map!(dispatch::is_null),
            IsNotNull => map!(dispatch::is_not_null),
            Not => map!(dispatch::is_not),
            IsUnique => map!(dispatch::is_unique),
            IsDuplicated => map!(dispatch::is_duplicated),
            Coalesce => map_as_slice!(fill_null::coalesce),
            ShrinkType => map_owned!(shrink_type::shrink),
            #[cfg(feature = "diff")]
            Diff(n, null_behavior) => map!(dispatch::diff, n, null_behavior),
            #[cfg(feature = "interpolate")]
            Interpolate(method) => {
                map!(dispatch::interpolate, method)
            }
            #[cfg(feature = "dot_product")]
            Dot => {
                map_as_slice!(dispatch::dot_impl)
            }
            #[cfg(feature = "log")]
            Entropy { base, normalize } => map!(dispatch::entropy, base, normalize),
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
            Strptime(options) => {
                map!(strings::strptime, &options)
            }
            #[cfg(feature = "concat_str")]
            ConcatVertical(delimiter) => map!(strings::concat, &delimiter),
            #[cfg(feature = "concat_str")]
            ConcatHorizontal(delimiter) => map_as_slice!(strings::concat_hor, &delimiter),
            #[cfg(feature = "regex")]
            Replace { all, literal } => map_as_slice!(strings::replace, literal, all),
            Uppercase => map!(strings::uppercase),
            Lowercase => map!(strings::lowercase),
            Strip(matches) => map!(strings::strip, matches.as_deref()),
            LStrip(matches) => map!(strings::lstrip, matches.as_deref()),
            RStrip(matches) => map!(strings::rstrip, matches.as_deref()),
            #[cfg(feature = "string_from_radix")]
            FromRadix(matches) => map!(strings::from_radix, matches),
        }
    }
}

#[cfg(feature = "dtype-binary")]
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
            IsoYear => map!(datetime::iso_year),
            Month => map!(datetime::month),
            Quarter => map!(datetime::quarter),
            Week => map!(datetime::week),
            WeekDay => map!(datetime::weekday),
            Day => map!(datetime::day),
            OrdinalDay => map!(datetime::ordinal_day),
            Hour => map!(datetime::hour),
            Minute => map!(datetime::minute),
            Second => map!(datetime::second),
            Millisecond => map!(datetime::millisecond),
            Microsecond => map!(datetime::microsecond),
            Nanosecond => map!(datetime::nanosecond),
            TimeStamp(tu) => map!(datetime::timestamp, tu),
            Truncate(every, offset) => map!(datetime::truncate, &every, &offset),
            Round(every, offset) => map!(datetime::round, &every, &offset),
            #[cfg(feature = "timezones")]
            CastTimezone(tz) => map!(datetime::replace_timezone, tz.as_deref()),
            #[cfg(feature = "timezones")]
            TzLocalize(tz) => map!(datetime::tz_localize, &tz),
            Combine(tu) => map_as_slice!(temporal::combine, tu),
            DateRange {
                name,
                every,
                closed,
                tz,
            } => {
                map_as_slice!(
                    datetime::date_range_dispatch,
                    name.as_ref(),
                    every,
                    closed,
                    tz.clone()
                )
            }
        }
    }
}
