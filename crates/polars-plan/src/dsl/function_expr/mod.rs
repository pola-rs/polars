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
#[cfg(feature = "random")]
mod random;
#[cfg(feature = "range")]
mod range;
#[cfg(all(feature = "rolling_window"))]
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
use std::hash::{Hash, Hasher};

#[cfg(feature = "dtype-array")]
pub(super) use array::ArrayFunction;
pub(crate) use correlation::CorrelationMethod;
#[cfg(feature = "fused")]
pub(crate) use fused::FusedOperator;
pub(super) use list::ListFunction;
use polars_core::prelude::*;
#[cfg(feature = "cutqcut")]
use polars_ops::prelude::{cut, qcut};
#[cfg(feature = "rle")]
use polars_ops::prelude::{rle, rle_id};
#[cfg(feature = "random")]
pub(crate) use random::RandomMethod;
use schema::FieldsMapper;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

pub(crate) use self::binary::BinaryFunction;
pub use self::boolean::BooleanFunction;
#[cfg(feature = "dtype-categorical")]
pub(crate) use self::cat::CategoricalFunction;
#[cfg(feature = "temporal")]
pub(super) use self::datetime::TemporalFunction;
pub(super) use self::pow::PowFunction;
#[cfg(feature = "range")]
pub(super) use self::range::RangeFunction;
#[cfg(all(feature = "rolling_window"))]
pub(super) use self::rolling::RollingFunction;
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
    Pow(PowFunction),
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
    #[cfg(feature = "range")]
    Range(RangeFunction),
    #[cfg(feature = "date_offset")]
    DateOffset,
    #[cfg(feature = "trigonometry")]
    Trigonometry(TrigonometricFunction),
    #[cfg(feature = "trigonometry")]
    Atan2,
    #[cfg(feature = "sign")]
    Sign,
    FillNull {
        super_type: DataType,
    },
    #[cfg(all(feature = "rolling_window"))]
    RollingExpr(RollingFunction),
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
    ApproxNUnique,
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
    #[cfg(feature = "cutqcut")]
    Cut {
        breaks: Vec<f64>,
        labels: Option<Vec<String>>,
        left_closed: bool,
        include_breaks: bool,
    },
    #[cfg(feature = "cutqcut")]
    QCut {
        probs: Vec<f64>,
        labels: Option<Vec<String>>,
        left_closed: bool,
        allow_duplicates: bool,
        include_breaks: bool,
    },
    #[cfg(feature = "rle")]
    RLE,
    #[cfg(feature = "rle")]
    RLEID,
    ToPhysical,
    #[cfg(feature = "random")]
    Random {
        method: random::RandomMethod,
        seed: Option<u64>,
    },
    SetSortedFlag(IsSorted),
}

impl Hash for FunctionExpr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            FunctionExpr::BinaryExpr(f) => f.hash(state),
            FunctionExpr::Boolean(f) => f.hash(state),
            #[cfg(feature = "strings")]
            FunctionExpr::StringExpr(f) => f.hash(state),
            #[cfg(feature = "random")]
            FunctionExpr::Random { method, .. } => method.hash(state),
            #[cfg(feature = "range")]
            FunctionExpr::Range(f) => f.hash(state),
            #[cfg(feature = "temporal")]
            FunctionExpr::TemporalExpr(f) => f.hash(state),
            #[cfg(feature = "trigonometry")]
            FunctionExpr::Trigonometry(f) => f.hash(state),
            #[cfg(feature = "fused")]
            FunctionExpr::Fused(f) => f.hash(state),
            #[cfg(feature = "interpolate")]
            FunctionExpr::Interpolate(f) => f.hash(state),
            #[cfg(feature = "dtype-categorical")]
            FunctionExpr::Categorical(f) => f.hash(state),
            _ => {},
        }
    }
}

impl Display for FunctionExpr {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use FunctionExpr::*;

        let s = match self {
            #[cfg(feature = "abs")]
            Abs => "abs",
            NullCount => "null_count",
            Pow(func) => return write!(f, "{func}"),
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
            #[cfg(feature = "range")]
            Range(func) => return write!(f, "{func}"),
            #[cfg(feature = "date_offset")]
            DateOffset => "dt.offset_by",
            #[cfg(feature = "trigonometry")]
            Trigonometry(func) => return write!(f, "{func}"),
            #[cfg(feature = "trigonometry")]
            Atan2 => return write!(f, "arctan2"),
            #[cfg(feature = "sign")]
            Sign => "sign",
            FillNull { .. } => "fill_null",
            #[cfg(all(feature = "rolling_window"))]
            RollingExpr(func, ..) => return write!(f, "{func}"),
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
            ApproxNUnique => "approx_n_unique",
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
            },
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
            #[cfg(feature = "cutqcut")]
            Cut { .. } => "cut",
            #[cfg(feature = "cutqcut")]
            QCut { .. } => "qcut",
            #[cfg(feature = "rle")]
            RLE => "rle",
            #[cfg(feature = "rle")]
            RLEID => "rle_id",
            ToPhysical => "to_physical",
            #[cfg(feature = "random")]
            Random { method, .. } => method.into(),
            SetSortedFlag(_) => "set_sorted",
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
#[macro_export]
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
            },
            Pow(func) => match func {
                PowFunction::Generic => wrap!(pow::pow),
                PowFunction::Sqrt => map!(pow::sqrt),
                PowFunction::Cbrt => map!(pow::cbrt),
            },
            #[cfg(feature = "row_hash")]
            Hash(k0, k1, k2, k3) => {
                map!(row_hash::row_hash, k0, k1, k2, k3)
            },
            #[cfg(feature = "arg_where")]
            ArgWhere => {
                wrap!(arg_where::arg_where)
            },
            #[cfg(feature = "search_sorted")]
            SearchSorted(side) => {
                map_as_slice!(search_sorted::search_sorted_impl, side)
            },
            #[cfg(feature = "strings")]
            StringExpr(s) => s.into(),
            BinaryExpr(s) => s.into(),
            #[cfg(feature = "temporal")]
            TemporalExpr(func) => func.into(),
            #[cfg(feature = "range")]
            Range(func) => func.into(),

            #[cfg(feature = "date_offset")]
            DateOffset => {
                map_as_slice!(temporal::date_offset)
            },

            #[cfg(feature = "trigonometry")]
            Trigonometry(trig_function) => {
                map!(trigonometry::apply_trigonometric_function, trig_function)
            },
            #[cfg(feature = "trigonometry")]
            Atan2 => {
                wrap!(trigonometry::apply_arctan2)
            },

            #[cfg(feature = "sign")]
            Sign => {
                map!(sign::sign)
            },
            FillNull { super_type } => {
                map_as_slice!(fill_null::fill_null, &super_type)
            },
            #[cfg(all(feature = "rolling_window"))]
            RollingExpr(f) => {
                use RollingFunction::*;
                match f {
                    Min(options) => map!(rolling::rolling_min, options.clone()),
                    MinBy(options) => map_as_slice!(rolling::rolling_min_by, options.clone()),
                    Max(options) => map!(rolling::rolling_max, options.clone()),
                    MaxBy(options) => map_as_slice!(rolling::rolling_max_by, options.clone()),
                    Mean(options) => map!(rolling::rolling_mean, options.clone()),
                    MeanBy(options) => map_as_slice!(rolling::rolling_mean_by, options.clone()),
                    Sum(options) => map!(rolling::rolling_sum, options.clone()),
                    SumBy(options) => map_as_slice!(rolling::rolling_sum_by, options.clone()),
                    Median(options) => map!(rolling::rolling_median, options.clone()),
                    MedianBy(options) => map_as_slice!(rolling::rolling_median_by, options.clone()),
                    Quantile(options) => map!(rolling::rolling_quantile, options.clone()),
                    QuantileBy(options) => {
                        map_as_slice!(rolling::rolling_quantile_by, options.clone())
                    },
                    Var(options) => map!(rolling::rolling_var, options.clone()),
                    VarBy(options) => map_as_slice!(rolling::rolling_var_by, options.clone()),
                    Std(options) => map!(rolling::rolling_std, options.clone()),
                    StdBy(options) => map_as_slice!(rolling::rolling_std_by, options.clone()),
                    #[cfg(all(feature = "rolling_window", feature = "moment"))]
                    Skew(window_size, bias) => map!(rolling::rolling_skew, window_size, bias),
                }
            },
            ShiftAndFill { periods } => {
                map_as_slice!(shift_and_fill::shift_and_fill, periods)
            },
            DropNans => map_owned!(nan::drop_nans),
            #[cfg(feature = "round_series")]
            Clip { min, max } => {
                map_owned!(clip::clip, min.clone(), max.clone())
            },
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
                    #[cfg(feature = "list_sets")]
                    SetOperation(s) => map_as_slice!(list::set_operation, s),
                    #[cfg(feature = "list_any_all")]
                    Any => map!(list::lst_any),
                    #[cfg(feature = "list_any_all")]
                    All => map!(list::lst_all),
                }
            },
            #[cfg(feature = "dtype-array")]
            ArrayExpr(lf) => {
                use ArrayFunction::*;
                match lf {
                    Min => map!(array::min),
                    Max => map!(array::max),
                    Sum => map!(array::sum),
                    Unique(stable) => map!(array::unique, stable),
                }
            },
            #[cfg(feature = "dtype-struct")]
            StructExpr(sf) => {
                use StructFunction::*;
                match sf {
                    FieldByIndex(index) => map!(struct_::get_by_index, index),
                    FieldByName(name) => map!(struct_::get_by_name, name.clone()),
                }
            },
            #[cfg(feature = "top_k")]
            TopK { k, descending } => {
                map!(top_k, k, descending)
            },
            Shift(periods) => map!(dispatch::shift, periods),
            Cumcount { reverse } => map!(cum::cumcount, reverse),
            Cumsum { reverse } => map!(cum::cumsum, reverse),
            Cumprod { reverse } => map!(cum::cumprod, reverse),
            Cummin { reverse } => map!(cum::cummin, reverse),
            Cummax { reverse } => map!(cum::cummax, reverse),
            Reverse => map!(dispatch::reverse),
            Boolean(func) => func.into(),
            #[cfg(feature = "approx_unique")]
            ApproxNUnique => map!(dispatch::approx_n_unique),
            #[cfg(feature = "dtype-categorical")]
            Categorical(func) => func.into(),
            Coalesce => map_as_slice!(fill_null::coalesce),
            ShrinkType => map_owned!(shrink_type::shrink),
            #[cfg(feature = "diff")]
            Diff(n, null_behavior) => map!(dispatch::diff, n, null_behavior),
            #[cfg(feature = "interpolate")]
            Interpolate(method) => {
                map!(dispatch::interpolate, method)
            },
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
            #[cfg(feature = "cutqcut")]
            Cut {
                breaks,
                labels,
                left_closed,
                include_breaks,
            } => map!(
                cut,
                breaks.clone(),
                labels.clone(),
                left_closed,
                include_breaks
            ),
            #[cfg(feature = "cutqcut")]
            QCut {
                probs,
                labels,
                left_closed,
                allow_duplicates,
                include_breaks,
            } => map!(
                qcut,
                probs.clone(),
                labels.clone(),
                left_closed,
                allow_duplicates,
                include_breaks
            ),
            #[cfg(feature = "rle")]
            RLE => map!(rle),
            #[cfg(feature = "rle")]
            RLEID => map!(rle_id),
            ToPhysical => map!(dispatch::to_physical),
            #[cfg(feature = "random")]
            Random { method, seed } => map!(random::random, method, seed),
            SetSortedFlag(sorted) => map!(dispatch::set_sorted_flag, sorted),
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
            CountMatch => {
                map_as_slice!(strings::count_match)
            },
            EndsWith { .. } => map_as_slice!(strings::ends_with),
            StartsWith { .. } => map_as_slice!(strings::starts_with),
            Extract { pat, group_index } => {
                map!(strings::extract, &pat, group_index)
            },
            ExtractAll => {
                map_as_slice!(strings::extract_all)
            },
            #[cfg(feature = "extract_groups")]
            ExtractGroups { pat, dtype } => {
                map!(strings::extract_groups, &pat, &dtype)
            },
            NChars => map!(strings::n_chars),
            Length => map!(strings::lengths),
            #[cfg(feature = "string_justify")]
            Zfill(alignment) => {
                map!(strings::zfill, alignment)
            },
            #[cfg(feature = "string_justify")]
            LJust { width, fillchar } => {
                map!(strings::ljust, width, fillchar)
            },
            #[cfg(feature = "string_justify")]
            RJust { width, fillchar } => {
                map!(strings::rjust, width, fillchar)
            },
            #[cfg(feature = "temporal")]
            Strptime(dtype, options) => {
                map_as_slice!(strings::strptime, dtype.clone(), &options)
            },
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
            },
            EndsWith(sub) => {
                map!(binary::ends_with, &sub)
            },
            StartsWith(sub) => {
                map!(binary::starts_with, &sub)
            },
        }
    }
}

#[cfg(feature = "temporal")]
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
            Truncate(truncate_options) => {
                map_as_slice!(datetime::truncate, &truncate_options)
            },
            #[cfg(feature = "date_offset")]
            MonthStart => map!(datetime::month_start),
            #[cfg(feature = "date_offset")]
            MonthEnd => map!(datetime::month_end),
            #[cfg(feature = "timezones")]
            BaseUtcOffset => map!(datetime::base_utc_offset),
            #[cfg(feature = "timezones")]
            DSTOffset => map!(datetime::dst_offset),
            Round(every, offset) => map!(datetime::round, &every, &offset),
            #[cfg(feature = "timezones")]
            ReplaceTimeZone(tz) => {
                map_as_slice!(dispatch::replace_time_zone, tz.as_deref())
            },
            Combine(tu) => map_as_slice!(temporal::combine, tu),
            DatetimeFunction {
                time_unit,
                time_zone,
            } => {
                map_as_slice!(temporal::datetime, &time_unit, time_zone.as_deref())
            },
        }
    }
}
