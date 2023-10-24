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
mod coerce;
mod concat;
mod correlation;
#[cfg(feature = "cum_agg")]
mod cum;
#[cfg(feature = "temporal")]
mod datetime;
mod dispatch;
#[cfg(feature = "ewma")]
mod ewm;
mod fill_null;
#[cfg(feature = "fused")]
mod fused;
mod list;
#[cfg(feature = "log")]
mod log;
mod nan;
#[cfg(feature = "peaks")]
mod peaks;
#[cfg(feature = "ffi_plugin")]
mod plugin;
mod pow;
#[cfg(feature = "random")]
mod random;
#[cfg(feature = "range")]
mod range;
#[cfg(all(feature = "rolling_window", feature = "moment"))]
mod rolling;
#[cfg(feature = "round_series")]
mod round;
#[cfg(feature = "row_hash")]
mod row_hash;
pub(super) mod schema;
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
    #[cfg(all(feature = "rolling_window", feature = "moment"))]
    // if we add more, make a sub enum
    RollingSkew {
        window_size: usize,
        bias: bool,
    },
    ShiftAndFill,
    Shift,
    DropNans,
    DropNulls,
    #[cfg(feature = "mode")]
    Mode,
    #[cfg(feature = "moment")]
    Skew(bool),
    #[cfg(feature = "moment")]
    Kurtosis(bool, bool),
    ArgUnique,
    #[cfg(feature = "rank")]
    Rank {
        options: RankOptions,
        seed: Option<u64>,
    },
    #[cfg(feature = "round_series")]
    Clip {
        has_min: bool,
        has_max: bool,
    },
    ListExpr(ListFunction),
    #[cfg(feature = "dtype-array")]
    ArrayExpr(ArrayFunction),
    #[cfg(feature = "dtype-struct")]
    StructExpr(StructFunction),
    #[cfg(feature = "dtype-struct")]
    AsStruct,
    #[cfg(feature = "top_k")]
    TopK(bool),
    #[cfg(feature = "cum_agg")]
    Cumcount {
        reverse: bool,
    },
    #[cfg(feature = "cum_agg")]
    Cumsum {
        reverse: bool,
    },
    #[cfg(feature = "cum_agg")]
    Cumprod {
        reverse: bool,
    },
    #[cfg(feature = "cum_agg")]
    Cummin {
        reverse: bool,
    },
    #[cfg(feature = "cum_agg")]
    Cummax {
        reverse: bool,
    },
    Reverse,
    #[cfg(feature = "dtype-struct")]
    ValueCounts {
        sort: bool,
        parallel: bool,
    },
    #[cfg(feature = "unique_counts")]
    UniqueCounts,
    Boolean(BooleanFunction),
    #[cfg(feature = "approx_unique")]
    ApproxNUnique,
    #[cfg(feature = "dtype-categorical")]
    Categorical(CategoricalFunction),
    Coalesce,
    ShrinkType,
    #[cfg(feature = "diff")]
    Diff(i64, NullBehavior),
    #[cfg(feature = "pct_change")]
    PctChange,
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
    #[cfg(feature = "peaks")]
    PeakMin,
    #[cfg(feature = "peaks")]
    PeakMax,
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
    #[cfg(feature = "ffi_plugin")]
    /// Creating this node is unsafe
    /// This will lead to calls over FFI>
    FfiPlugin {
        /// Shared library.
        lib: Arc<str>,
        /// Identifier in the shared lib.
        symbol: Arc<str>,
        /// Pickle serialized keyword arguments.
        kwargs: Arc<[u8]>,
    },
    BackwardFill {
        limit: FillNullLimit,
    },
    ForwardFill {
        limit: FillNullLimit,
    },
    SumHorizontal,
    MaxHorizontal,
    MinHorizontal,
    #[cfg(feature = "ewma")]
    EwmMean {
        options: EWMOptions,
    },
    #[cfg(feature = "ewma")]
    EwmStd {
        options: EWMOptions,
    },
    #[cfg(feature = "ewma")]
    EwmVar {
        options: EWMOptions,
    },
}

impl Hash for FunctionExpr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            FunctionExpr::Pow(f) => f.hash(state),
            #[cfg(feature = "search_sorted")]
            FunctionExpr::SearchSorted(f) => f.hash(state),
            FunctionExpr::BinaryExpr(f) => f.hash(state),
            FunctionExpr::Boolean(f) => f.hash(state),
            #[cfg(feature = "strings")]
            FunctionExpr::StringExpr(f) => f.hash(state),
            FunctionExpr::ListExpr(f) => f.hash(state),
            #[cfg(feature = "dtype-array")]
            FunctionExpr::ArrayExpr(f) => f.hash(state),
            #[cfg(feature = "dtype-struct")]
            FunctionExpr::StructExpr(f) => f.hash(state),
            #[cfg(feature = "random")]
            FunctionExpr::Random { method, .. } => method.hash(state),
            FunctionExpr::Correlation { method, .. } => method.hash(state),
            #[cfg(feature = "range")]
            FunctionExpr::Range(f) => f.hash(state),
            #[cfg(feature = "temporal")]
            FunctionExpr::TemporalExpr(f) => f.hash(state),
            #[cfg(feature = "trigonometry")]
            FunctionExpr::Trigonometry(f) => f.hash(state),
            #[cfg(feature = "fused")]
            FunctionExpr::Fused(f) => f.hash(state),
            #[cfg(feature = "diff")]
            FunctionExpr::Diff(_, null_behavior) => null_behavior.hash(state),
            #[cfg(feature = "interpolate")]
            FunctionExpr::Interpolate(f) => f.hash(state),
            #[cfg(feature = "dtype-categorical")]
            FunctionExpr::Categorical(f) => f.hash(state),
            #[cfg(feature = "ffi_plugin")]
            FunctionExpr::FfiPlugin {
                lib,
                symbol,
                kwargs,
            } => {
                kwargs.hash(state);
                lib.hash(state);
                symbol.hash(state);
            },
            FunctionExpr::SumHorizontal
            | FunctionExpr::MaxHorizontal
            | FunctionExpr::MinHorizontal
            | FunctionExpr::DropNans
            | FunctionExpr::DropNulls
            | FunctionExpr::Reverse
            | FunctionExpr::ArgUnique
            | FunctionExpr::Shift
            | FunctionExpr::ShiftAndFill => {},
            #[cfg(feature = "mode")]
            FunctionExpr::Mode => {},
            #[cfg(feature = "abs")]
            FunctionExpr::Abs => {},
            FunctionExpr::NullCount => {},
            #[cfg(feature = "date_offset")]
            FunctionExpr::DateOffset => {},
            #[cfg(feature = "arg_where")]
            FunctionExpr::ArgWhere => {},
            #[cfg(feature = "trigonometry")]
            FunctionExpr::Atan2 => {},
            #[cfg(feature = "dtype-struct")]
            FunctionExpr::AsStruct => {},
            #[cfg(feature = "sign")]
            FunctionExpr::Sign => {},
            #[cfg(feature = "row_hash")]
            FunctionExpr::Hash(a, b, c, d) => (a, b, c, d).hash(state),
            FunctionExpr::FillNull { super_type } => super_type.hash(state),
            #[cfg(all(feature = "rolling_window", feature = "moment"))]
            FunctionExpr::RollingSkew { window_size, bias } => {
                window_size.hash(state);
                bias.hash(state);
            },
            #[cfg(feature = "moment")]
            FunctionExpr::Skew(a) => a.hash(state),
            #[cfg(feature = "moment")]
            FunctionExpr::Kurtosis(a, b) => {
                a.hash(state);
                b.hash(state);
            },
            #[cfg(feature = "rank")]
            FunctionExpr::Rank { options, seed } => {
                options.hash(state);
                seed.hash(state);
            },
            #[cfg(feature = "round_series")]
            FunctionExpr::Clip { has_min, has_max } => {
                has_min.hash(state);
                has_max.hash(state);
            },
            #[cfg(feature = "top_k")]
            FunctionExpr::TopK(a) => a.hash(state),
            #[cfg(feature = "cum_agg")]
            FunctionExpr::Cumcount { reverse } => reverse.hash(state),
            #[cfg(feature = "cum_agg")]
            FunctionExpr::Cumsum { reverse } => reverse.hash(state),
            #[cfg(feature = "cum_agg")]
            FunctionExpr::Cumprod { reverse } => reverse.hash(state),
            #[cfg(feature = "cum_agg")]
            FunctionExpr::Cummin { reverse } => reverse.hash(state),
            #[cfg(feature = "cum_agg")]
            FunctionExpr::Cummax { reverse } => reverse.hash(state),
            #[cfg(feature = "dtype-struct")]
            FunctionExpr::ValueCounts { sort, parallel } => {
                sort.hash(state);
                parallel.hash(state);
            },
            #[cfg(feature = "unique_counts")]
            FunctionExpr::UniqueCounts => {},
            #[cfg(feature = "approx_unique")]
            FunctionExpr::ApproxNUnique => {},
            FunctionExpr::Coalesce => {},
            FunctionExpr::ShrinkType => {},
            #[cfg(feature = "pct_change")]
            FunctionExpr::PctChange => {},
            #[cfg(feature = "log")]
            FunctionExpr::Entropy { base, normalize } => {
                base.to_bits().hash(state);
                normalize.hash(state);
            },
            #[cfg(feature = "log")]
            FunctionExpr::Log { base } => base.to_bits().hash(state),
            #[cfg(feature = "log")]
            FunctionExpr::Log1p => {},
            #[cfg(feature = "log")]
            FunctionExpr::Exp => {},
            FunctionExpr::Unique(a) => a.hash(state),
            #[cfg(feature = "round_series")]
            FunctionExpr::Round { decimals } => decimals.hash(state),
            #[cfg(feature = "round_series")]
            FunctionExpr::Floor => {},
            #[cfg(feature = "round_series")]
            FunctionExpr::Ceil => {},
            FunctionExpr::UpperBound => {},
            FunctionExpr::LowerBound => {},
            FunctionExpr::ConcatExpr(a) => a.hash(state),
            #[cfg(feature = "peaks")]
            FunctionExpr::PeakMin => {},
            #[cfg(feature = "peaks")]
            FunctionExpr::PeakMax => {},
            #[cfg(feature = "cutqcut")]
            FunctionExpr::Cut {
                breaks,
                labels,
                left_closed,
                include_breaks,
            } => {
                let slice = bytemuck::cast_slice::<_, u64>(breaks);
                slice.hash(state);
                labels.hash(state);
                left_closed.hash(state);
                include_breaks.hash(state);
            },
            #[cfg(feature = "cutqcut")]
            FunctionExpr::QCut {
                probs,
                labels,
                left_closed,
                allow_duplicates,
                include_breaks,
            } => {
                let slice = bytemuck::cast_slice::<_, u64>(probs);
                slice.hash(state);
                labels.hash(state);
                left_closed.hash(state);
                allow_duplicates.hash(state);
                include_breaks.hash(state);
            },
            #[cfg(feature = "rle")]
            FunctionExpr::RLE => {},
            #[cfg(feature = "rle")]
            FunctionExpr::RLEID => {},
            FunctionExpr::ToPhysical => {},
            FunctionExpr::SetSortedFlag(is_sorted) => is_sorted.hash(state),
            FunctionExpr::BackwardFill { limit } | FunctionExpr::ForwardFill { limit } => {
                limit.hash(state)
            },
            #[cfg(feature = "ewma")]
            FunctionExpr::EwmMean { options } => options.hash(state),
            #[cfg(feature = "ewma")]
            FunctionExpr::EwmStd { options } => options.hash(state),
            #[cfg(feature = "ewma")]
            FunctionExpr::EwmVar { options } => options.hash(state),
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
            #[cfg(all(feature = "rolling_window", feature = "moment"))]
            RollingSkew { .. } => "rolling_skew",
            ShiftAndFill => "shift_and_fill",
            DropNans => "drop_nans",
            DropNulls => "drop_nulls",
            #[cfg(feature = "mode")]
            Mode => "mode",
            #[cfg(feature = "moment")]
            Skew(_) => "skew",
            #[cfg(feature = "moment")]
            Kurtosis(..) => "kurtosis",
            ArgUnique => "arg_unique",
            #[cfg(feature = "rank")]
            Rank { .. } => "rank",
            #[cfg(feature = "round_series")]
            Clip { has_min, has_max } => match (has_min, has_max) {
                (true, true) => "clip",
                (false, true) => "clip_max",
                (true, false) => "clip_min",
                _ => unreachable!(),
            },
            ListExpr(func) => return write!(f, "{func}"),
            #[cfg(feature = "dtype-struct")]
            StructExpr(func) => return write!(f, "{func}"),
            #[cfg(feature = "dtype-struct")]
            AsStruct => "as_struct",
            #[cfg(feature = "top_k")]
            TopK(descending) => {
                if *descending {
                    "bottom_k"
                } else {
                    "top_k"
                }
            },
            Shift => "shift",
            #[cfg(feature = "cum_agg")]
            Cumcount { .. } => "cumcount",
            #[cfg(feature = "cum_agg")]
            Cumsum { .. } => "cumsum",
            #[cfg(feature = "cum_agg")]
            Cumprod { .. } => "cumprod",
            #[cfg(feature = "cum_agg")]
            Cummin { .. } => "cummin",
            #[cfg(feature = "cum_agg")]
            Cummax { .. } => "cummax",
            #[cfg(feature = "dtype-struct")]
            ValueCounts { .. } => "value_counts",
            #[cfg(feature = "unique_counts")]
            UniqueCounts => "unique_counts",
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
            #[cfg(feature = "pct_change")]
            PctChange => "pct_change",
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
            #[cfg(feature = "peaks")]
            PeakMin => "peak_min",
            #[cfg(feature = "peaks")]
            PeakMax => "peak_max",
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
            #[cfg(feature = "ffi_plugin")]
            FfiPlugin { lib, symbol, .. } => return write!(f, "{lib}:{symbol}"),
            BackwardFill { .. } => "backward_fill",
            ForwardFill { .. } => "forward_fill",
            SumHorizontal => "sum_horizontal",
            MaxHorizontal => "max_horizontal",
            MinHorizontal => "min_horizontal",
            #[cfg(feature = "ewma")]
            EwmMean { .. } => "ewm_mean",
            #[cfg(feature = "ewma")]
            EwmStd { .. } => "ewm_std",
            #[cfg(feature = "ewma")]
            EwmVar { .. } => "ewm_var",
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

            #[cfg(all(feature = "rolling_window", feature = "moment"))]
            RollingSkew { window_size, bias } => {
                map!(rolling::rolling_skew, window_size, bias)
            },
            ShiftAndFill => {
                map_as_slice!(shift_and_fill::shift_and_fill)
            },
            DropNans => map_owned!(nan::drop_nans),
            DropNulls => map!(dispatch::drop_nulls),
            #[cfg(feature = "round_series")]
            Clip { has_min, has_max } => {
                map_as_slice!(clip::clip, has_min, has_max)
            },
            #[cfg(feature = "mode")]
            Mode => map!(dispatch::mode),
            #[cfg(feature = "moment")]
            Skew(bias) => map!(dispatch::skew, bias),
            #[cfg(feature = "moment")]
            Kurtosis(fisher, bias) => map!(dispatch::kurtosis, fisher, bias),
            ArgUnique => map!(dispatch::arg_unique),
            #[cfg(feature = "rank")]
            Rank { options, seed } => map!(dispatch::rank, options, seed),
            ListExpr(lf) => {
                use ListFunction::*;
                match lf {
                    Concat => wrap!(list::concat),
                    #[cfg(feature = "is_in")]
                    Contains => wrap!(list::contains),
                    #[cfg(feature = "list_drop_nulls")]
                    DropNulls => map!(list::drop_nulls),
                    #[cfg(feature = "list_sample")]
                    Sample {
                        is_fraction,
                        with_replacement,
                        shuffle,
                        seed,
                    } => {
                        if is_fraction {
                            map_as_slice!(list::sample_fraction, with_replacement, shuffle, seed)
                        } else {
                            map_as_slice!(list::sample_n, with_replacement, shuffle, seed)
                        }
                    },
                    Slice => wrap!(list::slice),
                    Shift => map_as_slice!(list::shift),
                    Get => wrap!(list::get),
                    #[cfg(feature = "list_take")]
                    Take(null_ob_oob) => map_as_slice!(list::take, null_ob_oob),
                    #[cfg(feature = "list_count")]
                    CountMatches => map_as_slice!(list::count_matches),
                    Sum => map!(list::sum),
                    Length => map!(list::length),
                    Max => map!(list::max),
                    Min => map!(list::min),
                    Mean => map!(list::mean),
                    ArgMin => map!(list::arg_min),
                    ArgMax => map!(list::arg_max),
                    #[cfg(feature = "diff")]
                    Diff { n, null_behavior } => map!(list::diff, n, null_behavior),
                    Sort(options) => map!(list::sort, options),
                    Reverse => map!(list::reverse),
                    Unique(is_stable) => map!(list::unique, is_stable),
                    #[cfg(feature = "list_sets")]
                    SetOperation(s) => map_as_slice!(list::set_operation, s),
                    #[cfg(feature = "list_any_all")]
                    Any => map!(list::lst_any),
                    #[cfg(feature = "list_any_all")]
                    All => map!(list::lst_all),
                    Join => map_as_slice!(list::join),
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
                    RenameFields(names) => map!(struct_::rename_fields, names.clone()),
                }
            },
            #[cfg(feature = "dtype-struct")]
            AsStruct => {
                map_as_slice!(coerce::as_struct)
            },
            #[cfg(feature = "top_k")]
            TopK(descending) => {
                map_as_slice!(top_k, descending)
            },
            Shift => map_as_slice!(shift_and_fill::shift),
            #[cfg(feature = "cum_agg")]
            Cumcount { reverse } => map!(cum::cumcount, reverse),
            #[cfg(feature = "cum_agg")]
            Cumsum { reverse } => map!(cum::cumsum, reverse),
            #[cfg(feature = "cum_agg")]
            Cumprod { reverse } => map!(cum::cumprod, reverse),
            #[cfg(feature = "cum_agg")]
            Cummin { reverse } => map!(cum::cummin, reverse),
            #[cfg(feature = "cum_agg")]
            Cummax { reverse } => map!(cum::cummax, reverse),
            #[cfg(feature = "dtype-struct")]
            ValueCounts { sort, parallel } => map!(dispatch::value_counts, sort, parallel),
            #[cfg(feature = "unique_counts")]
            UniqueCounts => map!(dispatch::unique_counts),
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
            #[cfg(feature = "pct_change")]
            PctChange => map_as_slice!(dispatch::pct_change),
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
            #[cfg(feature = "peaks")]
            PeakMin => map!(peaks::peak_min),
            #[cfg(feature = "peaks")]
            PeakMax => map!(peaks::peak_max),
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
            Random { method, seed } => {
                use RandomMethod::*;
                match method {
                    Shuffle => map!(random::shuffle, seed),
                    Sample {
                        is_fraction,
                        with_replacement,
                        shuffle,
                    } => {
                        if is_fraction {
                            map_as_slice!(random::sample_frac, with_replacement, shuffle, seed)
                        } else {
                            map_as_slice!(random::sample_n, with_replacement, shuffle, seed)
                        }
                    },
                }
            },
            SetSortedFlag(sorted) => map!(dispatch::set_sorted_flag, sorted),
            #[cfg(feature = "ffi_plugin")]
            FfiPlugin {
                lib,
                symbol,
                kwargs,
            } => unsafe {
                map_as_slice!(
                    plugin::call_plugin,
                    lib.as_ref(),
                    symbol.as_ref(),
                    kwargs.as_ref()
                )
            },
            BackwardFill { limit } => map!(dispatch::backward_fill, limit),
            ForwardFill { limit } => map!(dispatch::forward_fill, limit),
            SumHorizontal => map_as_slice!(dispatch::sum_horizontal),
            MaxHorizontal => wrap!(dispatch::max_horizontal),
            MinHorizontal => wrap!(dispatch::min_horizontal),
            #[cfg(feature = "ewma")]
            EwmMean { options } => map!(ewm::ewm_mean, options),
            #[cfg(feature = "ewma")]
            EwmStd { options } => map!(ewm::ewm_std, options),
            #[cfg(feature = "ewma")]
            EwmVar { options } => map!(ewm::ewm_var, options),
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
            CountMatches(literal) => {
                map_as_slice!(strings::count_matches, literal)
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
            LenBytes => map!(strings::len_bytes),
            LenChars => map!(strings::len_chars),
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
            Split(inclusive) => {
                map_as_slice!(strings::split, inclusive)
            },
            #[cfg(feature = "dtype-struct")]
            SplitExact { n, inclusive } => map_as_slice!(strings::split_exact, n, inclusive),
            #[cfg(feature = "dtype-struct")]
            SplitN(n) => map_as_slice!(strings::splitn, n),
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
            StripChars => map_as_slice!(strings::strip_chars),
            StripCharsStart => map_as_slice!(strings::strip_chars_start),
            StripCharsEnd => map_as_slice!(strings::strip_chars_end),
            StripPrefix => map_as_slice!(strings::strip_prefix),
            StripSuffix => map_as_slice!(strings::strip_suffix),
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
            Contains => {
                map_as_slice!(binary::contains)
            },
            EndsWith => {
                map_as_slice!(binary::ends_with)
            },
            StartsWith => {
                map_as_slice!(binary::starts_with)
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
            ToString(format) => map!(datetime::to_string, &format),
            TimeStamp(tu) => map!(datetime::timestamp, tu),
            #[cfg(feature = "timezones")]
            ConvertTimeZone(tz) => map!(datetime::convert_time_zone, &tz),
            WithTimeUnit(tu) => map!(datetime::with_time_unit, tu),
            CastTimeUnit(tu) => map!(datetime::cast_time_unit, tu),
            Truncate(offset) => {
                map_as_slice!(datetime::truncate, &offset)
            },
            #[cfg(feature = "date_offset")]
            MonthStart => map!(datetime::month_start),
            #[cfg(feature = "date_offset")]
            MonthEnd => map!(datetime::month_end),
            #[cfg(feature = "timezones")]
            BaseUtcOffset => map!(datetime::base_utc_offset),
            #[cfg(feature = "timezones")]
            DSTOffset => map!(datetime::dst_offset),
            Round(every, offset) => map_as_slice!(datetime::round, &every, &offset),
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
