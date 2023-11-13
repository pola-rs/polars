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
#[cfg(feature = "dtype-struct")]
mod coerce;
mod concat;
#[cfg(feature = "cov")]
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
#[cfg(feature = "rolling_window")]
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
#[cfg(feature = "cov")]
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
#[cfg(feature = "rolling_window")]
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
    // Namespaces
    #[cfg(feature = "dtype-array")]
    ArrayExpr(ArrayFunction),
    BinaryExpr(BinaryFunction),
    #[cfg(feature = "dtype-categorical")]
    Categorical(CategoricalFunction),
    ListExpr(ListFunction),
    #[cfg(feature = "strings")]
    StringExpr(StringFunction),
    #[cfg(feature = "dtype-struct")]
    StructExpr(StructFunction),
    #[cfg(feature = "temporal")]
    TemporalExpr(TemporalFunction),

    // Other expressions
    Boolean(BooleanFunction),
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
    #[cfg(feature = "rolling_window")]
    RollingExpr(RollingFunction),
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
    Reshape(Vec<i64>),
    #[cfg(feature = "repeat_by")]
    RepeatBy,
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
    #[cfg(feature = "approx_unique")]
    ApproxNUnique,
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
    RoundSF {
        digits: i32,
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
    #[cfg(feature = "cov")]
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
        use FunctionExpr::*;
        match self {
            // Namespaces
            #[cfg(feature = "dtype-array")]
            ArrayExpr(f) => f.hash(state),
            BinaryExpr(f) => f.hash(state),
            #[cfg(feature = "dtype-categorical")]
            Categorical(f) => f.hash(state),
            ListExpr(f) => f.hash(state),
            #[cfg(feature = "strings")]
            StringExpr(f) => f.hash(state),
            #[cfg(feature = "dtype-struct")]
            StructExpr(f) => f.hash(state),
            #[cfg(feature = "temporal")]
            TemporalExpr(f) => f.hash(state),

            // Other expressions
            Boolean(f) => f.hash(state),
            Pow(f) => f.hash(state),
            #[cfg(feature = "search_sorted")]
            SearchSorted(f) => f.hash(state),
            #[cfg(feature = "random")]
            Random { method, .. } => method.hash(state),
            #[cfg(feature = "cov")]
            Correlation { method, .. } => method.hash(state),
            #[cfg(feature = "range")]
            Range(f) => f.hash(state),
            #[cfg(feature = "trigonometry")]
            Trigonometry(f) => f.hash(state),
            #[cfg(feature = "fused")]
            Fused(f) => f.hash(state),
            #[cfg(feature = "diff")]
            Diff(_, null_behavior) => null_behavior.hash(state),
            #[cfg(feature = "interpolate")]
            Interpolate(f) => f.hash(state),
            #[cfg(feature = "ffi_plugin")]
            FfiPlugin {
                lib,
                symbol,
                kwargs,
            } => {
                kwargs.hash(state);
                lib.hash(state);
                symbol.hash(state);
            },
            SumHorizontal | MaxHorizontal | MinHorizontal | DropNans | DropNulls | Reverse
            | ArgUnique | Shift | ShiftAndFill => {},
            #[cfg(feature = "mode")]
            Mode => {},
            #[cfg(feature = "abs")]
            Abs => {},
            NullCount => {},
            #[cfg(feature = "date_offset")]
            DateOffset => {},
            #[cfg(feature = "arg_where")]
            ArgWhere => {},
            #[cfg(feature = "trigonometry")]
            Atan2 => {},
            #[cfg(feature = "dtype-struct")]
            AsStruct => {},
            #[cfg(feature = "sign")]
            Sign => {},
            #[cfg(feature = "row_hash")]
            Hash(a, b, c, d) => (a, b, c, d).hash(state),
            FillNull { super_type } => super_type.hash(state),
            #[cfg(feature = "rolling_window")]
            RollingExpr(f) => {
                f.hash(state);
            },
            #[cfg(feature = "moment")]
            Skew(a) => a.hash(state),
            #[cfg(feature = "moment")]
            Kurtosis(a, b) => {
                a.hash(state);
                b.hash(state);
            },
            #[cfg(feature = "rank")]
            Rank { options, seed } => {
                options.hash(state);
                seed.hash(state);
            },
            #[cfg(feature = "round_series")]
            Clip { has_min, has_max } => {
                has_min.hash(state);
                has_max.hash(state);
            },
            #[cfg(feature = "top_k")]
            TopK(a) => a.hash(state),
            #[cfg(feature = "cum_agg")]
            Cumcount { reverse } => reverse.hash(state),
            #[cfg(feature = "cum_agg")]
            Cumsum { reverse } => reverse.hash(state),
            #[cfg(feature = "cum_agg")]
            Cumprod { reverse } => reverse.hash(state),
            #[cfg(feature = "cum_agg")]
            Cummin { reverse } => reverse.hash(state),
            #[cfg(feature = "cum_agg")]
            Cummax { reverse } => reverse.hash(state),
            #[cfg(feature = "dtype-struct")]
            ValueCounts { sort, parallel } => {
                sort.hash(state);
                parallel.hash(state);
            },
            #[cfg(feature = "unique_counts")]
            UniqueCounts => {},
            #[cfg(feature = "approx_unique")]
            ApproxNUnique => {},
            Coalesce => {},
            ShrinkType => {},
            #[cfg(feature = "pct_change")]
            PctChange => {},
            #[cfg(feature = "log")]
            Entropy { base, normalize } => {
                base.to_bits().hash(state);
                normalize.hash(state);
            },
            #[cfg(feature = "log")]
            Log { base } => base.to_bits().hash(state),
            #[cfg(feature = "log")]
            Log1p => {},
            #[cfg(feature = "log")]
            Exp => {},
            Unique(a) => a.hash(state),
            #[cfg(feature = "round_series")]
            Round { decimals } => decimals.hash(state),
            #[cfg(feature = "round_series")]
            FunctionExpr::RoundSF { digits } => digits.hash(state),
            #[cfg(feature = "round_series")]
            FunctionExpr::Floor => {},
            #[cfg(feature = "round_series")]
            Ceil => {},
            UpperBound => {},
            LowerBound => {},
            ConcatExpr(a) => a.hash(state),
            #[cfg(feature = "peaks")]
            PeakMin => {},
            #[cfg(feature = "peaks")]
            PeakMax => {},
            #[cfg(feature = "cutqcut")]
            Cut {
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
            Reshape(dims) => {
                dims.hash(state);
            },
            #[cfg(feature = "repeat_by")]
            RepeatBy => {},
            #[cfg(feature = "cutqcut")]
            QCut {
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
            RLE => {},
            #[cfg(feature = "rle")]
            RLEID => {},
            ToPhysical => {},
            SetSortedFlag(is_sorted) => is_sorted.hash(state),
            BackwardFill { limit } | ForwardFill { limit } => limit.hash(state),
            #[cfg(feature = "ewma")]
            EwmMean { options } => options.hash(state),
            #[cfg(feature = "ewma")]
            EwmStd { options } => options.hash(state),
            #[cfg(feature = "ewma")]
            EwmVar { options } => options.hash(state),
        }
    }
}

impl Display for FunctionExpr {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use FunctionExpr::*;
        let s = match self {
            // Namespaces
            #[cfg(feature = "dtype-array")]
            ArrayExpr(func) => return write!(f, "{func}"),
            BinaryExpr(func) => return write!(f, "{func}"),
            #[cfg(feature = "dtype-categorical")]
            Categorical(func) => return write!(f, "{func}"),
            ListExpr(func) => return write!(f, "{func}"),
            #[cfg(feature = "strings")]
            StringExpr(func) => return write!(f, "{func}"),
            #[cfg(feature = "dtype-struct")]
            StructExpr(func) => return write!(f, "{func}"),
            #[cfg(feature = "temporal")]
            TemporalExpr(func) => return write!(f, "{func}"),

            // Other expressions
            Boolean(func) => return write!(f, "{func}"),
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
            #[cfg(feature = "rolling_window")]
            RollingExpr(func, ..) => return write!(f, "{func}"),
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
            #[cfg(feature = "approx_unique")]
            ApproxNUnique => "approx_n_unique",
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
            RoundSF { .. } => "round_sig_figs",
            #[cfg(feature = "round_series")]
            Floor => "floor",
            #[cfg(feature = "round_series")]
            Ceil => "ceil",
            UpperBound => "upper_bound",
            LowerBound => "lower_bound",
            #[cfg(feature = "fused")]
            Fused(fused) => return Display::fmt(fused, f),
            ConcatExpr(_) => "concat_expr",
            #[cfg(feature = "cov")]
            Correlation { method, .. } => return Display::fmt(method, f),
            #[cfg(feature = "peaks")]
            PeakMin => "peak_min",
            #[cfg(feature = "peaks")]
            PeakMax => "peak_max",
            #[cfg(feature = "cutqcut")]
            Cut { .. } => "cut",
            #[cfg(feature = "cutqcut")]
            QCut { .. } => "qcut",
            Reshape(_) => "reshape",
            #[cfg(feature = "repeat_by")]
            RepeatBy => "repeat_by",
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
            // Namespaces
            #[cfg(feature = "dtype-array")]
            ArrayExpr(func) => func.into(),
            BinaryExpr(func) => func.into(),
            #[cfg(feature = "dtype-categorical")]
            Categorical(func) => func.into(),
            ListExpr(func) => func.into(),
            #[cfg(feature = "strings")]
            StringExpr(func) => func.into(),
            #[cfg(feature = "dtype-struct")]
            StructExpr(func) => func.into(),
            #[cfg(feature = "temporal")]
            TemporalExpr(func) => func.into(),

            // Other expressions
            Boolean(func) => func.into(),
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
            #[cfg(feature = "rolling_window")]
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
                    #[cfg(feature = "moment")]
                    Skew(window_size, bias) => map!(rolling::rolling_skew, window_size, bias),
                }
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
            #[cfg(feature = "approx_unique")]
            ApproxNUnique => map!(dispatch::approx_n_unique),
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
            RoundSF { digits } => map!(round::round_sig_figs, digits),
            #[cfg(feature = "round_series")]
            Floor => map!(round::floor),
            #[cfg(feature = "round_series")]
            Ceil => map!(round::ceil),
            UpperBound => map!(bounds::upper_bound),
            LowerBound => map!(bounds::lower_bound),
            #[cfg(feature = "fused")]
            Fused(op) => map_as_slice!(fused::fused, op),
            ConcatExpr(rechunk) => map_as_slice!(concat::concat_expr, rechunk),
            #[cfg(feature = "cov")]
            Correlation { method, ddof } => map_as_slice!(correlation::corr, ddof, method),
            #[cfg(feature = "peaks")]
            PeakMin => map!(peaks::peak_min),
            #[cfg(feature = "peaks")]
            PeakMax => map!(peaks::peak_max),
            #[cfg(feature = "repeat_by")]
            RepeatBy => map_as_slice!(dispatch::repeat_by),
            Reshape(dims) => map!(dispatch::reshape, dims.clone()),
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
