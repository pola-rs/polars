#[cfg(feature = "abs")]
mod abs;
#[cfg(feature = "arg_where")]
mod arg_where;
#[cfg(feature = "dtype-array")]
mod array;
mod binary;
mod boolean;
mod bounds;
#[cfg(feature = "business")]
mod business;
#[cfg(feature = "dtype-categorical")]
pub mod cat;
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
#[cfg(feature = "ewma_by")]
mod ewm_by;
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
pub mod pow;
#[cfg(feature = "random")]
mod random;
#[cfg(feature = "range")]
mod range;
#[cfg(feature = "rolling_window")]
pub mod rolling;
#[cfg(feature = "rolling_window_by")]
pub mod rolling_by;
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
#[cfg(feature = "temporal")]
mod temporal;
#[cfg(feature = "trigonometry")]
pub mod trigonometry;
mod unique;

use std::fmt::{Display, Formatter};
use std::hash::{Hash, Hasher};

#[cfg(feature = "dtype-array")]
pub(crate) use array::ArrayFunction;
#[cfg(feature = "cov")]
pub(crate) use correlation::CorrelationMethod;
#[cfg(feature = "fused")]
pub(crate) use fused::FusedOperator;
pub(crate) use list::ListFunction;
use polars_core::prelude::*;
#[cfg(feature = "random")]
pub(crate) use random::RandomMethod;
use schema::FieldsMapper;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

pub(crate) use self::binary::BinaryFunction;
pub use self::boolean::BooleanFunction;
#[cfg(feature = "business")]
pub(super) use self::business::BusinessFunction;
#[cfg(feature = "dtype-categorical")]
pub use self::cat::CategoricalFunction;
#[cfg(feature = "temporal")]
pub use self::datetime::TemporalFunction;
pub use self::pow::PowFunction;
#[cfg(feature = "range")]
pub(super) use self::range::RangeFunction;
#[cfg(feature = "rolling_window")]
pub(super) use self::rolling::RollingFunction;
#[cfg(feature = "rolling_window_by")]
pub(super) use self::rolling_by::RollingFunctionBy;
#[cfg(feature = "strings")]
pub use self::strings::StringFunction;
#[cfg(feature = "dtype-struct")]
pub use self::struct_::StructFunction;
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
    #[cfg(feature = "business")]
    Business(BusinessFunction),
    #[cfg(feature = "abs")]
    Abs,
    Negate,
    #[cfg(feature = "hist")]
    Hist {
        bin_count: Option<usize>,
        include_category: bool,
        include_breakpoint: bool,
    },
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
    #[cfg(feature = "trigonometry")]
    Trigonometry(TrigonometricFunction),
    #[cfg(feature = "trigonometry")]
    Atan2,
    #[cfg(feature = "sign")]
    Sign,
    FillNull,
    FillNullWithStrategy(FillNullStrategy),
    #[cfg(feature = "rolling_window")]
    RollingExpr(RollingFunction),
    #[cfg(feature = "rolling_window_by")]
    RollingExprBy(RollingFunctionBy),
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
    Reshape(Vec<i64>, NestedType),
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
    TopK {
        descending: bool,
    },
    #[cfg(feature = "top_k")]
    TopKBy {
        descending: Vec<bool>,
    },
    #[cfg(feature = "cum_agg")]
    CumCount {
        reverse: bool,
    },
    #[cfg(feature = "cum_agg")]
    CumSum {
        reverse: bool,
    },
    #[cfg(feature = "cum_agg")]
    CumProd {
        reverse: bool,
    },
    #[cfg(feature = "cum_agg")]
    CumMin {
        reverse: bool,
    },
    #[cfg(feature = "cum_agg")]
    CumMax {
        reverse: bool,
    },
    Reverse,
    #[cfg(feature = "dtype-struct")]
    ValueCounts {
        sort: bool,
        parallel: bool,
        name: String,
        normalize: bool,
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
    #[cfg(feature = "interpolate_by")]
    InterpolateBy,
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
    /// This will lead to calls over FFI.
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
    MaxHorizontal,
    MinHorizontal,
    SumHorizontal,
    MeanHorizontal,
    #[cfg(feature = "ewma")]
    EwmMean {
        options: EWMOptions,
    },
    #[cfg(feature = "ewma_by")]
    EwmMeanBy {
        half_life: Duration,
    },
    #[cfg(feature = "ewma")]
    EwmStd {
        options: EWMOptions,
    },
    #[cfg(feature = "ewma")]
    EwmVar {
        options: EWMOptions,
    },
    #[cfg(feature = "replace")]
    Replace,
    #[cfg(feature = "replace")]
    ReplaceStrict {
        return_dtype: Option<DataType>,
    },
    GatherEvery {
        n: usize,
        offset: usize,
    },
    #[cfg(feature = "reinterpret")]
    Reinterpret(bool),
    ExtendConstant,
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
            #[cfg(feature = "business")]
            Business(f) => f.hash(state),
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
            #[cfg(feature = "interpolate_by")]
            InterpolateBy => {},
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
            MaxHorizontal | MinHorizontal | SumHorizontal | MeanHorizontal | DropNans
            | DropNulls | Reverse | ArgUnique | Shift | ShiftAndFill => {},
            #[cfg(feature = "mode")]
            Mode => {},
            #[cfg(feature = "abs")]
            Abs => {},
            Negate => {},
            NullCount => {},
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
            FillNull => {},
            #[cfg(feature = "rolling_window")]
            RollingExpr(f) => {
                f.hash(state);
            },
            #[cfg(feature = "rolling_window_by")]
            RollingExprBy(f) => {
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
            TopK { descending } => descending.hash(state),
            #[cfg(feature = "cum_agg")]
            CumCount { reverse } => reverse.hash(state),
            #[cfg(feature = "cum_agg")]
            CumSum { reverse } => reverse.hash(state),
            #[cfg(feature = "cum_agg")]
            CumProd { reverse } => reverse.hash(state),
            #[cfg(feature = "cum_agg")]
            CumMin { reverse } => reverse.hash(state),
            #[cfg(feature = "cum_agg")]
            CumMax { reverse } => reverse.hash(state),
            #[cfg(feature = "dtype-struct")]
            ValueCounts {
                sort,
                parallel,
                name,
                normalize,
            } => {
                sort.hash(state);
                parallel.hash(state);
                name.hash(state);
                normalize.hash(state);
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
            Reshape(dims, nested) => {
                dims.hash(state);
                nested.hash(state);
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
            #[cfg(feature = "ewma_by")]
            EwmMeanBy { half_life } => (half_life).hash(state),
            #[cfg(feature = "ewma")]
            EwmStd { options } => options.hash(state),
            #[cfg(feature = "ewma")]
            EwmVar { options } => options.hash(state),
            #[cfg(feature = "hist")]
            Hist {
                bin_count,
                include_category,
                include_breakpoint,
            } => {
                bin_count.hash(state);
                include_category.hash(state);
                include_breakpoint.hash(state);
            },
            #[cfg(feature = "replace")]
            Replace => {},
            #[cfg(feature = "replace")]
            ReplaceStrict { return_dtype } => return_dtype.hash(state),
            FillNullWithStrategy(strategy) => strategy.hash(state),
            GatherEvery { n, offset } => (n, offset).hash(state),
            #[cfg(feature = "reinterpret")]
            Reinterpret(signed) => signed.hash(state),
            ExtendConstant => {},
            #[cfg(feature = "top_k")]
            TopKBy { descending } => descending.hash(state),
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
            #[cfg(feature = "business")]
            Business(func) => return write!(f, "{func}"),
            #[cfg(feature = "abs")]
            Abs => "abs",
            Negate => "negate",
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
            #[cfg(feature = "trigonometry")]
            Trigonometry(func) => return write!(f, "{func}"),
            #[cfg(feature = "trigonometry")]
            Atan2 => return write!(f, "arctan2"),
            #[cfg(feature = "sign")]
            Sign => "sign",
            FillNull { .. } => "fill_null",
            #[cfg(feature = "rolling_window")]
            RollingExpr(func, ..) => return write!(f, "{func}"),
            #[cfg(feature = "rolling_window_by")]
            RollingExprBy(func, ..) => return write!(f, "{func}"),
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
            TopK { descending } => {
                if *descending {
                    "bottom_k"
                } else {
                    "top_k"
                }
            },
            #[cfg(feature = "top_k")]
            TopKBy { .. } => "top_k_by",
            Shift => "shift",
            #[cfg(feature = "cum_agg")]
            CumCount { .. } => "cum_count",
            #[cfg(feature = "cum_agg")]
            CumSum { .. } => "cum_sum",
            #[cfg(feature = "cum_agg")]
            CumProd { .. } => "cum_prod",
            #[cfg(feature = "cum_agg")]
            CumMin { .. } => "cum_min",
            #[cfg(feature = "cum_agg")]
            CumMax { .. } => "cum_max",
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
            #[cfg(feature = "interpolate_by")]
            InterpolateBy => "interpolate_by",
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
            Reshape(_, _) => "reshape",
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
            MaxHorizontal => "max_horizontal",
            MinHorizontal => "min_horizontal",
            SumHorizontal => "sum_horizontal",
            MeanHorizontal => "mean_horizontal",
            #[cfg(feature = "ewma")]
            EwmMean { .. } => "ewm_mean",
            #[cfg(feature = "ewma_by")]
            EwmMeanBy { .. } => "ewm_mean_by",
            #[cfg(feature = "ewma")]
            EwmStd { .. } => "ewm_std",
            #[cfg(feature = "ewma")]
            EwmVar { .. } => "ewm_var",
            #[cfg(feature = "hist")]
            Hist { .. } => "hist",
            #[cfg(feature = "replace")]
            Replace => "replace",
            #[cfg(feature = "replace")]
            ReplaceStrict { .. } => "replace_strict",
            FillNullWithStrategy(_) => "fill_null_with_strategy",
            GatherEvery { .. } => "gather_every",
            #[cfg(feature = "reinterpret")]
            Reinterpret(_) => "reinterpret",
            ExtendConstant => "extend_constant",
        };
        write!(f, "{s}")
    }
}

#[macro_export]
macro_rules! wrap {
    ($e:expr) => {
        SpecialEq::new(Arc::new($e))
    };

    ($e:expr, $($args:expr),*) => {{
        let f = move |s: &mut [Series]| {
            $e(s, $($args),*)
        };

        SpecialEq::new(Arc::new(f))
    }};
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
            #[cfg(feature = "business")]
            Business(func) => func.into(),
            #[cfg(feature = "abs")]
            Abs => map!(abs::abs),
            Negate => map!(dispatch::negate),
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
            FillNull => {
                map_as_slice!(fill_null::fill_null)
            },
            #[cfg(feature = "rolling_window")]
            RollingExpr(f) => {
                use RollingFunction::*;
                match f {
                    Min(options) => map!(rolling::rolling_min, options.clone()),
                    Max(options) => map!(rolling::rolling_max, options.clone()),
                    Mean(options) => map!(rolling::rolling_mean, options.clone()),
                    Sum(options) => map!(rolling::rolling_sum, options.clone()),
                    Quantile(options) => map!(rolling::rolling_quantile, options.clone()),
                    Var(options) => map!(rolling::rolling_var, options.clone()),
                    Std(options) => map!(rolling::rolling_std, options.clone()),
                    #[cfg(feature = "moment")]
                    Skew(window_size, bias) => map!(rolling::rolling_skew, window_size, bias),
                }
            },
            #[cfg(feature = "rolling_window_by")]
            RollingExprBy(f) => {
                use RollingFunctionBy::*;
                match f {
                    MinBy(options) => map_as_slice!(rolling_by::rolling_min_by, options.clone()),
                    MaxBy(options) => map_as_slice!(rolling_by::rolling_max_by, options.clone()),
                    MeanBy(options) => map_as_slice!(rolling_by::rolling_mean_by, options.clone()),
                    SumBy(options) => map_as_slice!(rolling_by::rolling_sum_by, options.clone()),
                    QuantileBy(options) => {
                        map_as_slice!(rolling_by::rolling_quantile_by, options.clone())
                    },
                    VarBy(options) => map_as_slice!(rolling_by::rolling_var_by, options.clone()),
                    StdBy(options) => map_as_slice!(rolling_by::rolling_std_by, options.clone()),
                }
            },
            #[cfg(feature = "hist")]
            Hist {
                bin_count,
                include_category,
                include_breakpoint,
            } => {
                map_as_slice!(
                    dispatch::hist,
                    bin_count,
                    include_category,
                    include_breakpoint
                )
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
            TopK { descending } => {
                map_as_slice!(top_k, descending)
            },
            #[cfg(feature = "top_k")]
            TopKBy { descending } => map_as_slice!(top_k_by, descending.clone()),
            Shift => map_as_slice!(shift_and_fill::shift),
            #[cfg(feature = "cum_agg")]
            CumCount { reverse } => map!(cum::cum_count, reverse),
            #[cfg(feature = "cum_agg")]
            CumSum { reverse } => map!(cum::cum_sum, reverse),
            #[cfg(feature = "cum_agg")]
            CumProd { reverse } => map!(cum::cum_prod, reverse),
            #[cfg(feature = "cum_agg")]
            CumMin { reverse } => map!(cum::cum_min, reverse),
            #[cfg(feature = "cum_agg")]
            CumMax { reverse } => map!(cum::cum_max, reverse),
            #[cfg(feature = "dtype-struct")]
            ValueCounts {
                sort,
                parallel,
                name,
                normalize,
            } => map!(
                dispatch::value_counts,
                sort,
                parallel,
                name.clone(),
                normalize
            ),
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
            #[cfg(feature = "interpolate_by")]
            InterpolateBy => {
                map_as_slice!(dispatch::interpolate_by)
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
            Reshape(dims, nested) => map!(dispatch::reshape, &dims, &nested),
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
            MaxHorizontal => wrap!(dispatch::max_horizontal),
            MinHorizontal => wrap!(dispatch::min_horizontal),
            SumHorizontal => wrap!(dispatch::sum_horizontal),
            MeanHorizontal => wrap!(dispatch::mean_horizontal),
            #[cfg(feature = "ewma")]
            EwmMean { options } => map!(ewm::ewm_mean, options),
            #[cfg(feature = "ewma_by")]
            EwmMeanBy { half_life } => map_as_slice!(ewm_by::ewm_mean_by, half_life),
            #[cfg(feature = "ewma")]
            EwmStd { options } => map!(ewm::ewm_std, options),
            #[cfg(feature = "ewma")]
            EwmVar { options } => map!(ewm::ewm_var, options),
            #[cfg(feature = "replace")]
            Replace => {
                map_as_slice!(dispatch::replace)
            },
            #[cfg(feature = "replace")]
            ReplaceStrict { return_dtype } => {
                map_as_slice!(dispatch::replace_strict, return_dtype.clone())
            },

            FillNullWithStrategy(strategy) => map!(dispatch::fill_null_with_strategy, strategy),
            GatherEvery { n, offset } => map!(dispatch::gather_every, n, offset),
            #[cfg(feature = "reinterpret")]
            Reinterpret(signed) => map!(dispatch::reinterpret, signed),
            ExtendConstant => map_as_slice!(dispatch::extend_constant),
        }
    }
}
