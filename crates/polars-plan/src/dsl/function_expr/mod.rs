#[cfg(feature = "dtype-array")]
mod array;
mod binary;
#[cfg(feature = "bitwise")]
mod bitwise;
mod boolean;
#[cfg(feature = "business")]
mod business;
#[cfg(feature = "dtype-categorical")]
mod cat;
#[cfg(feature = "cov")]
mod correlation;
#[cfg(feature = "temporal")]
mod datetime;
#[cfg(feature = "dtype-extension")]
mod extension;
mod list;
mod pow;
#[cfg(feature = "random")]
mod random;
#[cfg(feature = "range")]
mod range;
#[cfg(feature = "rolling_window")]
mod rolling;
#[cfg(feature = "rolling_window_by")]
mod rolling_by;
#[cfg(feature = "strings")]
mod strings;
#[cfg(feature = "dtype-struct")]
mod struct_;
#[cfg(feature = "trigonometry")]
mod trigonometry;

use std::fmt::{Display, Formatter};
use std::hash::{Hash, Hasher};

#[cfg(feature = "dtype-array")]
pub use array::ArrayFunction;
#[cfg(feature = "cov")]
pub use correlation::CorrelationMethod;
pub use list::ListFunction;
pub use polars_core::datatypes::ReshapeDimension;
use polars_core::prelude::*;
#[cfg(feature = "random")]
pub use random::RandomMethod;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

pub use self::binary::BinaryFunction;
#[cfg(feature = "bitwise")]
pub use self::bitwise::BitwiseFunction;
pub use self::boolean::BooleanFunction;
#[cfg(feature = "business")]
pub use self::business::BusinessFunction;
#[cfg(feature = "dtype-categorical")]
pub use self::cat::CategoricalFunction;
#[cfg(feature = "temporal")]
pub use self::datetime::TemporalFunction;
#[cfg(feature = "dtype-extension")]
pub use self::extension::ExtensionFunction;
pub use self::pow::PowFunction;
#[cfg(feature = "range")]
pub use self::range::{DateRangeArgs, RangeFunction};
#[cfg(feature = "rolling_window")]
pub use self::rolling::RollingFunction;
#[cfg(feature = "rolling_window_by")]
pub use self::rolling_by::RollingFunctionBy;
#[cfg(feature = "strings")]
pub use self::strings::StringFunction;
#[cfg(feature = "dtype-struct")]
pub use self::struct_::StructFunction;
#[cfg(feature = "trigonometry")]
pub use self::trigonometry::TrigonometricFunction;
use super::*;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Clone, PartialEq, Debug)]
pub enum FunctionExpr {
    // Namespaces
    #[cfg(feature = "dtype-array")]
    ArrayExpr(ArrayFunction),
    BinaryExpr(BinaryFunction),
    #[cfg(feature = "dtype-categorical")]
    Categorical(CategoricalFunction),
    #[cfg(feature = "dtype-extension")]
    Extension(ExtensionFunction),
    ListExpr(ListFunction),
    #[cfg(feature = "strings")]
    StringExpr(StringFunction),
    #[cfg(feature = "dtype-struct")]
    StructExpr(StructFunction),
    #[cfg(feature = "temporal")]
    TemporalExpr(TemporalFunction),
    #[cfg(feature = "bitwise")]
    Bitwise(BitwiseFunction),

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
    #[cfg(feature = "index_of")]
    IndexOf,
    #[cfg(feature = "search_sorted")]
    SearchSorted {
        side: SearchSortedSide,
        descending: bool,
    },
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
    RollingExpr {
        function: RollingFunction,
        options: RollingOptionsFixedWindow,
    },
    #[cfg(feature = "rolling_window_by")]
    RollingExprBy {
        function_by: RollingFunctionBy,
        options: RollingOptionsDynamicWindow,
    },
    Rechunk,
    Append {
        upcast: bool,
    },
    ShiftAndFill,
    Shift,
    DropNans,
    DropNulls,
    #[cfg(feature = "mode")]
    Mode {
        maintain_order: bool,
    },
    #[cfg(feature = "moment")]
    Skew(bool),
    #[cfg(feature = "moment")]
    Kurtosis(bool, bool),
    #[cfg(feature = "dtype-array")]
    Reshape(Vec<ReshapeDimension>),
    #[cfg(feature = "repeat_by")]
    RepeatBy,
    ArgUnique,
    ArgMin,
    ArgMax,
    ArgSort {
        descending: bool,
        nulls_last: bool,
    },
    MinBy,
    MaxBy,
    Product,
    #[cfg(feature = "rank")]
    Rank {
        options: RankOptions,
        seed: Option<u64>,
    },
    Repeat,
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
    #[cfg(feature = "cum_agg")]
    CumMean {
        reverse: bool,
    },
    Reverse,
    #[cfg(feature = "dtype-struct")]
    ValueCounts {
        sort: bool,
        parallel: bool,
        name: PlSmallStr,
        normalize: bool,
    },
    #[cfg(feature = "unique_counts")]
    UniqueCounts,
    #[cfg(feature = "approx_unique")]
    ApproxNUnique,
    Coalesce,
    #[cfg(feature = "diff")]
    Diff(NullBehavior),
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
    Log,
    #[cfg(feature = "log")]
    Log1p,
    #[cfg(feature = "log")]
    Exp,
    Unique(bool),
    #[cfg(feature = "round_series")]
    Round {
        decimals: u32,
        mode: RoundMode,
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
    ConcatExpr(bool),
    #[cfg(feature = "cov")]
    Correlation {
        method: correlation::CorrelationMethod,
    },
    #[cfg(feature = "peaks")]
    PeakMin,
    #[cfg(feature = "peaks")]
    PeakMax,
    #[cfg(feature = "cutqcut")]
    Cut {
        breaks: Vec<f64>,
        labels: Option<Vec<PlSmallStr>>,
        left_closed: bool,
        include_breaks: bool,
    },
    #[cfg(feature = "cutqcut")]
    QCut {
        probs: Vec<f64>,
        labels: Option<Vec<PlSmallStr>>,
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
        flags: FunctionOptions,
        /// Shared library.
        lib: PlSmallStr,
        /// Identifier in the shared lib.
        symbol: PlSmallStr,
        /// Pickle serialized keyword arguments.
        kwargs: Arc<[u8]>,
    },

    FoldHorizontal {
        callback: PlanCallback<(Series, Series), Series>,
        returns_scalar: bool,
        return_dtype: Option<DataTypeExpr>,
    },
    ReduceHorizontal {
        callback: PlanCallback<(Series, Series), Series>,
        returns_scalar: bool,
        return_dtype: Option<DataTypeExpr>,
    },
    #[cfg(feature = "dtype-struct")]
    CumReduceHorizontal {
        callback: PlanCallback<(Series, Series), Series>,
        returns_scalar: bool,
        return_dtype: Option<DataTypeExpr>,
    },
    #[cfg(feature = "dtype-struct")]
    CumFoldHorizontal {
        callback: PlanCallback<(Series, Series), Series>,
        returns_scalar: bool,
        return_dtype: Option<DataTypeExpr>,
        include_init: bool,
    },

    MaxHorizontal,
    MinHorizontal,
    SumHorizontal {
        ignore_nulls: bool,
    },
    MeanHorizontal {
        ignore_nulls: bool,
    },
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
        return_dtype: Option<DataTypeExpr>,
    },
    GatherEvery {
        n: usize,
        offset: usize,
    },
    #[cfg(feature = "reinterpret")]
    Reinterpret(bool),
    ExtendConstant,

    RowEncode(RowEncodingVariant),
    #[cfg(feature = "dtype-struct")]
    RowDecode(Vec<(PlSmallStr, DataTypeExpr)>, RowEncodingVariant),
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
            #[cfg(feature = "dtype-extension")]
            Extension(f) => f.hash(state),
            ListExpr(f) => f.hash(state),
            #[cfg(feature = "strings")]
            StringExpr(f) => f.hash(state),
            #[cfg(feature = "dtype-struct")]
            StructExpr(f) => f.hash(state),
            #[cfg(feature = "temporal")]
            TemporalExpr(f) => f.hash(state),
            #[cfg(feature = "bitwise")]
            Bitwise(f) => f.hash(state),

            // Other expressions
            Boolean(f) => f.hash(state),
            #[cfg(feature = "business")]
            Business(f) => f.hash(state),
            Pow(f) => f.hash(state),
            #[cfg(feature = "index_of")]
            IndexOf => {},
            #[cfg(feature = "search_sorted")]
            SearchSorted { side, descending } => {
                side.hash(state);
                descending.hash(state);
            },
            #[cfg(feature = "random")]
            Random { method, .. } => method.hash(state),
            #[cfg(feature = "cov")]
            Correlation { method, .. } => method.hash(state),
            #[cfg(feature = "range")]
            Range(f) => f.hash(state),
            #[cfg(feature = "trigonometry")]
            Trigonometry(f) => f.hash(state),
            #[cfg(feature = "diff")]
            Diff(null_behavior) => null_behavior.hash(state),
            #[cfg(feature = "interpolate")]
            Interpolate(f) => f.hash(state),
            #[cfg(feature = "interpolate_by")]
            InterpolateBy => {},
            #[cfg(feature = "ffi_plugin")]
            FfiPlugin {
                flags: _,
                lib,
                symbol,
                kwargs,
            } => {
                kwargs.hash(state);
                lib.hash(state);
                symbol.hash(state);
            },

            FoldHorizontal {
                callback,
                returns_scalar,
                return_dtype,
            }
            | ReduceHorizontal {
                callback,
                returns_scalar,
                return_dtype,
            } => {
                callback.hash(state);
                returns_scalar.hash(state);
                return_dtype.hash(state);
            },
            #[cfg(feature = "dtype-struct")]
            CumReduceHorizontal {
                callback,
                returns_scalar,
                return_dtype,
            } => {
                callback.hash(state);
                returns_scalar.hash(state);
                return_dtype.hash(state);
            },
            #[cfg(feature = "dtype-struct")]
            CumFoldHorizontal {
                callback,
                returns_scalar,
                return_dtype,
                include_init,
            } => {
                callback.hash(state);
                returns_scalar.hash(state);
                return_dtype.hash(state);
                include_init.hash(state);
            },
            SumHorizontal { ignore_nulls } | MeanHorizontal { ignore_nulls } => {
                ignore_nulls.hash(state)
            },
            MaxHorizontal | MinHorizontal | DropNans | DropNulls | Reverse | ArgUnique | ArgMin
            | ArgMax | Product | Shift | ShiftAndFill | Rechunk | MinBy | MaxBy => {},
            Append { upcast } => upcast.hash(state),
            ArgSort {
                descending,
                nulls_last,
            } => {
                descending.hash(state);
                nulls_last.hash(state);
            },
            #[cfg(feature = "mode")]
            Mode { maintain_order } => maintain_order.hash(state),
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
            RollingExpr { function, options } => {
                function.hash(state);
                options.hash(state);
            },
            #[cfg(feature = "rolling_window_by")]
            RollingExprBy {
                function_by,
                options,
            } => {
                function_by.hash(state);
                options.hash(state);
            },
            #[cfg(feature = "moment")]
            Skew(a) => a.hash(state),
            #[cfg(feature = "moment")]
            Kurtosis(a, b) => {
                a.hash(state);
                b.hash(state);
            },
            Repeat => {},
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
            #[cfg(feature = "cum_agg")]
            CumMean { reverse } => reverse.hash(state),
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
            #[cfg(feature = "pct_change")]
            PctChange => {},
            #[cfg(feature = "log")]
            Entropy { base, normalize } => {
                base.to_bits().hash(state);
                normalize.hash(state);
            },
            #[cfg(feature = "log")]
            Log => {},
            #[cfg(feature = "log")]
            Log1p => {},
            #[cfg(feature = "log")]
            Exp => {},
            Unique(a) => a.hash(state),
            #[cfg(feature = "round_series")]
            Round { decimals, mode } => {
                decimals.hash(state);
                mode.hash(state);
            },
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
            #[cfg(feature = "dtype-array")]
            Reshape(dims) => dims.hash(state),
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

            RowEncode(variants) => variants.hash(state),
            #[cfg(feature = "dtype-struct")]
            RowDecode(fs, variants) => {
                fs.hash(state);
                variants.hash(state);
            },
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
            #[cfg(feature = "dtype-extension")]
            Extension(func) => return write!(f, "{func}"),
            ListExpr(func) => return write!(f, "{func}"),
            #[cfg(feature = "strings")]
            StringExpr(func) => return write!(f, "{func}"),
            #[cfg(feature = "dtype-struct")]
            StructExpr(func) => return write!(f, "{func}"),
            #[cfg(feature = "temporal")]
            TemporalExpr(func) => return write!(f, "{func}"),
            #[cfg(feature = "bitwise")]
            Bitwise(func) => return write!(f, "bitwise_{func}"),

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
            #[cfg(feature = "index_of")]
            IndexOf => "index_of",
            #[cfg(feature = "search_sorted")]
            SearchSorted { .. } => "search_sorted",
            #[cfg(feature = "range")]
            Range(func) => return write!(f, "{func}"),
            #[cfg(feature = "trigonometry")]
            Trigonometry(func) => return write!(f, "{func}"),
            #[cfg(feature = "trigonometry")]
            Atan2 => return write!(f, "arctan2"),
            #[cfg(feature = "sign")]
            Sign => "sign",
            FillNull => "fill_null",
            #[cfg(feature = "rolling_window")]
            RollingExpr { function, .. } => return write!(f, "{function}"),
            #[cfg(feature = "rolling_window_by")]
            RollingExprBy { function_by, .. } => return write!(f, "{function_by}"),
            Rechunk => "rechunk",
            Append { .. } => "upcast",
            ShiftAndFill => "shift_and_fill",
            DropNans => "drop_nans",
            DropNulls => "drop_nulls",
            #[cfg(feature = "mode")]
            Mode { maintain_order } => {
                if *maintain_order {
                    "mode_stable"
                } else {
                    "mode"
                }
            },
            #[cfg(feature = "moment")]
            Skew(_) => "skew",
            #[cfg(feature = "moment")]
            Kurtosis(..) => "kurtosis",
            ArgUnique => "arg_unique",
            ArgMin => "arg_min",
            ArgMax => "arg_max",
            ArgSort { .. } => "arg_sort",
            MinBy => "min_by",
            MaxBy => "max_by",
            Product => "product",
            Repeat => "repeat",
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
            #[cfg(feature = "cum_agg")]
            CumMean { .. } => "cum_mean",
            #[cfg(feature = "dtype-struct")]
            ValueCounts { .. } => "value_counts",
            #[cfg(feature = "unique_counts")]
            UniqueCounts => "unique_counts",
            Reverse => "reverse",
            #[cfg(feature = "approx_unique")]
            ApproxNUnique => "approx_n_unique",
            Coalesce => "coalesce",
            #[cfg(feature = "diff")]
            Diff(_) => "diff",
            #[cfg(feature = "pct_change")]
            PctChange => "pct_change",
            #[cfg(feature = "interpolate")]
            Interpolate(_) => "interpolate",
            #[cfg(feature = "interpolate_by")]
            InterpolateBy => "interpolate_by",
            #[cfg(feature = "log")]
            Entropy { .. } => "entropy",
            #[cfg(feature = "log")]
            Log => "log",
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
            #[cfg(feature = "dtype-array")]
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
            FoldHorizontal { .. } => "fold",
            ReduceHorizontal { .. } => "reduce",
            #[cfg(feature = "dtype-struct")]
            CumReduceHorizontal { .. } => "cum_reduce",
            #[cfg(feature = "dtype-struct")]
            CumFoldHorizontal { .. } => "cum_fold",
            MaxHorizontal => "max_horizontal",
            MinHorizontal => "min_horizontal",
            SumHorizontal { .. } => "sum_horizontal",
            MeanHorizontal { .. } => "mean_horizontal",
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

            RowEncode(..) => "row_encode",
            #[cfg(feature = "dtype-struct")]
            RowDecode(..) => "row_decode",
        };
        write!(f, "{s}")
    }
}

#[cfg(any(feature = "array_to_struct", feature = "list_to_struct"))]
pub type DslNameGenerator = PlanCallback<usize, String>;
