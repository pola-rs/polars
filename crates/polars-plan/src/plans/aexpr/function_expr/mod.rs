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
#[cfg(feature = "cum_agg")]
mod cum;
#[cfg(feature = "temporal")]
mod datetime;
#[cfg(feature = "dtype-extension")]
mod extension;
#[cfg(feature = "fused")]
mod fused;
mod list;
#[cfg(feature = "ffi_plugin")]
pub mod plugin;
mod pow;
#[cfg(feature = "random")]
mod random;
#[cfg(feature = "range")]
mod range;
#[cfg(feature = "rolling_window")]
mod rolling;
#[cfg(feature = "rolling_window_by")]
mod rolling_by;
mod row_encode;
pub(super) mod schema;
#[cfg(feature = "strings")]
mod strings;
#[cfg(feature = "dtype-struct")]
mod struct_;
#[cfg(feature = "trigonometry")]
mod trigonometry;

use std::fmt::{Display, Formatter};
use std::hash::{Hash, Hasher};

#[cfg(feature = "dtype-array")]
pub use array::IRArrayFunction;
#[cfg(feature = "cov")]
pub use correlation::IRCorrelationMethod;
#[cfg(feature = "fused")]
pub use fused::FusedOperator;
pub use list::IRListFunction;
pub use polars_core::datatypes::ReshapeDimension;
use polars_core::prelude::*;
use polars_core::series::IsSorted;
use polars_core::series::ops::NullBehavior;
use polars_core::utils::SuperTypeFlags;
#[cfg(feature = "random")]
pub use random::IRRandomMethod;
use schema::FieldsMapper;

pub use self::binary::IRBinaryFunction;
#[cfg(feature = "bitwise")]
pub use self::bitwise::IRBitwiseFunction;
pub use self::boolean::IRBooleanFunction;
#[cfg(feature = "business")]
pub use self::business::IRBusinessFunction;
#[cfg(feature = "dtype-categorical")]
pub use self::cat::IRCategoricalFunction;
#[cfg(feature = "temporal")]
pub use self::datetime::IRTemporalFunction;
#[cfg(feature = "dtype-extension")]
pub use self::extension::IRExtensionFunction;
pub use self::pow::IRPowFunction;
#[cfg(feature = "range")]
pub use self::range::IRRangeFunction;
#[cfg(feature = "rolling_window")]
pub use self::rolling::IRRollingFunction;
#[cfg(feature = "rolling_window_by")]
pub use self::rolling_by::IRRollingFunctionBy;
pub use self::row_encode::RowEncodingVariant;
#[cfg(feature = "strings")]
pub use self::strings::IRStringFunction;
#[cfg(all(feature = "strings", feature = "regex", feature = "timezones"))]
pub use self::strings::TZ_AWARE_RE;
#[cfg(feature = "dtype-struct")]
pub use self::struct_::IRStructFunction;
#[cfg(feature = "trigonometry")]
pub use self::trigonometry::IRTrigonometricFunction;
use super::*;

#[cfg_attr(feature = "ir_serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, PartialEq, Debug)]
pub enum IRFunctionExpr {
    // Namespaces
    #[cfg(feature = "dtype-array")]
    ArrayExpr(IRArrayFunction),
    BinaryExpr(IRBinaryFunction),
    #[cfg(feature = "dtype-categorical")]
    Categorical(IRCategoricalFunction),
    #[cfg(feature = "dtype-extension")]
    Extension(IRExtensionFunction),
    ListExpr(IRListFunction),
    #[cfg(feature = "strings")]
    StringExpr(IRStringFunction),
    #[cfg(feature = "dtype-struct")]
    StructExpr(IRStructFunction),
    #[cfg(feature = "temporal")]
    TemporalExpr(IRTemporalFunction),
    #[cfg(feature = "bitwise")]
    Bitwise(IRBitwiseFunction),

    // Other expressions
    Boolean(IRBooleanFunction),
    #[cfg(feature = "business")]
    Business(IRBusinessFunction),
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
    Pow(IRPowFunction),
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
    Range(IRRangeFunction),
    #[cfg(feature = "trigonometry")]
    Trigonometry(IRTrigonometricFunction),
    #[cfg(feature = "trigonometry")]
    Atan2,
    #[cfg(feature = "sign")]
    Sign,
    FillNull,
    FillNullWithStrategy(FillNullStrategy),
    #[cfg(feature = "rolling_window")]
    RollingExpr {
        function: IRRollingFunction,
        options: RollingOptionsFixedWindow,
    },
    #[cfg(feature = "rolling_window_by")]
    RollingExprBy {
        function_by: IRRollingFunctionBy,
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
    Unique(/* maintain_order */ bool),
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
    #[cfg(feature = "fused")]
    Fused(fused::FusedOperator),
    ConcatExpr(bool),
    #[cfg(feature = "cov")]
    Correlation {
        method: correlation::IRCorrelationMethod,
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
        method: IRRandomMethod,
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
        return_dtype: Option<DataType>,
    },
    ReduceHorizontal {
        callback: PlanCallback<(Series, Series), Series>,
        returns_scalar: bool,
        return_dtype: Option<DataType>,
    },
    #[cfg(feature = "dtype-struct")]
    CumReduceHorizontal {
        callback: PlanCallback<(Series, Series), Series>,
        returns_scalar: bool,
        return_dtype: Option<DataType>,
    },
    #[cfg(feature = "dtype-struct")]
    CumFoldHorizontal {
        callback: PlanCallback<(Series, Series), Series>,
        returns_scalar: bool,
        return_dtype: Option<DataType>,
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
        return_dtype: Option<DataType>,
    },
    GatherEvery {
        n: usize,
        offset: usize,
    },
    #[cfg(feature = "reinterpret")]
    Reinterpret(bool),
    ExtendConstant,

    RowEncode(Vec<DataType>, RowEncodingVariant),
    #[cfg(feature = "dtype-struct")]
    RowDecode(Vec<Field>, RowEncodingVariant),
}

impl Hash for IRFunctionExpr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        use IRFunctionExpr::*;
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
            #[cfg(feature = "fused")]
            Fused(f) => f.hash(state),
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
            Append { upcast } => {
                upcast.hash(state);
            },
            ArgSort {
                descending,
                nulls_last,
            } => {
                descending.hash(state);
                nulls_last.hash(state);
            },
            #[cfg(feature = "mode")]
            Mode { maintain_order } => {
                maintain_order.hash(state);
            },
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
            IRFunctionExpr::RoundSF { digits } => digits.hash(state),
            #[cfg(feature = "round_series")]
            IRFunctionExpr::Floor => {},
            #[cfg(feature = "round_series")]
            Ceil => {},
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

            RowEncode(dts, variants) => {
                dts.hash(state);
                variants.hash(state);
            },
            #[cfg(feature = "dtype-struct")]
            RowDecode(fs, variants) => {
                fs.hash(state);
                variants.hash(state);
            },
        }
    }
}

impl Display for IRFunctionExpr {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use IRFunctionExpr::*;
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
            Append { .. } => "append",
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

#[macro_export]
macro_rules! wrap {
    ($e:expr) => {
        SpecialEq::new(Arc::new($e))
    };

    ($e:expr, $($args:expr),*) => {{
        let f = move |s: &mut [Column]| {
            $e(s, $($args),*)
        };

        SpecialEq::new(Arc::new(f))
    }};
}

/// `Fn(&[Column], args)`
/// * all expression arguments are in the slice.
/// * the first element is the root expression.
#[macro_export]
macro_rules! map_as_slice {
    ($func:path) => {{
        let f = move |s: &mut [Column]| {
            $func(s)
        };

        SpecialEq::new(Arc::new(f))
    }};

    ($func:path, $($args:expr),*) => {{
        let f = move |s: &mut [Column]| {
            $func(s, $($args),*)
        };

        SpecialEq::new(Arc::new(f))
    }};
}

/// * `FnOnce(Series)`
/// * `FnOnce(Series, args)`
#[macro_export]
macro_rules! map_owned {
    ($func:path) => {{
        let f = move |c: &mut [Column]| {
            let c = std::mem::take(&mut c[0]);
            $func(c)
        };

        SpecialEq::new(Arc::new(f))
    }};

    ($func:path, $($args:expr),*) => {{
        let f = move |c: &mut [Column]| {
            let c = std::mem::take(&mut c[0]);
            $func(c, $($args),*)
        };

        SpecialEq::new(Arc::new(f))
    }};
}

/// `Fn(&Series, args)`
#[macro_export]
macro_rules! map {
    ($func:path) => {{
        let f = move |c: &mut [Column]| {
            let c = &c[0];
            $func(c)
        };

        SpecialEq::new(Arc::new(f))
    }};

    ($func:path, $($args:expr),*) => {{
        let f = move |c: &mut [Column]| {
            let c = &c[0];
            $func(c, $($args),*)
        };

        SpecialEq::new(Arc::new(f))
    }};
}

impl IRFunctionExpr {
    pub fn function_options(&self) -> FunctionOptions {
        use IRFunctionExpr as F;
        match self {
            #[cfg(feature = "dtype-array")]
            F::ArrayExpr(e) => e.function_options(),
            F::BinaryExpr(e) => e.function_options(),
            #[cfg(feature = "dtype-categorical")]
            F::Categorical(e) => e.function_options(),
            #[cfg(feature = "dtype-extension")]
            F::Extension(e) => e.function_options(),
            F::ListExpr(e) => e.function_options(),
            #[cfg(feature = "strings")]
            F::StringExpr(e) => e.function_options(),
            #[cfg(feature = "dtype-struct")]
            F::StructExpr(e) => e.function_options(),
            #[cfg(feature = "temporal")]
            F::TemporalExpr(e) => e.function_options(),
            #[cfg(feature = "bitwise")]
            F::Bitwise(e) => e.function_options(),
            F::Boolean(e) => e.function_options(),
            #[cfg(feature = "business")]
            F::Business(e) => e.function_options(),
            F::Pow(e) => e.function_options(),
            #[cfg(feature = "range")]
            F::Range(e) => e.function_options(),
            #[cfg(feature = "abs")]
            F::Abs => FunctionOptions::elementwise(),
            F::Negate => FunctionOptions::elementwise(),
            #[cfg(feature = "hist")]
            F::Hist { .. } => FunctionOptions::groupwise(),
            F::NullCount => FunctionOptions::aggregation().flag(FunctionFlags::NON_ORDER_OBSERVING),
            #[cfg(feature = "row_hash")]
            F::Hash(_, _, _, _) => FunctionOptions::elementwise(),
            #[cfg(feature = "arg_where")]
            F::ArgWhere => FunctionOptions::groupwise(),
            #[cfg(feature = "index_of")]
            F::IndexOf => {
                FunctionOptions::aggregation().with_casting_rules(CastingRules::FirstArgLossless)
            },
            #[cfg(feature = "search_sorted")]
            F::SearchSorted { .. } => FunctionOptions::groupwise().with_supertyping(
                (SuperTypeFlags::default() & !SuperTypeFlags::ALLOW_PRIMITIVE_TO_STRING).into(),
            ),
            #[cfg(feature = "trigonometry")]
            F::Trigonometry(_) => FunctionOptions::elementwise(),
            #[cfg(feature = "trigonometry")]
            F::Atan2 => FunctionOptions::elementwise(),
            #[cfg(feature = "sign")]
            F::Sign => FunctionOptions::elementwise(),
            F::FillNull => FunctionOptions::elementwise().with_supertyping(Default::default()),
            F::FillNullWithStrategy(strategy) if strategy.is_elementwise() => {
                FunctionOptions::elementwise()
            },
            F::FillNullWithStrategy(_) => FunctionOptions::length_preserving(),
            #[cfg(feature = "rolling_window")]
            F::RollingExpr { .. } => FunctionOptions::length_preserving(),
            #[cfg(feature = "rolling_window_by")]
            F::RollingExprBy { .. } => FunctionOptions::length_preserving(),
            F::Rechunk => FunctionOptions::length_preserving(),
            F::Append { .. } => FunctionOptions::groupwise(),
            F::ShiftAndFill => FunctionOptions::length_preserving(),
            F::Shift => FunctionOptions::length_preserving(),
            F::DropNans => {
                FunctionOptions::row_separable().flag(FunctionFlags::NON_ORDER_PRODUCING)
            },
            F::DropNulls => FunctionOptions::row_separable()
                .flag(FunctionFlags::ALLOW_EMPTY_INPUTS | FunctionFlags::NON_ORDER_PRODUCING),
            #[cfg(feature = "mode")]
            F::Mode { maintain_order } => FunctionOptions::groupwise().with_flags(|f| {
                let f = f | FunctionFlags::NON_ORDER_PRODUCING;

                if !*maintain_order {
                    f | FunctionFlags::NON_ORDER_OBSERVING | FunctionFlags::TERMINATES_INPUT_ORDER
                } else {
                    f
                }
            }),
            #[cfg(feature = "moment")]
            F::Skew(_) => FunctionOptions::aggregation().flag(FunctionFlags::NON_ORDER_OBSERVING),
            #[cfg(feature = "moment")]
            F::Kurtosis(_, _) => {
                FunctionOptions::aggregation().flag(FunctionFlags::NON_ORDER_OBSERVING)
            },
            #[cfg(feature = "dtype-array")]
            F::Reshape(dims) => {
                if dims.len() == 1 && dims[0] == ReshapeDimension::Infer {
                    FunctionOptions::row_separable()
                } else {
                    FunctionOptions::groupwise()
                }
            },
            #[cfg(feature = "repeat_by")]
            F::RepeatBy => FunctionOptions::elementwise(),
            F::ArgUnique => FunctionOptions::groupwise(),
            F::ArgMin | F::ArgMax => FunctionOptions::aggregation(),
            F::ArgSort { .. } => FunctionOptions::length_preserving(),
            F::MinBy | F::MaxBy => FunctionOptions::aggregation(),
            F::Product => FunctionOptions::aggregation().flag(FunctionFlags::NON_ORDER_OBSERVING),
            #[cfg(feature = "rank")]
            F::Rank { .. } => FunctionOptions::length_preserving(),
            F::Repeat => {
                FunctionOptions::groupwise().with_flags(|f| f | FunctionFlags::ALLOW_RENAME)
            },
            #[cfg(feature = "round_series")]
            F::Clip { .. } => FunctionOptions::elementwise(),
            #[cfg(feature = "dtype-struct")]
            F::AsStruct => FunctionOptions::elementwise().with_flags(|f| {
                f | FunctionFlags::PASS_NAME_TO_APPLY | FunctionFlags::INPUT_WILDCARD_EXPANSION
            }),
            #[cfg(feature = "top_k")]
            F::TopK { .. } => FunctionOptions::groupwise(),
            #[cfg(feature = "top_k")]
            F::TopKBy { .. } => FunctionOptions::groupwise(),
            #[cfg(feature = "cum_agg")]
            F::CumCount { .. }
            | F::CumSum { .. }
            | F::CumProd { .. }
            | F::CumMin { .. }
            | F::CumMax { .. }
            | F::CumMean { .. } => FunctionOptions::length_preserving(),
            F::Reverse => FunctionOptions::length_preserving()
                .with_flags(|f| f | FunctionFlags::NON_ORDER_OBSERVING),
            #[cfg(feature = "dtype-struct")]
            F::ValueCounts { sort, .. } => FunctionOptions::groupwise().with_flags(|mut f| {
                if !sort {
                    f |= FunctionFlags::TERMINATES_INPUT_ORDER | FunctionFlags::NON_ORDER_PRODUCING
                }
                f | FunctionFlags::PASS_NAME_TO_APPLY | FunctionFlags::NON_ORDER_OBSERVING
            }),
            #[cfg(feature = "unique_counts")]
            F::UniqueCounts => FunctionOptions::groupwise(),
            #[cfg(feature = "approx_unique")]
            F::ApproxNUnique => {
                FunctionOptions::aggregation().flag(FunctionFlags::NON_ORDER_OBSERVING)
            },
            F::Coalesce => FunctionOptions::elementwise()
                .with_flags(|f| f | FunctionFlags::INPUT_WILDCARD_EXPANSION)
                .with_supertyping(Default::default()),
            #[cfg(feature = "diff")]
            F::Diff(NullBehavior::Drop) => FunctionOptions::groupwise(),
            #[cfg(feature = "diff")]
            F::Diff(NullBehavior::Ignore) => FunctionOptions::length_preserving(),
            #[cfg(feature = "pct_change")]
            F::PctChange => FunctionOptions::length_preserving(),
            #[cfg(feature = "interpolate")]
            F::Interpolate(_) => FunctionOptions::length_preserving(),
            #[cfg(feature = "interpolate_by")]
            F::InterpolateBy => FunctionOptions::length_preserving(),
            #[cfg(feature = "log")]
            F::Log | F::Log1p | F::Exp => FunctionOptions::elementwise(),
            #[cfg(feature = "log")]
            F::Entropy { .. } => {
                FunctionOptions::aggregation().flag(FunctionFlags::NON_ORDER_OBSERVING)
            },
            F::Unique(maintain_order) => FunctionOptions::groupwise().with_flags(|f| {
                let f = f | FunctionFlags::NON_ORDER_PRODUCING;

                if !*maintain_order {
                    f | FunctionFlags::NON_ORDER_OBSERVING | FunctionFlags::TERMINATES_INPUT_ORDER
                } else {
                    f
                }
            }),
            #[cfg(feature = "round_series")]
            F::Round { .. } | F::RoundSF { .. } | F::Floor | F::Ceil => {
                FunctionOptions::elementwise()
            },
            #[cfg(feature = "fused")]
            F::Fused(_) => FunctionOptions::elementwise(),
            F::ConcatExpr(_) => FunctionOptions::groupwise()
                .with_flags(|f| f | FunctionFlags::INPUT_WILDCARD_EXPANSION)
                .with_supertyping(Default::default()),
            #[cfg(feature = "cov")]
            F::Correlation { .. } => {
                FunctionOptions::aggregation().with_supertyping(Default::default())
            },
            #[cfg(feature = "peaks")]
            F::PeakMin | F::PeakMax => FunctionOptions::length_preserving(),
            #[cfg(feature = "cutqcut")]
            F::Cut { .. } | F::QCut { .. } => FunctionOptions::length_preserving()
                .with_flags(|f| f | FunctionFlags::PASS_NAME_TO_APPLY),
            #[cfg(feature = "rle")]
            F::RLE => FunctionOptions::groupwise(),
            #[cfg(feature = "rle")]
            F::RLEID => FunctionOptions::length_preserving(),
            F::ToPhysical => FunctionOptions::elementwise(),
            #[cfg(feature = "random")]
            F::Random {
                method: IRRandomMethod::Sample { .. },
                ..
            } => FunctionOptions::groupwise(),
            #[cfg(feature = "random")]
            F::Random {
                method: IRRandomMethod::Shuffle,
                ..
            } => FunctionOptions::length_preserving(),
            F::SetSortedFlag(_) => FunctionOptions::elementwise(),
            #[cfg(feature = "ffi_plugin")]
            F::FfiPlugin { flags, .. } => *flags,
            F::MaxHorizontal | F::MinHorizontal => FunctionOptions::elementwise().with_flags(|f| {
                f | FunctionFlags::INPUT_WILDCARD_EXPANSION | FunctionFlags::ALLOW_RENAME
            }),
            F::MeanHorizontal { .. } | F::SumHorizontal { .. } => FunctionOptions::elementwise()
                .with_flags(|f| f | FunctionFlags::INPUT_WILDCARD_EXPANSION),

            F::FoldHorizontal { returns_scalar, .. }
            | F::ReduceHorizontal { returns_scalar, .. } => FunctionOptions::groupwise()
                .with_flags(|mut f| {
                    f |= FunctionFlags::INPUT_WILDCARD_EXPANSION;
                    if *returns_scalar {
                        f |= FunctionFlags::RETURNS_SCALAR;
                    }
                    f
                }),
            #[cfg(feature = "dtype-struct")]
            F::CumFoldHorizontal { returns_scalar, .. }
            | F::CumReduceHorizontal { returns_scalar, .. } => FunctionOptions::groupwise()
                .with_flags(|mut f| {
                    f |= FunctionFlags::INPUT_WILDCARD_EXPANSION;
                    if *returns_scalar {
                        f |= FunctionFlags::RETURNS_SCALAR;
                    }
                    f
                }),
            #[cfg(feature = "ewma")]
            F::EwmMean { .. } | F::EwmStd { .. } | F::EwmVar { .. } => {
                FunctionOptions::length_preserving()
            },
            #[cfg(feature = "ewma_by")]
            F::EwmMeanBy { .. } => FunctionOptions::length_preserving(),
            #[cfg(feature = "replace")]
            F::Replace => FunctionOptions::elementwise(),
            #[cfg(feature = "replace")]
            F::ReplaceStrict { .. } => FunctionOptions::elementwise(),
            F::GatherEvery { .. } => FunctionOptions::groupwise(),
            #[cfg(feature = "reinterpret")]
            F::Reinterpret(_) => FunctionOptions::elementwise(),
            F::ExtendConstant => FunctionOptions::groupwise(),

            F::RowEncode(..) => FunctionOptions::elementwise(),
            #[cfg(feature = "dtype-struct")]
            F::RowDecode(..) => FunctionOptions::elementwise(),
        }
    }
}
