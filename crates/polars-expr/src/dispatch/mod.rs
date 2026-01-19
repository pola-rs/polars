use std::sync::Arc;

use polars_core::error::PolarsResult;
use polars_core::frame::DataFrame;
use polars_core::prelude::{Column, GroupPositions};
use polars_plan::dsl::{ColumnsUdf, SpecialEq};
use polars_plan::plans::{IRBooleanFunction, IRFunctionExpr, IRPowFunction};
use polars_utils::IdxSize;

use crate::prelude::{AggregationContext, PhysicalExpr};
use crate::state::ExecutionState;

#[macro_export]
macro_rules! wrap {
    ($e:expr) => {
        SpecialEq::new(Arc::new($e))
    };

    ($e:expr, $($args:expr),*) => {{
        let f = move |s: &mut [::polars_core::prelude::Column]| {
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
        let f = move |s: &mut [::polars_core::prelude::Column]| {
            $func(s)
        };

        SpecialEq::new(Arc::new(f))
    }};

    ($func:path, $($args:expr),*) => {{
        let f = move |s: &mut [::polars_core::prelude::Column]| {
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
        let f = move |c: &mut [::polars_core::prelude::Column]| {
            let c = std::mem::take(&mut c[0]);
            $func(c)
        };

        SpecialEq::new(Arc::new(f))
    }};

    ($func:path, $($args:expr),*) => {{
        let f = move |c: &mut [::polars_core::prelude::Column]| {
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
        let f = move |c: &mut [::polars_core::prelude::Column]| {
            let c = &c[0];
            $func(c)
        };

        SpecialEq::new(Arc::new(f))
    }};

    ($func:path, $($args:expr),*) => {{
        let f = move |c: &mut [::polars_core::prelude::Column]| {
            let c = &c[0];
            $func(c, $($args),*)
        };

        SpecialEq::new(Arc::new(f))
    }};
}

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
#[cfg(feature = "cum_agg")]
mod cum;
#[cfg(feature = "temporal")]
mod datetime;
#[cfg(feature = "dtype-extension")]
mod extension;
mod groups_dispatch;
mod horizontal;
mod list;
mod misc;
mod pow;
#[cfg(feature = "random")]
mod random;
#[cfg(feature = "range")]
mod range;
#[cfg(feature = "rolling_window")]
mod rolling;
#[cfg(feature = "rolling_window_by")]
mod rolling_by;
#[cfg(feature = "round_series")]
mod round;
mod shift_and_fill;
#[cfg(feature = "strings")]
mod strings;
#[cfg(feature = "dtype-struct")]
pub(crate) mod struct_;
#[cfg(feature = "temporal")]
mod temporal;
#[cfg(feature = "trigonometry")]
mod trigonometry;

pub use groups_dispatch::drop_items;

pub fn function_expr_to_udf(func: IRFunctionExpr) -> SpecialEq<Arc<dyn ColumnsUdf>> {
    use IRFunctionExpr as F;
    match func {
        // Namespaces
        #[cfg(feature = "dtype-array")]
        F::ArrayExpr(func) => array::function_expr_to_udf(func),
        F::BinaryExpr(func) => binary::function_expr_to_udf(func),
        #[cfg(feature = "dtype-categorical")]
        F::Categorical(func) => cat::function_expr_to_udf(func),
        #[cfg(feature = "dtype-extension")]
        F::Extension(func) => extension::function_expr_to_udf(func),
        F::ListExpr(func) => list::function_expr_to_udf(func),
        #[cfg(feature = "strings")]
        F::StringExpr(func) => strings::function_expr_to_udf(func),
        #[cfg(feature = "dtype-struct")]
        F::StructExpr(func) => struct_::function_expr_to_udf(func),
        #[cfg(feature = "temporal")]
        F::TemporalExpr(func) => temporal::temporal_func_to_udf(func),
        #[cfg(feature = "bitwise")]
        F::Bitwise(func) => bitwise::function_expr_to_udf(func),

        // Other expressions
        F::Boolean(func) => boolean::function_expr_to_udf(func),
        #[cfg(feature = "business")]
        F::Business(func) => business::function_expr_to_udf(func),
        #[cfg(feature = "abs")]
        F::Abs => map!(misc::abs),
        F::Negate => map!(misc::negate),
        F::NullCount => {
            let f = |s: &mut [Column]| {
                let s = &s[0];
                Ok(Column::new(s.name().clone(), [s.null_count() as IdxSize]))
            };
            wrap!(f)
        },
        F::Pow(func) => match func {
            IRPowFunction::Generic => wrap!(pow::pow),
            IRPowFunction::Sqrt => map!(pow::sqrt),
            IRPowFunction::Cbrt => map!(pow::cbrt),
        },
        #[cfg(feature = "row_hash")]
        F::Hash(k0, k1, k2, k3) => {
            map!(misc::row_hash, k0, k1, k2, k3)
        },
        #[cfg(feature = "arg_where")]
        F::ArgWhere => {
            wrap!(misc::arg_where)
        },
        #[cfg(feature = "index_of")]
        F::IndexOf => {
            map_as_slice!(misc::index_of)
        },
        #[cfg(feature = "search_sorted")]
        F::SearchSorted { side, descending } => {
            map_as_slice!(misc::search_sorted_impl, side, descending)
        },
        #[cfg(feature = "range")]
        F::Range(func) => range::function_expr_to_udf(func),

        #[cfg(feature = "trigonometry")]
        F::Trigonometry(trig_function) => {
            map!(trigonometry::apply_trigonometric_function, trig_function)
        },
        #[cfg(feature = "trigonometry")]
        F::Atan2 => {
            wrap!(trigonometry::apply_arctan2)
        },

        #[cfg(feature = "sign")]
        F::Sign => {
            map!(misc::sign)
        },
        F::FillNull => {
            map_as_slice!(misc::fill_null)
        },
        #[cfg(feature = "rolling_window")]
        F::RollingExpr { function, options } => {
            use IRRollingFunction::*;
            use polars_plan::plans::IRRollingFunction;
            match function {
                Min => map!(rolling::rolling_min, options.clone()),
                Max => map!(rolling::rolling_max, options.clone()),
                Mean => map!(rolling::rolling_mean, options.clone()),
                Sum => map!(rolling::rolling_sum, options.clone()),
                Quantile => map!(rolling::rolling_quantile, options.clone()),
                Var => map!(rolling::rolling_var, options.clone()),
                Std => map!(rolling::rolling_std, options.clone()),
                Rank => map!(rolling::rolling_rank, options.clone()),
                #[cfg(feature = "moment")]
                Skew => map!(rolling::rolling_skew, options.clone()),
                #[cfg(feature = "moment")]
                Kurtosis => map!(rolling::rolling_kurtosis, options.clone()),
                #[cfg(feature = "cov")]
                CorrCov {
                    corr_cov_options,
                    is_corr,
                } => {
                    map_as_slice!(
                        rolling::rolling_corr_cov,
                        options.clone(),
                        corr_cov_options,
                        is_corr
                    )
                },
                Map(f) => {
                    map!(rolling::rolling_map, options.clone(), f.clone())
                },
            }
        },
        #[cfg(feature = "rolling_window_by")]
        F::RollingExprBy {
            function_by,
            options,
        } => {
            use IRRollingFunctionBy::*;
            use polars_plan::plans::IRRollingFunctionBy;
            match function_by {
                MinBy => map_as_slice!(rolling_by::rolling_min_by, options.clone()),
                MaxBy => map_as_slice!(rolling_by::rolling_max_by, options.clone()),
                MeanBy => map_as_slice!(rolling_by::rolling_mean_by, options.clone()),
                SumBy => map_as_slice!(rolling_by::rolling_sum_by, options.clone()),
                QuantileBy => {
                    map_as_slice!(rolling_by::rolling_quantile_by, options.clone())
                },
                VarBy => map_as_slice!(rolling_by::rolling_var_by, options.clone()),
                StdBy => map_as_slice!(rolling_by::rolling_std_by, options.clone()),
                RankBy => map_as_slice!(rolling_by::rolling_rank_by, options.clone()),
            }
        },
        #[cfg(feature = "hist")]
        F::Hist {
            bin_count,
            include_category,
            include_breakpoint,
        } => {
            map_as_slice!(misc::hist, bin_count, include_category, include_breakpoint)
        },
        F::Rechunk => map!(misc::rechunk),
        F::Append { upcast } => map_as_slice!(misc::append, upcast),
        F::ShiftAndFill => {
            map_as_slice!(shift_and_fill::shift_and_fill)
        },
        F::DropNans => map_owned!(misc::drop_nans),
        F::DropNulls => map!(misc::drop_nulls),
        #[cfg(feature = "round_series")]
        F::Clip { has_min, has_max } => {
            map_as_slice!(misc::clip, has_min, has_max)
        },
        #[cfg(feature = "mode")]
        F::Mode { maintain_order } => map!(misc::mode, maintain_order),
        #[cfg(feature = "moment")]
        F::Skew(bias) => map!(misc::skew, bias),
        #[cfg(feature = "moment")]
        F::Kurtosis(fisher, bias) => map!(misc::kurtosis, fisher, bias),
        F::ArgUnique => map!(misc::arg_unique),
        F::ArgMin => map!(misc::arg_min),
        F::ArgMax => map!(misc::arg_max),
        F::ArgSort {
            descending,
            nulls_last,
        } => map!(misc::arg_sort, descending, nulls_last),
        F::MinBy => map_as_slice!(misc::min_by),
        F::MaxBy => map_as_slice!(misc::max_by),
        F::Product => map!(misc::product),
        F::Repeat => map_as_slice!(misc::repeat),
        #[cfg(feature = "rank")]
        F::Rank { options, seed } => map!(misc::rank, options, seed),
        #[cfg(feature = "dtype-struct")]
        F::AsStruct => {
            map_as_slice!(misc::as_struct)
        },
        #[cfg(feature = "top_k")]
        F::TopK { descending } => {
            map_as_slice!(polars_ops::prelude::top_k, descending)
        },
        #[cfg(feature = "top_k")]
        F::TopKBy { descending } => {
            map_as_slice!(polars_ops::prelude::top_k_by, descending.clone())
        },
        F::Shift => map_as_slice!(shift_and_fill::shift),
        #[cfg(feature = "cum_agg")]
        F::CumCount { reverse } => map!(cum::cum_count, reverse),
        #[cfg(feature = "cum_agg")]
        F::CumSum { reverse } => map!(cum::cum_sum, reverse),
        #[cfg(feature = "cum_agg")]
        F::CumProd { reverse } => map!(cum::cum_prod, reverse),
        #[cfg(feature = "cum_agg")]
        F::CumMin { reverse } => map!(cum::cum_min, reverse),
        #[cfg(feature = "cum_agg")]
        F::CumMax { reverse } => map!(cum::cum_max, reverse),
        #[cfg(feature = "cum_agg")]
        F::CumMean { reverse } => map!(cum::cum_mean, reverse),
        #[cfg(feature = "dtype-struct")]
        F::ValueCounts {
            sort,
            parallel,
            name,
            normalize,
        } => map!(misc::value_counts, sort, parallel, name.clone(), normalize),
        #[cfg(feature = "unique_counts")]
        F::UniqueCounts => map!(misc::unique_counts),
        F::Reverse => map!(misc::reverse),
        #[cfg(feature = "approx_unique")]
        F::ApproxNUnique => map!(misc::approx_n_unique),
        F::Coalesce => map_as_slice!(misc::coalesce),
        #[cfg(feature = "diff")]
        F::Diff(null_behavior) => map_as_slice!(misc::diff, null_behavior),
        #[cfg(feature = "pct_change")]
        F::PctChange => map_as_slice!(misc::pct_change),
        #[cfg(feature = "interpolate")]
        F::Interpolate(method) => {
            map!(misc::interpolate, method)
        },
        #[cfg(feature = "interpolate_by")]
        F::InterpolateBy => {
            map_as_slice!(misc::interpolate_by)
        },
        #[cfg(feature = "log")]
        F::Entropy { base, normalize } => map!(misc::entropy, base, normalize),
        #[cfg(feature = "log")]
        F::Log => map_as_slice!(misc::log),
        #[cfg(feature = "log")]
        F::Log1p => map!(misc::log1p),
        #[cfg(feature = "log")]
        F::Exp => map!(misc::exp),
        F::Unique(stable) => map!(misc::unique, stable),
        #[cfg(feature = "round_series")]
        F::Round { decimals, mode } => map!(round::round, decimals, mode),
        #[cfg(feature = "round_series")]
        F::RoundSF { digits } => map!(round::round_sig_figs, digits),
        #[cfg(feature = "round_series")]
        F::Floor => map!(round::floor),
        #[cfg(feature = "round_series")]
        F::Ceil => map!(round::ceil),
        #[cfg(feature = "fused")]
        F::Fused(op) => map_as_slice!(misc::fused, op),
        F::ConcatExpr(rechunk) => map_as_slice!(misc::concat_expr, rechunk),
        #[cfg(feature = "cov")]
        F::Correlation { method } => map_as_slice!(misc::corr, method),
        #[cfg(feature = "peaks")]
        F::PeakMin => map!(misc::peak_min),
        #[cfg(feature = "peaks")]
        F::PeakMax => map!(misc::peak_max),
        #[cfg(feature = "repeat_by")]
        F::RepeatBy => map_as_slice!(misc::repeat_by),
        #[cfg(feature = "dtype-array")]
        F::Reshape(dims) => map!(misc::reshape, &dims),
        #[cfg(feature = "cutqcut")]
        F::Cut {
            breaks,
            labels,
            left_closed,
            include_breaks,
        } => map!(
            misc::cut,
            breaks.clone(),
            labels.clone(),
            left_closed,
            include_breaks
        ),
        #[cfg(feature = "cutqcut")]
        F::QCut {
            probs,
            labels,
            left_closed,
            allow_duplicates,
            include_breaks,
        } => map!(
            misc::qcut,
            probs.clone(),
            labels.clone(),
            left_closed,
            allow_duplicates,
            include_breaks
        ),
        #[cfg(feature = "rle")]
        F::RLE => map!(polars_ops::series::rle),
        #[cfg(feature = "rle")]
        F::RLEID => map!(polars_ops::series::rle_id),
        F::ToPhysical => map!(misc::to_physical),
        #[cfg(feature = "random")]
        F::Random { method, seed } => {
            use IRRandomMethod::*;
            use polars_plan::plans::IRRandomMethod;
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
        F::SetSortedFlag(sorted) => map!(misc::set_sorted_flag, sorted),
        #[cfg(feature = "ffi_plugin")]
        F::FfiPlugin {
            flags: _,
            lib,
            symbol,
            kwargs,
        } => unsafe {
            map_as_slice!(
                polars_plan::plans::plugin::call_plugin,
                lib.as_ref(),
                symbol.as_ref(),
                kwargs.as_ref()
            )
        },

        F::FoldHorizontal {
            callback,
            returns_scalar,
            return_dtype,
        } => map_as_slice!(
            horizontal::fold,
            &callback,
            returns_scalar,
            return_dtype.as_ref()
        ),
        F::ReduceHorizontal {
            callback,
            returns_scalar,
            return_dtype,
        } => map_as_slice!(
            horizontal::reduce,
            &callback,
            returns_scalar,
            return_dtype.as_ref()
        ),
        #[cfg(feature = "dtype-struct")]
        F::CumReduceHorizontal {
            callback,
            returns_scalar,
            return_dtype,
        } => map_as_slice!(
            horizontal::cum_reduce,
            &callback,
            returns_scalar,
            return_dtype.as_ref()
        ),
        #[cfg(feature = "dtype-struct")]
        F::CumFoldHorizontal {
            callback,
            returns_scalar,
            return_dtype,
            include_init,
        } => map_as_slice!(
            horizontal::cum_fold,
            &callback,
            returns_scalar,
            return_dtype.as_ref(),
            include_init
        ),

        F::MaxHorizontal => wrap!(misc::max_horizontal),
        F::MinHorizontal => wrap!(misc::min_horizontal),
        F::SumHorizontal { ignore_nulls } => wrap!(misc::sum_horizontal, ignore_nulls),
        F::MeanHorizontal { ignore_nulls } => wrap!(misc::mean_horizontal, ignore_nulls),
        #[cfg(feature = "ewma")]
        F::EwmMean { options } => map!(misc::ewm_mean, options),
        #[cfg(feature = "ewma_by")]
        F::EwmMeanBy { half_life } => map_as_slice!(misc::ewm_mean_by, half_life),
        #[cfg(feature = "ewma")]
        F::EwmStd { options } => map!(misc::ewm_std, options),
        #[cfg(feature = "ewma")]
        F::EwmVar { options } => map!(misc::ewm_var, options),
        #[cfg(feature = "replace")]
        F::Replace => {
            map_as_slice!(misc::replace)
        },
        #[cfg(feature = "replace")]
        F::ReplaceStrict { return_dtype } => {
            map_as_slice!(misc::replace_strict, return_dtype.clone())
        },

        F::FillNullWithStrategy(strategy) => map!(misc::fill_null_with_strategy, strategy),
        F::GatherEvery { n, offset } => map!(misc::gather_every, n, offset),
        #[cfg(feature = "reinterpret")]
        F::Reinterpret(signed) => map!(misc::reinterpret, signed),
        F::ExtendConstant => map_as_slice!(misc::extend_constant),

        F::RowEncode(dts, variants) => {
            map_as_slice!(misc::row_encode, dts.clone(), variants.clone())
        },
        #[cfg(feature = "dtype-struct")]
        F::RowDecode(fs, variants) => {
            map_as_slice!(misc::row_decode, fs.clone(), variants.clone())
        },
    }
}

pub trait GroupsUdf: Send + Sync + 'static {
    fn evaluate_on_groups<'a>(
        &self,
        inputs: &[Arc<dyn PhysicalExpr>],
        df: &DataFrame,
        groups: &'a GroupPositions,
        state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>>;
}

pub fn function_expr_to_groups_udf(func: &IRFunctionExpr) -> Option<SpecialEq<Arc<dyn GroupsUdf>>> {
    macro_rules! wrap_groups {
        ($f:expr$(, ($arg:expr, $n:ident:$ty:ty))*) => {{
            struct Wrap($($ty),*);
            impl GroupsUdf for Wrap {
                fn evaluate_on_groups<'a>(
                    &self,
                    inputs: &[Arc<dyn PhysicalExpr>],
                    df: &DataFrame,
                    groups: &'a GroupPositions,
                    state: &ExecutionState,
                ) -> PolarsResult<AggregationContext<'a>> {
                    let Wrap($($n),*) = self;
                    $f(inputs, df, groups, state$(, *$n)*)
                }
            }

            SpecialEq::new(Arc::new(Wrap($($arg),*)) as Arc<dyn GroupsUdf>)
        }};
    }
    use IRFunctionExpr as F;
    Some(match func {
        F::NullCount => wrap_groups!(groups_dispatch::null_count),
        F::Reverse => wrap_groups!(groups_dispatch::reverse),
        F::Boolean(IRBooleanFunction::Any { ignore_nulls }) => {
            let ignore_nulls = *ignore_nulls;
            wrap_groups!(groups_dispatch::any, (ignore_nulls, v: bool))
        },
        F::Boolean(IRBooleanFunction::All { ignore_nulls }) => {
            let ignore_nulls = *ignore_nulls;
            wrap_groups!(groups_dispatch::all, (ignore_nulls, v: bool))
        },
        #[cfg(feature = "bitwise")]
        F::Bitwise(f) => {
            use polars_plan::plans::IRBitwiseFunction as B;
            match f {
                B::And => wrap_groups!(groups_dispatch::bitwise_and),
                B::Or => wrap_groups!(groups_dispatch::bitwise_or),
                B::Xor => wrap_groups!(groups_dispatch::bitwise_xor),
                _ => return None,
            }
        },
        F::DropNans => wrap_groups!(groups_dispatch::drop_nans),
        F::DropNulls => wrap_groups!(groups_dispatch::drop_nulls),

        #[cfg(feature = "moment")]
        F::Skew(bias) => wrap_groups!(groups_dispatch::skew, (*bias, v: bool)),
        #[cfg(feature = "moment")]
        F::Kurtosis(fisher, bias) => {
            wrap_groups!(groups_dispatch::kurtosis, (*fisher, v1: bool), (*bias, v2: bool))
        },

        F::Unique(stable) => wrap_groups!(groups_dispatch::unique, (*stable, v: bool)),
        F::FillNullWithStrategy(polars_core::prelude::FillNullStrategy::Forward(limit)) => {
            wrap_groups!(groups_dispatch::forward_fill_null, (*limit, v: Option<IdxSize>))
        },
        F::FillNullWithStrategy(polars_core::prelude::FillNullStrategy::Backward(limit)) => {
            wrap_groups!(groups_dispatch::backward_fill_null, (*limit, v: Option<IdxSize>))
        },

        _ => return None,
    })
}
