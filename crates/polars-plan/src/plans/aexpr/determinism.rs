use polars_utils::idx_vec::UnitVec;
use polars_utils::unitvec;

use super::*;

/// Returns `true` if evaluating `root`'s subtree may produce different
/// values across calls for a fundamental reason: random draws, opaque
/// user UDFs, FFI plugins, runtime-injected predicates.
///
/// Returns `false` for non-determinism polars permits by policy
/// (`Expr.unique()` output order, `sum` over floats not being
/// bitwise-reproducible across parallel reductions). Optimizer rewrites
/// may freely factor those out.
///
/// Used as a correctness gate by rewrites that change the per-row
/// evaluation count of a subexpression, for example OR factoring
/// `(A ∧ X) ∨ (A ∧ Y) → A ∧ (X ∨ Y)`, which is sound only when `A`
/// is not inherently non-deterministic. A newly added `AExpr` or
/// `IRFunctionExpr` variant fails to compile here until explicitly
/// classified, so the helper cannot silently misclassify an unfamiliar
/// variant.
pub fn is_inherently_nondeterministic(root: Node, arena: &Arena<AExpr>) -> bool {
    let mut stack: UnitVec<Node> = unitvec![];
    let mut ae = arena.get(root);
    loop {
        // Exhaustive match: a newly added `AExpr` variant must trigger a
        // compile error here so its classification gets a fresh decision.
        match ae {
            // Opaque user code: cannot inspect, assume non-deterministic.
            AExpr::AnonymousFunction { .. } | AExpr::AnonymousAgg { .. } => return true,

            // Per-function classification, then fall through to recurse into inputs.
            AExpr::Function { function, .. } => {
                if is_inherently_nondeterministic_fn(function) {
                    return true;
                }
            },

            // Fall through to `inputs_rev` to recurse into children.
            AExpr::Column(_)
            | AExpr::Literal(_)
            | AExpr::Len
            | AExpr::Element
            | AExpr::BinaryExpr { .. }
            | AExpr::Cast { .. }
            | AExpr::Ternary { .. }
            | AExpr::Sort { .. }
            | AExpr::SortBy { .. }
            | AExpr::Filter { .. }
            | AExpr::Gather { .. }
            | AExpr::Slice { .. }
            | AExpr::Explode { .. }
            | AExpr::Agg(_)
            | AExpr::Over { .. }
            | AExpr::Eval { .. } => {},

            #[cfg(feature = "dtype-struct")]
            AExpr::StructField(_) | AExpr::StructEval { .. } => {},

            #[cfg(feature = "dynamic_group_by")]
            AExpr::Rolling { .. } => {},
        }
        ae.inputs_rev(&mut stack);
        let Some(node) = stack.pop() else {
            return false;
        };
        ae = arena.get(node);
    }
}

fn is_inherently_nondeterministic_fn(f: &IRFunctionExpr) -> bool {
    use IRFunctionExpr as F;
    match f {
        #[cfg(feature = "dtype-array")]
        F::ArrayExpr(a) => is_inherently_nondeterministic_array_fn(a),
        F::BinaryExpr(b) => is_inherently_nondeterministic_binary_fn(b),
        #[cfg(feature = "dtype-categorical")]
        F::Categorical(_) => false,
        #[cfg(feature = "dtype-extension")]
        F::Extension(_) => false,
        F::ListExpr(l) => is_inherently_nondeterministic_list_fn(l),
        #[cfg(feature = "strings")]
        F::StringExpr(_) => false,
        #[cfg(feature = "dtype-struct")]
        F::StructExpr(_) => false,
        #[cfg(feature = "temporal")]
        F::TemporalExpr(_) => false,
        #[cfg(feature = "bitwise")]
        F::Bitwise(_) => false,

        F::Boolean(_) => false,
        #[cfg(feature = "business")]
        F::Business(_) => false,
        #[cfg(feature = "abs")]
        F::Abs => false,
        F::Negate => false,
        #[cfg(feature = "hist")]
        F::Hist { .. } => false,
        F::NullCount => false,
        F::Pow(_) => false,
        #[cfg(feature = "row_hash")]
        F::Hash(..) => false,
        #[cfg(feature = "arg_where")]
        F::ArgWhere => false,
        #[cfg(feature = "index_of")]
        F::IndexOf => false,
        #[cfg(feature = "search_sorted")]
        F::SearchSorted { .. } => false,
        #[cfg(feature = "range")]
        F::Range(_) => false,
        #[cfg(feature = "trigonometry")]
        F::Trigonometry(_) => false,
        #[cfg(feature = "trigonometry")]
        F::Atan2 => false,
        #[cfg(feature = "sign")]
        F::Sign => false,
        F::FillNull => false,
        F::FillNullWithStrategy(_) => false,
        #[cfg(feature = "rolling_window")]
        F::RollingExpr { function, .. } => is_inherently_nondeterministic_rolling_fn(function),
        #[cfg(feature = "rolling_window_by")]
        F::RollingExprBy { .. } => false,
        F::Rechunk => false,
        F::ShiftAndFill => false,
        F::Shift => false,
        F::DropNans => false,
        F::DropNulls => false,
        F::Quantile { .. } => false,
        #[cfg(feature = "mode")]
        F::Mode { .. } => false,
        #[cfg(feature = "moment")]
        F::Skew(_) => false,
        #[cfg(feature = "moment")]
        F::Kurtosis(..) => false,
        #[cfg(feature = "dtype-array")]
        F::Reshape(_) => false,
        #[cfg(feature = "repeat_by")]
        F::RepeatBy => false,
        F::ArgUnique => false,
        F::ArgMin | F::ArgMax => false,
        F::ArgSort { .. } => false,
        F::MinBy | F::MaxBy => false,
        F::Product => false,
        // Only the `Random` tie-breaker is inherently non-deterministic.
        // The other `RankMethod` variants (Average, Min, Max, Dense,
        // Ordinal) are deterministic.
        #[cfg(all(feature = "rank", feature = "random"))]
        F::Rank { options, .. } => {
            matches!(options.method, polars_ops::series::RankMethod::Random)
        },
        #[cfg(all(feature = "rank", not(feature = "random")))]
        F::Rank { .. } => false,
        F::Repeat => false,
        #[cfg(feature = "round_series")]
        F::Clip { .. } => false,
        #[cfg(feature = "dtype-struct")]
        F::AsStruct => false,
        #[cfg(feature = "top_k")]
        F::TopK { .. } => false,
        #[cfg(feature = "top_k")]
        F::TopKBy { .. } => false,
        #[cfg(feature = "cum_agg")]
        F::CumCount { .. }
        | F::CumSum { .. }
        | F::CumProd { .. }
        | F::CumMin { .. }
        | F::CumMax { .. } => false,
        F::Reverse => false,
        #[cfg(feature = "dtype-struct")]
        F::ValueCounts { .. } => false,
        #[cfg(feature = "unique_counts")]
        F::UniqueCounts => false,
        #[cfg(feature = "approx_unique")]
        F::ApproxNUnique => false,
        F::Coalesce => false,
        #[cfg(feature = "diff")]
        F::Diff(_) => false,
        #[cfg(feature = "pct_change")]
        F::PctChange => false,
        #[cfg(feature = "interpolate")]
        F::Interpolate(_) => false,
        #[cfg(feature = "interpolate_by")]
        F::InterpolateBy => false,
        #[cfg(feature = "log")]
        F::Entropy { .. } => false,
        #[cfg(feature = "log")]
        F::Log | F::Log1p | F::Exp => false,
        F::Unique(_) => false,
        #[cfg(feature = "round_series")]
        F::Round { .. } | F::RoundSF { .. } | F::Truncate { .. } | F::Floor | F::Ceil => false,
        #[cfg(feature = "fused")]
        F::Fused(_) => false,
        F::ConcatExpr { .. } => false,
        #[cfg(feature = "cov")]
        F::Correlation { .. } => false,
        #[cfg(feature = "peaks")]
        F::PeakMin | F::PeakMax => false,
        #[cfg(feature = "cutqcut")]
        F::Cut { .. } | F::QCut { .. } => false,
        #[cfg(feature = "rle")]
        F::RLE | F::RLEID => false,
        F::ToPhysical => false,
        F::SetSortedFlag(_) => false,
        F::MaxHorizontal | F::MinHorizontal => false,
        F::SumHorizontal { .. } | F::MeanHorizontal { .. } => false,
        #[cfg(feature = "ewma")]
        F::EwmMean { .. } | F::EwmStd { .. } | F::EwmVar { .. } => false,
        #[cfg(feature = "ewma_by")]
        F::EwmMeanBy { .. } => false,
        #[cfg(feature = "replace")]
        F::Replace | F::ReplaceStrict { .. } => false,
        F::GatherEvery { .. } => false,
        #[cfg(feature = "reinterpret")]
        F::Reinterpret(_) => false,
        F::ExtendConstant => false,
        F::RowEncode(..) => false,
        #[cfg(feature = "dtype-struct")]
        F::RowDecode(..) => false,

        #[cfg(feature = "random")]
        F::Random { .. } => true,

        #[cfg(feature = "ffi_plugin")]
        F::FfiPlugin { .. } => true,
        F::FoldHorizontal { .. } | F::ReduceHorizontal { .. } => true,
        #[cfg(feature = "dtype-struct")]
        F::CumFoldHorizontal { .. } | F::CumReduceHorizontal { .. } => true,
        F::DynamicPred { .. } => true,
    }
}

#[cfg(feature = "dtype-array")]
fn is_inherently_nondeterministic_array_fn(f: &IRArrayFunction) -> bool {
    // Exhaustive match: no current variant is inherently non-deterministic,
    // but a new variant must trigger a compile error so its classification
    // gets a fresh decision.
    use IRArrayFunction as A;
    match f {
        A::Length
        | A::Min
        | A::Max
        | A::Sum
        | A::ToList
        | A::Unique(_)
        | A::NUnique
        | A::Std(_)
        | A::Var(_)
        | A::Mean
        | A::Median
        | A::Sort(_)
        | A::Reverse
        | A::ArgMin
        | A::ArgMax
        | A::Get(_)
        | A::Join(_)
        | A::Shift
        | A::Explode(_)
        | A::Concat
        | A::Slice(..) => false,
        #[cfg(feature = "is_in")]
        A::Contains { .. } => false,
        #[cfg(feature = "array_count")]
        A::CountMatches => false,
        #[cfg(feature = "array_to_struct")]
        A::ToStruct(_) => false,
    }
}

fn is_inherently_nondeterministic_binary_fn(f: &IRBinaryFunction) -> bool {
    // Exhaustive match: no current variant is inherently non-deterministic,
    // but a new variant must trigger a compile error so its classification
    // gets a fresh decision.
    use IRBinaryFunction as B;
    match f {
        B::Contains
        | B::StartsWith
        | B::EndsWith
        | B::Size
        | B::Slice
        | B::Head
        | B::Tail
        | B::Get(_) => false,
        #[cfg(feature = "binary_encoding")]
        B::HexDecode(_)
        | B::HexEncode
        | B::Base64Decode(_)
        | B::Base64Encode
        | B::Reinterpret(..) => false,
    }
}

fn is_inherently_nondeterministic_list_fn(f: &IRListFunction) -> bool {
    use IRListFunction as L;
    match f {
        L::Concat
        | L::Slice
        | L::Shift
        | L::Get(_)
        | L::Sum
        | L::Length
        | L::Max
        | L::Min
        | L::Mean
        | L::Median
        | L::Std(_)
        | L::Var(_)
        | L::ArgMin
        | L::ArgMax
        | L::Sort(_)
        | L::Reverse
        | L::Unique(_)
        | L::NUnique
        | L::Join(_) => false,
        #[cfg(feature = "is_in")]
        L::Contains { .. } => false,
        #[cfg(feature = "list_drop_nulls")]
        L::DropNulls => false,
        #[cfg(feature = "list_gather")]
        L::Gather(_) | L::GatherEvery => false,
        #[cfg(feature = "list_count")]
        L::CountMatches => false,
        #[cfg(feature = "diff")]
        L::Diff { .. } => false,
        #[cfg(feature = "list_sets")]
        L::SetOperation(_) => false,
        #[cfg(feature = "dtype-array")]
        L::ToArray(_) => false,
        #[cfg(feature = "list_to_struct")]
        L::ToStruct(_) => false,

        // Inherently non-deterministic: draws random samples.
        #[cfg(feature = "list_sample")]
        L::Sample { .. } => true,
    }
}

#[cfg(feature = "rolling_window")]
fn is_inherently_nondeterministic_rolling_fn(f: &IRRollingFunction) -> bool {
    use IRRollingFunction as R;
    match f {
        R::Min | R::Max | R::Mean | R::Sum | R::Quantile | R::Var | R::Std | R::Rank => false,
        #[cfg(feature = "moment")]
        R::Skew | R::Kurtosis => false,
        #[cfg(feature = "cov")]
        R::CorrCov { .. } => false,

        // Opaque rolling-window user callback.
        R::Map(_) => true,
    }
}
