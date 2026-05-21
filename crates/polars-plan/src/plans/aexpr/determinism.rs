use super::*;

/// Returns `true` iff evaluating `root`'s subtree on the same input row twice
/// is guaranteed to yield the same value.
///
/// Used by optimizer rewrites that change the number of times a subexpression
/// is evaluated per row, for example the OR-factoring distributivity rule
/// `(A ∧ X) ∨ (A ∧ Y) → A ∧ (X ∨ Y)`, which is only sound when `A` is
/// deterministic. Conservative on unknowns: any `AExpr` variant or
/// `IRFunctionExpr` not explicitly classified pure returns `false`, so a
/// newly added variant cannot silently break a rewrite that consumes this
/// helper. It only leaves optimization on the table until classified.
pub fn is_deterministic(root: Node, arena: &Arena<AExpr>) -> bool {
    let mut stack = vec![root];
    while let Some(n) = stack.pop() {
        match arena.get(n) {
            // Trivial leaves.
            AExpr::Column(_) | AExpr::Literal(_) | AExpr::Len => {},

            // Deterministic iff children are. Push children, keep walking.
            AExpr::BinaryExpr { left, right, .. } => {
                stack.push(*left);
                stack.push(*right);
            },
            AExpr::Cast { expr, .. } => stack.push(*expr),
            AExpr::Ternary {
                predicate,
                truthy,
                falsy,
            } => {
                stack.push(*predicate);
                stack.push(*truthy);
                stack.push(*falsy);
            },

            // Per-function classification, then recurse into the inputs.
            AExpr::Function {
                input, function, ..
            } => {
                if !is_deterministic_fn(function) {
                    return false;
                }
                for inp in input {
                    stack.push(inp.node());
                }
            },

            // Opaque user UDF.
            AExpr::AnonymousFunction { .. } => return false,

            // Everything else returns false. Agg, Sort, SortBy, Gather, Filter,
            // Slice, Explode, Over, Rolling work across rows, not one row at a
            // time, so "same row twice" has no answer. AnonymousAgg is an
            // opaque user function. Eval, StructEval, Element, StructField wrap
            // inner expressions we don't walk into. The rewrites using this
            // helper wouldn't try to move any of these around anyway.
            _ => return false,
        }
    }
    true
}

fn is_deterministic_fn(f: &IRFunctionExpr) -> bool {
    use IRFunctionExpr as F;
    match f {
        #[cfg(feature = "dtype-array")]
        F::ArrayExpr(a) => is_deterministic_array_fn(a),
        F::BinaryExpr(b) => is_deterministic_binary_fn(b),
        #[cfg(feature = "dtype-categorical")]
        F::Categorical(_) => true,
        #[cfg(feature = "dtype-extension")]
        F::Extension(_) => true,
        F::ListExpr(l) => is_deterministic_list_fn(l),
        #[cfg(feature = "strings")]
        F::StringExpr(_) => true,
        #[cfg(feature = "dtype-struct")]
        F::StructExpr(_) => true,
        #[cfg(feature = "temporal")]
        F::TemporalExpr(_) => true,
        #[cfg(feature = "bitwise")]
        F::Bitwise(_) => true,

        F::Boolean(_) => true,
        #[cfg(feature = "business")]
        F::Business(_) => true,
        #[cfg(feature = "abs")]
        F::Abs => true,
        F::Negate => true,
        #[cfg(feature = "hist")]
        F::Hist { .. } => true,
        F::NullCount => true,
        F::Pow(_) => true,
        #[cfg(feature = "row_hash")]
        F::Hash(..) => true,
        #[cfg(feature = "arg_where")]
        F::ArgWhere => true,
        #[cfg(feature = "index_of")]
        F::IndexOf => true,
        #[cfg(feature = "search_sorted")]
        F::SearchSorted { .. } => true,
        #[cfg(feature = "range")]
        F::Range(_) => true,
        #[cfg(feature = "trigonometry")]
        F::Trigonometry(_) => true,
        #[cfg(feature = "trigonometry")]
        F::Atan2 => true,
        #[cfg(feature = "sign")]
        F::Sign => true,
        F::FillNull => true,
        F::FillNullWithStrategy(_) => true,
        #[cfg(feature = "rolling_window")]
        F::RollingExpr { function, .. } => is_deterministic_rolling_fn(function),
        #[cfg(feature = "rolling_window_by")]
        F::RollingExprBy { .. } => true,
        F::Rechunk => true,
        F::ShiftAndFill => true,
        F::Shift => true,
        F::DropNans => true,
        F::DropNulls => true,
        F::Quantile { .. } => true,
        #[cfg(feature = "mode")]
        F::Mode { .. } => true,
        #[cfg(feature = "moment")]
        F::Skew(_) => true,
        #[cfg(feature = "moment")]
        F::Kurtosis(..) => true,
        #[cfg(feature = "dtype-array")]
        F::Reshape(_) => true,
        #[cfg(feature = "repeat_by")]
        F::RepeatBy => true,
        F::ArgUnique => true,
        F::ArgMin | F::ArgMax => true,
        F::ArgSort { .. } => true,
        F::MinBy | F::MaxBy => true,
        F::Product => true,
        // `Rank { method: RankMethod::Random, .. }` uses a random tie-breaker.
        // Bail false unconditionally. The `Random` variant of `RankMethod` is
        // gated on `polars-ops/random`, which polars-plan does not forward,
        // so a feature-conditional check is unreliable. Rank in a per-row
        // predicate is rare enough that losing optimisation here is fine.
        #[cfg(feature = "rank")]
        F::Rank { .. } => false,
        F::Repeat => true,
        #[cfg(feature = "round_series")]
        F::Clip { .. } => true,
        #[cfg(feature = "dtype-struct")]
        F::AsStruct => true,
        #[cfg(feature = "top_k")]
        F::TopK { .. } => true,
        #[cfg(feature = "top_k")]
        F::TopKBy { .. } => true,
        #[cfg(feature = "cum_agg")]
        F::CumCount { .. }
        | F::CumSum { .. }
        | F::CumProd { .. }
        | F::CumMin { .. }
        | F::CumMax { .. } => true,
        F::Reverse => true,
        #[cfg(feature = "dtype-struct")]
        F::ValueCounts { .. } => true,
        #[cfg(feature = "unique_counts")]
        F::UniqueCounts => true,
        #[cfg(feature = "approx_unique")]
        F::ApproxNUnique => true,
        F::Coalesce => true,
        #[cfg(feature = "diff")]
        F::Diff(_) => true,
        #[cfg(feature = "pct_change")]
        F::PctChange => true,
        #[cfg(feature = "interpolate")]
        F::Interpolate(_) => true,
        #[cfg(feature = "interpolate_by")]
        F::InterpolateBy => true,
        #[cfg(feature = "log")]
        F::Entropy { .. } => true,
        #[cfg(feature = "log")]
        F::Log | F::Log1p | F::Exp => true,
        F::Unique(_) => true,
        #[cfg(feature = "round_series")]
        F::Round { .. } | F::RoundSF { .. } | F::Truncate { .. } | F::Floor | F::Ceil => true,
        #[cfg(feature = "fused")]
        F::Fused(_) => true,
        F::ConcatExpr { .. } => true,
        #[cfg(feature = "cov")]
        F::Correlation { .. } => true,
        #[cfg(feature = "peaks")]
        F::PeakMin | F::PeakMax => true,
        #[cfg(feature = "cutqcut")]
        F::Cut { .. } | F::QCut { .. } => true,
        #[cfg(feature = "rle")]
        F::RLE | F::RLEID => true,
        F::ToPhysical => true,
        F::SetSortedFlag(_) => true,
        F::MaxHorizontal | F::MinHorizontal => true,
        F::SumHorizontal { .. } | F::MeanHorizontal { .. } => true,
        #[cfg(feature = "ewma")]
        F::EwmMean { .. } | F::EwmStd { .. } | F::EwmVar { .. } => true,
        #[cfg(feature = "ewma_by")]
        F::EwmMeanBy { .. } => true,
        #[cfg(feature = "replace")]
        F::Replace | F::ReplaceStrict { .. } => true,
        F::GatherEvery { .. } => true,
        #[cfg(feature = "reinterpret")]
        F::Reinterpret(_) => true,
        F::ExtendConstant => true,
        F::RowEncode(..) => true,
        #[cfg(feature = "dtype-struct")]
        F::RowDecode(..) => true,

        #[cfg(feature = "random")]
        F::Random { .. } => false,

        #[cfg(feature = "ffi_plugin")]
        F::FfiPlugin { .. } => false,
        F::FoldHorizontal { .. } | F::ReduceHorizontal { .. } => false,
        #[cfg(feature = "dtype-struct")]
        F::CumFoldHorizontal { .. } | F::CumReduceHorizontal { .. } => false,
        F::DynamicPred { .. } => false,
    }
}

#[cfg(feature = "dtype-array")]
fn is_deterministic_array_fn(f: &IRArrayFunction) -> bool {
    // Exhaustive match: every variant is currently pure, but a new variant
    // must trigger a compile error so its determinism gets a fresh decision.
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
        | A::Slice(..) => true,
        #[cfg(feature = "is_in")]
        A::Contains { .. } => true,
        #[cfg(feature = "array_count")]
        A::CountMatches => true,
        #[cfg(feature = "array_to_struct")]
        A::ToStruct(_) => true,
    }
}

fn is_deterministic_binary_fn(f: &IRBinaryFunction) -> bool {
    // Exhaustive match: every variant is currently pure, but a new variant
    // must trigger a compile error so its determinism gets a fresh decision.
    use IRBinaryFunction as B;
    match f {
        B::Contains
        | B::StartsWith
        | B::EndsWith
        | B::Size
        | B::Slice
        | B::Head
        | B::Tail
        | B::Get(_) => true,
        #[cfg(feature = "binary_encoding")]
        B::HexDecode(_)
        | B::HexEncode
        | B::Base64Decode(_)
        | B::Base64Encode
        | B::Reinterpret(..) => true,
    }
}

fn is_deterministic_list_fn(f: &IRListFunction) -> bool {
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
        | L::Join(_) => true,
        #[cfg(feature = "is_in")]
        L::Contains { .. } => true,
        #[cfg(feature = "list_drop_nulls")]
        L::DropNulls => true,
        #[cfg(feature = "list_gather")]
        L::Gather(_) | L::GatherEvery => true,
        #[cfg(feature = "list_count")]
        L::CountMatches => true,
        #[cfg(feature = "diff")]
        L::Diff { .. } => true,
        #[cfg(feature = "list_sets")]
        L::SetOperation(_) => true,
        #[cfg(feature = "dtype-array")]
        L::ToArray(_) => true,
        #[cfg(feature = "list_to_struct")]
        L::ToStruct(_) => true,

        // Volatile.
        #[cfg(feature = "list_sample")]
        L::Sample { .. } => false,
    }
}

#[cfg(feature = "rolling_window")]
fn is_deterministic_rolling_fn(f: &IRRollingFunction) -> bool {
    use IRRollingFunction as R;
    match f {
        R::Min | R::Max | R::Mean | R::Sum | R::Quantile | R::Var | R::Std | R::Rank => true,
        #[cfg(feature = "moment")]
        R::Skew | R::Kurtosis => true,
        #[cfg(feature = "cov")]
        R::CorrCov { .. } => true,

        // Opaque rolling-window user callback.
        R::Map(_) => false,
    }
}
