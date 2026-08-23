use std::hash::Hash;

use bitflags::bitflags;
use polars_core::prelude::*;
use polars_core::utils::SuperTypeOptions;
#[cfg(feature = "iejoin")]
use polars_ops::frame::IEJoinOptions;
use polars_ops::frame::{CrossJoinFilter, CrossJoinOptions, JoinArgs, JoinTypeOptions};
use polars_utils::bool::UnsafeBool;
use polars_utils::itertools::Itertools;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use strum_macros::IntoStaticStr;

use crate::dsl::JoinOptions;
#[cfg(feature = "cse")]
use crate::plans::ExpressionHasher;
use crate::plans::ir::inputs::{Exprs, ExprsMut};
use crate::plans::{ExprIR, ExpressionComparator, PlSmallStr};

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "ir_serde", derive(Serialize, Deserialize))]
pub struct DistinctOptionsIR {
    /// Subset of columns that will be taken into account.
    pub subset: Option<Arc<[PlSmallStr]>>,
    /// This will maintain the order of the input.
    /// Note that this is more expensive.
    /// `maintain_order` is not supported in the streaming
    /// engine.
    pub maintain_order: bool,
    /// Which rows to keep.
    pub keep_strategy: UniqueKeepStrategy,
    /// Take only a slice of the result
    pub slice: Option<(i64, usize)>,
}

#[cfg(feature = "dsl-schema")]
impl schemars::JsonSchema for FunctionFlags {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "FunctionFlags".into()
    }

    fn schema_id() -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed(concat!(module_path!(), "::", "FunctionFlags"))
    }

    fn json_schema(_generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        use schemars::json_schema;
        use serde_json::{Map, Value};

        // Add a map of flag names and bit patterns to detect schema changes
        let name_to_bits: Map<String, Value> = Self::all()
            .iter_names()
            .map(|(name, flag)| (name.to_owned(), flag.bits().into()))
            .collect();

        json_schema!({
            "type": "string",
            "format": "bitflags",
            "bitflags": name_to_bits
        })
    }
}

bitflags!(
        #[repr(transparent)]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
        pub struct FunctionFlags: u16 {
            /// The physical expression may rename the output of this function.
            /// If set to `false` the physical engine will ensure the left input
            /// expression is the output name.
            const ALLOW_RENAME = 1 << 0;
            /// if set, then the `Series` passed to the function in the group_by operation
            /// will ensure the name is set. This is an extra heap allocation per group.
            const PASS_NAME_TO_APPLY = 1 << 1;
            /// There can be two ways of expanding wildcards:
            ///
            /// Say the schema is 'a', 'b' and there is a function `f`. In this case, `f('*')` can expand
            /// to:
            /// 1. `f('a', 'b')`
            /// 2. `f('a'), f('b')`
            ///
            /// Setting this to true, will lead to behavior 1.
            ///
            /// This also accounts for regex expansion.
            const INPUT_WILDCARD_EXPANSION = 1 << 2;
            /// Automatically explode on unit length if it ran as final aggregation.
            ///
            /// this is the case for aggregations like sum, min, covariance etc.
            /// We need to know this because we cannot see the difference between
            /// the following functions based on the output type and number of elements:
            ///
            /// x: {1, 2, 3}
            ///
            /// head_1(x) -> {1}
            /// sum(x) -> {4}
            ///
            /// mutually exclusive with `RETURNS_SCALAR`
            const RETURNS_SCALAR = 1 << 3;
            /// This can happen with UDF's that use Polars within the UDF.
            /// This can lead to recursively entering the engine and sometimes deadlocks.
            /// This flag must be set to handle that.
            const OPTIONAL_RE_ENTRANT = 1 << 4;
            /// Whether this function allows no inputs.
            const ALLOW_EMPTY_INPUTS = 1 << 5;

            /// Given a function f and a column of values [v1, ..., vn]
            /// f is row-separable i.f.f.
            /// f([v1, ..., vn]) = concat(f(v1, ... vm), f(vm+1, ..., vn))
            const ROW_SEPARABLE = 1 << 6;
            /// Given a function f and a column of values [v1, ..., vn]
            /// f is length preserving i.f.f. len(f([v1, ..., vn])) = n
            ///
            /// mutually exclusive with `RETURNS_SCALAR`
            const LENGTH_PRESERVING = 1 << 7;
            /// NULLs on the first input are propagated to the output.
            const PRESERVES_NULL_FIRST_INPUT = 1 << 8;
            /// NULLs on any input are propagated to the output.
            const PRESERVES_NULL_ALL_INPUTS = 1 << 9;

            /// Indicates that this expression does not observe the ordering of its input(s).
            const NON_ORDER_OBSERVING = 1 << 10;

            /// Indicates that the ordering of the inputs to this expression is not observable
            /// in its output.
            const TERMINATES_INPUT_ORDER = 1 << 11;

            /// Indicates that this expression does not produce any ordering into its output.
            const NON_ORDER_PRODUCING = 1 << 12;

            /// Produces a RANGE based on its inputs.
            const RANGE = 1 << 13;
        }
);

impl FunctionFlags {
    pub fn set_elementwise(&mut self) {
        *self |= Self::ROW_SEPARABLE | Self::LENGTH_PRESERVING;
    }

    pub fn is_elementwise(self) -> bool {
        self.contains(Self::ROW_SEPARABLE | Self::LENGTH_PRESERVING)
    }

    pub fn is_row_separable(self) -> bool {
        self.contains(Self::ROW_SEPARABLE)
    }

    pub fn is_length_preserving(self) -> bool {
        self.contains(Self::LENGTH_PRESERVING)
    }

    pub fn observes_input_order(self) -> bool {
        let non_order_observing =
            self.contains(Self::NON_ORDER_OBSERVING) | self.contains(Self::ROW_SEPARABLE);

        !non_order_observing
    }

    pub fn terminates_input_order(self) -> bool {
        self.contains(Self::TERMINATES_INPUT_ORDER) | self.contains(Self::RETURNS_SCALAR)
    }

    pub fn non_order_producing(self) -> bool {
        self.contains(Self::NON_ORDER_PRODUCING)
            | self.contains(Self::RETURNS_SCALAR)
            | self.is_elementwise()
    }

    pub fn returns_scalar(self) -> bool {
        self.contains(Self::RETURNS_SCALAR)
    }

    pub fn is_range(self) -> bool {
        self.contains(Self::RANGE)
    }
}

impl Default for FunctionFlags {
    fn default() -> Self {
        Self::from_bits_truncate(0)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum CastingRules {
    /// Whether information may be lost during cast. E.g. a float to int is considered lossy,
    /// whereas int to int is considered lossless.
    /// Overflowing is not considered in this flag, that's handled in `strict` casting
    FirstArgLossless,
    Supertype(SuperTypeOptions),
}

impl CastingRules {
    pub fn cast_to_supertypes() -> CastingRules {
        Self::Supertype(Default::default())
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, Default)]
#[cfg_attr(any(feature = "serde"), derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub struct FunctionOptions {
    // Validate the output of a `map`.
    // this should always be true or we could OOB
    pub check_lengths: UnsafeBool,
    pub flags: FunctionFlags,

    /// Options used when deciding how to cast the arguments of the function.
    #[cfg_attr(any(feature = "serde", feature = "dsl-schema"), serde(skip))]
    pub cast_options: Option<CastingRules>,
}

impl FunctionOptions {
    #[cfg(feature = "fused")]
    pub(crate) unsafe fn no_check_lengths(&mut self) {
        unsafe { self.check_lengths = UnsafeBool::new_false() };
    }
    pub fn check_lengths(&self) -> bool {
        *self.check_lengths
    }

    pub fn set_elementwise(&mut self) {
        self.flags.set_elementwise();
    }

    pub fn is_elementwise(&self) -> bool {
        self.flags.is_elementwise()
    }

    pub fn is_length_preserving(&self) -> bool {
        self.flags.contains(FunctionFlags::LENGTH_PRESERVING)
    }

    pub fn is_row_separable(&self) -> bool {
        self.flags.is_row_separable()
    }

    pub fn returns_scalar(&self) -> bool {
        self.flags.returns_scalar()
    }

    pub fn elementwise() -> FunctionOptions {
        FunctionOptions {
            ..Default::default()
        }
        .with_flags(|f| f | FunctionFlags::ROW_SEPARABLE | FunctionFlags::LENGTH_PRESERVING)
    }

    pub fn elementwise_with_infer() -> FunctionOptions {
        Self::length_preserving()
    }

    pub fn row_separable() -> FunctionOptions {
        FunctionOptions {
            ..Default::default()
        }
        .with_flags(|f| f | FunctionFlags::ROW_SEPARABLE)
    }

    pub fn length_preserving() -> FunctionOptions {
        FunctionOptions {
            ..Default::default()
        }
        .with_flags(|f| f | FunctionFlags::LENGTH_PRESERVING)
    }

    /// Will respect group boundaries. Shift, Reverse, etc.
    pub fn groupwise() -> FunctionOptions {
        FunctionOptions {
            ..Default::default()
        }
    }

    pub fn aggregation() -> FunctionOptions {
        let mut options = Self::groupwise();
        options.flags |= FunctionFlags::RETURNS_SCALAR;
        options
    }

    pub fn with_supertyping(self, supertype_options: SuperTypeOptions) -> FunctionOptions {
        self.with_casting_rules(CastingRules::Supertype(supertype_options))
    }

    pub fn with_casting_rules(mut self, casting_rules: CastingRules) -> FunctionOptions {
        self.cast_options = Some(casting_rules);
        self
    }

    pub fn flag(mut self, flags: FunctionFlags) -> FunctionOptions {
        self.flags |= flags;
        self
    }

    pub fn with_flags(mut self, f: impl Fn(FunctionFlags) -> FunctionFlags) -> FunctionOptions {
        self.flags = f(self.flags);
        self
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ProjectionOptions {
    pub run_parallel: bool,
    pub duplicate_check: bool,
    // Should length-1 Series be broadcast to the length of the dataframe.
    // Only used by CSE optimizer
    pub should_broadcast: bool,
}

impl Default for ProjectionOptions {
    fn default() -> Self {
        Self {
            run_parallel: true,
            duplicate_check: true,
            should_broadcast: true,
        }
    }
}

impl ProjectionOptions {
    /// Conservatively merge the options of two [`ProjectionOptions`]
    pub fn merge_options(&self, other: &Self) -> Self {
        Self {
            run_parallel: self.run_parallel & other.run_parallel,
            duplicate_check: self.duplicate_check & other.duplicate_check,
            should_broadcast: self.should_broadcast | other.should_broadcast,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Hash)]
#[cfg_attr(feature = "ir_serde", derive(Serialize, Deserialize))]
pub struct JoinOptionsIR {
    pub allow_parallel: bool,
    pub force_parallel: bool,
    pub args: JoinArgs,
    pub options: JoinTypeOptionsIR,
}

impl JoinOptionsIR {
    /// The match condition is exactly `left_on == right_on`.
    pub fn is_pure_equi(&self) -> bool {
        !self.options.is_non_equi()
    }

    /// The match condition has a non-equality component, held in `options`.
    pub fn is_non_equi(&self) -> bool {
        self.options.is_non_equi()
    }

    pub(crate) fn shallow_eq(&self, other: &Self, expr_cmp: &impl ExpressionComparator) -> bool {
        let Self {
            allow_parallel,
            force_parallel,
            args,
            options,
        } = self;

        *allow_parallel == other.allow_parallel
            && *force_parallel == other.force_parallel
            && *args == other.args
            && options.shallow_eq(&other.options, expr_cmp)
    }

    #[cfg(feature = "cse")]
    pub(crate) fn shallow_hash<H: std::hash::Hasher>(
        &self,
        state: &mut H,
        expr_hash: &impl ExpressionHasher,
    ) {
        let Self {
            allow_parallel,
            force_parallel,
            args,
            options,
        } = self;

        allow_parallel.hash(state);
        force_parallel.hash(state);
        args.hash(state);
        options.shallow_hash(state, expr_hash);
    }
}

#[derive(Clone, PartialEq, Eq, IntoStaticStr, Debug)]
#[cfg_attr(feature = "ir_serde", derive(Serialize, Deserialize))]
#[strum(serialize_all = "snake_case")]
pub enum JoinTypeOptionsIR {
    /// The match condition is `left == right` for every key pair.
    ///
    /// An empty `on` is a plain cross join.
    Equi { on: Vec<(ExprIR, ExprIR)> },
    /// Backwards/forwards/nearest match on a single key pair. The strategy, tolerance
    /// and `by` group keys live in [`JoinType::AsOf`].
    #[cfg(feature = "asof_join")]
    AsOf { on: Vec<(ExprIR, ExprIR)> },
    /// Inequality join over one or two arbitrary predicates.
    ///
    /// `operator1` relates the first key pair, `operator2` the second.
    #[cfg(feature = "iejoin")]
    IEJoin {
        ie_options: IEJoinOptions,
        left_on: Vec<ExprIR>,
        right_on: Vec<ExprIR>,
    },
    /// `point ∈ [lower, upper]`. Only reached by the streaming engine.
    ///
    /// The side with one key holds the point, the other holds the bounds:
    /// - double bounded: bound side has 2 keys, `operator1` is the lower bound op and
    ///   `operator2` the upper.
    /// - single bounded: both sides have 1 key and `operator2` is `None`.
    #[cfg(feature = "iejoin")]
    Range {
        ie_options: IEJoinOptions,
        left_on: Vec<ExprIR>,
        right_on: Vec<ExprIR>,
    },
    // Fused cross join and filter. Executed by the in-memory engine; the streaming
    // engine has no native node for this and falls back to it via `InMemoryJoin`.
    CrossAndFilter {
        predicate: ExprIR, // Must be elementwise.
    },
}

impl Default for JoinTypeOptionsIR {
    fn default() -> Self {
        Self::Equi { on: Vec::new() }
    }
}

impl Hash for JoinTypeOptionsIR {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);

        #[cfg(feature = "iejoin")]
        if let Self::IEJoin { ie_options, .. } | Self::Range { ie_options, .. } = self {
            ie_options.hash(state);
        }

        self.exprs().count().hash(state);
        for expr in self.exprs() {
            expr.node().hash(state);
        }
    }
}

impl JoinTypeOptionsIR {
    pub(crate) fn shallow_eq(&self, other: &Self, expr_cmp: &impl ExpressionComparator) -> bool {
        if std::mem::discriminant(self) != std::mem::discriminant(other) {
            return false;
        }

        #[cfg(feature = "iejoin")]
        if let (
            Self::IEJoin { ie_options, .. } | Self::Range { ie_options, .. },
            Self::IEJoin {
                ie_options: other_ie,
                ..
            }
            | Self::Range {
                ie_options: other_ie,
                ..
            },
        ) = (self, other)
            && ie_options != other_ie
        {
            return false;
        }

        // Equal discriminants, so both sides hold the same shape and `exprs()` walks them
        // in the same order.
        self.exprs()
            .eq_by_(other.exprs(), |l, r| expr_cmp.equals(l, r))
    }

    #[cfg(feature = "cse")]
    pub(crate) fn shallow_hash<H: std::hash::Hasher>(
        &self,
        state: &mut H,
        expr_hash: &impl ExpressionHasher,
    ) {
        std::mem::discriminant(self).hash(state);

        #[cfg(feature = "iejoin")]
        if let Self::IEJoin { ie_options, .. } | Self::Range { ie_options, .. } = self {
            ie_options.hash(state);
        }

        self.exprs().count().hash(state);
        for expr in self.exprs() {
            expr_hash.hash_expr(expr, state);
        }
    }

    pub fn compile<C: FnOnce(&ExprIR) -> PolarsResult<Arc<dyn CrossJoinFilter>>>(
        self,
        plan: C,
    ) -> PolarsResult<Option<JoinTypeOptions>> {
        use JoinTypeOptionsIR::*;
        match self {
            CrossAndFilter { predicate } => {
                let predicate = plan(&predicate)?;

                Ok(Some(JoinTypeOptions::Cross(CrossJoinOptions { predicate })))
            },
            #[cfg(feature = "iejoin")]
            IEJoin { ie_options, .. } | Range { ie_options, .. } => {
                Ok(Some(JoinTypeOptions::IEJoin(ie_options)))
            },
            Equi { .. } => Ok(None),
            #[cfg(feature = "asof_join")]
            AsOf { .. } => Ok(None),
        }
    }

    /// The keys of the two variants that store them as pairs.
    ///
    /// The only place `Equi`/`AsOf` are matched apart; every accessor below goes through it.
    /// They cannot share an or-pattern arm because rustc rejects `#[cfg]` on one alternative.
    fn key_pairs(&self) -> Option<&Vec<(ExprIR, ExprIR)>> {
        match self {
            Self::Equi { on } => Some(on),
            #[cfg(feature = "asof_join")]
            Self::AsOf { on } => Some(on),
            _ => None,
        }
    }

    /// See [`Self::key_pairs`].
    fn key_pairs_mut(&mut self) -> Option<&mut Vec<(ExprIR, ExprIR)>> {
        match self {
            Self::Equi { on } => Some(on),
            #[cfg(feature = "asof_join")]
            Self::AsOf { on } => Some(on),
            _ => None,
        }
    }

    /// The left-hand side keys, in positional order.
    ///
    /// For [`Self::Range`] this can differ in length from [`Self::right_on`], so only zip the
    /// two sides once the condition is known not to be a range.
    pub fn left_on(&self) -> Exprs<'_> {
        if let Some(on) = self.key_pairs() {
            return Exprs::pair_lhs(on);
        }
        match self {
            #[cfg(feature = "iejoin")]
            Self::IEJoin { left_on, .. } | Self::Range { left_on, .. } => Exprs::slice(left_on),
            _ => Exprs::Empty,
        }
    }

    /// The right-hand side keys, in positional order. Same caveat as [`Self::left_on`].
    pub fn right_on(&self) -> Exprs<'_> {
        if let Some(on) = self.key_pairs() {
            return Exprs::pair_rhs(on);
        }
        match self {
            #[cfg(feature = "iejoin")]
            Self::IEJoin { right_on, .. } | Self::Range { right_on, .. } => Exprs::slice(right_on),
            _ => Exprs::Empty,
        }
    }

    pub fn left_on_len(&self) -> usize {
        if let Some(on) = self.key_pairs() {
            return on.len();
        }
        match self {
            #[cfg(feature = "iejoin")]
            Self::IEJoin { left_on, .. } | Self::Range { left_on, .. } => left_on.len(),
            _ => 0,
        }
    }

    pub fn right_on_len(&self) -> usize {
        if let Some(on) = self.key_pairs() {
            return on.len();
        }
        match self {
            #[cfg(feature = "iejoin")]
            Self::IEJoin { right_on, .. } | Self::Range { right_on, .. } => right_on.len(),
            _ => 0,
        }
    }

    pub fn key_vecs(&self) -> (Vec<ExprIR>, Vec<ExprIR>) {
        (
            self.left_on().cloned().collect(),
            self.right_on().cloned().collect(),
        )
    }

    /// Every left key, then every right key, then any fused predicate.
    ///
    /// The order must stay in sync with [`Self::exprs_mut`].
    pub fn exprs(&self) -> Exprs<'_> {
        if let Some(on) = self.key_pairs() {
            return Exprs::pair_sides(on);
        }
        match self {
            #[cfg(feature = "iejoin")]
            Self::IEJoin {
                left_on, right_on, ..
            }
            | Self::Range {
                left_on, right_on, ..
            } => Exprs::double_slice(left_on, right_on),
            Self::CrossAndFilter { predicate } => Exprs::single(predicate),
            _ => Exprs::Empty,
        }
    }

    /// See [`Self::exprs`]. Yields in the same order.
    pub fn exprs_mut(&mut self) -> ExprsMut<'_> {
        // Checked first so the mutable borrow does not span the match below.
        if self.key_pairs().is_some() {
            return ExprsMut::pair_sides(self.key_pairs_mut().unwrap());
        }
        match self {
            #[cfg(feature = "iejoin")]
            Self::IEJoin {
                left_on, right_on, ..
            }
            | Self::Range {
                left_on, right_on, ..
            } => ExprsMut::double_slice(left_on, right_on),
            Self::CrossAndFilter { predicate } => ExprsMut::single(predicate),
            _ => ExprsMut::Empty,
        }
    }

    /// Replace the keys, keeping the current variant.
    ///
    /// # Panics
    /// On [`Self::CrossAndFilter`], which holds no keys, or on a length mismatch where the
    /// variant stores keys in pairs.
    pub fn set_keys(&mut self, left: Vec<ExprIR>, right: Vec<ExprIR>) {
        if let Some(on) = self.key_pairs_mut() {
            *on = left.into_iter().zip_eq(right).collect();
            return;
        }
        match self {
            #[cfg(feature = "iejoin")]
            Self::IEJoin {
                left_on, right_on, ..
            }
            | Self::Range {
                left_on, right_on, ..
            } => {
                *left_on = left;
                *right_on = right;
            },
            _ => panic!("cross join filter holds no keys"),
        }
    }

    /// The match condition is exactly `left == right` for every key pair.
    ///
    /// True for [`Self::AsOf`] too: its strategy and tolerance live in [`JoinType::AsOf`],
    /// not in the match condition.
    pub fn is_pure_equi(&self) -> bool {
        !self.is_non_equi()
    }

    /// The match condition has a non-equality component.
    pub fn is_non_equi(&self) -> bool {
        self.key_pairs().is_none()
    }
}

impl From<JoinOptions> for JoinOptionsIR {
    fn from(opts: JoinOptions) -> Self {
        Self {
            allow_parallel: opts.allow_parallel,
            force_parallel: opts.force_parallel,
            args: opts.args,
            options: Default::default(),
        }
    }
}

impl From<JoinOptionsIR> for JoinOptions {
    fn from(opts: JoinOptionsIR) -> Self {
        Self {
            allow_parallel: opts.allow_parallel,
            force_parallel: opts.force_parallel,
            args: opts.args,
        }
    }
}
