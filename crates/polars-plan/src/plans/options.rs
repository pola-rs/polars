use std::hash::Hash;

use bitflags::bitflags;
use polars_core::prelude::*;
use polars_core::utils::SuperTypeOptions;
#[cfg(feature = "asof_join")]
use polars_defs::join::AsOfOptions;
#[cfg(feature = "iejoin")]
use polars_defs::join::IEJoinOptions;
use polars_defs::join::{
    CrossJoinFilter, CrossJoinOptions, JoinArgs, JoinBuildSide, JoinCoalesce, JoinType,
    JoinTypeOptions, JoinValidation, MaintainOrderJoin,
};
#[cfg(feature = "dynamic_group_by")]
use polars_defs::time::group_by::{DynamicGroupOptionsIR, RollingGroupOptionsIR};
use polars_utils::bool::UnsafeBool;
use polars_utils::itertools::Itertools;
use polars_utils::slice_enum::Slice;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use strum_macros::IntoStaticStr;

use crate::dsl::GroupbyOptions;
#[cfg(feature = "cse")]
use crate::plans::ExpressionHasher;
use crate::plans::ir::inputs::{ExprSchema, Exprs, ExprsMut};
use crate::plans::{DynamicPred, ExprIR, ExpressionComparator, PlSmallStr};

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
    pub slice: Option<Slice>,
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
    // Maintain the input height on empty select(). Used by drop().
    pub maintain_dataframe_height: bool,
}

impl Default for ProjectionOptions {
    fn default() -> Self {
        Self {
            run_parallel: true,
            duplicate_check: true,
            should_broadcast: true,
            maintain_dataframe_height: false,
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
            maintain_dataframe_height: self.maintain_dataframe_height
                & other.maintain_dataframe_height,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Default, Hash)]
#[cfg_attr(feature = "ir_serde", derive(Serialize, Deserialize))]
pub struct GroupbyOptionsIR {
    #[cfg(feature = "dynamic_group_by")]
    pub dynamic: Option<DynamicGroupOptionsIR>,
    #[cfg(feature = "dynamic_group_by")]
    pub rolling: Option<RollingGroupOptionsIR>,
    /// Take only a slice of the result
    pub slice: Option<Slice>,
}

impl GroupbyOptionsIR {
    pub fn is_rolling(&self) -> bool {
        #[cfg(feature = "dynamic_group_by")]
        {
            self.rolling.is_some()
        }
        #[cfg(not(feature = "dynamic_group_by"))]
        {
            false
        }
    }

    pub fn is_dynamic(&self) -> bool {
        #[cfg(feature = "dynamic_group_by")]
        {
            self.dynamic.is_some()
        }
        #[cfg(not(feature = "dynamic_group_by"))]
        {
            false
        }
    }
}

impl From<GroupbyOptions> for GroupbyOptionsIR {
    fn from(opts: GroupbyOptions) -> Self {
        Self {
            #[cfg(feature = "dynamic_group_by")]
            dynamic: opts.dynamic.map(Into::into),
            #[cfg(feature = "dynamic_group_by")]
            rolling: opts.rolling.map(Into::into),
            slice: opts.slice,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Hash)]
#[cfg_attr(feature = "ir_serde", derive(Serialize, Deserialize))]
pub struct JoinOptionsIR {
    pub allow_parallel: bool,
    pub force_parallel: bool,
    /// The join type together with its match condition, which includes the join keys.
    pub join_type: JoinTypeIR,
    pub args: JoinArgsIR,
    /// Ranges of build-side keys the join publishes to scans below its probe side once
    /// the build is done. Only set together with a forced `args.build_side`.
    pub runtime_filters: Vec<RuntimeFilter>,
    /// The estimated rows of the left and right input of a join that has no
    /// `args.build_side`.
    #[cfg_attr(feature = "ir_serde", serde(default))]
    pub estimated_rows: Option<[u64; 2]>,
}

/// The settings of a join that apply to every join type.
///
/// The settings that only some join types have are in their [`JoinTypeIR`] variant.
#[derive(Clone, Debug, PartialEq, Hash, Default)]
#[cfg_attr(feature = "ir_serde", derive(Serialize, Deserialize))]
pub struct JoinArgsIR {
    pub suffix: Option<PlSmallStr>,
    pub slice: Option<Slice>,
    pub maintain_order: MaintainOrderJoin,
    pub build_side: Option<JoinBuildSide>,
}

impl JoinArgsIR {
    pub fn suffix(&self) -> &PlSmallStr {
        const DEFAULT: &PlSmallStr = &PlSmallStr::from_static("_right");
        self.suffix.as_ref().unwrap_or(DEFAULT)
    }
}

/// Largest ratio of distinct build keys to distinct probe keys for which a
/// bloom filter is published.
pub const MAX_BUILD_PROBE_DISTINCT_RATIO: f64 = 0.3;

/// The build-side key at `key_idx` of the join's `on`, published through
/// `pred` for the probe side: its range, and a bloom filter over its values
/// when `bloom_keys` gives the number of distinct keys to size it for.
#[derive(Clone, Debug, PartialEq, Hash)]
#[cfg_attr(feature = "ir_serde", derive(Serialize, Deserialize))]
pub struct RuntimeFilter {
    pub key_idx: usize,
    pub pred: DynamicPred,
    pub bloom_keys: Option<usize>,
    /// The probe side's distinct keys, when the plan can estimate them.
    pub probe_distinct: Option<usize>,
}

impl JoinOptionsIR {
    /// The match condition is exactly `left_on == right_on`.
    pub fn is_pure_equi(&self) -> bool {
        self.join_type.is_pure_equi()
    }

    /// The match condition has a non-equality component.
    pub fn is_non_equi(&self) -> bool {
        self.join_type.is_non_equi()
    }

    /// Whether the left and right key columns are merged into one output column.
    pub fn should_coalesce(&self) -> bool {
        self.join_type.should_coalesce()
    }

    /// The arguments in the form that the executors take.
    pub fn to_join_args(&self) -> JoinArgs {
        let JoinArgsIR {
            suffix,
            slice,
            maintain_order,
            build_side,
        } = self.args.clone();
        JoinArgs {
            how: self.join_type.to_join_type(),
            validation: self.join_type.validation(),
            suffix,
            slice,
            nulls_equal: self.join_type.nulls_equal(),
            coalesce: self.join_type.coalesce(),
            maintain_order,
            build_side,
        }
    }

    /// Errors if no engine can execute this join.
    ///
    /// Whether a non-equi condition ends up executable depends on how far the optimizer
    /// managed to lower it, so this can only be answered once the plan is optimized.
    pub fn ensure_executable(&self) -> PolarsResult<()> {
        let supported = match &self.join_type {
            JoinTypeIR::Equi {
                how,
                on,
                fused_predicate: Some(_),
                ..
            } => {
                // An outer join must evaluate the predicate per candidate pair, since a
                // filter afterwards would drop its unmatched rows. Only the unordered hash
                // join does that.
                !on.is_empty()
                    && self.args.slice.is_none()
                    && match how {
                        EquiJoinType::Inner => true,
                        EquiJoinType::Left | EquiJoinType::Right | EquiJoinType::Full => {
                            self.args.maintain_order == MaintainOrderJoin::None
                        },
                        #[cfg(feature = "semi_anti_join")]
                        EquiJoinType::Semi | EquiJoinType::Anti => false,
                    }
            },
            _ => true,
        };

        polars_ensure!(
            supported,
            InvalidOperation:
            "'{}' join is not supported with non-equi join conditions",
            self.join_type.how(),
        );
        Ok(())
    }

    pub(crate) fn shallow_eq(&self, other: &Self, expr_cmp: &impl ExpressionComparator) -> bool {
        let Self {
            allow_parallel,
            force_parallel,
            join_type,
            args,
            runtime_filters,
            estimated_rows,
        } = self;

        *allow_parallel == other.allow_parallel
            && *force_parallel == other.force_parallel
            && join_type.shallow_eq(&other.join_type, expr_cmp)
            && *args == other.args
            && *runtime_filters == other.runtime_filters
            && *estimated_rows == other.estimated_rows
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
            join_type,
            args,
            runtime_filters,
            estimated_rows,
        } = self;

        allow_parallel.hash(state);
        force_parallel.hash(state);
        join_type.shallow_hash(state, expr_hash);
        args.hash(state);
        runtime_filters.hash(state);
        estimated_rows.hash(state);
    }
}

/// Which rows an equi join emits.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "ir_serde", derive(Serialize, Deserialize))]
pub enum EquiJoinType {
    Inner,
    Left,
    Right,
    Full,
    #[cfg(feature = "semi_anti_join")]
    Semi,
    #[cfg(feature = "semi_anti_join")]
    Anti,
}

/// Which rows a join with a non-equality match condition emits.
///
/// Semi and anti joins only exist with an equality condition.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "ir_serde", derive(Serialize, Deserialize))]
pub enum NonEquiJoinType {
    Inner,
    Left,
    Right,
    Full,
}

impl From<NonEquiJoinType> for EquiJoinType {
    fn from(how: NonEquiJoinType) -> Self {
        match how {
            NonEquiJoinType::Inner => Self::Inner,
            NonEquiJoinType::Left => Self::Left,
            NonEquiJoinType::Right => Self::Right,
            NonEquiJoinType::Full => Self::Full,
        }
    }
}

impl From<EquiJoinType> for JoinHow {
    fn from(how: EquiJoinType) -> Self {
        match how {
            EquiJoinType::Inner => Self::Inner,
            EquiJoinType::Left => Self::Left,
            EquiJoinType::Right => Self::Right,
            EquiJoinType::Full => Self::Full,
            #[cfg(feature = "semi_anti_join")]
            EquiJoinType::Semi => Self::Semi,
            #[cfg(feature = "semi_anti_join")]
            EquiJoinType::Anti => Self::Anti,
        }
    }
}

impl From<NonEquiJoinType> for JoinHow {
    fn from(how: NonEquiJoinType) -> Self {
        EquiJoinType::from(how).into()
    }
}

/// The join type of a [`JoinTypeIR`] as a plain value, named as [`JoinType`] names it.
///
/// An outer join is named by the rows it emits. An inner join with a non-equality
/// condition is named by its algorithm instead: `Cross`, `IEJoin` or `Range`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, IntoStaticStr)]
pub enum JoinHow {
    Inner,
    Left,
    Right,
    Full,
    #[cfg(feature = "semi_anti_join")]
    Semi,
    #[cfg(feature = "semi_anti_join")]
    Anti,
    #[cfg(feature = "asof_join")]
    AsOf,
    #[cfg(feature = "iejoin")]
    IEJoin,
    #[cfg(feature = "iejoin")]
    Range,
    Cross,
}

impl JoinHow {
    pub fn is_equi(self) -> bool {
        matches!(self, Self::Inner | Self::Left | Self::Right | Self::Full)
    }

    pub fn is_semi_anti(self) -> bool {
        self.is_semi() || self.is_anti()
    }

    pub fn is_semi(self) -> bool {
        #[cfg(feature = "semi_anti_join")]
        {
            matches!(self, Self::Semi)
        }
        #[cfg(not(feature = "semi_anti_join"))]
        {
            false
        }
    }

    pub fn is_anti(self) -> bool {
        #[cfg(feature = "semi_anti_join")]
        {
            matches!(self, Self::Anti)
        }
        #[cfg(not(feature = "semi_anti_join"))]
        {
            false
        }
    }

    pub fn is_asof(self) -> bool {
        #[cfg(feature = "asof_join")]
        {
            matches!(self, Self::AsOf)
        }
        #[cfg(not(feature = "asof_join"))]
        {
            false
        }
    }

    pub fn is_inner(self) -> bool {
        matches!(self, Self::Inner)
    }

    pub fn is_cross(self) -> bool {
        matches!(self, Self::Cross)
    }

    pub fn is_ie(self) -> bool {
        #[cfg(feature = "iejoin")]
        {
            matches!(self, Self::IEJoin)
        }
        #[cfg(not(feature = "iejoin"))]
        {
            false
        }
    }

    pub fn is_range(self) -> bool {
        #[cfg(feature = "iejoin")]
        {
            matches!(self, Self::Range)
        }
        #[cfg(not(feature = "iejoin"))]
        {
            false
        }
    }

    /// Unmatched rows of the left input appear in the output.
    pub fn emits_unmatched_left(self) -> bool {
        matches!(self, Self::Left | Self::Full) || self.is_anti()
    }

    /// Unmatched rows of the right input appear in the output.
    pub fn emits_unmatched_right(self) -> bool {
        matches!(self, Self::Right | Self::Full)
    }

    /// Whether a join of this type merges its left and right key columns, given the
    /// `coalesce` setting. Mirrors [`JoinCoalesce::coalesce`].
    fn should_coalesce(self, coalesce: JoinCoalesce) -> bool {
        use JoinCoalesce::*;
        match self {
            Self::Left | Self::Inner | Self::Right => {
                matches!(coalesce, JoinSpecific | CoalesceColumns)
            },
            Self::Full => matches!(coalesce, CoalesceColumns),
            #[cfg(feature = "asof_join")]
            Self::AsOf => matches!(coalesce, JoinSpecific | CoalesceColumns),
            #[cfg(feature = "iejoin")]
            Self::IEJoin | Self::Range => false,
            Self::Cross => false,
            #[cfg(feature = "semi_anti_join")]
            Self::Semi | Self::Anti => false,
        }
    }
}

impl std::fmt::Display for JoinHow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let val = match self {
            Self::Inner => "INNER",
            Self::Left => "LEFT",
            Self::Right => "RIGHT",
            Self::Full => "FULL",
            #[cfg(feature = "semi_anti_join")]
            Self::Semi => "SEMI",
            #[cfg(feature = "semi_anti_join")]
            Self::Anti => "ANTI",
            #[cfg(feature = "asof_join")]
            Self::AsOf => "ASOF",
            #[cfg(feature = "iejoin")]
            Self::IEJoin => "IEJOIN",
            #[cfg(feature = "iejoin")]
            Self::Range => "RANGE",
            Self::Cross => "CROSS",
        };
        f.write_str(val)
    }
}

/// The type of a join together with its match condition.
///
/// Each variant holds only the join types that its condition supports.
#[derive(Clone, PartialEq, IntoStaticStr, Debug)]
#[cfg_attr(feature = "ir_serde", derive(Serialize, Deserialize))]
#[strum(serialize_all = "snake_case")]
pub enum JoinTypeIR {
    /// The match condition is `left == right` for every key pair, and, if there is a
    /// `fused_predicate`, that predicate as well.
    ///
    /// `on` is never empty: a join without keys is a [`Self::Cross`].
    /// Join keys may be scalar. Execution broadcasts keys to their input's height.
    Equi {
        how: EquiJoinType,
        on: Vec<(ExprIR, ExprIR)>,
        /// Boolean match condition in the join's output namespace, evaluated per
        /// candidate pair. A pair survives only on `true`; `false` and null reject it.
        ///
        /// Never set on a semi or anti join, or on a join with an attached slice.
        fused_predicate: Option<ExprIR>,
        /// Null keys match each other.
        nulls_equal: bool,
        /// Which side's keys must be unique. Always many-to-many on a semi or anti join.
        validation: JoinValidation,
        coalesce: JoinCoalesce,
    },
    /// Each left row is matched to at most one right row: the backward, forward or nearest
    /// match of the key pair `on`, within the `by` groups that `options` gives.
    ///
    /// The options are boxed because they are much larger than the other variants.
    #[cfg(feature = "asof_join")]
    AsOf {
        options: Box<AsOfOptions>,
        on: (ExprIR, ExprIR),
        coalesce: JoinCoalesce,
    },
    /// Inequality join over one or two arbitrary predicates.
    ///
    /// `operator1` relates the first key pair, `operator2` the second.
    #[cfg(feature = "iejoin")]
    IEJoin {
        how: NonEquiJoinType,
        ie_options: IEJoinOptions,
        left_on: Vec<ExprIR>,
        right_on: Vec<ExprIR>,
    },
    /// Inner join on `point ∈ [lower, upper]`. Only reached by the streaming engine.
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
    /// Every left row paired with every right row.
    Cross,
    /// A cross join that keeps the pairs for which `predicate` holds. Executed by the
    /// in-memory engine. The streaming engine has no native node for this and falls back
    /// to it via `InMemoryJoin`.
    CrossAndFilter {
        how: NonEquiJoinType,
        /// Must be elementwise.
        predicate: ExprIR,
    },
}

impl Hash for JoinTypeIR {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.hash_non_exprs(state);
        self.exprs().count().hash(state);
        for expr in self.exprs() {
            expr.node().hash(state);
        }
    }
}

impl JoinTypeIR {
    /// See [`JoinHow`].
    pub fn how(&self) -> JoinHow {
        match self {
            Self::Equi { how, .. } => (*how).into(),
            #[cfg(feature = "asof_join")]
            Self::AsOf { .. } => JoinHow::AsOf,
            #[cfg(feature = "iejoin")]
            Self::IEJoin {
                how: NonEquiJoinType::Inner,
                ..
            } => JoinHow::IEJoin,
            #[cfg(feature = "iejoin")]
            Self::IEJoin { how, .. } => (*how).into(),
            #[cfg(feature = "iejoin")]
            Self::Range { .. } => JoinHow::Range,
            Self::Cross
            | Self::CrossAndFilter {
                how: NonEquiJoinType::Inner,
                ..
            } => JoinHow::Cross,
            Self::CrossAndFilter { how, .. } => (*how).into(),
        }
    }

    /// The join type that the executors take, as [`Self::how`] names it.
    pub fn to_join_type(&self) -> JoinType {
        #[cfg(feature = "asof_join")]
        if let Self::AsOf { options, .. } = self {
            return JoinType::AsOf(options.clone());
        }
        match self.how() {
            JoinHow::Inner => JoinType::Inner,
            JoinHow::Left => JoinType::Left,
            JoinHow::Right => JoinType::Right,
            JoinHow::Full => JoinType::Full,
            #[cfg(feature = "semi_anti_join")]
            JoinHow::Semi => JoinType::Semi,
            #[cfg(feature = "semi_anti_join")]
            JoinHow::Anti => JoinType::Anti,
            #[cfg(feature = "asof_join")]
            JoinHow::AsOf => unreachable!(),
            #[cfg(feature = "iejoin")]
            JoinHow::IEJoin => JoinType::IEJoin,
            #[cfg(feature = "iejoin")]
            JoinHow::Range => JoinType::Range,
            JoinHow::Cross => JoinType::Cross,
        }
    }

    /// Builds the join that `args` describe on the key pairs `on`, and splits off the
    /// settings that apply to every join type.
    ///
    /// Settings that do not apply to the join type are dropped: e.g. `validation` on a
    /// cross join.
    ///
    /// Errors on [`JoinType::IEJoin`] and [`JoinType::Range`], which only the optimizer
    /// creates.
    ///
    /// # Panics
    /// If the number of key pairs does not suit `how`: a cross join takes none, an as-of
    /// join exactly one and any other join at least one.
    pub fn from_args(
        args: JoinArgs,
        on: Vec<(ExprIR, ExprIR)>,
    ) -> PolarsResult<(Self, JoinArgsIR)> {
        let JoinArgs {
            how,
            validation,
            suffix,
            slice,
            nulls_equal,
            coalesce,
            maintain_order,
            build_side,
        } = args;
        let args = JoinArgsIR {
            suffix,
            slice,
            maintain_order,
            build_side,
        };
        let how = match how {
            JoinType::Inner => EquiJoinType::Inner,
            JoinType::Left => EquiJoinType::Left,
            JoinType::Right => EquiJoinType::Right,
            JoinType::Full => EquiJoinType::Full,
            #[cfg(feature = "semi_anti_join")]
            JoinType::Semi => EquiJoinType::Semi,
            #[cfg(feature = "semi_anti_join")]
            JoinType::Anti => EquiJoinType::Anti,
            #[cfg(feature = "asof_join")]
            JoinType::AsOf(options) => {
                let [on] = <[_; 1]>::try_from(on)
                    .unwrap_or_else(|_| panic!("an as-of join takes exactly one key pair"));
                return Ok((
                    Self::AsOf {
                        options,
                        on,
                        coalesce,
                    },
                    args,
                ));
            },
            #[cfg(feature = "iejoin")]
            JoinType::IEJoin | JoinType::Range => {
                polars_bail!(InvalidOperation: "'{how}' join cannot be requested directly")
            },
            JoinType::Cross => {
                assert!(on.is_empty(), "a cross join takes no keys");
                return Ok((Self::Cross, args));
            },
        };
        assert!(!on.is_empty(), "an equi join takes at least one key pair");
        let join_type = Self::Equi {
            how,
            on,
            fused_predicate: None,
            nulls_equal,
            validation,
            coalesce,
        };
        Ok((join_type, args))
    }

    /// Null keys match each other. Always false for a join without equality keys.
    pub fn nulls_equal(&self) -> bool {
        match self {
            Self::Equi { nulls_equal, .. } => *nulls_equal,
            _ => false,
        }
    }

    /// Which side's keys must be unique. Many-to-many for every join but an equi join.
    pub fn validation(&self) -> JoinValidation {
        match self {
            Self::Equi { validation, .. } => *validation,
            _ => JoinValidation::ManyToMany,
        }
    }

    /// The `coalesce` setting. Joins that never coalesce report
    /// [`JoinCoalesce::KeepColumns`].
    pub fn coalesce(&self) -> JoinCoalesce {
        match self {
            Self::Equi { coalesce, .. } => *coalesce,
            #[cfg(feature = "asof_join")]
            Self::AsOf { coalesce, .. } => *coalesce,
            _ => JoinCoalesce::KeepColumns,
        }
    }

    /// Whether the left and right key columns are merged into one output column.
    pub fn should_coalesce(&self) -> bool {
        self.how().should_coalesce(self.coalesce())
    }

    /// Sets the `coalesce` setting.
    ///
    /// # Panics
    /// On joins other than equi and as-of joins, which never coalesce.
    pub fn set_coalesce(&mut self, new: JoinCoalesce) {
        match self {
            Self::Equi { coalesce, .. } => *coalesce = new,
            #[cfg(feature = "asof_join")]
            Self::AsOf { coalesce, .. } => *coalesce = new,
            _ => panic!("a '{}' join never coalesces", self.how()),
        }
    }

    #[cfg(feature = "asof_join")]
    pub fn asof_options(&self) -> Option<&AsOfOptions> {
        match self {
            Self::AsOf { options, .. } => Some(options),
            _ => None,
        }
    }

    /// Feeds everything except the expressions into `state`.
    fn hash_non_exprs<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        self.how().hash(state);
        self.nulls_equal().hash(state);
        self.validation().hash(state);
        self.coalesce().hash(state);
        match self {
            #[cfg(feature = "asof_join")]
            Self::AsOf { options, .. } => options.hash(state),
            #[cfg(feature = "iejoin")]
            Self::IEJoin { ie_options, .. } | Self::Range { ie_options, .. } => {
                ie_options.hash(state)
            },
            _ => {},
        }
    }

    /// Compares everything except the expressions.
    fn non_exprs_eq(&self, other: &Self) -> bool {
        if std::mem::discriminant(self) != std::mem::discriminant(other)
            || self.how() != other.how()
            || self.nulls_equal() != other.nulls_equal()
            || self.validation() != other.validation()
            || self.coalesce() != other.coalesce()
        {
            return false;
        }
        match (self, other) {
            #[cfg(feature = "asof_join")]
            (Self::AsOf { options: l, .. }, Self::AsOf { options: r, .. }) => l == r,
            #[cfg(feature = "iejoin")]
            (
                Self::IEJoin { ie_options: l, .. } | Self::Range { ie_options: l, .. },
                Self::IEJoin { ie_options: r, .. } | Self::Range { ie_options: r, .. },
            ) => l == r,
            _ => true,
        }
    }

    pub(crate) fn shallow_eq(&self, other: &Self, expr_cmp: &impl ExpressionComparator) -> bool {
        // Equal discriminants, so both sides hold the same shape and `exprs()` walks them
        // in the same order.
        self.non_exprs_eq(other)
            && self
                .exprs()
                .eq_by_(other.exprs(), |l, r| expr_cmp.equals(l, r))
    }

    #[cfg(feature = "cse")]
    pub(crate) fn shallow_hash<H: std::hash::Hasher>(
        &self,
        state: &mut H,
        expr_hash: &impl ExpressionHasher,
    ) {
        self.hash_non_exprs(state);
        self.exprs().count().hash(state);
        for expr in self.exprs() {
            expr_hash.hash_expr(expr, state);
        }
    }

    pub fn compile<C: FnOnce(&ExprIR) -> PolarsResult<Arc<dyn CrossJoinFilter>>>(
        self,
        plan: C,
    ) -> PolarsResult<Option<JoinTypeOptions>> {
        match self {
            Self::CrossAndFilter { predicate, .. } => {
                let predicate = plan(&predicate)?;

                Ok(Some(JoinTypeOptions::Cross(CrossJoinOptions { predicate })))
            },
            #[cfg(feature = "iejoin")]
            Self::IEJoin { ie_options, .. } | Self::Range { ie_options, .. } => {
                Ok(Some(JoinTypeOptions::IEJoin(ie_options)))
            },
            Self::Equi {
                fused_predicate: Some(fused_predicate),
                ..
            } => {
                let predicate = plan(&fused_predicate)?;

                Ok(Some(JoinTypeOptions::FusedPredicate(CrossJoinOptions {
                    predicate,
                })))
            },
            Self::Equi { .. } | Self::Cross => Ok(None),
            #[cfg(feature = "asof_join")]
            Self::AsOf { .. } => Ok(None),
        }
    }

    /// The keys as `(left, right)` pairs, or `None` if the condition does not equate them.
    /// A [`Self::Cross`] has no pairs, which is an empty slice.
    ///
    /// The only place `Equi`/`AsOf` are matched apart; every accessor below goes through it.
    /// They cannot share an or-pattern arm because rustc rejects `#[cfg]` on one alternative.
    pub fn key_pairs(&self) -> Option<&[(ExprIR, ExprIR)]> {
        match self {
            Self::Equi { on, .. } => Some(on),
            #[cfg(feature = "asof_join")]
            Self::AsOf { on, .. } => Some(std::slice::from_ref(on)),
            Self::Cross => Some(&[]),
            _ => None,
        }
    }

    /// See [`Self::key_pairs`].
    fn key_pairs_mut(&mut self) -> Option<&mut [(ExprIR, ExprIR)]> {
        match self {
            Self::Equi { on, .. } => Some(on),
            #[cfg(feature = "asof_join")]
            Self::AsOf { on, .. } => Some(std::slice::from_mut(on)),
            Self::Cross => Some(&mut []),
            _ => None,
        }
    }

    /// The left-hand side keys, in positional order.
    ///
    /// For [`Self::Range`] this can differ in length from [`Self::right_on`]; to pair the
    /// two sides, use [`Self::key_pairs`] instead of zipping them.
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
        if let Self::Equi {
            on,
            fused_predicate,
            ..
        } = self
        {
            return Exprs::pair_sides_then(on, fused_predicate.as_ref());
        }
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
            Self::CrossAndFilter { predicate, .. } => Exprs::single(predicate),
            _ => Exprs::Empty,
        }
    }

    /// See [`Self::exprs`]. Yields in the same order.
    pub fn exprs_mut(&mut self) -> ExprsMut<'_> {
        // Checked first so the mutable borrow does not span the match below.
        if let Self::Equi {
            on,
            fused_predicate,
            ..
        } = self
        {
            return ExprsMut::pair_sides_then(on, fused_predicate.as_mut());
        }
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
            Self::CrossAndFilter { predicate, .. } => ExprsMut::single(predicate),
            _ => ExprsMut::Empty,
        }
    }

    /// The schema that the `idx`-th expression of [`Self::exprs`] resolves against.
    pub fn expr_schema(&self, idx: usize) -> ExprSchema {
        let n_left = self.left_on_len();
        if idx < n_left {
            ExprSchema::Input(0)
        } else if idx < n_left + self.right_on_len() {
            ExprSchema::Input(1)
        } else {
            ExprSchema::Output
        }
    }

    /// Replace the keys, keeping the current variant.
    ///
    /// # Panics
    /// On [`Self::CrossAndFilter`] and on a [`Self::Cross`] given keys, as neither holds any.
    /// Also on a length mismatch where the variant stores keys in pairs.
    pub fn set_keys(&mut self, left: Vec<ExprIR>, right: Vec<ExprIR>) {
        match self {
            Self::Equi { on, .. } => *on = left.into_iter().zip_eq(right).collect(),
            #[cfg(feature = "asof_join")]
            Self::AsOf { on, .. } => {
                let pairs: Vec<_> = left.into_iter().zip_eq(right).collect();
                [*on] = <[_; 1]>::try_from(pairs)
                    .unwrap_or_else(|_| panic!("an as-of join takes exactly one key pair"));
            },
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
            Self::Cross => assert!(
                left.is_empty() && right.is_empty(),
                "cross join holds no keys"
            ),
            Self::CrossAndFilter { .. } => panic!("cross join holds no keys"),
        }
    }

    /// Replaces which rows the join emits, keeping its match condition.
    ///
    /// # Panics
    /// On as-of, range and plain cross joins, whose condition fixes the rows they emit.
    pub fn set_how(&mut self, new: NonEquiJoinType) {
        match self {
            Self::Equi { how, .. } => *how = new.into(),
            #[cfg(feature = "iejoin")]
            Self::IEJoin { how, .. } => *how = new,
            Self::CrossAndFilter { how, .. } => *how = new,
            _ => panic!("cannot change the type of a '{}' join", self.how()),
        }
    }

    /// The match condition is exactly `left == right` for every key pair.
    ///
    /// True for [`Self::Cross`], which has no keys, and for [`Self::AsOf`]: its strategy
    /// and tolerance are not part of the match condition. False once a fused predicate is
    /// attached.
    pub fn is_pure_equi(&self) -> bool {
        !self.is_non_equi()
    }

    /// The match condition has a non-equality component.
    ///
    /// Use [`Self::key_pairs`] instead where what is needed is paired keys: a fused predicate
    /// join has those as well.
    pub fn is_non_equi(&self) -> bool {
        self.key_pairs().is_none() || self.has_fused_predicate()
    }

    pub fn has_fused_predicate(&self) -> bool {
        self.fused_predicate().is_some()
    }

    /// The predicate of a [`Self::CrossAndFilter`].
    pub fn cross_predicate(&self) -> Option<&ExprIR> {
        match self {
            Self::CrossAndFilter { predicate, .. } => Some(predicate),
            _ => None,
        }
    }

    /// The fused match condition, if any. See [`Self::Equi`].
    pub fn fused_predicate(&self) -> Option<&ExprIR> {
        match self {
            Self::Equi {
                fused_predicate, ..
            } => fused_predicate.as_ref(),
            _ => None,
        }
    }

    pub(crate) fn push_key_pair(&mut self, left: ExprIR, right: ExprIR) {
        let Self::Equi { on, .. } = self else {
            panic!("key pairs can only be added to equi joins")
        };
        on.push((left, right));
    }

    pub(crate) fn set_fused_predicate(&mut self, new: ExprIR) {
        let Self::Equi {
            fused_predicate, ..
        } = self
        else {
            panic!("a fused predicate is only supported on equi joins")
        };
        assert!(
            fused_predicate.is_none(),
            "fused_predicate would be overwritten"
        );
        *fused_predicate = Some(new);
    }
}
