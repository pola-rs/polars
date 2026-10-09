//! Join argument and option definitions.

use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;
use std::sync::Arc;

use polars_core::datatypes::BooleanChunked;
use polars_core::frame::DataFrame;
#[cfg(feature = "asof_join")]
use polars_core::scalar::Scalar;
use polars_error::{PolarsResult, polars_ensure};
use polars_utils::pl_str::PlSmallStr;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use strum_macros::IntoStaticStr;

/// Parameters for which side to use as the build side in a join. Currently only
/// respected by the streaming engine.
#[derive(Clone, PartialEq, Debug, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum JoinBuildSide {
    /// Unless there's a very good reason to believe that the right side is
    /// smaller, use the left side.
    PreferLeft,
    /// Regardless of other heuristics, use the left side as build side.
    ForceLeft,

    // Similar to above.
    PreferRight,
    ForceRight,
}

#[derive(Clone, PartialEq, Debug, Hash, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub struct JoinArgs {
    pub how: JoinType,
    pub validation: JoinValidation,
    pub suffix: Option<PlSmallStr>,
    pub slice: Option<(i64, usize)>,
    pub nulls_equal: bool,
    pub coalesce: JoinCoalesce,
    pub maintain_order: MaintainOrderJoin,
    pub build_side: Option<JoinBuildSide>,
}

impl JoinArgs {
    pub fn should_coalesce(&self) -> bool {
        self.coalesce.coalesce(&self.how)
    }
}

#[derive(Clone, PartialEq, Hash, Default, IntoStaticStr)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum JoinType {
    #[default]
    Inner,
    Left,
    Right,
    Full,
    // Box is okay because this is inside a `Arc<JoinOptionsIR>`
    #[cfg(feature = "asof_join")]
    AsOf(Box<AsOfOptions>),
    #[cfg(feature = "semi_anti_join")]
    Semi,
    #[cfg(feature = "semi_anti_join")]
    Anti,
    #[cfg(feature = "iejoin")]
    /// Inequality join with two arbitrary predicates
    // Options are set by optimizer/planner in Options
    IEJoin,
    #[cfg(feature = "iejoin")]
    /// Inequality join with col ∈ [lo, hi] predicate
    // Options are set by optimizer/planner in Options
    Range,
    // Options are set by optimizer/planner in Options
    Cross,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash, Default, IntoStaticStr)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum JoinCoalesce {
    #[default]
    JoinSpecific,
    CoalesceColumns,
    KeepColumns,
}

impl JoinCoalesce {
    pub fn coalesce(&self, join_type: &JoinType) -> bool {
        use JoinCoalesce::*;
        use JoinType::*;
        match join_type {
            Left | Inner | Right => {
                matches!(self, JoinSpecific | CoalesceColumns)
            },
            Full => {
                matches!(self, CoalesceColumns)
            },
            #[cfg(feature = "asof_join")]
            AsOf(_) => matches!(self, JoinSpecific | CoalesceColumns),
            #[cfg(feature = "iejoin")]
            IEJoin | Range => false,
            Cross => false,
            #[cfg(feature = "semi_anti_join")]
            Semi | Anti => false,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash, Default, IntoStaticStr)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[strum(serialize_all = "snake_case")]
pub enum MaintainOrderJoin {
    #[default]
    None,
    Left,
    Right,
    LeftRight,
    RightLeft,
}

impl MaintainOrderJoin {
    pub fn flip(&self) -> Self {
        match self {
            MaintainOrderJoin::None => MaintainOrderJoin::None,
            MaintainOrderJoin::Left => MaintainOrderJoin::Right,
            MaintainOrderJoin::Right => MaintainOrderJoin::Left,
            MaintainOrderJoin::LeftRight => MaintainOrderJoin::RightLeft,
            MaintainOrderJoin::RightLeft => MaintainOrderJoin::LeftRight,
        }
    }
}

impl JoinArgs {
    pub fn new(how: JoinType) -> Self {
        Self {
            how,
            validation: Default::default(),
            suffix: None,
            slice: None,
            nulls_equal: false,
            coalesce: Default::default(),
            maintain_order: Default::default(),
            build_side: None,
        }
    }

    pub fn with_coalesce(mut self, coalesce: JoinCoalesce) -> Self {
        self.coalesce = coalesce;
        self
    }

    pub fn with_maintain_order(mut self, maintain_order: MaintainOrderJoin) -> Self {
        self.maintain_order = maintain_order;
        self
    }

    pub fn with_suffix(mut self, suffix: Option<PlSmallStr>) -> Self {
        self.suffix = suffix;
        self
    }

    pub fn with_build_side(mut self, build_side: Option<JoinBuildSide>) -> Self {
        self.build_side = build_side;
        self
    }

    pub fn suffix(&self) -> &PlSmallStr {
        const DEFAULT: &PlSmallStr = &PlSmallStr::from_static("_right");
        self.suffix.as_ref().unwrap_or(DEFAULT)
    }
}

impl From<JoinType> for JoinArgs {
    fn from(value: JoinType) -> Self {
        JoinArgs::new(value)
    }
}

pub trait CrossJoinFilter: Send + Sync {
    /// Evaluates the filter predicate on `df`, returning a boolean mask.
    fn evaluate(&self, df: &DataFrame) -> PolarsResult<BooleanChunked>;

    fn apply(&self, df: DataFrame, parallel: bool) -> PolarsResult<DataFrame> {
        let mask = self.evaluate(&df)?;
        if parallel {
            df.filter(&mask)
        } else {
            df.filter_seq(&mask)
        }
    }
}

impl<T> CrossJoinFilter for T
where
    T: Fn(&DataFrame) -> PolarsResult<BooleanChunked> + Send + Sync,
{
    fn evaluate(&self, df: &DataFrame) -> PolarsResult<BooleanChunked> {
        self(df)
    }
}

#[derive(Clone)]
pub struct CrossJoinOptions {
    pub predicate: Arc<dyn CrossJoinFilter>,
}

impl CrossJoinOptions {
    fn as_ptr_ref(&self) -> *const dyn CrossJoinFilter {
        Arc::as_ptr(&self.predicate)
    }
}

impl Eq for CrossJoinOptions {}

impl PartialEq for CrossJoinOptions {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::addr_eq(self.as_ptr_ref(), other.as_ptr_ref())
    }
}

impl Hash for CrossJoinOptions {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.as_ptr_ref().hash(state);
    }
}

impl Debug for CrossJoinOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "CrossJoinOptions",)
    }
}

#[derive(Clone, PartialEq, Eq, Hash, IntoStaticStr, Debug)]
#[strum(serialize_all = "snake_case")]
pub enum JoinTypeOptions {
    #[cfg(feature = "iejoin")]
    IEJoin(IEJoinOptions),
    Cross(CrossJoinOptions),
    /// A predicate fused into an equi join's match condition, on top of its keys.
    FusedPredicate(CrossJoinOptions),
}

impl Display for JoinType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use JoinType::*;
        let val = match self {
            Left => "LEFT",
            Right => "RIGHT",
            Inner => "INNER",
            Full => "FULL",
            #[cfg(feature = "asof_join")]
            AsOf(_) => "ASOF",
            #[cfg(feature = "iejoin")]
            IEJoin => "IEJOIN",
            #[cfg(feature = "iejoin")]
            Range => "RANGE",
            Cross => "CROSS",
            #[cfg(feature = "semi_anti_join")]
            Semi => "SEMI",
            #[cfg(feature = "semi_anti_join")]
            Anti => "ANTI",
        };
        write!(f, "{val}")
    }
}

impl Debug for JoinType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self}")
    }
}

impl JoinType {
    pub fn is_equi(&self) -> bool {
        matches!(
            self,
            JoinType::Inner | JoinType::Left | JoinType::Right | JoinType::Full
        )
    }

    pub fn is_semi_anti(&self) -> bool {
        #[cfg(feature = "semi_anti_join")]
        {
            matches!(self, JoinType::Semi | JoinType::Anti)
        }
        #[cfg(not(feature = "semi_anti_join"))]
        {
            false
        }
    }

    pub fn is_semi(&self) -> bool {
        #[cfg(feature = "semi_anti_join")]
        {
            matches!(self, JoinType::Semi)
        }
        #[cfg(not(feature = "semi_anti_join"))]
        {
            false
        }
    }

    pub fn is_anti(&self) -> bool {
        #[cfg(feature = "semi_anti_join")]
        {
            matches!(self, JoinType::Anti)
        }
        #[cfg(not(feature = "semi_anti_join"))]
        {
            false
        }
    }

    pub fn is_asof(&self) -> bool {
        #[cfg(feature = "asof_join")]
        {
            matches!(self, JoinType::AsOf(_))
        }
        #[cfg(not(feature = "asof_join"))]
        {
            false
        }
    }

    pub fn is_inner(&self) -> bool {
        matches!(self, JoinType::Inner)
    }

    pub fn is_cross(&self) -> bool {
        matches!(self, JoinType::Cross)
    }

    pub fn is_ie(&self) -> bool {
        #[cfg(feature = "iejoin")]
        {
            matches!(self, JoinType::IEJoin)
        }
        #[cfg(not(feature = "iejoin"))]
        {
            false
        }
    }

    pub fn is_range(&self) -> bool {
        #[cfg(feature = "iejoin")]
        {
            matches!(self, JoinType::Range)
        }
        #[cfg(not(feature = "iejoin"))]
        {
            false
        }
    }

    /// Unmatched rows of the left input appear in the output.
    pub fn emits_unmatched_left(&self) -> bool {
        #[cfg(feature = "semi_anti_join")]
        {
            matches!(self, JoinType::Left | JoinType::Full | JoinType::Anti)
        }
        #[cfg(not(feature = "semi_anti_join"))]
        {
            matches!(self, JoinType::Left | JoinType::Full)
        }
    }

    /// Unmatched rows of the right input appear in the output.
    pub fn emits_unmatched_right(&self) -> bool {
        matches!(self, JoinType::Right | JoinType::Full)
    }

    /// Joins supported in join where with non-equi conditions
    pub fn supports_non_equi(&self) -> bool {
        matches!(self, JoinType::Inner | JoinType::Left | JoinType::Right)
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Default, Hash, IntoStaticStr)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum JoinValidation {
    /// No unique checks
    #[default]
    ManyToMany,
    /// Check if join keys are unique in right dataset.
    ManyToOne,
    /// Check if join keys are unique in left dataset.
    OneToMany,
    /// Check if join keys are unique in both left and right datasets
    OneToOne,
}

impl JoinValidation {
    pub fn needs_checks(&self) -> bool {
        !matches!(self, JoinValidation::ManyToMany)
    }

    pub fn swap(self, swap: bool) -> Self {
        use JoinValidation::*;
        if swap {
            match self {
                ManyToMany => ManyToMany,
                ManyToOne => OneToMany,
                OneToMany => ManyToOne,
                OneToOne => OneToOne,
            }
        } else {
            self
        }
    }

    pub fn is_valid_join(&self, join_type: &JoinType) -> PolarsResult<()> {
        if !self.needs_checks() {
            return Ok(());
        }
        polars_ensure!(matches!(join_type, JoinType::Inner | JoinType::Full | JoinType::Left),
                      ComputeError: "{self} validation on a {join_type} join is not supported");
        Ok(())
    }
}

impl Display for JoinValidation {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            JoinValidation::ManyToMany => "m:m",
            JoinValidation::ManyToOne => "m:1",
            JoinValidation::OneToMany => "1:m",
            JoinValidation::OneToOne => "1:1",
        };
        write!(f, "{s}")
    }
}

impl Debug for JoinValidation {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "JoinValidation: {self}")
    }
}

#[cfg(feature = "asof_join")]
#[derive(Clone, Debug, PartialEq, Default, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub struct AsOfOptions {
    pub strategy: AsofStrategy,
    /// A tolerance in the same unit as the asof column
    pub tolerance: Option<Scalar>,
    /// A time duration specified as a string, for example:
    /// - "5m"
    /// - "2h15m"
    /// - "1d6h"
    pub tolerance_str: Option<PlSmallStr>,
    pub left_by: Option<Vec<PlSmallStr>>,
    pub right_by: Option<Vec<PlSmallStr>>,
    /// Allow equal matches
    pub allow_eq: bool,
    pub check_sortedness: bool,
}

#[cfg(feature = "asof_join")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Hash, strum_macros::IntoStaticStr)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum AsofStrategy {
    /// selects the last row in the right DataFrame whose ‘on’ key is less than or equal to the left’s key
    #[default]
    Backward,
    /// selects the first row in the right DataFrame whose ‘on’ key is greater than or equal to the left’s key.
    Forward,
    /// selects the right in the right DataFrame whose 'on' key is nearest to the left's key.
    Nearest,
}

#[cfg(feature = "iejoin")]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash, strum_macros::IntoStaticStr)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum InequalityOperator {
    #[default]
    Lt,
    LtEq,
    Gt,
    GtEq,
}

#[cfg(feature = "iejoin")]
impl InequalityOperator {
    pub fn is_strict(&self) -> bool {
        matches!(self, InequalityOperator::Gt | InequalityOperator::Lt)
    }

    pub fn flip(&self) -> InequalityOperator {
        use InequalityOperator::*;
        match self {
            Lt => Gt,
            LtEq => GtEq,
            Gt => Lt,
            GtEq => LtEq,
        }
    }
}

#[cfg(feature = "iejoin")]
#[derive(Clone, Debug, PartialEq, Eq, Default, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct IEJoinOptions {
    pub operator1: InequalityOperator,
    pub operator2: Option<InequalityOperator>,
}

#[cfg(feature = "iejoin")]
impl IEJoinOptions {
    /// The options such that matching with the left/right inputs swapped produces the
    /// same pairs as matching with the original inputs and options.
    pub fn flip(&self) -> IEJoinOptions {
        IEJoinOptions {
            operator1: self.operator1.flip(),
            operator2: self.operator2.map(|op| op.flip()),
        }
    }
}
