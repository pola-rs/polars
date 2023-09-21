use super::*;

pub(super) type JoinIds = Vec<IdxSize>;
pub type LeftJoinIds = (ChunkJoinIds, ChunkJoinOptIds);
pub type InnerJoinIds = (JoinIds, JoinIds);

#[cfg(feature = "chunked_ids")]
pub(super) type ChunkJoinIds = Either<Vec<IdxSize>, Vec<ChunkId>>;
#[cfg(feature = "chunked_ids")]
pub type ChunkJoinOptIds = Either<Vec<Option<IdxSize>>, Vec<Option<ChunkId>>>;

#[cfg(not(feature = "chunked_ids"))]
pub type ChunkJoinOptIds = Vec<Option<IdxSize>>;

#[cfg(not(feature = "chunked_ids"))]
pub type ChunkJoinIds = Vec<IdxSize>;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "asof_join")]
use super::asof::AsOfOptions;

#[derive(Clone, PartialEq, Eq, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct JoinArgs {
    pub how: JoinType,
    pub validation: JoinValidation,
    pub suffix: Option<String>,
    pub slice: Option<(i64, usize)>,
}

impl JoinArgs {
    pub fn new(how: JoinType) -> Self {
        Self {
            how,
            validation: Default::default(),
            suffix: None,
            slice: None,
        }
    }

    pub fn suffix(&self) -> &str {
        self.suffix.as_deref().unwrap_or("_right")
    }
}

#[derive(Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum JoinType {
    Left,
    Inner,
    Outer,
    #[cfg(feature = "asof_join")]
    AsOf(AsOfOptions),
    Cross,
    #[cfg(feature = "semi_anti_join")]
    Semi,
    #[cfg(feature = "semi_anti_join")]
    Anti,
}

impl From<JoinType> for JoinArgs {
    fn from(value: JoinType) -> Self {
        JoinArgs::new(value)
    }
}

impl Display for JoinType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use JoinType::*;
        let val = match self {
            Left => "LEFT",
            Inner => "INNER",
            Outer => "OUTER",
            #[cfg(feature = "asof_join")]
            AsOf(_) => "ASOF",
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

#[derive(Copy, Clone, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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

    fn swap(self, swap: bool) -> Self {
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

    pub fn is_valid_join(&self, join_type: &JoinType, n_keys: usize) -> PolarsResult<()> {
        if !self.needs_checks() {
            return Ok(());
        }
        polars_ensure!(n_keys == 1, ComputeError: "{self} validation on a {join_type} is not yet supported for multiple keys");
        polars_ensure!(matches!(join_type, JoinType::Inner | JoinType::Outer | JoinType::Left),
                      ComputeError: "{self} validation on a {join_type} join is not supported");
        Ok(())
    }

    pub(super) fn validate_probe(
        &self,
        s_left: &Series,
        s_right: &Series,
        build_shortest_table: bool,
    ) -> PolarsResult<()> {
        // In default, probe is the left series.
        //
        // In inner join and outer join, the shortest relation will be used to create a hash table.
        // In left join, always use the right side to create.
        //
        // If `build_shortest_table` and left is shorter, swap. Then rhs will be the probe.
        // If left == right, swap too. (apply the same logic as `det_hash_prone_order`)
        let should_swap = build_shortest_table && s_left.len() <= s_right.len();
        let probe = if should_swap { s_right } else { s_left };

        use JoinValidation::*;
        let valid = match self.swap(should_swap) {
            // Only check the `build` side.
            // The other side use `validate_build` to check
            ManyToMany | ManyToOne => true,
            OneToMany | OneToOne => probe.n_unique()? == probe.len(),
        };
        polars_ensure!(valid, ComputeError: "the join keys did not fulfil {} validation", self);
        Ok(())
    }

    pub(super) fn validate_build(
        &self,
        build_size: usize,
        expected_size: usize,
        swapped: bool,
    ) -> PolarsResult<()> {
        use JoinValidation::*;

        // In default, build is in rhs.
        let valid = match self.swap(swapped) {
            // Only check the `build` side.
            // The other side use `validate_prone` to check
            ManyToMany | OneToMany => true,
            ManyToOne | OneToOne => build_size == expected_size,
        };
        polars_ensure!(valid, ComputeError: "the join keys did not fulfil {} validation", self);
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
