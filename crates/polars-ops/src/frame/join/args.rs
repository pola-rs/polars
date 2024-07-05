use super::*;

pub(super) type JoinIds = Vec<IdxSize>;
pub type LeftJoinIds = (ChunkJoinIds, ChunkJoinOptIds);
pub type InnerJoinIds = (JoinIds, JoinIds);

#[cfg(feature = "chunked_ids")]
pub(super) type ChunkJoinIds = Either<Vec<IdxSize>, Vec<ChunkId>>;
#[cfg(feature = "chunked_ids")]
pub type ChunkJoinOptIds = Either<Vec<NullableIdxSize>, Vec<ChunkId>>;

#[cfg(not(feature = "chunked_ids"))]
pub type ChunkJoinOptIds = Vec<NullableIdxSize>;

#[cfg(not(feature = "chunked_ids"))]
pub type ChunkJoinIds = Vec<IdxSize>;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct JoinArgs {
    pub how: JoinType,
    pub validation: JoinValidation,
    pub suffix: Option<String>,
    pub slice: Option<(i64, usize)>,
    pub join_nulls: bool,
    pub coalesce: JoinCoalesce,
}

impl JoinArgs {
    pub fn should_coalesce(&self) -> bool {
        self.coalesce.coalesce(&self.how)
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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
            Full { .. } => {
                matches!(self, CoalesceColumns)
            },
            #[cfg(feature = "asof_join")]
            AsOf(_) => matches!(self, JoinSpecific | CoalesceColumns),
            Cross => false,
            #[cfg(feature = "semi_anti_join")]
            Semi | Anti => false,
        }
    }
}

impl Default for JoinArgs {
    fn default() -> Self {
        Self {
            how: JoinType::Inner,
            validation: Default::default(),
            suffix: None,
            slice: None,
            join_nulls: false,
            coalesce: Default::default(),
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
            join_nulls: false,
            coalesce: Default::default(),
        }
    }

    pub fn with_coalesce(mut self, coalesce: JoinCoalesce) -> Self {
        self.coalesce = coalesce;
        self
    }

    pub fn with_suffix(mut self, suffix: Option<String>) -> Self {
        self.suffix = suffix;
        self
    }

    pub fn suffix(&self) -> &str {
        self.suffix.as_deref().unwrap_or("_right")
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum JoinType {
    Inner,
    Left,
    Right,
    Full,
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
            Right => "RIGHT",
            Inner => "INNER",
            Full { .. } => "FULL",
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

impl JoinType {
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
}

#[derive(Copy, Clone, PartialEq, Eq, Default, Hash)]
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

    pub fn is_valid_join(&self, join_type: &JoinType) -> PolarsResult<()> {
        if !self.needs_checks() {
            return Ok(());
        }
        polars_ensure!(matches!(join_type, JoinType::Inner | JoinType::Full{..} | JoinType::Left),
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
        polars_ensure!(valid, ComputeError: "join keys did not fulfill {} validation", self);
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
        polars_ensure!(valid, ComputeError: "join keys did not fulfill {} validation", self);
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
