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

/// [ChunkIdx, DfIdx]
pub type ChunkId = [IdxSize; 2];

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
        polars_ensure!(n_keys == 1, ComputeError: "{validation} not yet supported for multiple keys");
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
        // the shortest relation is built
        let swap = build_shortest_table && s_left.len() > s_right.len();

        use JoinValidation::*;
        // all rhs `Many`s are valid
        // rhs `One`s need to be checked
        let valid = match self.swap(swap) {
            ManyToMany | OneToMany => true,
            ManyToOne | OneToOne => {
                let s = if swap { s_left } else { s_right };
                s.n_unique()? == s.len()
            }
        };
        polars_ensure!(valid, ComputeError: "the join keys did not fulfil {} validation", self);
        Ok(())
    }

    pub(super) fn validate_build(
        &self,
        build_size: usize,
        expected_size: usize,
        swap: bool,
    ) -> PolarsResult<()> {
        use JoinValidation::*;

        // all lhs `Many`s are valid
        // lhs `One`s need to be checked
        let valid = match self.swap(swap) {
            ManyToMany | ManyToOne => true,
            OneToMany | OneToOne => build_size == expected_size,
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
