use super::*;

pub type LeftJoinIds = (JoinIds, JoinOptIds);

#[cfg(feature = "chunked_ids")]
pub(super) type JoinIds = Either<Vec<IdxSize>, Vec<ChunkId>>;
#[cfg(feature = "chunked_ids")]
pub type JoinOptIds = Either<Vec<Option<IdxSize>>, Vec<Option<ChunkId>>>;

#[cfg(not(feature = "chunked_ids"))]
pub type JoinOptIds = Vec<Option<IdxSize>>;

#[cfg(not(feature = "chunked_ids"))]
pub type JoinIds = Vec<IdxSize>;

/// [ChunkIdx, DfIdx]
pub type ChunkId = [IdxSize; 2];

#[derive(Clone, PartialEq, Eq)]
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
