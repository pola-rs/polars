use polars_utils::IdxSize;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RowIndex {
    pub name: String,
    pub offset: IdxSize,
}
