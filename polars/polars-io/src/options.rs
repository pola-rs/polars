use polars_arrow::prelude::IdxSize;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RowCount {
    pub name: String,
    pub offset: IdxSize,
}
