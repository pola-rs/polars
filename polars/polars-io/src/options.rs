use polars_arrow::prelude::IdxSize;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RowCount {
    pub name: String,
    pub offset: IdxSize,
}

impl RowCount {
    pub fn new(name: &str, offset: IdxSize) -> Self {
        Self {
            name: name.to_string(),
            offset,
        }
    }
}

impl From<&str> for RowCount {
    fn from(name: &str) -> Self {
        Self {
            name: name.to_string(),
            offset: 0,
        }
    }
}

impl From<(&str, IdxSize)> for RowCount {
    fn from((name, offset): (&str, IdxSize)) -> Self {
        Self {
            name: name.to_string(),
            offset,
        }
    }
}
