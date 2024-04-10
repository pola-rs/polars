#[cfg(feature = "serde-lazy")]
use serde::{Deserialize, Serialize};
pub use slice::*;

use crate::prelude::*;

#[derive(Copy, Clone, Eq, PartialEq, Debug, Hash)]
#[cfg_attr(feature = "serde-lazy", derive(Serialize, Deserialize))]
pub struct SortOptions {
    pub descending: bool,
    pub nulls_last: bool,
    pub multithreaded: bool,
    pub maintain_order: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde-lazy", derive(Serialize, Deserialize))]
pub struct SortMultipleOptions {
    pub descending: Vec<bool>,
    pub nulls_last: bool,
    pub multithreaded: bool,
    pub maintain_order: bool,
}

impl Default for SortOptions {
    fn default() -> Self {
        Self {
            descending: false,
            nulls_last: false,
            multithreaded: true,
            maintain_order: false,
        }
    }
}

impl Default for SortMultipleOptions {
    fn default() -> Self {
        Self {
            descending: vec![false],
            nulls_last: false,
            multithreaded: true,
            maintain_order: false,
        }
    }
}

impl SortMultipleOptions {
    pub fn with_order(&mut self, descending: Vec<bool>) -> &mut Self {
        self.descending = descending;
        self
    }
}

impl From<SortOptions> for SortMultipleOptions {
    fn from(value: SortOptions) -> Self {
        SortMultipleOptions {
            descending: vec![value.descending],
            nulls_last: value.nulls_last,
            multithreaded: value.multithreaded,
            maintain_order: value.maintain_order,
        }
    }
}
