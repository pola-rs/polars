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
    pub fn with_order(mut self, descending: Vec<bool>) -> Self {
        self.descending = descending;
        self
    }

    pub fn descending(self, descending: bool) -> Self {
        self.with_order(vec![descending])
    }

    pub fn nulls_last(mut self, enabled: bool) -> Self {
        self.nulls_last = enabled;
        self
    }

    pub fn multithreaded(mut self, enabled: bool) -> Self {
        self.multithreaded = enabled;
        self
    }

    pub fn maintain_order(mut self, enabled: bool) -> Self {
        self.maintain_order = enabled;
        self
    }
}

impl SortOptions {
    pub fn descending(mut self, enabled: bool) -> Self {
        self.descending = enabled;
        self
    }

    pub fn nulls_last(mut self, enabled: bool) -> Self {
        self.nulls_last = enabled;
        self
    }

    pub fn multithreaded(mut self, enabled: bool) -> Self {
        self.multithreaded = enabled;
        self
    }

    pub fn maintain_order(mut self, enabled: bool) -> Self {
        self.maintain_order = enabled;
        self
    }
}

impl From<&SortOptions> for SortMultipleOptions {
    fn from(value: &SortOptions) -> Self {
        SortMultipleOptions {
            descending: vec![value.descending],
            nulls_last: value.nulls_last,
            multithreaded: value.multithreaded,
            maintain_order: value.maintain_order,
        }
    }
}
