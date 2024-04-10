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
    /// Order of the columns.
    ///
    /// If only one value is given, it will boardcast to all columns.
    ///
    /// Use [`SortMultipleOptions::with_orders`] or [`SortMultipleOptions::with_order`] to modify.
    ///
    /// # Safety
    ///
    /// Len must matches the number of columns or equal to 1.
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
    pub fn new() -> Self {
        Self::default()
    }

    /// Specify order for each columns
    ///
    /// # Safety
    ///
    /// Len must matches the number of columns or equal to 1.
    pub fn with_orders(mut self, descending: impl IntoIterator<Item = bool>) -> Self {
        self.descending = descending.into_iter().collect();
        self
    }

    /// Implement order for all columns
    pub fn with_order(mut self, descending: bool) -> Self {
        self.descending = vec![descending];
        self
    }

    pub fn with_nulls_last(mut self, enabled: bool) -> Self {
        self.nulls_last = enabled;
        self
    }

    pub fn with_multithreaded(mut self, enabled: bool) -> Self {
        self.multithreaded = enabled;
        self
    }

    pub fn with_maintain_order(mut self, enabled: bool) -> Self {
        self.maintain_order = enabled;
        self
    }

    pub fn with_order_reversed(mut self) -> Self {
        self.descending.iter_mut().for_each(|x| *x = !*x);
        self
    }
}

impl SortOptions {
    pub fn with_order(mut self, enabled: bool) -> Self {
        self.descending = enabled;
        self
    }

    pub fn with_nulls_last(mut self, enabled: bool) -> Self {
        self.nulls_last = enabled;
        self
    }

    pub fn with_multithreaded(mut self, enabled: bool) -> Self {
        self.multithreaded = enabled;
        self
    }

    pub fn with_maintain_order(mut self, enabled: bool) -> Self {
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

impl From<&SortMultipleOptions> for SortOptions {
    fn from(value: &SortMultipleOptions) -> Self {
        SortOptions {
            descending: value.descending.first().copied().unwrap_or(false),
            nulls_last: value.nulls_last,
            multithreaded: value.multithreaded,
            maintain_order: value.maintain_order,
        }
    }
}
