#[cfg(feature = "serde-lazy")]
use serde::{Deserialize, Serialize};
pub use slice::*;

use crate::prelude::*;

/// Options for single series sorting.
///
/// Indicating the order of sorting, nulls position, multithreading, and maintaining order.
///
/// # Example
///
/// ```
/// # use polars_core::prelude::*;
/// let s = Series::new("a", [Some(5), Some(2), Some(3), Some(4), None].as_ref());
/// let sorted = s
///     .sort(
///         SortOptions::default()
///             .with_order_descending(true)
///             .with_nulls_last(true)
///             .with_multithreaded(false),
///     )
///     .unwrap();
/// assert_eq!(
///     sorted,
///     Series::new("a", [Some(5), Some(4), Some(3), Some(2), None].as_ref())
/// );
/// ```
#[derive(Copy, Clone, Eq, PartialEq, Debug, Hash)]
#[cfg_attr(feature = "serde-lazy", derive(Serialize, Deserialize))]
pub struct SortOptions {
    /// If true sort in descending order.
    /// Default `false`.
    pub descending: bool,
    /// Whether place null values last.
    /// Default `false`.
    pub nulls_last: bool,
    /// If true sort in multiple threads.
    /// Default `true`.
    pub multithreaded: bool,
    /// If true maintain the order of equal elements.
    /// Default `false`.
    pub maintain_order: bool,
}

/// Sort options for multi-series sorting.
///
/// Indicating the order of sorting, nulls position, multithreading, and maintaining order.
///
/// # Example
/// ```
/// # use polars_core::prelude::*;
///
/// # fn main() -> PolarsResult<()> {
/// let df = df! {
///     "a" => [Some(1), Some(2), None, Some(4), None],
///     "b" => [Some(5), None, Some(3), Some(2), Some(1)]
/// }?;
///
/// let out = df
///     .sort(
///         ["a", "b"],
///         SortMultipleOptions::default()
///             .with_maintain_order(true)
///             .with_multithreaded(false)
///             .with_order_descending_multi([false, true])
///             .with_nulls_last(true),
///     )?;
///
/// let expected = df! {
///     "a" => [Some(1), Some(2), Some(4), None, None],
///     "b" => [Some(5), None, Some(2), Some(3), Some(1)]
/// }?;
///
/// assert_eq!(out, expected);
///
/// # Ok(())
/// # }
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde-lazy", derive(Serialize, Deserialize))]
pub struct SortMultipleOptions {
    /// Order of the columns. Default all `false``.
    ///
    /// If only one value is given, it will broadcast to all columns.
    ///
    /// Use [`SortMultipleOptions::with_order_descending_multi`]
    /// or [`SortMultipleOptions::with_order_descending`] to modify.
    ///
    /// # Safety
    ///
    /// Len must match the number of columns, or equal 1.
    pub descending: Vec<bool>,
    /// Whether place null values last. Default `false`.
    pub nulls_last: Vec<bool>,
    /// Whether sort in multiple threads. Default `true`.
    pub multithreaded: bool,
    /// Whether maintain the order of equal elements. Default `false`.
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
            nulls_last: vec![false],
            multithreaded: true,
            maintain_order: false,
        }
    }
}

impl SortMultipleOptions {
    /// Create `SortMultipleOptions` with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Specify order for each column. Defaults all `false`.
    ///
    /// # Safety
    ///
    /// Len must match the number of columns, or be equal to 1.
    pub fn with_order_descending_multi(
        mut self,
        descending: impl IntoIterator<Item = bool>,
    ) -> Self {
        self.descending = descending.into_iter().collect();
        self
    }

    /// Implement order for all columns. Default `false`.
    pub fn with_order_descending(mut self, descending: bool) -> Self {
        self.descending = vec![descending];
        self
    }

    /// Specify whether to place nulls last, per-column. Defaults all `false`.
    ///
    /// # Safety
    ///
    /// Len must match the number of columns, or be equal to 1.
    pub fn with_nulls_last_multi(mut self, nulls_last: impl IntoIterator<Item = bool>) -> Self {
        self.nulls_last = nulls_last.into_iter().collect();
        self
    }

    /// Whether to place null values last. Default `false`.
    pub fn with_nulls_last(mut self, enabled: bool) -> Self {
        self.nulls_last = vec![enabled];
        self
    }

    /// Whether to sort in multiple threads. Default `true`.
    pub fn with_multithreaded(mut self, enabled: bool) -> Self {
        self.multithreaded = enabled;
        self
    }

    /// Whether to maintain the order of equal elements. Default `false`.
    pub fn with_maintain_order(mut self, enabled: bool) -> Self {
        self.maintain_order = enabled;
        self
    }

    /// Reverse the order of sorting for each column.
    pub fn with_order_reversed(mut self) -> Self {
        self.descending.iter_mut().for_each(|x| *x = !*x);
        self
    }
}

impl SortOptions {
    /// Create `SortOptions` with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Specify sorting order for the column. Default `false`.
    pub fn with_order_descending(mut self, enabled: bool) -> Self {
        self.descending = enabled;
        self
    }

    /// Whether place null values last. Default `false`.
    pub fn with_nulls_last(mut self, enabled: bool) -> Self {
        self.nulls_last = enabled;
        self
    }

    /// Whether sort in multiple threads. Default `true`.
    pub fn with_multithreaded(mut self, enabled: bool) -> Self {
        self.multithreaded = enabled;
        self
    }

    /// Whether maintain the order of equal elements. Default `false`.
    pub fn with_maintain_order(mut self, enabled: bool) -> Self {
        self.maintain_order = enabled;
        self
    }

    /// Reverse the order of sorting.
    pub fn with_order_reversed(mut self) -> Self {
        self.descending = !self.descending;
        self
    }
}

impl From<&SortOptions> for SortMultipleOptions {
    fn from(value: &SortOptions) -> Self {
        SortMultipleOptions {
            descending: vec![value.descending],
            nulls_last: vec![value.nulls_last],
            multithreaded: value.multithreaded,
            maintain_order: value.maintain_order,
        }
    }
}

impl From<&SortMultipleOptions> for SortOptions {
    fn from(value: &SortMultipleOptions) -> Self {
        SortOptions {
            descending: value.descending.first().copied().unwrap_or(false),
            nulls_last: value.nulls_last.first().copied().unwrap_or(false),
            multithreaded: value.multithreaded,
            maintain_order: value.maintain_order,
        }
    }
}
