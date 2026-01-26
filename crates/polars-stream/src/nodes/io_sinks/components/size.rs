use std::num::NonZeroU64;

use polars_core::frame::DataFrame;
use polars_error::{PolarsResult, polars_err};
use polars_utils::IdxSize;
use polars_utils::index::NonZeroIdxSize;

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct RowCountAndSize {
    pub num_rows: IdxSize,
    pub num_bytes: u64,
}

impl RowCountAndSize {
    pub const MAX: RowCountAndSize = RowCountAndSize {
        num_rows: IdxSize::MAX,
        num_bytes: u64::MAX,
    };

    pub fn new_from_df(df: &DataFrame) -> Self {
        Self {
            num_rows: IdxSize::try_from(df.height()).unwrap(),
            num_bytes: u64::try_from(df.estimated_size()).unwrap(),
        }
    }

    /// How many rows from `other` can fit into `self`.
    ///
    /// # Parameters
    /// * `byte_size_min_rows`: Row limit calculated from byte size will be at least this value
    pub fn num_rows_takeable_from(self, other: Self, byte_size_min_rows: IdxSize) -> IdxSize {
        let mut max_rows = self.num_rows.min(other.num_rows);

        let limit_according_to_byte_size =
            byte_size_min_rows.max(if self.num_bytes < other.row_byte_size() {
                0
            } else {
                IdxSize::try_from(self.num_bytes.div_ceil(other.row_byte_size().max(1)))
                    .unwrap_or(IdxSize::MAX)
            });

        if self.num_bytes < u64::MAX {
            max_rows = max_rows.min(limit_according_to_byte_size)
        }

        max_rows
    }

    /// Byte size of a single row. If `self.num_rows > 0`, the returned size will be at least 1.
    pub fn row_byte_size(&self) -> u64 {
        if self.num_rows == 0 {
            0
        } else {
            #[cfg_attr(feature = "bigidx", expect(clippy::useless_conversion))]
            (self.num_bytes / u64::from(self.num_rows)).max(1)
        }
    }

    /// Returns an error if the resulting row count exceeds `IdxSize::MAX`. `num_bytes` will use
    /// saturating addition.
    pub fn add(self, rhs: Self) -> PolarsResult<Self> {
        let num_rows = self.num_rows.checked_add(rhs.num_rows).ok_or_else(|| {
            let consider_installing_64 = if cfg!(feature = "bigidx") {
                ""
            } else {
                ". Consider installing 'polars[rt64]'."
            };

            let counter = u128::saturating_add(self.num_rows.into(), rhs.num_rows.into());

            polars_err!(
                ComputeError:
                "row count ({}) exceeded maximum supported of {}{}",
                counter, IdxSize::MAX, consider_installing_64
            )
        })?;

        let num_bytes = self.num_bytes.saturating_add(rhs.num_bytes);

        Ok(Self {
            num_rows,
            num_bytes,
        })
    }

    /// Increment this `RowCountAndSize` by `num_rows`. The increment of `self.num_bytes` will be
    /// calculated according to `total.num_bytes - self.num_bytes`.
    ///
    /// Returns `None` if the incremented result would exceed `total.num_rows`.
    pub fn add_delta(self, num_rows: IdxSize, total: Self) -> Option<Self> {
        let available = total.checked_sub(self)?;

        if num_rows > available.num_rows {
            return None;
        }

        let num_bytes = u64::min(
            available.row_byte_size().saturating_mul(
                #[cfg_attr(feature = "bigidx", expect(clippy::useless_conversion))]
                u64::from(num_rows),
            ),
            available.num_bytes,
        );

        // `total.checked_sub(self)` guarantees no overflow below.
        Some(Self {
            num_rows: self.num_rows + num_rows,
            num_bytes: self.num_bytes + num_bytes,
        })
    }

    pub fn checked_sub(self, rhs: Self) -> Option<Self> {
        Some(Self {
            num_rows: IdxSize::checked_sub(self.num_rows, rhs.num_rows)?,
            num_bytes: u64::checked_sub(self.num_bytes, rhs.num_bytes)?,
        })
    }

    pub fn saturating_add(self, rhs: Self) -> Self {
        Self {
            num_rows: IdxSize::saturating_add(self.num_rows, rhs.num_rows),
            num_bytes: u64::saturating_add(self.num_bytes, rhs.num_bytes),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct NonZeroRowCountAndSize {
    num_rows: NonZeroIdxSize,
    num_bytes: NonZeroU64,
}

impl NonZeroRowCountAndSize {
    pub const MAX: NonZeroRowCountAndSize = NonZeroRowCountAndSize {
        num_rows: NonZeroIdxSize::MAX,
        num_bytes: NonZeroU64::MAX,
    };

    pub fn new(size: RowCountAndSize) -> Option<Self> {
        Some(Self {
            num_rows: NonZeroIdxSize::new(size.num_rows)?,
            num_bytes: NonZeroU64::new(size.num_bytes)?,
        })
    }

    pub fn min(self, other: Self) -> Self {
        Self {
            num_rows: self.num_rows.min(other.num_rows),
            num_bytes: self.num_bytes.min(other.num_bytes),
        }
    }

    #[inline]
    pub fn get(self) -> RowCountAndSize {
        RowCountAndSize {
            num_rows: self.num_rows.get(),
            num_bytes: self.num_bytes.get(),
        }
    }
}

/// Calculates how many rows can be sent in a morsel to a writer.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct TakeableRowsProvider {
    pub max_size: NonZeroRowCountAndSize,
    pub byte_size_min_rows: NonZeroIdxSize,
    /// If `true`, allows sending buffered rows if there are at least `self.byte_size_min_rows` number
    /// of them. Effectively disables strict morsel combining to `self.max_size` while retaining
    /// splitting of morsels larger than `self.max_size`.
    pub allow_non_max_size: bool,
}

impl TakeableRowsProvider {
    pub fn num_rows_takeable_from(self, from: RowCountAndSize, flush: bool) -> Option<IdxSize> {
        let num_rows = self
            .max_size
            .get()
            .num_rows_takeable_from(from, self.byte_size_min_rows.get());

        let have_full_chunk = num_rows < from.num_rows || num_rows == self.max_size.get().num_rows;

        (num_rows > 0
            && (flush
                || have_full_chunk
                || (self.allow_non_max_size && num_rows >= self.byte_size_min_rows.get())))
        .then_some(num_rows)
    }
}
