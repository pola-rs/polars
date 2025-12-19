use polars_core::frame::DataFrame;
use polars_error::{PolarsResult, polars_err};
use polars_utils::IdxSize;

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

    pub fn min(self, other: Self) -> Self {
        Self {
            num_rows: self.num_rows.min(other.num_rows),
            num_bytes: self.num_bytes.min(other.num_bytes),
        }
    }

    /// How many rows from `other` can fit into `self`.
    pub fn num_rows_takeable_from(self, other: Self) -> IdxSize {
        let mut max_rows = self.num_rows.min(other.num_rows);

        let limit_according_to_byte_size =
            IdxSize::try_from(self.num_bytes.div_ceil(other.row_byte_size().max(1)))
                .unwrap_or(IdxSize::MAX);

        if self.num_bytes < u64::MAX {
            max_rows = max_rows.min(limit_according_to_byte_size.max(16384))
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

        let num_bytes = std::cmp::min(
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
