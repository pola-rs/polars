use polars_core::frame::DataFrame;
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
        let max_rows = self.num_rows.min(other.num_rows);

        if self.num_bytes == u64::MAX {
            max_rows
        } else if self.num_bytes < other.row_byte_size() {
            0
        } else {
            max_rows.min(
                IdxSize::try_from(self.num_bytes.div_ceil(other.row_byte_size().max(1)))
                    .unwrap_or(IdxSize::MAX),
            )
        }
    }

    /// Byte size of a single row. If `self.num_rows > 0`, the returned size will be at least 1.
    pub fn row_byte_size(&self) -> u64 {
        if self.num_rows == 0 {
            0
        } else {
            #[allow(clippy::useless_conversion)]
            (self.num_bytes / u64::from(self.num_rows)).max(1)
        }
    }

    pub fn checked_add(self, rhs: Self) -> Option<Self> {
        Some(Self {
            num_rows: IdxSize::checked_add(self.num_rows, rhs.num_rows)?,
            num_bytes: u64::checked_add(self.num_bytes, rhs.num_bytes)?,
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
