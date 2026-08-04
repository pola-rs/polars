use std::cmp;
use std::num::NonZeroU64;

use polars_core::frame::DataFrame;
use polars_error::{PolarsResult, polars_err};
use polars_utils::IdxSize;
use polars_utils::calc_morsel_split::{PartSizesIter, calc_n_parts};
use polars_utils::index::{NonZeroIdxSize, idxsize_to_u64};

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
    /// * `byte_size_min_rows`: Row limit calculated from byte size will not fall below this value.
    pub fn num_rows_takeable_from(self, other: Self, byte_size_min_rows: IdxSize) -> IdxSize {
        let mut max_rows = self.num_rows.min(other.num_rows);

        let limit_according_to_byte_size =
            byte_size_min_rows.max(if self.num_bytes < other.row_byte_size() {
                0
            } else {
                IdxSize::try_from(self.num_bytes.div_ceil(u64::max(1, other.row_byte_size())))
                    .unwrap_or(IdxSize::MAX)
            });

        if self.num_bytes < u64::MAX {
            max_rows = IdxSize::min(max_rows, limit_according_to_byte_size)
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
        self.checked_add(rhs).ok_or_else(|| {
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
        })
    }

    pub fn checked_add(self, rhs: Self) -> Option<Self> {
        let num_rows = self.num_rows.checked_add(rhs.num_rows)?;
        let num_bytes = self.num_bytes.saturating_add(rhs.num_bytes);

        Some(Self {
            num_rows,
            num_bytes,
        })
    }

    /// Increment this `RowCountAndSize` by `num_rows`. The increment of `self.num_bytes` will be
    /// calculated according to `total.num_bytes - self.num_bytes`.
    ///
    /// Returns `None` if the incremented result would exceed `total.num_rows`.
    pub fn add_delta(self, num_rows: IdxSize, total: Self) -> Option<Self> {
        self.checked_add(self.calc_delta(num_rows, total)?)
    }

    /// Returns `None` if `num_rows` exceeds `total.num_rows - self.num_rows`.
    pub fn calc_delta(self, num_rows: IdxSize, total: Self) -> Option<Self> {
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

        Some(Self {
            num_rows,
            num_bytes,
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
    pub num_rows: NonZeroIdxSize,
    pub num_bytes: NonZeroU64,
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

    #[expect(unused)]
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

#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub enum SplitMode {
    Approximate,
    #[default]
    Exact,
}

/// Target sink morsel size to split to.
#[derive(Debug, Clone, PartialEq)]
pub struct TargetSinkMorselSize {
    pub target_num_rows: NonZeroIdxSize,
    /// Note: Disabled if set to u64::MAX.
    pub target_num_bytes: NonZeroU64,
    /// Row limit calculated from `target_num_bytes` will not fall below this value.
    pub target_num_bytes_min_rows: NonZeroIdxSize,
    /// * `Exact`: Split to exactly `target_num_rows`, unless either no more rows
    ///   are available (i.e. last chunk), or if `target_num_bytes != u64::MAX` and
    ///   the limit calculated against the `target_num_bytes` is less than
    ///   `target_num_rows`.
    /// * `Approximate`: Splits do not need to be exactly `target_num_rows`.
    pub target_num_rows_mode: SplitMode,
}

#[derive(Debug, Default, PartialEq)]
enum LimitedBy {
    #[default]
    Rows,
    ByteSize,
}

impl TargetSinkMorselSize {
    pub fn calc_next_splits(
        &self,
        buffered_size: RowCountAndSize,
        incoming_size: RowCountAndSize,
    ) -> (bool, PartSizesIter) {
        let combined_size = buffered_size.checked_add(incoming_size).unwrap();

        let mut flush_buffered_as_one_split = false;
        let (mut part_sizes_iter, mut limited_by) = self.build_part_sizes_iter(combined_size);

        // Note: We assume the buffered amount `buffered_size` does not exceed the configured target
        // sizes (i.e., it should always be a residual of the target size as it is what remains
        // from the last round of sending).
        if incoming_size.num_rows != 0
            && !(self.target_num_rows_mode == SplitMode::Exact && limited_by == LimitedBy::Rows)
            && match part_sizes_iter.len() {
                0 => false,
                1 => {
                    incoming_size.num_rows > buffered_size.num_rows
                        && buffered_size.num_rows != 0
                        && incoming_size.num_rows / buffered_size.num_rows
                            > self.target_num_rows.get() / combined_size.num_rows
                },
                _ => true,
            }
        {
            flush_buffered_as_one_split = buffered_size.num_rows != 0;
            (part_sizes_iter, limited_by) = self.build_part_sizes_iter(incoming_size);
        }

        if limited_by == LimitedBy::Rows
            && part_sizes_iter.len() <= 1
            && self.target_num_rows_mode != SplitMode::Exact
            && part_sizes_iter.base_part_size().checked_mul(2).is_some_and(
                // base_part_size < (4/3) * target_num_rows
                |double_base_part_size| {
                    u64::abs_diff(
                        double_base_part_size,
                        idxsize_to_u64(self.target_num_rows.get()),
                    ) < u64::abs_diff(
                        part_sizes_iter.base_part_size(),
                        idxsize_to_u64(self.target_num_rows.get()),
                    )
                },
            )
        {
            // Wait for more data to have a fuller chunk.
            part_sizes_iter = PartSizesIter::default()
        }

        (flush_buffered_as_one_split, part_sizes_iter)
    }

    fn build_part_sizes_iter(&self, size: RowCountAndSize) -> (PartSizesIter, LimitedBy) {
        if size.num_rows == 0 {
            return (PartSizesIter::default(), LimitedBy::default());
        }

        let n_parts_by_num_rows = if self.target_num_rows_mode == SplitMode::Exact {
            u64::max(
                1,
                idxsize_to_u64(size.num_rows / self.target_num_rows.get()),
            )
        } else {
            calc_n_parts(
                idxsize_to_u64(size.num_rows),
                NonZeroU64::new(idxsize_to_u64(self.target_num_rows.get())).unwrap(),
            )
        };

        let mut max_parts_by_num_bytes = 0;
        let mut n_parts_by_num_bytes = 0;

        if self.target_num_bytes.get() != u64::MAX {
            max_parts_by_num_bytes =
                idxsize_to_u64(size.num_rows / self.target_num_bytes_min_rows.get());
            n_parts_by_num_bytes = calc_n_parts(size.num_bytes, self.target_num_bytes);
        };

        if match u64::cmp(
            &n_parts_by_num_rows,
            &u64::min(n_parts_by_num_bytes, max_parts_by_num_bytes),
        ) {
            cmp::Ordering::Greater => true,
            cmp::Ordering::Equal => self.target_num_rows_mode == SplitMode::Exact,
            cmp::Ordering::Less => false,
        } {
            (
                match self.target_num_rows_mode {
                    SplitMode::Approximate => PartSizesIter::new_from_total_size(
                        idxsize_to_u64(size.num_rows),
                        n_parts_by_num_rows as usize,
                    ),
                    SplitMode::Exact => {
                        if size.num_rows < self.target_num_rows.get() {
                            PartSizesIter::default()
                        } else {
                            PartSizesIter::new_from_part_size(
                                idxsize_to_u64(self.target_num_rows.get()),
                                n_parts_by_num_rows as usize,
                            )
                        }
                    },
                },
                LimitedBy::Rows,
            )
        } else {
            (
                if n_parts_by_num_bytes < max_parts_by_num_bytes {
                    PartSizesIter::new_from_total_size(
                        idxsize_to_u64(size.num_rows),
                        n_parts_by_num_bytes as usize,
                    )
                } else {
                    PartSizesIter::new_from_part_size(
                        idxsize_to_u64(self.target_num_bytes_min_rows.get()),
                        max_parts_by_num_bytes as usize,
                    )
                },
                LimitedBy::ByteSize,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU64;

    use polars_utils::index::NonZeroIdxSize;

    use crate::nodes::io_sinks::components::size::{
        RowCountAndSize, SplitMode, TargetSinkMorselSize,
    };

    fn calc_splits(
        target_size: &TargetSinkMorselSize,
        buffered_size: RowCountAndSize,
        incoming_size: RowCountAndSize,
    ) -> (bool, Vec<u64>) {
        let (a, b) = target_size.calc_next_splits(buffered_size, incoming_size);

        (a, b.collect())
    }

    #[test]
    fn test_target_sink_morsel_size() {
        let target_size = TargetSinkMorselSize {
            target_num_rows: NonZeroIdxSize::new(100).unwrap(),
            target_num_bytes: NonZeroU64::new(100).unwrap(),
            target_num_bytes_min_rows: NonZeroIdxSize::new(5).unwrap(),
            target_num_rows_mode: SplitMode::Exact,
        };

        assert_eq!(
            calc_splits(
                &target_size,
                RowCountAndSize {
                    num_rows: 5,
                    num_bytes: 5,
                },
                RowCountAndSize {
                    num_rows: 5,
                    num_bytes: 5,
                },
            ),
            (false, vec![])
        )
    }
}
