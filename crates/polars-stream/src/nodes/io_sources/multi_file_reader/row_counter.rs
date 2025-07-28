use std::fmt::Debug;

use polars_error::{PolarsResult, polars_err};
use polars_utils::IdxSize;

/// Tracker counting physical and deleted rows.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RowCounter {
    /// Number of rows physically present in the file.
    physical_rows: usize,
    /// Number of rows deleted from the file.
    deleted_rows: usize,
}

impl RowCounter {
    /// `usize::MAX` physical rows, 0 deleted rows
    #[expect(unused)]
    pub const MAX: Self = Self {
        physical_rows: usize::MAX,
        deleted_rows: 0,
    };

    /// Does not check if `physical_rows < deleted_rows`.
    ///
    /// # Safety
    /// This does not represent a valid row position and should not be used as such.
    ///
    /// # Panics
    /// Panics if [`usize`] conversion fails.
    #[inline]
    unsafe fn new_unchecked<P, D>(physical_rows: P, deleted_rows: D) -> Self
    where
        usize: TryFrom<P> + TryFrom<D>,
        <usize as TryFrom<P>>::Error: Debug,
        <usize as TryFrom<D>>::Error: Debug,
    {
        Self {
            physical_rows: usize::try_from(physical_rows).unwrap(),
            deleted_rows: usize::try_from(deleted_rows).unwrap(),
        }
    }

    /// # Panics
    /// Panics if `deleted_rows > physical_rows`, or if [`usize`] conversion fails.
    #[inline]
    pub fn new<P, D>(physical_rows: P, deleted_rows: D) -> Self
    where
        usize: TryFrom<P> + TryFrom<D>,
        <usize as TryFrom<P>>::Error: Debug,
        <usize as TryFrom<D>>::Error: Debug,
    {
        let slf = unsafe { Self::new_unchecked(physical_rows, deleted_rows) };

        // Trigger validation
        slf.num_rows().unwrap();

        slf
    }

    /// # Safety
    /// The caller is responsible for ensuring the value is correct.
    ///
    /// # Panics
    /// Panics if `self.physical_rows < self.deleted_rows`
    pub unsafe fn set_deleted_rows<D>(&mut self, deleted_rows: D)
    where
        usize: TryFrom<D>,
        <usize as TryFrom<D>>::Error: Debug,
    {
        self.deleted_rows = usize::try_from(deleted_rows).unwrap();
        self.num_rows().unwrap();
    }

    /// Performs a saturating add if there are no deleted rows, otherwise performs a checked add.
    ///
    /// # Panics
    /// Panics if there are deleted rows and addition overflows.
    #[inline]
    pub fn add(self, other: Self) -> Self {
        (|| {
            let physical_rows = self.physical_rows.checked_add(other.physical_rows);
            let deleted_rows = self.deleted_rows.checked_add(other.deleted_rows)?;

            let physical_rows = if deleted_rows == 0 {
                physical_rows.unwrap_or(usize::MAX)
            } else {
                // If there are row deletions we cannot saturate the position properly (the
                // `num_rows()` will start to decrease).
                physical_rows?
            };

            Some(Self {
                physical_rows,
                deleted_rows,
            })
        })()
        .unwrap_or_else(|| panic!("addition overflow: {self:?} + {other:?}"))
    }

    /// # Panics
    /// Panics if subtraction overflows.
    #[inline]
    pub fn sub(self, other: Self) -> Self {
        let func = |a: usize, b: usize| {
            a.checked_sub(b)
                .unwrap_or_else(|| panic!("subtraction overflow: {self:?} - {other:?}"))
        };

        Self {
            physical_rows: func(self.physical_rows, other.physical_rows),
            deleted_rows: func(self.deleted_rows, other.deleted_rows),
        }
    }

    /// Returns the number of rows after applying deletions. This returns an
    /// error if there are more deleted rows than physical rows.
    pub fn num_rows(&self) -> PolarsResult<usize> {
        self.physical_rows
            .checked_sub(self.deleted_rows)
            .ok_or_else(|| {
                polars_err!(
                    ComputeError: "RowCounter: Invalid state: \
                    number of rows removed by deletion files ({}) \
                    is greater than the number of rows physically present ({})",
                    self.deleted_rows, self.physical_rows,
                )
            })
    }

    /// Returns [`RowCounter::num_rows`] as a usize.
    ///
    /// # Panics
    /// Panics if `usize` to `IdxSize` conversion fails.
    #[inline]
    #[expect(unused)]
    pub fn num_rows_idxsize(&self) -> PolarsResult<IdxSize> {
        self.num_rows().map(|x| IdxSize::try_from(x).unwrap())
    }

    #[inline]
    /// Saturates to `IdxSize::MAX` if conversion fails
    pub fn num_rows_idxsize_saturating(&self) -> PolarsResult<IdxSize> {
        self.num_rows()
            .map(|x| IdxSize::try_from(x).unwrap_or(IdxSize::MAX))
    }

    /// Returns the number of rows physically present in the file.
    #[inline]
    pub fn num_physical_rows(&self) -> usize {
        self.physical_rows
    }

    #[inline]
    #[expect(unused)]
    pub fn num_physical_rows_idxsize_saturating(&self) -> IdxSize {
        IdxSize::try_from(self.physical_rows).unwrap_or(IdxSize::MAX)
    }
}
