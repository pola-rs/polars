use crate::prelude::*;
use itertools::Itertools;

#[derive(Debug, Clone, PartialEq)]
pub struct Row<'a>(Vec<AnyValue<'a>>);

impl DataFrame {
    /// Get a row from a DataFrame. Use of this is discouraged as it will likely be slow.
    pub fn get_row(&self, idx: usize) -> Row {
        let values = self.columns.iter().map(|s| s.get(idx)).collect_vec();
        Row(values)
    }

    /// Amortize allocations by reusing a row.
    /// The caller is responsible for the making sure the row has at least capacity for the number
    /// of columns in the DataFrame
    pub fn get_row_amortized<'a>(&'a self, idx: usize, row: &mut Row<'a>) {
        self.columns
            .iter()
            .zip(&mut row.0)
            .for_each(|(s, any_val)| {
                *any_val = s.get(idx);
            });
    }

    /// Amortize allocations by reusing a row.
    /// The caller is responsible for the making sure the row has at least capacity for the number
    /// of columns in the DataFrame
    ///
    /// # Safety
    /// Does not do any bounds checking.
    pub unsafe fn get_row_amortized_unchecked<'a>(&'a self, idx: usize, row: &mut Row<'a>) {
        self.columns
            .iter()
            .zip(&mut row.0)
            .for_each(|(s, any_val)| {
                *any_val = s.get_unchecked(idx);
            });
    }
}
