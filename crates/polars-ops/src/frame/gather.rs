use polars_core::prelude::*;

use crate::series::convert_and_bound_index;

pub trait GatherDf {
    /// Selects from this DataFrame the rows at the indices in the given column.
    fn gather_with_column(&self, idxs: &Column, null_on_oob: bool) -> PolarsResult<DataFrame>;

    /// Selects from this DataFrame the rows at the indices in the given series.
    fn gather_with_series(&self, idxs: &Series, null_on_oob: bool) -> PolarsResult<DataFrame>;
}

impl GatherDf for DataFrame {
    fn gather_with_column(&self, idxs: &Column, null_on_oob: bool) -> PolarsResult<DataFrame> {
        match idxs {
            Column::Series(s) => self.gather_with_series(s, null_on_oob),
            Column::Scalar(idx_c) => {
                if idx_c.is_empty() {
                    return Ok(self.clear());
                }

                let idx_s = idx_c.as_single_value_series();
                let idx_ca = convert_and_bound_index(&idx_s, self.height(), null_on_oob)?;
                match idx_ca.get(0) {
                    Some(idx) => Ok(self.new_from_index(idx as usize, idx_c.len())),
                    None => Ok(DataFrame::full_null(self.schema(), idx_c.len())),
                }
            },
        }
    }

    fn gather_with_series(&self, idxs: &Series, null_on_oob: bool) -> PolarsResult<DataFrame> {
        if idxs.is_empty() {
            return Ok(self.clear());
        }

        let idx_ca = convert_and_bound_index(idxs, self.height(), null_on_oob)?;
        Ok(unsafe { self.take_unchecked_impl(&idx_ca, false) })
    }
}
