use super::*;

pub trait ZipOuterJoinColumn {
    unsafe fn zip_outer_join_column(
        &self,
        _right_column: &Series,
        _opt_join_tuples: &[(Option<IdxSize>, Option<IdxSize>)],
    ) -> Series {
        unimplemented!()
    }
}

impl<T> ZipOuterJoinColumn for ChunkedArray<T>
where
    T: PolarsDataType,
    ChunkedArray<T>: IntoSeries,
{
    unsafe fn zip_outer_join_column(
        &self,
        right_column: &Series,
        opt_join_tuples: &[(Option<IdxSize>, Option<IdxSize>)],
    ) -> Series {
        let right_ca = self.unpack_series_matching_type(right_column).unwrap();

        if self.null_count() == 0 && right_ca.null_count() == 0 {
            opt_join_tuples
                .iter()
                .map(|(opt_left_idx, opt_right_idx)| {
                    if let Some(left_idx) = opt_left_idx {
                        unsafe { self.value_unchecked(*left_idx as usize) }
                    } else {
                        unsafe {
                            let right_idx = opt_right_idx.unwrap_unchecked();
                            right_ca.value_unchecked(right_idx as usize)
                        }
                    }
                })
                .collect_ca_like(self)
                .into_series()
        } else {
            opt_join_tuples
                .iter()
                .map(|(opt_left_idx, opt_right_idx)| {
                    if let Some(left_idx) = opt_left_idx {
                        unsafe { self.get_unchecked(*left_idx as usize) }
                    } else {
                        unsafe {
                            let right_idx = opt_right_idx.unwrap_unchecked();
                            right_ca.get_unchecked(right_idx as usize)
                        }
                    }
                })
                .collect_ca_like(self)
                .into_series()
        }
    }
}
