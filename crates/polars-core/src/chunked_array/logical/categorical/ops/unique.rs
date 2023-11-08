use super::*;
use crate::frame::group_by::IntoGroupsProxy;

impl CategoricalChunked {
    pub fn unique(&self) -> PolarsResult<Self> {
        let cat_map = self.get_rev_map();
        if self.can_fast_unique() {
            let ca = match &**cat_map {
                RevMapping::Local(a) => {
                    UInt32Chunked::from_iter_values(self.physical().name(), 0..(a.len() as u32))
                },
                RevMapping::Global(map, _, _) => {
                    UInt32Chunked::from_iter_values(self.physical().name(), map.keys().copied())
                },
            };
            // safety:
            // we only removed some indexes so we are still in bounds
            unsafe {
                let mut out =
                    CategoricalChunked::from_cats_and_rev_map_unchecked(ca, cat_map.clone());
                out.set_fast_unique(true);
                Ok(out)
            }
        } else {
            let ca = self.physical().unique()?;
            // safety:
            // we only removed some indexes so we are still in bounds
            unsafe {
                Ok(CategoricalChunked::from_cats_and_rev_map_unchecked(
                    ca,
                    cat_map.clone(),
                ))
            }
        }
    }

    pub fn n_unique(&self) -> PolarsResult<usize> {
        if self.can_fast_unique() {
            Ok(self.get_rev_map().len())
        } else {
            self.physical().n_unique()
        }
    }

    pub fn value_counts(&self) -> PolarsResult<DataFrame> {
        let groups = self.physical().group_tuples(true, false).unwrap();
        let physical_values = unsafe {
            self.physical()
                .clone()
                .into_series()
                .agg_first(&groups)
                .u32()
                .unwrap()
                .clone()
        };

        let mut values = self.clone();
        *values.physical_mut() = physical_values;

        let mut counts = groups.group_count();
        counts.rename("counts");
        let cols = vec![values.into_series(), counts.into_series()];
        let df = DataFrame::new_no_checks(cols);
        df.sort(["counts"], true, false)
    }
}
