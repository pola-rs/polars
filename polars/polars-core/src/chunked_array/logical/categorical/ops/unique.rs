use super::*;
use crate::prelude::groupby::IntoGroupsProxy;
use crate::utils::NoNull;

impl CategoricalChunked {
    pub fn unique(&self) -> Result<Self> {
        let cat_map = self.get_rev_map();
        if self.can_fast_unique() {
            let ca = match &**cat_map {
                RevMapping::Local(a) => {
                    UInt32Chunked::from_iter_values(self.logical().name(), 0..(a.len() as u32))
                }
                RevMapping::Global(map, _, _) => {
                    UInt32Chunked::from_iter_values(self.logical().name(), map.keys().copied())
                }
            };
            let mut out = CategoricalChunked::from_cats_and_rev_map(ca, cat_map.clone());
            out.set_fast_unique(true);
            Ok(out)
        } else {
            let ca = self.logical().unique()?;
            Ok(CategoricalChunked::from_cats_and_rev_map(
                ca,
                cat_map.clone(),
            ))
        }
    }

    pub fn n_unique(&self) -> Result<usize> {
        if self.can_fast_unique() {
            Ok(self.get_rev_map().len())
        } else {
            self.logical().n_unique()
        }
    }

    pub fn value_counts(&self) -> Result<DataFrame> {
        let group_tuples = self.logical().group_tuples(true, false).into_idx();
        let logical_values = unsafe {
            self.logical
                .take_unchecked(group_tuples.iter().map(|t| t.0 as usize).into())
        };

        let mut values = self.clone();
        *values.logical_mut() = logical_values;

        let mut counts: NoNull<IdxCa> = group_tuples
            .into_iter()
            .map(|(_, groups)| groups.len() as IdxSize)
            .collect();
        counts.rename("counts");
        let cols = vec![values.into_series(), counts.into_inner().into_series()];
        let df = DataFrame::new_no_checks(cols);
        df.sort(&["counts"], true)
    }
}
