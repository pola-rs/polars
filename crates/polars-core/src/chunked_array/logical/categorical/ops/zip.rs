use polars_utils::iter::EnumerateIdxTrait;

use super::*;

impl CategoricalChunked {
    pub(crate) fn zip_with(
        &self,
        mask: &BooleanChunked,
        other: &CategoricalChunked,
    ) -> PolarsResult<Self> {
        let (new_physical, new_rev_map) = match (&**self.get_rev_map(), &**other.get_rev_map()) {
            (RevMapping::Global(_, _, idl), RevMapping::Global(_, _, idr)) if idl == idr => {
                let new_rev_map = self._merge_categorical_map(other)?;
                let new_physical = self.physical().zip_with(mask, other.physical())?;
                (new_physical, new_rev_map)
            },
            (RevMapping::Local(cats_left), RevMapping::Local(cats_right)) => {
                // We need to merge two potentially overlapping local rev_maps
                let cats_left_hashmap = PlHashMap::from_iter(
                    cats_left.iter().enumerate_idx().map(|(k, v)| (v, k as u32)),
                );
                let mut new_categories: MutableUtf8Array<i64> =
                    MutableUtf8Array::from_iter(cats_left.iter());
                let mut idx_mapping = PlHashMap::with_capacity(cats_right.len());

                for (idx, s) in cats_right.iter().enumerate() {
                    if let Some(v) = cats_left_hashmap.get(&s) {
                        idx_mapping.insert(idx, *v);
                    } else {
                        idx_mapping.insert(idx, new_categories.len() as u32);
                        new_categories.push(s);
                    }
                }
                let new_rev_map = Arc::new(RevMapping::Local(new_categories.into()));

                // Fastpath there are no overlapping categories we can just do an addition
                let new_physical = if new_rev_map.len() == cats_left.len() + cats_right.len() {
                    self.physical()
                        .zip_with(mask, &(other.physical() + cats_left.len() as u32))?
                } else {
                    let cats_right_remapped = other
                        .physical()
                        .into_iter()
                        .map(|z: Option<u32>| Some(*idx_mapping.get(&(z? as usize)).unwrap()))
                        .collect();
                    self.physical().zip_with(mask, &cats_right_remapped)?
                };
                (new_physical, new_rev_map)
            },
            _ => return polars_bail!(string_cache_mismatch),
        };

        // Safety: physical and rev map are correctly constructed above
        unsafe {
            return Ok(CategoricalChunked::from_cats_and_rev_map_unchecked(
                new_physical,
                new_rev_map,
            ));
        }
    }
}
