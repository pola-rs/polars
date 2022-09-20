use super::*;

impl CategoricalChunked {
    pub(crate) fn zip_with(
        &self,
        mask: &BooleanChunked,
        other: &CategoricalChunked,
    ) -> PolarsResult<Self> {
        let cats = match &**self.get_rev_map() {
            RevMapping::Local(rev_map) => {
                // the logic for merging the rev maps will concatenate utf8 arrays
                // to make sure the indexes still make sense we need to offset the right hand side
                self.logical()
                    .zip_with(mask, &(other.logical() + rev_map.len() as u32))?
            }
            _ => self.logical().zip_with(mask, other.logical())?,
        };
        let new_state = self.merge_categorical_map(other)?;

        // Safety:
        // we checked the rev_maps.
        unsafe {
            Ok(CategoricalChunked::from_cats_and_rev_map_unchecked(
                cats, new_state,
            ))
        }
    }
}
