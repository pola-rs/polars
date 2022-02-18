use super::*;

impl CategoricalChunked {
    fn unique(&self) -> Result<Self> {
        let cat_map = self.rev_map();
        if self.can_fast_unique() {
            let ca = match &*cat_map {
                RevMapping::Local(a) => {
                    UInt32Chunked::from_iter_values(self.name(), 0..(a.len() as u32))
                }
                RevMapping::Global(map, _, _) => {
                    UInt32Chunked::from_iter_values(self.name(), map.keys().copied())
                }
            };
            let mut out = CategoricalChunked::from_cats_and_rev_map(ca, cat_map.clone());
            out.set_fast_unique(true);
            Ok(out)
        } else {
            let ca = self.logical().unique()?;
            Ok(CategoricalChunked::from_cats_and_rev_map(ca, cat_map.clone()))
        }
    }
}
