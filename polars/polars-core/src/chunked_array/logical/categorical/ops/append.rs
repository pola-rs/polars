use super::*;
use crate::chunked_array::ops::append::new_chunks;
use crate::series::IsSorted;

impl CategoricalChunked {
    pub fn append(&mut self, other: &Self) -> PolarsResult<()> {
        if self.logical.null_count() == self.len() && other.logical.null_count() == other.len() {
            let len = self.len();
            self.logical_mut().length += other.len() as IdxSize;
            new_chunks(&mut self.logical.chunks, &other.logical().chunks, len);
            return Ok(());
        }
        let is_local_different_source =
            match (self.get_rev_map().as_ref(), other.get_rev_map().as_ref()) {
                (RevMapping::Local(arr_l), RevMapping::Local(arr_r)) => !std::ptr::eq(arr_l, arr_r),
                _ => false,
            };

        if is_local_different_source {
            polars_bail!(
            ComputeError: r#"
cannot concat categoricals coming from a different source, consider setting a global StringCache.

Help: if you're using Python, this may look something like:

    with pl.StringCache():
        df1 = pl.DataFrame({'a': ['1', '2']}, schema={'a': pl.Categorical})
        df2 = pl.DataFrame({'a': ['1', '3']}, schema={'a': pl.Categorical})
        pl.concat([df1, df2])

Alternatively, if the performance cost is acceptable, you could just set:

    pl.enable_string_cache(True)

on startup.
"#.trim_start()
            );
        } else {
            let len = self.len();
            let new_rev_map = self.merge_categorical_map(other)?;
            unsafe { self.set_rev_map(new_rev_map, false) };

            self.logical_mut().length += other.len() as IdxSize;
            new_chunks(&mut self.logical.chunks, &other.logical().chunks, len);
        }
        self.logical.set_sorted_flag(IsSorted::Not);
        Ok(())
    }
}
