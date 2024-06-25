use super::{Metadata, MetadataEnv};
use crate::chunked_array::{ChunkAgg, ChunkedArray, PolarsDataType, PolarsNumericType};
use crate::series::IsSorted;

pub trait MetadataCollectable<T>: Sized {
    fn collect_cheap_metadata(&mut self) {}

    #[inline(always)]
    fn with_cheap_metadata(mut self) -> Self {
        self.collect_cheap_metadata();
        self
    }
}

impl<T> MetadataCollectable<T> for ChunkedArray<T>
where
    T: PolarsDataType,
    T: PolarsNumericType,
    ChunkedArray<T>: ChunkAgg<T::Native>,
{
    fn collect_cheap_metadata(&mut self) {
        if !MetadataEnv::experimental_enabled() {
            return;
        }

        if self.len() < 32 {
            let (min, max) = self
                .min_max()
                .map_or((None, None), |(l, r)| (Some(l), Some(r)));

            let has_one_value = self.len() - self.null_count() == 1;

            let md = Metadata::DEFAULT
                .sorted_opt(has_one_value.then_some(IsSorted::Ascending))
                .min_value_opt(min)
                .max_value_opt(max)
                .distinct_count_opt(has_one_value.then_some(1));

            if !md.is_empty() {
                mdlog!("Initializing cheap metadata");
            }

            self.merge_metadata(md);
        }
    }
}
