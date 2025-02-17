use super::*;

struct CategoricalZipWith<'a>(&'a BooleanChunked);

impl CategoricalMergeOperation for CategoricalZipWith<'_> {
    fn finish(self, lhs: &UInt32Chunked, rhs: &UInt32Chunked) -> PolarsResult<UInt32Chunked> {
        lhs.zip_with(self.0, rhs)
    }
}
impl CategoricalChunked {
    pub(crate) fn zip_with(
        &self,
        mask: &BooleanChunked,
        other: &CategoricalChunked,
    ) -> PolarsResult<Self> {
        call_categorical_merge_operation(self, other, CategoricalZipWith(mask))
    }
}
