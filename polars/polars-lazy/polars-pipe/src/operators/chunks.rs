use super::*;

#[derive(Clone, Debug)]
pub struct DataChunk {
    pub chunk_index: IdxSize,
    pub data: DataFrame,
}

impl DataChunk {
    pub(crate) fn with_data(&self, data: DataFrame) -> Self {
        DataChunk {
            chunk_index: self.chunk_index,
            data,
        }
    }
}
