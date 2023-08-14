use super::*;

pub enum SourceResult {
    Finished,
    GotMoreData(Vec<DataChunk>),
}

pub trait Source: Send + Sync {
    fn get_batches(&mut self, context: &PExecutionContext) -> PolarsResult<SourceResult>;

    fn fmt(&self) -> &str;
}
