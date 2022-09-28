use super::*;

pub enum SourceResult {
    Finished,
    GotMoreData(DataChunk),
}

pub trait Source {
    fn get_batches(context: &PExecutionContext) -> PolarsResult<SourceResult>;
}
