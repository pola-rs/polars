use polars_core::error::PolarsResult;

use crate::operators::{PExecutionContext, Source, SourceResult};

pub struct UnionSource {
    sources: Vec<Box<dyn Source>>,
    source_index: usize,
}

impl UnionSource {
    pub(crate) fn new(sources: Vec<Box<dyn Source>>) -> Self {
        Self {
            sources,
            source_index: 0,
        }
    }
}

impl Source for UnionSource {
    fn get_batches(&mut self, context: &PExecutionContext) -> PolarsResult<SourceResult> {
        // early return if we have data
        // if no data we deplete the loop and are finished
        while self.source_index < self.sources.len() {
            let src = &mut self.sources[self.source_index];
            match src.get_batches(context)? {
                SourceResult::Finished => self.source_index += 1,
                SourceResult::GotMoreData(chunks) => return Ok(SourceResult::GotMoreData(chunks)),
            }
        }
        Ok(SourceResult::Finished)
    }
    fn fmt(&self) -> &str {
        "union"
    }
}
