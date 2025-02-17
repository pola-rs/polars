use polars_core::prelude::SchemaRef;

use crate::executors::operators::reproject_chunk;
use crate::operators::{PExecutionContext, PolarsResult, Source, SourceResult};

/// A source that will ensure we keep the schema order
pub(crate) struct ReProjectSource {
    schema: SchemaRef,
    source: Box<dyn Source>,
    positions: Vec<usize>,
}

impl ReProjectSource {
    pub(crate) fn new(schema: SchemaRef, source: Box<dyn Source>) -> Self {
        ReProjectSource {
            schema,
            source,
            positions: vec![],
        }
    }
}

impl Source for ReProjectSource {
    fn get_batches(&mut self, context: &PExecutionContext) -> PolarsResult<SourceResult> {
        Ok(match self.source.get_batches(context)? {
            SourceResult::Finished => SourceResult::Finished,
            SourceResult::GotMoreData(mut chunks) => {
                for chunk in &mut chunks {
                    reproject_chunk(chunk, &mut self.positions, self.schema.as_ref())?;
                }
                SourceResult::GotMoreData(chunks)
            },
        })
    }

    fn fmt(&self) -> &str {
        "re-project-source"
    }
}
