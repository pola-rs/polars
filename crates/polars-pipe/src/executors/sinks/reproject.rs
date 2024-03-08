use std::any::Any;

use polars_core::schema::SchemaRef;

use crate::executors::sources::ReProjectSource;
use crate::operators::{
    DataChunk, FinalizedSink, PExecutionContext, PolarsResult, Sink, SinkResult,
};

/// A sink that will ensure we keep the schema order
pub(crate) struct ReProjectSink {
    schema: SchemaRef,
    sink: Box<dyn Sink>,
}

impl ReProjectSink {
    pub(crate) fn new(schema: SchemaRef, sink: Box<dyn Sink>) -> Self {
        Self { schema, sink }
    }
}

impl Sink for ReProjectSink {
    fn sink(&mut self, context: &PExecutionContext, chunk: DataChunk) -> PolarsResult<SinkResult> {
        self.sink.sink(context, chunk)
    }

    fn combine(&mut self, other: &mut dyn Sink) {
        let other = other.as_any().downcast_mut::<Self>().unwrap();
        self.sink.combine(other.sink.as_mut())
    }

    fn split(&self, thread_no: usize) -> Box<dyn Sink> {
        let sink = self.sink.split(thread_no);
        Box::new(Self {
            schema: self.schema.clone(),
            sink,
        })
    }

    fn finalize(&mut self, context: &PExecutionContext) -> PolarsResult<FinalizedSink> {
        Ok(match self.sink.finalize(context)? {
            FinalizedSink::Finished(df) => {
                FinalizedSink::Finished(df.select(self.schema.iter_names())?)
            },
            FinalizedSink::Source(source) => {
                FinalizedSink::Source(Box::new(ReProjectSource::new(self.schema.clone(), source)))
            },
            _ => unimplemented!(),
        })
    }

    fn as_any(&mut self) -> &mut dyn Any {
        self
    }

    fn fmt(&self) -> &str {
        "re-project-sink"
    }
}
