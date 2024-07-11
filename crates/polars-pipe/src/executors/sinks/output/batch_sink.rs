use std::any::Any;

use crossbeam_channel::Sender;
use polars_core::prelude::*;

use crate::operators::{
    chunks_to_df_unchecked, DataChunk, FinalizedSink, PExecutionContext, Sink, SinkResult,
};

#[derive(Clone)]
pub struct BatchSink {
    sender: Sender<DataFrame>,
}

impl BatchSink {
    pub fn new(sender: Sender<DataFrame>) -> PolarsResult<Self> {
        Ok(Self { sender })
    }
}

impl Sink for BatchSink {
    fn sink(&mut self, _context: &PExecutionContext, chunk: DataChunk) -> PolarsResult<SinkResult> {
        let df: DataFrame = chunks_to_df_unchecked(vec![chunk]);
        let result = self.sender.send(df);
        match result {
            Ok(..) => Ok(SinkResult::CanHaveMoreInput),
            Err(..) => Ok(SinkResult::Finished),
        }
    }

    fn combine(&mut self, _other: &mut dyn Sink) {
        // Nothing to do
    }

    fn split(&self, _thread_no: usize) -> Box<dyn Sink> {
        Box::new(self.clone())
    }

    fn finalize(&mut self, _context: &PExecutionContext) -> PolarsResult<FinalizedSink> {
        let _ = self.sender.send(Default::default());
        Ok(FinalizedSink::Finished(Default::default()))
    }

    fn as_any(&mut self) -> &mut dyn Any {
        self
    }

    fn fmt(&self) -> &str {
        "batch_sink"
    }
}
