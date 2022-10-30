use std::any::Any;
use std::borrow::Cow;
use std::sync::{Arc, Mutex};
use polars_core::error::PolarsResult;
use polars_core::frame::DataFrame;
use polars_core::prelude::JoinType::Outer;
use polars_core::utils::accumulate_dataframes_vertical_unchecked;
use crate::operators::{chunks_to_df_unchecked, DataChunk, FinalizedSink, Operator, OperatorResult, PExecutionContext, Sink, SinkResult, Source, SourceResult};


#[derive(Default)]
pub struct CrossJoin {
    chunks: Vec<DataChunk>,
    suffix: Cow<'static, str>,
    shared: Arc<Mutex<DataFrame>>
}

impl CrossJoin {
    pub(crate) fn new(suffix: Cow<'static, str>) -> Self {
        CrossJoin {
            chunks: vec![],
            suffix,
            shared: Default::default()
        }
    }
}

impl Sink for CrossJoin {
    fn sink(&mut self, _context: &PExecutionContext, chunk: DataChunk) -> PolarsResult<SinkResult> {
        self.chunks.push(chunk);
        Ok(SinkResult::CanHaveMoreInput)
    }

    fn combine(&mut self, mut other: Box<dyn Sink>) {
        let other = other.as_any().downcast_mut::<Self>().unwrap();
        let other_chunks = std::mem::take(&mut other.chunks);
        self.chunks.extend(other_chunks.into_iter());
    }

    fn split(&self, _thread_no: usize) -> Box<dyn Sink> {
        let mut new = Self::default();
        // ensure the arc ptr is shared
        new.shared = self.shared.clone();
        Box::new(new)
    }

    fn finalize(&mut self) -> PolarsResult<FinalizedSink> {
        // todo! share sink
        Ok(FinalizedSink::Operator(
            Box::new(CrossJoinPhase2{
                df: chunks_to_df_unchecked(std::mem::take(&mut self.chunks)),
                suffix: std::mem::take(&mut self.suffix)
            })
        ))
    }

    fn as_any(&mut self) -> &mut dyn Any {
        self
    }

    fn into_source(&mut self) -> PolarsResult<Option<Box<dyn Source>>> {
        Ok(None)
    }
}

pub struct CrossJoinPhase2 {
    df: DataFrame,
    suffix: Cow<'static, str>
}

impl Operator for CrossJoinPhase2 {
    fn execute(&self, _context: &PExecutionContext, chunk: &DataChunk) -> PolarsResult<OperatorResult> {
        // todo! amortize left and right name creation
        let df = self.df.cross_join(&chunk.data, Some(self.suffix.to_string()), None)?;
        Ok(OperatorResult::Finished(chunk.with_data(df)))
    }
}
