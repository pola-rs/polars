use std::any::Any;
use polars_core::error::PolarsResult;
use polars_core::frame::DataFrame;
use polars_core::prelude::JoinType::Outer;
use polars_core::utils::accumulate_dataframes_vertical_unchecked;
use crate::operators::{chunks_to_df_unchecked, DataChunk, FinalizedSink, Operator, OperatorResult, PExecutionContext, Sink, SinkResult, Source, SourceResult};


#[derive(Default)]
pub struct OuterJoinPhase1 {
    chunks: Vec<DataChunk>,
    suffix: Option<String>
}

impl Sink for OuterJoinPhase1 {
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
        Box::new(Self::default())
    }

    fn finalize(&mut self) -> PolarsResult<FinalizedSink> {
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
    suffix: Option<String>
}

impl Operator for CrossJoinPhase2 {
    fn execute(&self, _context: &PExecutionContext, chunk: &DataChunk) -> PolarsResult<OperatorResult> {
        // todo! amortize left and right name creation
        let df = self.df.cross_join(&chunk.data, self.suffix.clone(), None)?;
        Ok(OperatorResult::Finished(chunk.with_data(df)))
    }
}
