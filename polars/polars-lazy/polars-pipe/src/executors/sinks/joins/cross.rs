use std::any::Any;
use std::borrow::Cow;
use std::sync::{Arc, Mutex};
use std::vec;

use polars_core::error::PolarsResult;
use polars_core::frame::DataFrame;
use polars_core::utils::{split_df, split_df_as_ref};

use crate::operators::{
    chunks_to_df_unchecked, DataChunk, FinalizedSink, Operator, OperatorResult, PExecutionContext,
    Sink, SinkResult,
};

#[derive(Default)]
pub struct CrossJoin {
    chunks: Vec<DataChunk>,
    suffix: Cow<'static, str>,
    shared: Arc<Mutex<DataFrame>>,
}

impl CrossJoin {
    pub(crate) fn new(suffix: Cow<'static, str>) -> Self {
        CrossJoin {
            chunks: vec![],
            suffix,
            shared: Default::default(),
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
        Box::new(Self {
            suffix: self.suffix.clone(),
            shared: self.shared.clone(),
            ..Default::default()
        })
    }

    fn finalize(&mut self) -> PolarsResult<FinalizedSink> {
        // todo! share sink
        Ok(FinalizedSink::Operator(Box::new(CrossJoinPhase2 {
            df: Arc::new(chunks_to_df_unchecked(std::mem::take(&mut self.chunks))),
            suffix: Arc::from(self.suffix.as_ref()),
            in_process: None
        })))
    }

    fn as_any(&mut self) -> &mut dyn Any {
        self
    }
}

#[derive(Clone)]
pub struct CrossJoinPhase2 {
    df: Arc<DataFrame>,
    suffix: Arc<str>,
    in_process: Option<vec::IntoIter<DataFrame>>
}

impl Operator for CrossJoinPhase2 {
    fn execute(
        &mut self,
        _context: &PExecutionContext,
        chunk: &DataChunk,
    ) -> PolarsResult<OperatorResult> {
        if self.in_process.is_none() {
            let data_size = chunk.data.height();
            let left_len = self.df.height();
            let output_size = left_len * data_size;
            if output_size > crate::CHUNK_SIZE && left_len > 10 {
                let mut n_chunks = output_size / crate::CHUNK_SIZE;
                if n_chunks > self.df.height() {
                    n_chunks = 2
                }
                let chunks = split_df_as_ref(&self.df, n_chunks).unwrap();
                debug_assert_eq!(chunks.iter().map(|df|df.height()).sum::<usize>(), left_len);
                // set in process
                // it is used below
                self.in_process = Some(chunks.into_iter())
            }
                // we can do a single join
            else {
                dbg!("fast path");

                // todo! amortize left and right name creation
                let df = self
                    .df
                    .cross_join(&chunk.data, Some(self.suffix.to_string()), None)?;
                return Ok(OperatorResult::Finished(chunk.with_data(df)))
            }

        }
        // output size is large we process in chunks
        let iter = self.in_process.as_mut().unwrap();
        match iter.next() {
            Some(df) => {
                // todo! amortize left and right name creation
                let df = df
                    .cross_join(&chunk.data, Some(self.suffix.to_string()), None)?;

                dbg!(&df.shape());
                Ok(OperatorResult::HaveMoreOutPut(chunk.with_data(df)))
            }
            None => {
                self.in_process = None;
                Ok(OperatorResult::NeedsNewData)
            }

        }


    }
    fn split(&self, _thread_no: usize) -> Box<dyn Operator> {
        Box::new(self.clone())
    }
}
