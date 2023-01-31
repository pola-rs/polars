use std::any::Any;
use std::borrow::Cow;
use std::iter::StepBy;
use std::ops::Range;
use std::sync::Arc;
use std::vec;

use polars_core::error::PolarsResult;
use polars_core::frame::DataFrame;

use crate::operators::{
    chunks_to_df_unchecked, DataChunk, FinalizedSink, Operator, OperatorResult, PExecutionContext,
    Sink, SinkResult,
};

#[derive(Default)]
pub struct CrossJoin {
    chunks: Vec<DataChunk>,
    suffix: Cow<'static, str>,
}

impl CrossJoin {
    pub(crate) fn new(suffix: Cow<'static, str>) -> Self {
        CrossJoin {
            chunks: vec![],
            suffix,
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
            ..Default::default()
        })
    }

    fn finalize(&mut self, _context: &PExecutionContext) -> PolarsResult<FinalizedSink> {
        // todo! share sink
        Ok(FinalizedSink::Operator(Box::new(CrossJoinProbe {
            df: Arc::new(chunks_to_df_unchecked(std::mem::take(&mut self.chunks))),
            suffix: Arc::from(self.suffix.as_ref()),
            in_process_left: None,
            in_process_right: None,
            in_process_left_df: Default::default(),
            output_names: None,
        })))
    }

    fn as_any(&mut self) -> &mut dyn Any {
        self
    }

    fn fmt(&self) -> &str {
        "cross_join_sink"
    }
}

#[derive(Clone)]
pub struct CrossJoinProbe {
    df: Arc<DataFrame>,
    suffix: Arc<str>,
    in_process_left: Option<StepBy<Range<usize>>>,
    in_process_right: Option<StepBy<Range<usize>>>,
    in_process_left_df: DataFrame,
    output_names: Option<Vec<String>>,
}

impl Operator for CrossJoinProbe {
    fn execute(
        &mut self,
        _context: &PExecutionContext,
        chunk: &DataChunk,
    ) -> PolarsResult<OperatorResult> {
        // expected output size = size**2
        // so this is a small number
        let size = 250;

        if self.in_process_left.is_none() {
            let mut iter_left = (0..self.df.height()).step_by(size);
            let offset = iter_left.next().unwrap();
            self.in_process_left_df = self.df.slice(offset as i64, size);
            self.in_process_left = Some(iter_left);
        }
        if self.in_process_right.is_none() {
            self.in_process_right = Some((0..chunk.data.height()).step_by(size));
        }
        // output size is large we process in chunks
        let iter_left = self.in_process_left.as_mut().unwrap();
        let iter_right = self.in_process_right.as_mut().unwrap();

        match iter_right.next() {
            None => {
                self.in_process_right = None;

                // if right is depleted take the next left chunk
                match iter_left.next() {
                    None => {
                        self.in_process_left = None;
                        Ok(OperatorResult::NeedsNewData)
                    }
                    Some(offset) => {
                        self.in_process_left_df = self.df.slice(offset as i64, size);
                        self.in_process_right = Some((0..chunk.data.height()).step_by(size));
                        let iter_right = self.in_process_right.as_mut().unwrap();
                        let offset = iter_right.next().unwrap();
                        let right_df = chunk.data.slice(offset as i64, size);
                        let df = self.in_process_left_df.cross_join(
                            &right_df,
                            Some(self.suffix.as_ref()),
                            None,
                        )?;
                        Ok(OperatorResult::HaveMoreOutPut(chunk.with_data(df)))
                    }
                }
            }
            // deplete the right chunks over the current left chunk
            Some(offset) => {
                // this will be the branch of the first call

                let right_df = chunk.data.slice(offset as i64, size);

                // we use the first join to determine the output names
                // this we can amortize the name allocations.
                let df = match &self.output_names {
                    None => {
                        let df = self.in_process_left_df.cross_join(
                            &right_df,
                            Some(self.suffix.as_ref()),
                            None,
                        )?;
                        self.output_names = Some(df.get_column_names_owned());
                        df
                    }
                    Some(names) => self
                        .in_process_left_df
                        ._cross_join_with_names(&right_df, names)?,
                };

                Ok(OperatorResult::HaveMoreOutPut(chunk.with_data(df)))
            }
        }
    }
    fn split(&self, _thread_no: usize) -> Box<dyn Operator> {
        Box::new(self.clone())
    }

    fn fmt(&self) -> &str {
        "cross_join_probe"
    }
}
