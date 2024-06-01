use std::any::Any;
use std::iter::StepBy;
use std::ops::Range;
use std::sync::Arc;
use std::vec;

use polars_core::error::PolarsResult;
use polars_core::frame::DataFrame;
use polars_ops::prelude::CrossJoin as CrossJoinTrait;
use polars_utils::arena::Node;
use smartstring::alias::String as SmartString;

use crate::executors::operators::PlaceHolder;
use crate::operators::{
    chunks_to_df_unchecked, DataChunk, FinalizedSink, Operator, OperatorResult, PExecutionContext,
    Sink, SinkResult,
};

#[derive(Default)]
pub struct CrossJoin {
    chunks: Vec<DataChunk>,
    suffix: SmartString,
    swapped: bool,
    node: Node,
    placeholder: PlaceHolder,
}

impl CrossJoin {
    pub(crate) fn new(
        suffix: SmartString,
        swapped: bool,
        node: Node,
        placeholder: PlaceHolder,
    ) -> Self {
        CrossJoin {
            chunks: vec![],
            suffix,
            swapped,
            node,
            placeholder,
        }
    }
}

impl Sink for CrossJoin {
    fn node(&self) -> Node {
        self.node
    }
    fn is_join_build(&self) -> bool {
        true
    }

    fn sink(&mut self, _context: &PExecutionContext, chunk: DataChunk) -> PolarsResult<SinkResult> {
        self.chunks.push(chunk);
        Ok(SinkResult::CanHaveMoreInput)
    }

    fn combine(&mut self, other: &mut dyn Sink) {
        let other = other.as_any().downcast_mut::<Self>().unwrap();
        let other_chunks = std::mem::take(&mut other.chunks);
        self.chunks.extend(other_chunks);
    }

    fn split(&self, _thread_no: usize) -> Box<dyn Sink> {
        Box::new(Self {
            suffix: self.suffix.clone(),
            swapped: self.swapped,
            placeholder: self.placeholder.clone(),
            ..Default::default()
        })
    }

    fn finalize(&mut self, _context: &PExecutionContext) -> PolarsResult<FinalizedSink> {
        let op = Box::new(CrossJoinProbe {
            df: Arc::new(chunks_to_df_unchecked(std::mem::take(&mut self.chunks))),
            suffix: Arc::from(self.suffix.as_ref()),
            in_process_left: None,
            in_process_right: None,
            in_process_left_df: Default::default(),
            output_names: None,
            swapped: self.swapped,
        });
        self.placeholder.replace(op);

        Ok(FinalizedSink::Operator)
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
    output_names: Option<Vec<SmartString>>,
    swapped: bool,
}

impl Operator for CrossJoinProbe {
    fn execute(
        &mut self,
        _context: &PExecutionContext,
        chunk: &DataChunk,
    ) -> PolarsResult<OperatorResult> {
        // Expected output is size**2, so this needs to be a a small number.
        // However, if one of the DataFrames is much smaller than 250, we want
        // to take rather more from the other DataFrame so we don't end up with
        // overly small chunks.
        let mut size = 250;
        if chunk.data.height() > 0 {
            size *= (250 / chunk.data.height()).max(1);
        }
        if self.df.height() > 0 {
            size *= (250 / self.df.height()).max(1);
        }

        if self.in_process_left.is_none() {
            let mut iter_left = (0..self.df.height()).step_by(size);
            let offset = iter_left.next().unwrap_or(0);
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
                    },
                    Some(offset) => {
                        self.in_process_left_df = self.df.slice(offset as i64, size);
                        self.in_process_right = Some((0..chunk.data.height()).step_by(size));
                        let iter_right = self.in_process_right.as_mut().unwrap();
                        let offset = iter_right.next().unwrap_or(0);
                        let right_df = chunk.data.slice(offset as i64, size);

                        let (a, b) = if self.swapped {
                            (&right_df, &self.in_process_left_df)
                        } else {
                            (&self.in_process_left_df, &right_df)
                        };

                        let mut df = a.cross_join(b, Some(self.suffix.as_ref()), None)?;
                        // Cross joins can produce multiple chunks.
                        // No parallelize in operators
                        df.as_single_chunk();
                        Ok(OperatorResult::HaveMoreOutPut(chunk.with_data(df)))
                    },
                }
            },
            // deplete the right chunks over the current left chunk
            Some(offset) => {
                // this will be the branch of the first call

                let right_df = chunk.data.slice(offset as i64, size);

                let (a, b) = if self.swapped {
                    (&right_df, &self.in_process_left_df)
                } else {
                    (&self.in_process_left_df, &right_df)
                };

                // we use the first join to determine the output names
                // this we can amortize the name allocations.
                let mut df = match &self.output_names {
                    None => {
                        let df = a.cross_join(b, Some(self.suffix.as_ref()), None)?;
                        self.output_names = Some(df.get_column_names_owned());
                        df
                    },
                    Some(names) => a._cross_join_with_names(b, names)?,
                };
                // Cross joins can produce multiple chunks.
                df.as_single_chunk();

                Ok(OperatorResult::HaveMoreOutPut(chunk.with_data(df)))
            },
        }
    }
    fn split(&self, _thread_no: usize) -> Box<dyn Operator> {
        Box::new(self.clone())
    }

    fn fmt(&self) -> &str {
        "cross_join_probe"
    }
}
