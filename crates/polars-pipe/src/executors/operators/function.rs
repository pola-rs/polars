use std::collections::VecDeque;

use polars_core::error::PolarsResult;
use polars_core::utils::_split_offsets;
use polars_core::POOL;
use polars_plan::prelude::*;

use crate::operators::{DataChunk, Operator, OperatorResult, PExecutionContext};
use crate::pipeline::determine_chunk_size;

#[derive(Clone)]
pub struct FunctionOperator {
    n_threads: usize,
    chunk_size: usize,
    offsets: VecDeque<(usize, usize)>,
    function: FunctionIR,
}

impl FunctionOperator {
    pub(crate) fn new(function: FunctionIR) -> Self {
        FunctionOperator {
            n_threads: POOL.current_num_threads(),
            function,
            chunk_size: 128,
            offsets: VecDeque::new(),
        }
    }

    fn execute_no_expanding(&mut self, chunk: &DataChunk) -> PolarsResult<OperatorResult> {
        Ok(OperatorResult::Finished(
            chunk.with_data(self.function.evaluate(chunk.data.clone())?),
        ))
    }

    // Combine every two `(offset, len)` pairs so that we double the chunk size
    fn combine_offsets(&mut self) {
        self.offsets = self
            .offsets
            .make_contiguous()
            .chunks(2)
            .map(|chunk| {
                if chunk.len() == 2 {
                    let offset = chunk[0].0;
                    let len = chunk[0].1 + chunk[1].1;
                    (offset, len)
                } else {
                    chunk[0]
                }
            })
            .collect()
    }
}

impl Operator for FunctionOperator {
    fn execute(
        &mut self,
        context: &PExecutionContext,
        chunk: &DataChunk,
    ) -> PolarsResult<OperatorResult> {
        if self.function.expands_rows() {
            let input_height = chunk.data.height();
            // ideal chunk size we want to have
            // we cannot rely on input chunk size as that can increase due to multiple explode calls
            // for instance.
            let chunk_size_ambition = determine_chunk_size(chunk.data.width(), self.n_threads)?;

            if self.offsets.is_empty() {
                let n = input_height / self.chunk_size;
                if n > 1 {
                    self.offsets = _split_offsets(input_height, n).into();
                } else {
                    return self.execute_no_expanding(chunk);
                }
            }
            if let Some((offset, len)) = self.offsets.pop_front() {
                let df = chunk.data.slice(offset as i64, len);
                let output = self.function.evaluate(df)?;
                if output.height() * 2 < chunk.data.height()
                    && output.height() * 2 < chunk_size_ambition
                {
                    self.chunk_size *= 2;
                    // ensure that next slice is larger
                    self.combine_offsets();
                }
                // allow some increase in chunk size so that we don't toggle the chunk size
                // every iteration
                else if output.height() * 4 > chunk.data.height()
                    || output.height() > chunk_size_ambition * 2
                {
                    let new_chunk_size = self.chunk_size / 2;

                    if context.verbose && new_chunk_size < 5 {
                        eprintln!("chunk size in 'function operation' shrank to {new_chunk_size} and has been set to 5 as lower limit")
                    }
                    // ensure it is never 0
                    self.chunk_size = std::cmp::max(new_chunk_size, 5);
                };
                let output = chunk.with_data(output);
                if self.offsets.is_empty() {
                    Ok(OperatorResult::Finished(output))
                } else {
                    Ok(OperatorResult::HaveMoreOutPut(output))
                }
            } else {
                self.execute_no_expanding(chunk)
            }
        } else {
            self.execute_no_expanding(chunk)
        }
    }

    fn split(&self, _thread_no: usize) -> Box<dyn Operator> {
        Box::new(self.clone())
    }

    fn fmt(&self) -> &str {
        "function"
    }
}
