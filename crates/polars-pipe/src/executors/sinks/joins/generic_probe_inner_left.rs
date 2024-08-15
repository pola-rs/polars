use std::borrow::Cow;

use arrow::array::{Array, BinaryArray};
use polars_core::prelude::*;
use polars_core::series::IsSorted;
use polars_ops::chunked_array::DfTake;
use polars_ops::frame::join::_finish_join;
use polars_ops::prelude::{JoinArgs, JoinType};
use polars_utils::nulls::IsNull;
use smartstring::alias::String as SmartString;

use crate::executors::sinks::joins::generic_build::*;
use crate::executors::sinks::joins::row_values::RowValues;
use crate::executors::sinks::joins::{ExtraPayload, PartitionedMap, ToRow};
use crate::executors::sinks::utils::hash_rows;
use crate::expressions::PhysicalPipedExpr;
use crate::operators::{DataChunk, Operator, OperatorResult, PExecutionContext};

#[derive(Clone)]
pub struct GenericJoinProbe<K: ExtraPayload> {
    /// All chunks are stacked into a single dataframe
    /// the dataframe is not rechunked.
    df_a: Arc<DataFrame>,
    /// The join columns are all tightly packed
    /// the values of a join column(s) can be found
    /// by:
    /// first get the offset of the chunks and multiply that with the number of join
    /// columns
    ///      * chunk_offset = (idx * n_join_keys)
    ///      * end = (offset + n_join_keys)
    materialized_join_cols: Arc<[BinaryArray<i64>]>,
    suffix: Arc<str>,
    hb: PlRandomState,
    /// partitioned tables that will be used for probing
    /// stores the key and the chunk_idx, df_idx of the left table
    hash_tables: Arc<PartitionedMap<K>>,

    /// Amortize allocations
    /// In inner join these are the left table.
    /// In left join there are the right table.
    join_tuples_a: Vec<ChunkId>,
    /// in inner join these are the right table
    /// in left join there are the left table
    join_tuples_b: Vec<DfIdx>,
    hashes: Vec<u64>,
    /// the join order is swapped to ensure we hash the smaller table
    swapped_or_left: bool,
    /// cached output names
    output_names: Option<Vec<SmartString>>,
    args: JoinArgs,
    join_nulls: bool,
    row_values: RowValues,
}

impl<K: ExtraPayload> GenericJoinProbe<K> {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        mut df_a: DataFrame,
        materialized_join_cols: Arc<[BinaryArray<i64>]>,
        suffix: Arc<str>,
        hb: PlRandomState,
        hash_tables: Arc<PartitionedMap<K>>,
        join_columns_left: Arc<Vec<Arc<dyn PhysicalPipedExpr>>>,
        join_columns_right: Arc<Vec<Arc<dyn PhysicalPipedExpr>>>,
        swapped_or_left: bool,
        // Re-use the hashes allocation of the build side.
        amortized_hashes: Vec<u64>,
        context: &PExecutionContext,
        args: JoinArgs,
        join_nulls: bool,
    ) -> Self {
        if swapped_or_left && args.should_coalesce() {
            let tmp = DataChunk {
                data: df_a.slice(0, 1),
                chunk_index: 0,
            };

            // remove duplicate_names caused by joining
            // on the same column
            let names = join_columns_left
                .iter()
                .flat_map(|phys_e| {
                    phys_e
                        .evaluate(&tmp, &context.execution_state)
                        .ok()
                        .map(|s| s.name().to_string())
                })
                .collect::<Vec<_>>();
            df_a = df_a.drop_many(&names)
        }

        GenericJoinProbe {
            df_a: Arc::new(df_a),
            materialized_join_cols,
            suffix,
            hb,
            hash_tables,
            join_tuples_a: vec![],
            join_tuples_b: vec![],
            hashes: amortized_hashes,
            swapped_or_left,
            output_names: None,
            args,
            join_nulls,
            row_values: RowValues::new(join_columns_right, !swapped_or_left),
        }
    }

    fn finish_join(
        &mut self,
        mut left_df: DataFrame,
        right_df: DataFrame,
    ) -> PolarsResult<DataFrame> {
        Ok(match &self.output_names {
            None => {
                let out = _finish_join(left_df, right_df, Some(self.suffix.as_ref()))?;
                self.output_names = Some(out.get_column_names_owned());
                out
            },
            Some(names) => unsafe {
                // SAFETY:
                // if we have duplicate names, we overwrite
                // them in the next snippet
                left_df
                    .get_columns_mut()
                    .extend_from_slice(right_df.get_columns());
                left_df
                    .get_columns_mut()
                    .iter_mut()
                    .zip(names)
                    .for_each(|(s, name)| {
                        s.rename(name);
                    });
                left_df
            },
        })
    }

    fn match_left<'b, I, T>(&mut self, iter: I)
    where
        I: Iterator<Item = (usize, (&'b u64, T))> + 'b,
        T: IsNull
            // Temporary trait to get concrete &[u8]
            // Input is either &[u8] or Option<&[u8]>
            + ToRow,
    {
        for (i, (h, row)) in iter {
            let df_idx_left = i as IdxSize;

            let entry = if row.is_null() {
                None
            } else {
                let row = row.get_row();
                self.hash_tables
                    .raw_entry(*h)
                    .from_hash(*h, |key| {
                        compare_fn(key, *h, &self.materialized_join_cols, row)
                    })
                    .map(|key_val| key_val.1)
            };

            match entry {
                Some(indexes_right) => {
                    let indexes_right = &indexes_right.0;
                    self.join_tuples_a.extend_from_slice(indexes_right);
                    self.join_tuples_b
                        .extend(std::iter::repeat(df_idx_left).take(indexes_right.len()));
                },
                None => {
                    self.join_tuples_b.push(df_idx_left);
                    self.join_tuples_a.push(ChunkId::null());
                },
            }
        }
    }

    fn execute_left(
        &mut self,
        context: &PExecutionContext,
        chunk: &DataChunk,
    ) -> PolarsResult<OperatorResult> {
        // A left join holds the right table as build table
        // and streams the left table through. This allows us to maintain
        // the left table order

        self.join_tuples_a.clear();
        self.join_tuples_b.clear();
        let mut hashes = std::mem::take(&mut self.hashes);
        let rows = self
            .row_values
            .get_values(context, chunk, self.join_nulls)?;
        hash_rows(&rows, &mut hashes, &self.hb);

        if self.join_nulls || rows.null_count() == 0 {
            let iter = hashes.iter().zip(rows.values_iter()).enumerate();
            self.match_left(iter);
        } else {
            let iter = hashes.iter().zip(rows.iter()).enumerate();
            self.match_left(iter);
        }
        self.hashes = hashes;
        let right_df = self.df_a.as_ref();

        // join tuples of left joins are always sorted
        // this will ensure sorted flags maintain
        let left_df = unsafe {
            chunk
                .data
                ._take_unchecked_slice_sorted(&self.join_tuples_b, false, IsSorted::Ascending)
        };
        let right_df = unsafe { right_df._take_opt_chunked_unchecked_seq(&self.join_tuples_a) };

        let out = self.finish_join(left_df, right_df)?;

        // Clear memory.
        self.row_values.clear();
        self.hashes.clear();

        Ok(OperatorResult::Finished(chunk.with_data(out)))
    }

    fn match_inner<'b, I>(&mut self, iter: I)
    where
        I: Iterator<Item = (usize, (&'b u64, &'b [u8]))> + 'b,
    {
        for (i, (h, row)) in iter {
            let df_idx_right = i as IdxSize;

            let entry = self
                .hash_tables
                .raw_entry(*h)
                .from_hash(*h, |key| {
                    compare_fn(key, *h, &self.materialized_join_cols, row)
                })
                .map(|key_val| key_val.1);

            if let Some(indexes_left) = entry {
                let indexes_left = &indexes_left.0;
                self.join_tuples_a.extend_from_slice(indexes_left);
                self.join_tuples_b
                    .extend(std::iter::repeat(df_idx_right).take(indexes_left.len()));
            }
        }
    }

    fn execute_inner(
        &mut self,
        context: &PExecutionContext,
        chunk: &DataChunk,
    ) -> PolarsResult<OperatorResult> {
        self.join_tuples_a.clear();
        self.join_tuples_b.clear();
        let mut hashes = std::mem::take(&mut self.hashes);
        let rows = self
            .row_values
            .get_values(context, chunk, self.join_nulls)?;
        hash_rows(&rows, &mut hashes, &self.hb);

        if self.join_nulls || rows.null_count() == 0 {
            let iter = hashes.iter().zip(rows.values_iter()).enumerate();
            self.match_inner(iter);
        } else {
            let iter = hashes
                .iter()
                .zip(rows.iter())
                .enumerate()
                .filter_map(|(i, (h, row))| row.map(|row| (i, (h, row))));
            self.match_inner(iter);
        }
        self.hashes = hashes;

        let left_df = unsafe {
            self.df_a
                ._take_chunked_unchecked_seq(&self.join_tuples_a, IsSorted::Not)
        };
        let right_df = unsafe {
            let mut df = Cow::Borrowed(&chunk.data);
            if let Some(ids) = &self.row_values.join_column_idx {
                let mut tmp = df.into_owned();
                let cols = tmp.get_columns_mut();
                // we go from higher idx to lower so that lower indices remain untouched
                // by our mutation
                for idx in ids.iter().rev() {
                    let _ = cols.remove(*idx);
                }
                df = Cow::Owned(tmp);
            }
            df._take_unchecked_slice(&self.join_tuples_b, false)
        };

        let (a, b) = if self.swapped_or_left {
            (right_df, left_df)
        } else {
            (left_df, right_df)
        };
        let out = self.finish_join(a, b)?;

        // Clear memory.
        self.row_values.clear();
        self.hashes.clear();

        Ok(OperatorResult::Finished(chunk.with_data(out)))
    }
}

impl<K: ExtraPayload> Operator for GenericJoinProbe<K> {
    fn execute(
        &mut self,
        context: &PExecutionContext,
        chunk: &DataChunk,
    ) -> PolarsResult<OperatorResult> {
        match self.args.how {
            JoinType::Inner => self.execute_inner(context, chunk),
            JoinType::Left => self.execute_left(context, chunk),
            _ => unreachable!(),
        }
    }

    fn split(&self, _thread_no: usize) -> Box<dyn Operator> {
        let new = self.clone();
        Box::new(new)
    }
    fn fmt(&self) -> &str {
        "generic_join_probe"
    }
}
