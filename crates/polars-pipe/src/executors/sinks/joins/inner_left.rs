use std::borrow::Cow;
use std::sync::Arc;

use arrow::array::BinaryArray;
use polars_core::datatypes::ChunkId;
use polars_core::error::PolarsResult;
use polars_core::export::ahash::RandomState;
use polars_core::prelude::*;
use polars_core::series::IsSorted;
use polars_ops::frame::join::_finish_join;
use polars_ops::prelude::JoinType;
use polars_row::RowsEncoded;
use polars_utils::hash_to_partition;
use polars_utils::slice::GetSaferUnchecked;
use smartstring::alias::String as SmartString;

use crate::executors::sinks::joins::generic_build::*;
use crate::executors::sinks::utils::hash_rows;
use crate::expressions::PhysicalPipedExpr;
use crate::operators::{DataChunk, Operator, OperatorResult, PExecutionContext};

#[derive(Clone)]
pub struct GenericJoinProbe {
    // all chunks are stacked into a single dataframe
    // the dataframe is not rechunked.
    df_a: Arc<DataFrame>,
    // the join columns are all tightly packed
    // the values of a join column(s) can be found
    // by:
    // first get the offset of the chunks and multiply that with the number of join
    // columns
    //      * chunk_offset = (idx * n_join_keys)
    //      * end = (offset + n_join_keys)
    materialized_join_cols: Arc<Vec<BinaryArray<i64>>>,
    suffix: Arc<str>,
    hb: RandomState,
    // partitioned tables that will be used for probing
    // stores the key and the chunk_idx, df_idx of the left table
    hash_tables: Arc<Vec<PlIdHashMap<Key, Vec<ChunkId>>>>,

    // the columns that will be joined on
    join_columns_right: Arc<Vec<Arc<dyn PhysicalPipedExpr>>>,

    // amortize allocations
    current_rows: RowsEncoded,
    join_columns: Vec<ArrayRef>,
    // in inner join these are the left table
    // in left join there are the right table
    join_tuples_a: Vec<ChunkId>,
    join_tuples_a_left_join: Vec<Option<ChunkId>>,
    // in inner join these are the right table
    // in left join there are the left table
    join_tuples_b: Vec<DfIdx>,
    hashes: Vec<u64>,
    // the join order is swapped to ensure we hash the smaller table
    swapped_or_left: bool,
    // location of join columns.
    // these column locations need to be dropped from the rhs
    join_column_idx: Option<Vec<usize>>,
    // cached output names
    output_names: Option<Vec<SmartString>>,
    how: JoinType,
}

impl GenericJoinProbe {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        mut df_a: DataFrame,
        materialized_join_cols: Arc<Vec<BinaryArray<i64>>>,
        suffix: Arc<str>,
        hb: RandomState,
        hash_tables: Arc<Vec<PlIdHashMap<Key, Vec<ChunkId>>>>,
        join_columns_left: Arc<Vec<Arc<dyn PhysicalPipedExpr>>>,
        join_columns_right: Arc<Vec<Arc<dyn PhysicalPipedExpr>>>,
        swapped_or_left: bool,
        join_columns: Vec<ArrayRef>,
        hashes: Vec<u64>,
        context: &PExecutionContext,
        how: JoinType,
    ) -> Self {
        if swapped_or_left {
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
                        .evaluate(&tmp, context.execution_state.as_any())
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
            join_columns_right,
            join_columns,
            join_tuples_a: vec![],
            join_tuples_a_left_join: vec![],
            join_tuples_b: vec![],
            hashes,
            swapped_or_left,
            current_rows: Default::default(),
            join_column_idx: None,
            output_names: None,
            how,
        }
    }
    fn set_join_series(
        &mut self,
        context: &PExecutionContext,
        chunk: &DataChunk,
    ) -> PolarsResult<BinaryArray<i64>> {
        debug_assert!(self.join_columns.is_empty());

        let determine_idx = !self.swapped_or_left && self.join_column_idx.is_none();
        let mut names = vec![];

        for phys_e in self.join_columns_right.iter() {
            let s = phys_e.evaluate(chunk, context.execution_state.as_any())?;
            let s = s.to_physical_repr().rechunk();
            if determine_idx {
                names.push(s.name().to_string());
            }
            self.join_columns.push(s.array_ref(0).clone());
        }

        // we determine the indices of the columns that have to be removed
        // if swapped the join column is already removed from the `build_df` as that will
        // be the rhs one.
        if !self.swapped_or_left && self.join_column_idx.is_none() {
            let mut idx = names
                .iter()
                .filter_map(|name| chunk.data.find_idx_by_name(name))
                .collect::<Vec<_>>();
            // ensure that it is sorted so that we can later remove columns in
            // a predictable order
            idx.sort_unstable();
            self.join_column_idx = Some(idx);
        }
        polars_row::convert_columns_amortized_no_order(&self.join_columns, &mut self.current_rows);

        // safety: we keep rows-encode alive
        unsafe { Ok(self.current_rows.borrow_array()) }
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
                // safety:
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

    fn execute_left(
        &mut self,
        context: &PExecutionContext,
        chunk: &DataChunk,
    ) -> PolarsResult<OperatorResult> {
        // A left join holds the right table as build table
        // and streams the left table through. This allows us to maintain
        // the left table order

        self.join_tuples_a_left_join.clear();
        self.join_tuples_b.clear();
        let mut hashes = std::mem::take(&mut self.hashes);
        let rows = self.set_join_series(context, chunk)?;
        hash_rows(&rows, &mut hashes, &self.hb);
        self.hashes = hashes;

        for (i, (h, row)) in self.hashes.iter().zip(rows.values_iter()).enumerate() {
            let df_idx_left = i as IdxSize;
            // get the hashtable belonging by this hash partition
            let partition = hash_to_partition(*h, self.hash_tables.len());
            let current_table = unsafe { self.hash_tables.get_unchecked_release(partition) };

            let entry = current_table
                .raw_entry()
                .from_hash(*h, |key| {
                    compare_fn(key, *h, &self.materialized_join_cols, row)
                })
                .map(|key_val| key_val.1);

            match entry {
                Some(indexes_right) => {
                    self.join_tuples_a_left_join
                        .extend(indexes_right.iter().copied().map(Some));
                    self.join_tuples_b
                        .extend(std::iter::repeat(df_idx_left).take(indexes_right.len()));
                },
                None => {
                    self.join_tuples_b.push(df_idx_left);
                    self.join_tuples_a_left_join.push(None);
                },
            }
        }
        let right_df = self.df_a.as_ref();

        // join tuples of left joins are always sorted
        // this will ensure sorted flags maintain
        let left_df = unsafe {
            chunk
                .data
                ._take_unchecked_slice_sorted(&self.join_tuples_b, false, IsSorted::Ascending)
        };
        let right_df =
            unsafe { right_df._take_opt_chunked_unchecked_seq(&self.join_tuples_a_left_join) };

        let out = self.finish_join(left_df, right_df)?;

        // clear memory
        self.join_columns.clear();
        self.hashes.clear();

        Ok(OperatorResult::Finished(chunk.with_data(out)))
    }

    fn execute_inner(
        &mut self,
        context: &PExecutionContext,
        chunk: &DataChunk,
    ) -> PolarsResult<OperatorResult> {
        self.join_tuples_a.clear();
        self.join_tuples_b.clear();
        let mut hashes = std::mem::take(&mut self.hashes);
        let rows = self.set_join_series(context, chunk)?;
        hash_rows(&rows, &mut hashes, &self.hb);
        self.hashes = hashes;

        for (i, (h, row)) in self.hashes.iter().zip(rows.values_iter()).enumerate() {
            let df_idx_right = i as IdxSize;
            // get the hashtable belonging by this hash partition
            let partition = hash_to_partition(*h, self.hash_tables.len());
            let current_table = unsafe { self.hash_tables.get_unchecked_release(partition) };

            let entry = current_table
                .raw_entry()
                .from_hash(*h, |key| {
                    compare_fn(key, *h, &self.materialized_join_cols, row)
                })
                .map(|key_val| key_val.1);

            if let Some(indexes_left) = entry {
                self.join_tuples_a.extend_from_slice(indexes_left);
                self.join_tuples_b
                    .extend(std::iter::repeat(df_idx_right).take(indexes_left.len()));
            }
        }

        let left_df = unsafe {
            self.df_a
                ._take_chunked_unchecked_seq(&self.join_tuples_a, IsSorted::Not)
        };
        let right_df = unsafe {
            let mut df = Cow::Borrowed(&chunk.data);
            if let Some(ids) = &self.join_column_idx {
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

        // clear memory
        self.join_columns.clear();
        self.hashes.clear();

        Ok(OperatorResult::Finished(chunk.with_data(out)))
    }
}

impl Operator for GenericJoinProbe {
    fn execute(
        &mut self,
        context: &PExecutionContext,
        chunk: &DataChunk,
    ) -> PolarsResult<OperatorResult> {
        match self.how {
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
