use std::sync::Arc;

use parking_lot::Mutex;
use polars_async::executor::TaskMetricAggregator;
use polars_core::frame::column::{DictionaryMerger, try_concat_columns};
use polars_core::prelude::{
    Column, DataFrame, DataType, IntoColumn, IntoSeries, PlSmallStr, Series,
};
use polars_core::runtime::RAYON;
use polars_core::schema::Schema;
use polars_core::utils::accumulate_dataframes_vertical_unchecked_dictionary_aware;
use polars_ooc::{LeastRecentSpillContext, ParameterFreeSpillContext, SpillFrame};
use rayon::prelude::*;

use super::compute_node_prelude::*;

/// A column of `len` null codes.
fn null_codes(name: PlSmallStr, len: usize) -> Column {
    Column::full_null(name, len, &DataType::UInt32)
}

use crate::utils::in_memory_linearize::linearize;

#[derive(Debug)]
pub struct InMemorySinkNode {
    morsels_per_pipe: Mutex<Vec<Vec<Morsel>>>,
    schema: Arc<Schema>,
    spill_ctx: LeastRecentSpillContext,
    /// For each column, the merger of the dictionaries of its dictionary-coded morsels, when
    /// [`Self::merging_dictionaries`].
    mergers: Option<Vec<Mutex<DictionaryMerger>>>,
}

impl InMemorySinkNode {
    pub fn new(schema: Arc<Schema>, task_metrics: Option<Arc<TaskMetricAggregator>>) -> Self {
        Self {
            morsels_per_pipe: Mutex::default(),
            schema,
            spill_ctx: LeastRecentSpillContext::new("in-memory-sink".into(), task_metrics),
            mergers: None,
        }
    }

    /// A sink whose output is one frame ([`ComputeNode::get_output`]) that keeps dictionary-coded
    /// columns coded. Each morsel's codes are shifted to the merged dictionary as the morsel
    /// arrives, so that the work happens while the query runs on other threads, and the stored
    /// morsels hold the shifted codes as plain `UInt32` columns until the output is made. For
    /// this, the morsels must not be taken with [`Self::take_spill_frames`].
    pub fn merging_dictionaries(
        schema: Arc<Schema>,
        task_metrics: Option<Arc<TaskMetricAggregator>>,
    ) -> Self {
        let mergers = (0..schema.len()).map(|_| Mutex::default()).collect();
        Self {
            mergers: Some(mergers),
            ..Self::new(schema, task_metrics)
        }
    }

    /// Whether the column of `idx` is held as codes if it is a dictionary column: not if the
    /// codes would be mistaken for the values.
    fn is_mergeable(&self, idx: usize) -> bool {
        self.schema
            .get_at_index(idx)
            .is_some_and(|(_, dtype)| *dtype != DataType::UInt32)
    }

    /// Whether `c` is a column of nulls that has no values of its own (a column missing from a
    /// file), of the dtype of column `idx`.
    fn is_null_scalar(&self, idx: usize, c: &Column) -> bool {
        c.as_scalar_column().is_some_and(|s| {
            s.scalar().is_null()
                && self
                    .schema
                    .get_at_index(idx)
                    .is_some_and(|(_, dtype)| dtype == c.dtype())
        })
    }

    /// Whether `c`, a part of the column of `idx`, holds shifted (or null) codes.
    fn is_codes(&self, idx: usize, c: &Column) -> bool {
        self.is_mergeable(idx)
            && c.dtype() == &DataType::UInt32
            && c.as_dictionary_column().is_none()
    }

    /// Replaces the dictionary columns of `morsel` with their codes shifted to the merged
    /// dictionary. A column that has no codes to speak of (no rows, only nulls) becomes null
    /// codes, holding no dictionary; one the merger refuses is materialized here, so that no
    /// dictionary is held for it.
    async fn shift_dictionary_codes(
        &self,
        morsel: &mut Morsel,
        mergers: &[Mutex<DictionaryMerger>],
    ) {
        let mut df = morsel.df_mut().await;
        if df.width() != mergers.len()
            || !df.columns().iter().enumerate().any(|(i, c)| {
                self.is_mergeable(i)
                    && (c.as_dictionary_column().is_some() || self.is_null_scalar(i, c))
            })
        {
            return;
        }
        let height = df.height();
        let columns = std::mem::replace(&mut *df, DataFrame::empty()).into_columns();
        let columns = columns
            .into_iter()
            .enumerate()
            .map(|(i, c)| {
                if !self.is_mergeable(i) {
                    return c;
                }
                if self.is_null_scalar(i, &c) {
                    mergers[i].lock().skip_rows(c.len());
                    return null_codes(c.name().clone(), c.len());
                }
                let Column::Dictionary(d) = c else {
                    return c;
                };
                if d.null_count() == d.len() {
                    mergers[i].lock().skip_rows(d.len());
                    return null_codes(d.name().clone(), d.len());
                }
                if let Some(dtype) = self.schema.get_at_index(i).map(|(_, d)| d)
                    && d.dtype() != dtype
                {
                    mergers[i].lock().fail();
                    return d.take_materialized_series().into_column();
                }
                let offset = mergers[i].lock().register(&d);
                match offset {
                    Some(offset) => {
                        let name = d.name().clone();
                        Column::from(d.into_shifted_codes(offset).with_name(name).into_series())
                    },
                    None => d.take_materialized_series().into_column(),
                }
            })
            .collect();
        // SAFETY: the columns have the height of the frame they come from.
        *df = unsafe { DataFrame::new_unchecked(height, columns) };
    }

    /// The output of a sink that is [`Self::merging_dictionaries`].
    fn merge_output(
        &self,
        mut dfs: Vec<DataFrame>,
        mergers: &[Mutex<DictionaryMerger>],
    ) -> PolarsResult<DataFrame> {
        let height = dfs.iter().map(|df| df.height()).sum::<usize>();
        let mut columns = Vec::with_capacity(mergers.len());
        let mut per_df: Vec<std::vec::IntoIter<Column>> = dfs
            .drain(..)
            .map(|df| df.into_columns().into_iter())
            .collect();
        for (i, merger) in mergers.iter().enumerate() {
            let merger = merger.lock();
            let mut parts: Vec<Column> = per_df.iter_mut().map(|c| c.next().unwrap()).collect();
            if !parts.iter().any(|c| self.is_codes(i, c)) {
                columns.push(try_concat_columns(parts)?);
                continue;
            }
            let name = parts[0].name().clone();
            if merger.is_empty() {
                // Only null codes: nothing was registered.
                let dtype = self.schema.get_at_index(i).map(|(_, d)| d.clone()).unwrap();
                for c in parts.iter_mut() {
                    if self.is_codes(i, c) {
                        *c = Column::full_null(name.clone(), c.len(), &dtype);
                    }
                }
                columns.push(try_concat_columns(parts)?);
                continue;
            }
            // A column is coded when the merger took all its chunks, and the codes of each are
            // what is left of them.
            let all_codes = !merger.failed() && parts.iter().all(|c| self.is_codes(i, c));
            if all_codes && merger.is_dense_enough(height) {
                let chunks = parts.into_iter().map(|c| c.u32().unwrap().clone());
                columns.push(merger.into_column(name, chunks)?.into());
                continue;
            }
            // Gather the values, each chunk from its own dictionary.
            let series: Vec<PolarsResult<Series>> = RAYON.install(|| {
                parts
                    .into_par_iter()
                    .map(|c| {
                        if self.is_codes(i, &c) {
                            let codes = c.u32().unwrap().clone();
                            drop(c);
                            merger.gather_chunk(name.clone(), codes)
                        } else {
                            Ok(c.as_materialized_series().clone())
                        }
                    })
                    .collect()
            });
            let mut series = series.into_iter();
            let mut acc = series.next().unwrap()?;
            for s in series {
                acc.append_owned(s?)?;
            }
            columns.push(acc.into_column());
        }
        // SAFETY: every column has the summed height.
        Ok(unsafe { DataFrame::new_unchecked(height, columns) })
    }

    /// Takes the collected morsels in order, as frames registered in `spill_ctx`. They can be
    /// spilled while `spill_ctx` lives.
    pub fn take_spill_frames(&mut self, spill_ctx: &LeastRecentSpillContext) -> Arc<[SpillFrame]> {
        let morsels_per_pipe = core::mem::take(&mut *self.morsels_per_pipe.get_mut());
        linearize(morsels_per_pipe)
            .into_iter()
            .map(|morsel| {
                let sf = morsel.into_sf();
                spill_ctx.register_no_spill_check(&sf);
                sf
            })
            .collect()
    }
}

impl ComputeNode for InMemorySinkNode {
    fn name(&self) -> &str {
        "in-memory-sink"
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        _state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        assert!(send.is_empty());
        assert!(recv.len() == 1);

        // We are always ready to receive, unless the sender is done, then we're
        // also done.
        if recv[0] != PortState::Done {
            recv[0] = PortState::Ready;
        }
        Ok(())
    }

    fn memory_usage(&self) -> NodeMemoryUsage {
        NodeMemoryUsage::Accumulating
    }

    fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        recv_ports: &mut [Option<RecvPort<'_>>],
        send_ports: &mut [Option<SendPort<'_>>],
        _state: &'s StreamingExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert!(recv_ports.len() == 1 && send_ports.is_empty());
        let receivers = recv_ports[0].take().unwrap().parallel();

        for mut recv in receivers {
            let slf = &*self;
            join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                let mut morsels = Vec::new();
                while let Ok(mut morsel) = recv.recv().await {
                    morsel.take_consume_token();
                    if let Some(mergers) = &slf.mergers {
                        slf.shift_dictionary_codes(&mut morsel, mergers).await;
                    }
                    slf.spill_ctx.register(morsel.sf()).await;
                    morsels.push(morsel);
                }

                slf.morsels_per_pipe.lock().push(morsels);
                Ok(())
            }));
        }
    }

    fn get_output(&mut self) -> PolarsResult<Option<DataFrame>> {
        let morsels_per_pipe = core::mem::take(&mut *self.morsels_per_pipe.get_mut());
        let spillframes = linearize(morsels_per_pipe);
        if spillframes.is_empty() {
            Ok(Some(DataFrame::empty_with_schema(&self.schema)))
        } else {
            let dfs: Vec<DataFrame> = spillframes
                .into_iter()
                .map(|sf| sf.into_df_blocking())
                .collect();
            if let Some(mergers) = &self.mergers {
                return self.merge_output(dfs, mergers).map(Some);
            }
            Ok(Some(
                accumulate_dataframes_vertical_unchecked_dictionary_aware(dfs),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use polars_arrow::array::Utf8ViewArray;
    use polars_core::frame::column::{DictionaryColumn, SharedDictionary};
    use polars_core::prelude::*;
    use polars_core::runtime::ASYNC;
    use polars_error::PolarsResult;

    use super::*;
    use crate::morsel::{MorselSeq, SourceToken};

    fn str_dict(name: &str, dict: &[String], codes: &[Option<u32>]) -> Column {
        let values = Utf8ViewArray::from_slice_values(dict);
        let codes = UInt32Chunked::from_iter_options(PlSmallStr::EMPTY, codes.iter().copied());
        DictionaryColumn::try_new(name.into(), SharedDictionary::new(values), codes)
            .unwrap()
            .into_column()
    }

    fn int_dict(name: &str, dict: Vec<i64>, codes: &[Option<u32>]) -> Column {
        let codes = UInt32Chunked::from_iter_options(PlSmallStr::EMPTY, codes.iter().copied());
        let dict = SharedDictionary::try_new(Series::new(PlSmallStr::EMPTY, dict)).unwrap();
        DictionaryColumn::try_new(name.into(), dict, codes)
            .unwrap()
            .into_column()
    }

    fn schema(cols: &[&Column]) -> Arc<Schema> {
        Arc::new(
            cols.iter()
                .map(|c| Field::new(c.name().clone(), c.dtype().clone()))
                .collect(),
        )
    }

    /// Feeds `dfs` to a merging sink as the pipe tasks do and returns its output.
    fn run(schema: Arc<Schema>, dfs: Vec<DataFrame>) -> PolarsResult<DataFrame> {
        let mut sink = InMemorySinkNode::merging_dictionaries(schema, None);
        let mergers = sink.mergers.take().unwrap();
        let mut morsels = Vec::new();
        ASYNC.block_in_place_on(async {
            for (i, df) in dfs.into_iter().enumerate() {
                let mut morsel =
                    Morsel::new_unregistered(df, MorselSeq::new(i as u64), SourceToken::new());
                sink.shift_dictionary_codes(&mut morsel, &mergers).await;
                sink.spill_ctx.register(morsel.sf()).await;
                morsels.push(morsel);
            }
        });
        let dfs = morsels.into_iter().map(|m| m.into_sf().into_df_blocking());
        sink.merge_output(dfs.collect(), &mergers)
    }

    #[test]
    fn filtered_scan_does_not_hold_every_dictionary() {
        // 200 row groups of which a morsel has 2 rows, each with its own large dictionary.
        let n_dict = 200_000usize;
        let mut dfs = Vec::new();
        let mut expected: Vec<i64> = Vec::new();
        for g in 0..200i64 {
            let dict: Vec<i64> = (0..n_dict as i64).map(|v| v + g * 7).collect();
            expected.push(dict[3]);
            expected.push(dict[n_dict - 1]);
            let c = int_dict("a", dict, &[Some(3), Some(n_dict as u32 - 1)]);
            dfs.push(DataFrame::new(2, vec![c]).unwrap());
        }
        let sch = Arc::new(Schema::from_iter([Field::new("a".into(), DataType::Int64)]));
        let out = run(sch, dfs).unwrap();
        let a = out.column("a").unwrap();
        assert_eq!(a.dtype(), &DataType::Int64);
        assert_eq!(
            a.i64().unwrap().into_no_null_iter().collect::<Vec<_>>(),
            expected
        );
    }

    #[test]
    fn empty_and_null_chunks_hold_no_dictionary() {
        let dict: Vec<String> = (0..10).map(|i| format!("v{i}")).collect();
        let a = str_dict("s", &dict, &[Some(1), None, Some(9)]);
        let empty = str_dict("s", &dict, &[]);
        let nulls = str_dict("s", &dict, &[None, None]);
        let sch = schema(&[&a]);
        let dfs = vec![
            DataFrame::new(0, vec![empty]).unwrap(),
            DataFrame::new(3, vec![a.clone()]).unwrap(),
            DataFrame::new(2, vec![nulls]).unwrap(),
        ];
        let out = run(sch.clone(), dfs).unwrap();
        let c = out.column("s").unwrap();
        assert_eq!(c.len(), 5);
        assert!(c.as_dictionary_column().is_some(), "kept as codes");
        let exp = Series::new("s".into(), [Some("v1"), None, Some("v9"), None, None]);
        assert!(c.as_materialized_series().equals_missing(&exp));

        // Only nulls: a column of nulls of the dtype.
        let nulls = str_dict("s", &dict, &[None, None]);
        let out = run(sch, vec![DataFrame::new(2, vec![nulls]).unwrap()]).unwrap();
        let c = out.column("s").unwrap();
        assert_eq!(c.dtype(), &DataType::String);
        assert_eq!(c.null_count(), 2);
    }

    #[test]
    fn null_scalar_part_is_codes_of_nulls() {
        let dict: Vec<String> = (0..10).map(|i| format!("v{i}")).collect();
        let a = str_dict("s", &dict, &[Some(1), Some(2)]);
        let sch = schema(&[&a]);
        let null = Column::full_null("s".into(), 3, &DataType::String);
        assert!(null.as_scalar_column().is_some());
        let dfs = vec![
            DataFrame::new(2, vec![a.clone()]).unwrap(),
            DataFrame::new(3, vec![null.clone()]).unwrap(),
            DataFrame::new(2, vec![a]).unwrap(),
        ];
        let out = run(sch.clone(), dfs).unwrap();
        let c = out.column("s").unwrap();
        assert!(c.as_dictionary_column().is_some(), "kept as codes");
        let exp = Series::new(
            "s".into(),
            [
                Some("v1"),
                Some("v2"),
                None,
                None,
                None,
                Some("v1"),
                Some("v2"),
            ],
        );
        assert!(c.as_materialized_series().equals_missing(&exp));

        // Only the missing column: still a column of nulls.
        let out = run(sch, vec![DataFrame::new(3, vec![null]).unwrap()]).unwrap();
        let c = out.column("s").unwrap();
        assert_eq!((c.dtype(), c.null_count()), (&DataType::String, 3));
    }

    #[test]
    fn dictionary_of_another_dtype_than_the_schema_is_an_error() {
        let dict: Vec<String> = (0..3).map(|i| format!("v{i}")).collect();
        let s = str_dict("a", &dict, &[Some(0), Some(1)]);
        let i = int_dict("a", vec![1, 2, 3], &[Some(0), Some(1)]);
        let sch = schema(&[&s]);
        // The first chunk already differs from the schema.
        let dfs = vec![
            DataFrame::new(2, vec![i.clone()]).unwrap(),
            DataFrame::new(2, vec![s.clone()]).unwrap(),
        ];
        assert!(run(sch.clone(), dfs).is_err());
        let dfs = vec![
            DataFrame::new(2, vec![s]).unwrap(),
            DataFrame::new(2, vec![i]).unwrap(),
        ];
        assert!(run(sch, dfs).is_err());
    }

    /// Forces the spill of the collected morsels, which hold shifted codes as `UInt32`.
    #[test]
    fn spilled_morsels_unspill_with_their_values() {
        const MORSELS: usize = 8;
        let dict: Vec<String> = (0..1000).map(|i| format!("value-{i}")).collect();
        let n = 1 << 16;
        let codes = |k: usize| -> Vec<Option<u32>> {
            (0..n)
                .map(|i| {
                    if i % 11 == 0 {
                        None
                    } else {
                        Some(((i + k) % 1000) as u32)
                    }
                })
                .collect()
        };
        let mut dfs = Vec::new();
        let mut exp_s: Vec<Option<String>> = Vec::new();
        let mut exp_i: Vec<Option<i64>> = Vec::new();
        for k in 0..MORSELS {
            let c = codes(k);
            // Each morsel has dictionaries of its own, so the shifts are not all zero.
            let s = str_dict("s", &dict, &c);
            let ints: Vec<i64> = (0..1000).map(|v| v * 3 + k as i64).collect();
            let i = int_dict("i", ints.clone(), &c);
            for x in &c {
                exp_s.push(x.map(|x| dict[x as usize].clone()));
                exp_i.push(x.map(|x| ints[x as usize]));
            }
            dfs.push(DataFrame::new(n, vec![s, i]).unwrap());
        }
        let sch = Arc::new(Schema::from_iter([
            Field::new("s".into(), DataType::String),
            Field::new("i".into(), DataType::Int64),
        ]));

        // Over budget until the end of the test (a drop guard), so registering a morsel spills.
        let _guard = polars_ooc::test_support::test_lock();
        let _over_budget = polars_ooc::test_support::OverBudget::new();
        let mut sink = InMemorySinkNode::merging_dictionaries(sch, None);
        let mergers = sink.mergers.take().unwrap();
        let mut morsels = Vec::new();
        ASYNC.block_in_place_on(async {
            for (i, df) in dfs.into_iter().enumerate() {
                let mut morsel =
                    Morsel::new_unregistered(df, MorselSeq::new(i as u64), SourceToken::new());
                sink.shift_dictionary_codes(&mut morsel, &mergers).await;
                sink.spill_ctx.register(morsel.sf()).await;
                morsels.push(morsel);
            }
        });
        let spilled = morsels
            .iter()
            .filter(|m| m.sf().try_get().is_none())
            .count();
        assert!(spilled > 0, "no morsel was spilled");
        drop(_over_budget);

        let dfs = morsels.into_iter().map(|m| m.into_sf().into_df_blocking());
        let out = sink.merge_output(dfs.collect(), &mergers).unwrap();
        let s = out.column("s").unwrap();
        let i = out.column("i").unwrap();
        assert_eq!(s.dtype(), &DataType::String);
        assert_eq!(i.dtype(), &DataType::Int64);
        assert_eq!(
            s.str()
                .unwrap()
                .iter()
                .map(|v| v.map(str::to_owned))
                .collect::<Vec<_>>(),
            exp_s
        );
        assert_eq!(i.i64().unwrap().iter().collect::<Vec<_>>(), exp_i);
    }
}
