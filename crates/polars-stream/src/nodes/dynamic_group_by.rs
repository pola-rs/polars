use std::sync::Arc;

use arrow::legacy::time_zone::Tz;
use polars_core::frame::DataFrame;
use polars_core::prelude::{Column, DataType, GroupsType, Int64Chunked, IntoColumn, TimeUnit};
use polars_core::schema::Schema;
use polars_core::series::IsSorted;
use polars_error::{PolarsError, PolarsResult, polars_bail, polars_ensure};
use polars_expr::state::ExecutionState;
use polars_time::prelude::{GroupByDynamicWindower, Label, ensure_duration_matches_dtype};
use polars_time::{DynamicGroupOptions, LB_NAME, UB_NAME};
use polars_utils::IdxSize;
use polars_utils::pl_str::PlSmallStr;

use super::ComputeNode;
use crate::DEFAULT_DISTRIBUTOR_BUFFER_SIZE;
use crate::async_executor::{JoinHandle, TaskPriority, TaskScope};
use crate::async_primitives::distributor_channel::distributor_channel;
use crate::async_primitives::wait_group::WaitGroup;
use crate::execute::StreamingExecutionState;
use crate::expression::StreamExpr;
use crate::graph::PortState;
use crate::morsel::{Morsel, MorselSeq, SourceToken};
use crate::pipe::{RecvPort, SendPort};

type NextWindows = (Vec<[IdxSize; 2]>, Vec<i64>, Vec<i64>, DataFrame);

pub struct DynamicGroupBy {
    buf_df: DataFrame,
    /// How many `buf_df` rows did we discard of already?
    buf_df_offset: IdxSize,
    buf_index_column: Column,

    seq: MorselSeq,

    slice_offset: IdxSize,
    slice_length: IdxSize,

    group_by: Option<PlSmallStr>,
    index_column: PlSmallStr,
    index_column_idx: usize,
    label: Label,
    include_boundaries: bool,
    windower: GroupByDynamicWindower,
    aggs: Arc<[(PlSmallStr, StreamExpr)]>,
}
impl DynamicGroupBy {
    pub fn new(
        schema: Arc<Schema>,
        options: DynamicGroupOptions,
        aggs: Arc<[(PlSmallStr, StreamExpr)]>,
        slice: Option<(IdxSize, IdxSize)>,
    ) -> PolarsResult<Self> {
        let DynamicGroupOptions {
            index_column,
            every,
            period,
            offset,
            label,
            include_boundaries,
            closed_window,
            start_by,
        } = options;

        polars_ensure!(!every.negative(), ComputeError: "'every' argument must be positive");

        let (index_column_idx, _, index_dtype) = schema.get_full(&index_column).unwrap();
        ensure_duration_matches_dtype(every, index_dtype, "every")?;
        ensure_duration_matches_dtype(period, index_dtype, "period")?;
        ensure_duration_matches_dtype(offset, index_dtype, "offset")?;

        use DataType as DT;
        let (tu, tz) = match index_dtype {
            DT::Datetime(tu, tz) => (*tu, tz.clone()),
            DT::Date => (TimeUnit::Microseconds, None),
            DT::Int64 | DT::Int32 => (TimeUnit::Nanoseconds, None),
            dt => polars_bail!(
                ComputeError:
                "expected any of the following dtypes: {{ Date, Datetime, Int32, Int64 }}, got {}",
                dt
            ),
        };

        let buf_df = DataFrame::empty_with_arc_schema(schema.clone());
        let buf_index_column =
            Column::new_empty(index_column.clone(), &DT::Datetime(tu, tz.clone()));

        // @NOTE: This is a bit strange since it ignores errors, but it mirrors the in-memory
        // engine.
        let tz = tz.and_then(|tz| tz.parse::<Tz>().ok());
        let windower = GroupByDynamicWindower::new(
            period,
            offset,
            every,
            start_by,
            closed_window,
            tu,
            tz,
            include_boundaries || matches!(label, Label::Left),
            include_boundaries || matches!(label, Label::Right),
        );

        let (slice_offset, slice_length) = slice.unwrap_or((0, IdxSize::MAX));

        Ok(Self {
            buf_df,

            buf_df_offset: 0,
            buf_index_column,
            seq: MorselSeq::default(),

            slice_offset,
            slice_length,

            group_by: None,
            index_column,
            index_column_idx,
            label,
            include_boundaries,
            windower,
            aggs,
        })
    }

    #[expect(clippy::too_many_arguments)]
    async fn evaluate_one(
        windows: Vec<[IdxSize; 2]>,
        lower_bound: Vec<i64>,
        upper_bound: Vec<i64>,
        aggs: &[(PlSmallStr, StreamExpr)],
        state: &ExecutionState,
        mut df: DataFrame,

        group_by: Option<&str>,
        index_column_name: &str,
        index_column_idx: usize,
        label: Label,
        include_boundaries: bool,
    ) -> PolarsResult<DataFrame> {
        let height = windows.len();
        let groups = GroupsType::new_slice(windows, true, true).into_sliceable();

        // @NOTE:
        // Rechunk so we can use specialized rolling/dynamic kernels.
        df.rechunk_mut();

        let mut columns =
            Vec::with_capacity(if include_boundaries { 2 } else { 0 } + 1 + aggs.len());

        // Construct `lower_bound`, `upper_bound` and `key` columns that might be included in the
        // output dataframe.
        {
            let mut lower = Int64Chunked::new_vec(PlSmallStr::from_static(LB_NAME), lower_bound);
            let mut upper = Int64Chunked::new_vec(PlSmallStr::from_static(UB_NAME), upper_bound);
            if group_by.is_none() {
                lower.set_sorted_flag(IsSorted::Ascending);
                upper.set_sorted_flag(IsSorted::Ascending);
            }
            let mut lower = lower.into_column();
            let mut upper = upper.into_column();

            let index_column = &df.columns()[index_column_idx];
            let index_dtype = index_column.dtype();
            let mut bound_dtype_physical = index_dtype.to_physical();
            let mut bound_dtype = index_dtype;
            if index_dtype.is_date() {
                bound_dtype = &DataType::Datetime(TimeUnit::Microseconds, None);
                bound_dtype_physical = DataType::Int64;
            }
            lower = lower.cast(&bound_dtype_physical).unwrap();
            upper = upper.cast(&bound_dtype_physical).unwrap();
            (lower, upper) = unsafe {
                (
                    lower.from_physical_unchecked(bound_dtype)?,
                    upper.from_physical_unchecked(bound_dtype)?,
                )
            };

            let key = match label {
                Label::DataPoint => unsafe { index_column.agg_first(&groups) },
                Label::Left => lower
                    .cast(index_dtype)
                    .unwrap()
                    .with_name(index_column_name.into()),
                Label::Right => upper
                    .cast(index_dtype)
                    .unwrap()
                    .with_name(index_column_name.into()),
            };

            if include_boundaries {
                columns.extend([lower, upper]);
            }
            columns.push(key);
        }

        for (name, agg) in aggs.iter() {
            let mut agg = agg.evaluate_on_groups(&df, &groups, state).await?;
            let agg = agg.finalize();
            columns.push(agg.with_name(name.clone()));
        }

        Ok(unsafe { DataFrame::new_unchecked(height, columns) })
    }

    /// Progress the state and get the next available evaluation windows, data and key.
    fn next_windows(&mut self, finalize: bool) -> PolarsResult<Option<NextWindows>> {
        let mut windows = Vec::new();
        let mut lower_bound = Vec::new();
        let mut upper_bound = Vec::new();

        let num_retired = if finalize {
            self.windower
                .finalize(&mut windows, &mut lower_bound, &mut upper_bound);
            self.buf_df.height() as IdxSize
        } else {
            let mut offset = self.windower.num_seen() - self.buf_df_offset;
            let ca = self.buf_index_column.datetime()?;
            for arr in ca.physical().downcast_iter() {
                let arr_len = arr.len() as IdxSize;
                if offset >= arr_len {
                    offset -= arr_len;
                    continue;
                }

                self.windower.insert(
                    &arr.values().as_slice()[offset as usize..],
                    &mut windows,
                    &mut lower_bound,
                    &mut upper_bound,
                )?;
                offset = offset.saturating_sub(arr_len);
            }
            self.windower.lowest_needed_index() - self.buf_df_offset
        };

        if windows.is_empty() {
            if num_retired > 0 {
                self.buf_df = self.buf_df.slice(num_retired as i64, usize::MAX);
                self.buf_index_column = self.buf_index_column.slice(num_retired as i64, usize::MAX);
                self.buf_df_offset += num_retired;
            }

            return Ok(None);
        }

        // Prune the data that is not covered by the windows and update the windows accordingly.
        let offset = windows[0][0];
        let end = windows.last().unwrap();
        let end = end[0] + end[1];

        if self.slice_offset as usize > windows.len() {
            self.slice_offset -= windows.len() as IdxSize;
            windows.clear();
            lower_bound.clear();
            upper_bound.clear();
        } else if self.slice_offset > 0 {
            let offset = self.slice_offset as usize;
            self.slice_offset = self.slice_offset.saturating_sub(windows.len() as IdxSize);
            windows.drain(..offset);
            lower_bound.drain(..offset.min(lower_bound.len()));
            upper_bound.drain(..offset.min(upper_bound.len()));
        }

        let trunc_length = windows.len().min(self.slice_length as usize);
        windows.truncate(trunc_length);
        lower_bound.truncate(trunc_length);
        upper_bound.truncate(trunc_length);
        self.slice_length -= windows.len() as IdxSize;

        windows.iter_mut().for_each(|[s, _]| *s -= offset);
        let data = self.buf_df.slice(
            (offset - self.buf_df_offset) as i64,
            (end - self.buf_df_offset) as usize,
        );

        self.buf_df = self.buf_df.slice(num_retired as i64, usize::MAX);
        self.buf_index_column = self.buf_index_column.slice(num_retired as i64, usize::MAX);
        self.buf_df_offset += num_retired;

        if windows.is_empty() {
            return Ok(None);
        }

        Ok(Some((windows, lower_bound, upper_bound, data)))
    }
}

impl ComputeNode for DynamicGroupBy {
    fn name(&self) -> &str {
        "dynamic-group-by"
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        _state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        assert!(recv.len() == 1 && send.len() == 1);

        if self.slice_length == 0 {
            recv[0] = PortState::Done;
            send[0] = PortState::Done;
            std::mem::take(&mut self.buf_df);
            return Ok(());
        }

        if send[0] == PortState::Done {
            recv[0] = PortState::Done;
            std::mem::take(&mut self.buf_df);
        } else if recv[0] == PortState::Done {
            if self.buf_df.height() == 0 {
                send[0] = PortState::Done;
            } else {
                send[0] = PortState::Ready;
            }
        } else {
            recv.swap_with_slice(send);
        }

        Ok(())
    }

    fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        recv_ports: &mut [Option<RecvPort<'_>>],
        send_ports: &mut [Option<SendPort<'_>>],
        state: &'s StreamingExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert!(recv_ports.len() == 1 && send_ports.len() == 1);

        let Some(recv) = recv_ports[0].take() else {
            // We no longer have to receive data. Finalize and send all remaining data.
            assert!(self.buf_df.height() > 0);
            assert!(self.slice_length > 0);
            let mut send = send_ports[0].take().unwrap().serial();
            join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                if let Some((windows, lower_bound, upper_bound, df)) = self.next_windows(true)? {
                    let df = Self::evaluate_one(
                        windows,
                        lower_bound,
                        upper_bound,
                        &self.aggs,
                        &state.in_memory_exec_state,
                        df,
                        self.group_by.as_deref(),
                        self.index_column.as_str(),
                        self.index_column_idx,
                        self.label,
                        self.include_boundaries,
                    )
                    .await?;

                    _ = send
                        .send(Morsel::new(df, self.seq.successor(), SourceToken::new()))
                        .await;
                }

                self.buf_df = self.buf_df.clear();
                Ok(())
            }));
            return;
        };

        let mut recv = recv.serial();
        let send = send_ports[0].take().unwrap().parallel();

        let (mut distributor, rxs) =
            distributor_channel::<(Morsel, Vec<[IdxSize; 2]>, Vec<i64>, Vec<i64>)>(
                send.len(),
                *DEFAULT_DISTRIBUTOR_BUFFER_SIZE,
            );

        // Worker tasks.
        //
        // These evaluate the aggregations.
        join_handles.extend(rxs.into_iter().zip(send).map(|(mut rx, mut tx)| {
            let wg = WaitGroup::default();
            let aggs = self.aggs.clone();
            let state = state.in_memory_exec_state.split();

            let group_by = self.group_by.clone();
            let index_column = self.index_column.clone();
            let index_column_idx = self.index_column_idx;
            let label = self.label;
            let include_boundaries = self.include_boundaries;

            scope.spawn_task(TaskPriority::High, async move {
                while let Ok((mut morsel, windows, lower_bound, upper_bound)) = rx.recv().await {
                    morsel = morsel
                        .async_try_map::<PolarsError, _, _>(async |df| {
                            Self::evaluate_one(
                                windows,
                                lower_bound,
                                upper_bound,
                                &aggs,
                                &state,
                                df,
                                group_by.as_deref(),
                                index_column.as_str(),
                                index_column_idx,
                                label,
                                include_boundaries,
                            )
                            .await
                        })
                        .await?;
                    morsel.set_consume_token(wg.token());

                    if tx.send(morsel).await.is_err() {
                        break;
                    }
                    wg.wait().await;
                }

                Ok(())
            })
        }));

        // Distributor task.
        //
        // This finds boundaries to distribute to worker threads over.
        join_handles.push(scope.spawn_task(TaskPriority::High, async move {
            while let Ok(morsel) = recv.recv().await
                && self.slice_length > 0
            {
                let (df, seq, source_token, wait_token) = morsel.into_inner();
                self.seq = seq;
                drop(wait_token);

                if df.height() == 0 {
                    continue;
                }

                let morsel_index_column = df.column(&self.index_column)?;
                polars_ensure!(
                    morsel_index_column.null_count() == 0,
                    ComputeError: "null values in `group_by_dynamic` not supported, fill nulls."
                );

                use DataType as DT;
                let morsel_index_column = match morsel_index_column.dtype() {
                    DT::Datetime(_, _) => morsel_index_column.clone(),
                    DT::Date => {
                        morsel_index_column.cast(&DT::Datetime(TimeUnit::Microseconds, None))?
                    },
                    DT::Int32 => morsel_index_column
                        .cast(&DT::Int64)?
                        .cast(&DT::Datetime(TimeUnit::Nanoseconds, None))?,
                    DT::Int64 => {
                        morsel_index_column.cast(&DT::Datetime(TimeUnit::Nanoseconds, None))?
                    },
                    _ => unreachable!(),
                };

                self.buf_df.vstack_mut_owned(df)?;
                self.buf_index_column.append_owned(morsel_index_column)?;

                if let Some((windows, lower_bound, upper_bound, df)) = self.next_windows(false)? {
                    if distributor
                        .send((
                            Morsel::new(df, seq, source_token),
                            windows,
                            lower_bound,
                            upper_bound,
                        ))
                        .await
                        .is_err()
                    {
                        break;
                    }
                }
            }

            Ok(())
        }));
    }
}
