use std::sync::Arc;

use chrono_tz::Tz;
use polars_core::frame::DataFrame;
use polars_core::prelude::{Column, DataType, GroupsType, TimeUnit};
use polars_core::schema::Schema;
use polars_error::{PolarsError, PolarsResult, polars_bail, polars_ensure};
use polars_expr::state::ExecutionState;
use polars_time::prelude::{RollingWindower, ensure_duration_matches_dtype};
use polars_time::{ClosedWindow, Duration};
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

type NextWindows = (Vec<[IdxSize; 2]>, DataFrame, Column);

pub struct RollingGroupBy {
    buf_df: DataFrame,
    /// How many `buf_df` rows did we discard of already?
    buf_df_offset: IdxSize,
    /// Casted index column, which may need to keep around old values.
    buf_index_column: Column,
    /// Uncasted index column.
    buf_key_column: Column,

    seq: MorselSeq,

    index_column: PlSmallStr,
    windower: RollingWindower,
    aggs: Arc<[(PlSmallStr, StreamExpr)]>,
}
impl RollingGroupBy {
    pub fn new(
        schema: Arc<Schema>,
        index_column: PlSmallStr,
        period: Duration,
        offset: Duration,
        closed: ClosedWindow,
        aggs: Arc<[(PlSmallStr, StreamExpr)]>,
    ) -> PolarsResult<Self> {
        polars_ensure!(
            !period.is_zero() && !period.negative(),
            ComputeError: "rolling window period should be strictly positive",
        );

        let key_dtype = schema.get(&index_column).unwrap();
        ensure_duration_matches_dtype(period, key_dtype, "period")?;
        ensure_duration_matches_dtype(offset, key_dtype, "offset")?;

        use DataType as DT;
        let (tu, tz) = match key_dtype {
            DT::Datetime(tu, tz) => (*tu, tz.clone()),
            DT::Date => (TimeUnit::Microseconds, None),
            DT::UInt32 | DT::UInt64 | DT::Int64 | DT::Int32 => (TimeUnit::Nanoseconds, None),
            dt => polars_bail!(
                ComputeError:
                "expected any of the following dtypes: {{ Date, Datetime, Int32, Int64, UInt32, UInt64 }}, got {}",
                dt
            ),
        };

        let buf_df = DataFrame::empty_with_arc_schema(schema.clone());
        let buf_key_column = Column::new_empty(index_column.clone(), key_dtype);
        let buf_index_column =
            Column::new_empty(index_column.clone(), &DT::Datetime(tu, tz.clone()));

        // @NOTE: This is a bit strange since it ignores errors, but it mirrors the in-memory
        // engine.
        let tz = tz.and_then(|tz| tz.parse::<Tz>().ok());
        let windower = RollingWindower::new(period, offset, closed, tu, tz);

        Ok(Self {
            buf_df,
            buf_df_offset: 0,
            buf_index_column,
            buf_key_column,
            seq: MorselSeq::default(),
            index_column,
            windower,
            aggs,
        })
    }

    async fn evaluate_one(
        windows: Vec<[IdxSize; 2]>,
        key: Column,
        aggs: &[(PlSmallStr, StreamExpr)],
        state: &ExecutionState,
        mut df: DataFrame,
    ) -> PolarsResult<DataFrame> {
        assert_eq!(windows.len(), key.len());

        let groups = GroupsType::Slice {
            groups: windows,
            overlapping: true,
        }
        .into_sliceable();

        // @NOTE:
        // Rechunk so we can use specialized rolling kernels.
        //
        // This can be removed if / when the rolling kernels are chunking aware.
        df.rechunk_mut();

        let mut columns = Vec::with_capacity(1 + aggs.len());
        let height = key.len();
        columns.push(key);
        for (name, agg) in aggs.iter() {
            let mut agg = agg.evaluate_on_groups(&df, &groups, state).await?;
            let agg = agg.finalize();
            columns.push(agg.with_name(name.clone()));
        }

        Ok(unsafe { DataFrame::new_no_checks(height, columns) })
    }

    /// Progress the state and get the next available evaluation windows, data and key.
    fn next_windows(&mut self, finalize: bool) -> PolarsResult<Option<NextWindows>> {
        let buf_index_col_dt = self.buf_index_column.datetime()?;
        let mut time = Vec::new();
        time.extend(
            buf_index_col_dt
                .physical()
                .downcast_iter()
                .map(|arr| arr.values().as_slice()),
        );

        let mut windows = Vec::new();
        let num_retired = if finalize {
            self.windower.finalize(&time, &mut windows);
            self.buf_key_column.len() as IdxSize
        } else {
            self.windower.insert(&time, &mut windows)?
        };

        if num_retired == 0 && windows.is_empty() {
            return Ok(None);
        }

        let start_row_offset = self.buf_df_offset;

        self.buf_index_column = self.buf_index_column.slice(num_retired as i64, usize::MAX);
        let new_buf_df = self.buf_df.slice(num_retired as i64, usize::MAX);
        let data = std::mem::replace(&mut self.buf_df, new_buf_df);
        self.buf_df_offset += num_retired;

        if windows.is_empty() {
            return Ok(None);
        }

        // Prune the data that is not covered by the windows and update the windows accordingly.
        let offset = windows[0][0];
        let end = windows.last().unwrap();
        let end = end[0] + end[1];
        windows.iter_mut().for_each(|[s, _]| *s -= offset);
        let data = data.slice(
            (offset - start_row_offset) as i64,
            (end - start_row_offset) as usize,
        );

        let key;
        (key, self.buf_key_column) = self.buf_key_column.split_at(windows.len() as i64);

        Ok(Some((windows, data, key)))
    }
}

impl ComputeNode for RollingGroupBy {
    fn name(&self) -> &str {
        "rolling-group-by"
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        _state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        assert!(recv.len() == 1 && send.len() == 1);

        if send[0] == PortState::Done {
            recv[0] = PortState::Done;
            std::mem::take(&mut self.buf_df);
        } else if recv[0] == PortState::Done {
            if self.buf_df.is_empty() {
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
            assert!(!self.buf_df.is_empty());
            let mut send = send_ports[0].take().unwrap().serial();
            join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                if let Some((windows, df, key)) = self.next_windows(true)? {
                    let df = Self::evaluate_one(
                        windows,
                        key,
                        &self.aggs,
                        &state.in_memory_exec_state,
                        df,
                    )
                    .await?;

                    _ = send
                        .send(Morsel::new(df, self.seq.successor(), SourceToken::new()))
                        .await;
                }

                self.buf_df = self.buf_df.clear();
                self.buf_key_column = self.buf_key_column.clear();
                self.buf_index_column = self.buf_index_column.clear();

                Ok(())
            }));
            return;
        };

        let mut recv = recv.serial();
        let send = send_ports[0].take().unwrap().parallel();

        let (mut distributor, rxs) = distributor_channel::<(Morsel, Column, Vec<[IdxSize; 2]>)>(
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
            scope.spawn_task(TaskPriority::High, async move {
                while let Ok((mut morsel, key, windows)) = rx.recv().await {
                    morsel = morsel
                        .async_try_map::<PolarsError, _, _>(async |df| {
                            Self::evaluate_one(windows, key, &aggs, &state, df).await
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
            while let Ok(morsel) = recv.recv().await {
                let (df, seq, source_token, wait_token) = morsel.into_inner();
                self.seq = seq;
                drop(wait_token);

                if df.height() == 0 {
                    continue;
                }

                let morsel_index_column = df.column(&self.index_column)?;
                polars_ensure!(
                    morsel_index_column.null_count() == 0,
                    ComputeError: "null values in `rolling` not supported, fill nulls."
                );

                self.buf_key_column.append(morsel_index_column)?;

                use DataType as DT;
                let morsel_index_column = match morsel_index_column.dtype() {
                    DT::Datetime(_, _) => morsel_index_column.clone(),
                    DT::Date => {
                        morsel_index_column.cast(&DT::Datetime(TimeUnit::Microseconds, None))?
                    },
                    DT::UInt32 | DT::UInt64 | DT::Int32 => morsel_index_column
                        .cast(&DT::Int64)?
                        .cast(&DT::Datetime(TimeUnit::Nanoseconds, None))?,
                    DT::Int64 => {
                        morsel_index_column.cast(&DT::Datetime(TimeUnit::Nanoseconds, None))?
                    },
                    _ => unreachable!(),
                };
                self.buf_index_column.append(&morsel_index_column)?;
                self.buf_df.vstack_mut_owned(df)?;

                if let Some((windows, df, key)) = self.next_windows(false)? {
                    if distributor
                        .send((Morsel::new(df, seq, source_token), key, windows))
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
