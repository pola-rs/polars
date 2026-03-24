use polars_core::prelude::*;
use polars_plan::dsl::StrptimeOptions;
use polars_time::chunkedarray::StringMethods;
use polars_utils::pl_str::PlSmallStr;

use super::compute_node_prelude::*;
use crate::pipe::{PortReceiver, PortSender};

fn apply_strptime(
    col: &Column,
    dtype: &DataType,
    options: &StrptimeOptions,
) -> PolarsResult<Column> {
    debug_assert!(options.format.is_some());

    let str_col = col.str()?;

    let result: Column = match dtype {
        #[cfg(feature = "dtype-date")]
        DataType::Date => str_col
            .as_date(options.format.as_deref(), options.cache)?
            .into_column(),

        #[cfg(feature = "dtype-datetime")]
        DataType::Datetime(time_unit, time_zone) => {
            #[cfg(all(feature = "regex", feature = "timezones"))]
            let tz_aware = options
                .format
                .as_deref()
                .map(|f| polars_plan::plans::TZ_AWARE_RE.is_match(f))
                .unwrap_or(false);
            #[cfg(not(all(feature = "regex", feature = "timezones")))]
            let tz_aware = false;

            let ambiguous = StringChunked::full(PlSmallStr::EMPTY, "raise", str_col.len());
            str_col
                .as_datetime(
                    options.format.as_deref(),
                    *time_unit,
                    options.cache,
                    tz_aware,
                    time_zone.as_ref(),
                    &ambiguous,
                )?
                .into_column()
        },

        #[cfg(feature = "dtype-time")]
        DataType::Time => str_col
            .as_time(options.format.as_deref(), options.cache)?
            .into_column(),

        dt => polars_bail!(ComputeError: "strptime_infer: unsupported dtype {}", dt),
    };

    Ok(result.with_name(col.name().clone()))
}

#[derive(Clone, Copy, PartialEq)]
enum Phase {
    /// Receiving serially, scanning for the first non-null value to infer format.
    Inferring,
    /// Format known; passing morsels through in parallel.
    Parsing,
}

pub struct StrptimeInferNode {
    dtype: DataType,
    options: StrptimeOptions,
    phase: Phase,
}

impl StrptimeInferNode {
    pub fn new(dtype: DataType, options: StrptimeOptions) -> Self {
        Self {
            dtype,
            options,
            phase: Phase::Inferring,
        }
    }
}

impl ComputeNode for StrptimeInferNode {
    fn name(&self) -> &str {
        "strptime-infer"
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        _state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        assert!(recv.len() == 1 && send.len() == 1);

        // Transition once we found a format.
        if self.options.format.is_some() {
            self.phase = Phase::Parsing;
        }

        recv.swap_with_slice(send);
        Ok(())
    }

    fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        recv_ports: &mut [Option<RecvPort<'_>>],
        send_ports: &mut [Option<SendPort<'_>>],
        _state: &'s StreamingExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert!(recv_ports.len() == 1 && send_ports.len() == 1);

        match self.phase {
            Phase::Inferring => {
                let mut recv = recv_ports[0].take().unwrap().serial();
                let mut send = send_ports[0].take().unwrap().serial();
                let dtype = &self.dtype;
                let options = &mut self.options;
                join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                    while let Ok(morsel) = recv.recv().await {
                        if options.format.is_none() {
                            let col = &morsel.df().columns()[0];
                            let ca = col.str()?;
                            if ca.null_count() < ca.len() {
                                let found = ca.into_iter().flatten().find_map(|val| {
                                    polars_time::chunkedarray::string::infer::infer_format_for_dtype(val, dtype)
                                });
                                if let Some(fmt) = found {
                                    // Raise the same error as the in-memory path when
                                    // a tz-aware format is inferred but the dtype has no tz.
                                    #[cfg(feature = "dtype-datetime")]
                                    if matches!(dtype, DataType::Datetime(_, None))
                                        && (fmt.contains("%z") || fmt.contains("%Z") || fmt.contains("%#z"))
                                    {
                                        polars_bail!(to_datetime_tz_mismatch);
                                    }
                                    options.format = Some(PlSmallStr::from_static(fmt));
                                }
                            }
                        }

                        let out_morsel = if options.format.is_some() {
                            morsel.source_token().stop();
                            morsel.try_map(|df| apply_strptime(&df.columns()[0], dtype, options).map(Column::into_frame))?
                        } else {
                            morsel.try_map(|df| df.columns()[0].cast(dtype).map(Column::into_frame))?
                        };

                        if send.send(out_morsel).await.is_err() {
                            break;
                        }
                    }
                    Ok(())
                }));
            },

            Phase::Parsing => {
                let receivers = recv_ports[0].take().unwrap().parallel();
                let senders = send_ports[0].take().unwrap().parallel();
                for (mut recv, mut send) in receivers.into_iter().zip(senders) {
                    let dtype = &self.dtype;
                    let options = &self.options;
                    join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                        while let Ok(morsel) = recv.recv().await {
                            let out = morsel.try_map(|df| apply_strptime(&df.columns()[0], dtype, options).map(Column::into_frame))?;
                            if send.send(out).await.is_err() {
                                break;
                            }
                        }
                        Ok(())
                    }));
                }
            },
        }
    }
}
