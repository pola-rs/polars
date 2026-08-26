use polars_core::prelude::*;
use polars_plan::dsl::StrptimeOptions;
use polars_time::chunkedarray::StringMethods;
use polars_time::chunkedarray::string::Pattern;
use polars_time::chunkedarray::string::infer::{
    DatetimeInfer, TryFromWithUnit, coerce_string_to_date, coerce_string_to_datetime,
    infer_from_values, infer_pattern_date_single, infer_pattern_datetime_single, sniff_time_fmt,
};

use super::compute_node_prelude::*;

pub struct StrptimeInferNode {
    dtype: DataType,
    options: StrptimeOptions,
    infer: Option<FormatInfer>,
    phase: Phase,

    /// Name to report parse failures against; the input carries an internal
    /// one assigned during lowering.
    input_name: PlSmallStr,

    /// Ambiguous can be `raise`, `earliest`, `latest` and `null`.
    ///
    /// If it broadcast and it is `raise` or `null`, we can actually execute it here. So
    /// - `false` -> "null"
    /// - `true`  -> "raise"
    ambiguous_is_raise: bool,
}

#[derive(Clone, Copy, PartialEq)]
enum Phase {
    /// Serially receiving, infer the format and serially send morsel.
    Inferring,
    /// Format known and operation is row separable. Converting morsels in parallel.
    Parsing,
}

#[derive(Clone)]
enum FormatInfer {
    #[cfg(feature = "dtype-date")]
    Date(DatetimeInfer<Int32Type>),
    #[cfg(feature = "dtype-datetime")]
    Datetime(DatetimeInfer<Int64Type>, Option<TimeZone>),
    #[cfg(feature = "dtype-time")]
    /// (fmt, use_cache)
    Time(&'static str, bool),
}

impl StrptimeInferNode {
    pub fn new(
        dtype: DataType,
        options: StrptimeOptions,
        input_name: PlSmallStr,
        ambiguous_is_raise: bool,
    ) -> Self {
        Self {
            dtype,
            options,
            infer: None,
            phase: Phase::Inferring,
            input_name,
            ambiguous_is_raise,
        }
    }
}

impl FormatInfer {
    /// Scans `ca` rather than only its first value; which values are
    /// unparseable must not depend on where they sit in the column.
    fn try_new(
        ca: &StringChunked,
        dtype: &DataType,
        options: &StrptimeOptions,
    ) -> PolarsResult<Option<Self>> {
        match dtype {
            #[cfg(feature = "dtype-date")]
            DataType::Date => {
                let Some(pattern) = infer_from_values(ca, infer_pattern_date_single) else {
                    return Ok(None);
                };
                let infer = DatetimeInfer::<Int32Type>::try_from_with_unit(pattern, None)?;
                Ok(Some(FormatInfer::Date(infer)))
            },
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime(tu, tz) => {
                let Some(pattern) = infer_from_values(ca, infer_pattern_datetime_single) else {
                    return Ok(None);
                };
                if matches!(pattern, Pattern::DatetimeYMDZ) && tz.is_none() {
                    polars_bail!(to_datetime_tz_mismatch);
                }
                let infer = DatetimeInfer::<Int64Type>::try_from_with_unit(pattern, Some(*tu))?;
                Ok(Some(FormatInfer::Datetime(infer, tz.clone())))
            },
            #[cfg(feature = "dtype-time")]
            DataType::Time => {
                Ok(infer_from_values(ca, sniff_time_fmt)
                    .map(|f| FormatInfer::Time(f, options.cache)))
            },
            _ => Ok(None),
        }
    }

    fn apply(
        &mut self,
        col: &Column,
        ambiguous: &StringChunked,
        strict: bool,
        input_name: &PlSmallStr,
    ) -> PolarsResult<Column> {
        let ca = col.str()?;
        let name = col.name().clone();

        let result: Column = match self {
            #[cfg(feature = "dtype-date")]
            FormatInfer::Date(infer) => coerce_string_to_date(infer, ca)?
                .into_series()
                .with_name(name)
                .into_column(),

            #[cfg(feature = "dtype-datetime")]
            FormatInfer::Datetime(infer, tz) => {
                coerce_string_to_datetime(infer, ca, tz.as_ref(), ambiguous)?
                    .into_series()
                    .with_name(name)
                    .into_column()
            },

            #[cfg(feature = "dtype-time")]
            FormatInfer::Time(fmt, use_cache) => ca
                .as_time(Some(fmt), *use_cache)?
                .into_series()
                .with_name(name)
                .into_column(),
        };

        if strict && col.null_count() != result.null_count() {
            // reported against the name the user wrote, not the internal alias
            let reported = result.clone().with_name(input_name.clone());
            polars_core::utils::handle_casting_failures(
                col.as_materialized_series(),
                reported.as_materialized_series(),
            )?;
        }

        Ok(result)
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

        if self.infer.is_some() {
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

        let ambiguous = if self.ambiguous_is_raise {
            "raise"
        } else {
            "null"
        };
        let ambiguous = StringChunked::new(PlSmallStr::EMPTY, vec![ambiguous]);

        match self.phase {
            Phase::Inferring => {
                let mut recv = recv_ports[0].take().unwrap().serial();
                let mut send = send_ports[0].take().unwrap().serial();

                let dtype = &self.dtype;
                let options = &self.options;
                let input_name = &self.input_name;
                let infer_slot = &mut self.infer;
                join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                    while let Ok(morsel) = recv.recv().await {
                        if infer_slot.is_none() {
                            let df = morsel.df().await;
                            let ca = df.columns()[0].str()?;
                            if ca.first_non_null().is_some() {
                                *infer_slot = FormatInfer::try_new(ca, dtype, options)?;

                                let unit = if matches!(dtype, DataType::Time) {
                                    "time"
                                } else {
                                    "date"
                                };
                                polars_ensure!(
                                    infer_slot.is_some() || !options.strict,
                                    parse_fmt_idk = unit
                                );
                            }
                        }

                        // Request a stop, so switch to parsing in parallel.
                        if infer_slot.is_some() {
                            morsel.source_token().stop();
                        }

                        let morsel = morsel
                            .try_map(|df| {
                                let cols = df.columns();
                                if let Some(ref mut infer) = *infer_slot {
                                    infer
                                        .apply(&cols[0], &ambiguous, options.strict, input_name)
                                        .map(Column::into_frame)
                                } else {
                                    Ok(Column::full_null(
                                        cols[0].name().clone(),
                                        cols[0].len(),
                                        dtype,
                                    )
                                    .into_frame())
                                }
                            })
                            .await?;
                        if send.send(morsel).await.is_err() {
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
                    let strict = self.options.strict;
                    let input_name = self.input_name.clone();
                    let ambiguous = ambiguous.clone();
                    let mut infer = self.infer.clone().unwrap();
                    join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                        while let Ok(morsel) = recv.recv().await {
                            let morsel = morsel
                                .try_map(|df| {
                                    let cols = df.columns();
                                    infer
                                        .apply(&cols[0], &ambiguous, strict, &input_name)
                                        .map(Column::into_frame)
                                })
                                .await?;
                            if send.send(morsel).await.is_err() {
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
