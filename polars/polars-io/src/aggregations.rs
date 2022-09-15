use polars_core::prelude::*;

pub enum ScanAggregation {
    Sum {
        column: String,
        alias: Option<String>,
    },
    Min {
        column: String,
        alias: Option<String>,
    },
    Max {
        column: String,
        alias: Option<String>,
    },
    First {
        column: String,
        alias: Option<String>,
    },
    Last {
        column: String,
        alias: Option<String>,
    },
}

impl ScanAggregation {
    /// Evaluate the aggregations per batch.
    #[cfg(any(
        feature = "ipc",
        feature = "ipc_streaming",
        feature = "parquet",
        feature = "json",
        feature = "avro"
    ))]
    pub(crate) fn evaluate_batch(&self, df: &DataFrame) -> PolarsResult<Series> {
        use ScanAggregation::*;
        let s = match self {
            Sum { column, .. } => df.column(column)?.sum_as_series(),
            Min { column, .. } => df.column(column)?.min_as_series(),
            Max { column, .. } => df.column(column)?.max_as_series(),
            First { column, .. } => df.column(column)?.head(Some(1)),
            Last { column, .. } => df.column(column)?.tail(Some(1)),
        };
        Ok(s)
    }

    /// After all batches are concatenated the aggregation is determined for the whole set.
    pub(crate) fn finish(&self, df: &DataFrame) -> PolarsResult<Series> {
        use ScanAggregation::*;
        match self {
            Sum { column, alias } => {
                let mut s = df.column(column)?.sum_as_series();
                if let Some(alias) = alias {
                    s.rename(alias);
                }
                Ok(s)
            }
            Min { column, alias } => {
                let mut s = df.column(column)?.min_as_series();
                if let Some(alias) = alias {
                    s.rename(alias);
                }
                Ok(s)
            }
            Max { column, alias } => {
                let mut s = df.column(column)?.max_as_series();
                if let Some(alias) = alias {
                    s.rename(alias);
                }
                Ok(s)
            }
            First { column, alias } => {
                let mut s = df.column(column)?.head(Some(1));
                if let Some(alias) = alias {
                    s.rename(alias);
                }
                Ok(s)
            }
            Last { column, alias } => {
                let mut s = df.column(column)?.tail(Some(1));
                if let Some(alias) = alias {
                    s.rename(alias);
                }
                Ok(s)
            }
        }
    }
}

#[cfg(any(
    feature = "ipc",
    feature = "ipc_streaming",
    feature = "parquet",
    feature = "json",
    feature = "avro"
))]
pub(crate) fn apply_aggregations(
    df: &mut DataFrame,
    aggregate: Option<&[ScanAggregation]>,
) -> PolarsResult<()> {
    if let Some(aggregate) = aggregate {
        let cols = aggregate
            .iter()
            .map(|scan_agg| scan_agg.evaluate_batch(df))
            .collect::<PolarsResult<_>>()?;
        if cfg!(debug_assertions) {
            *df = DataFrame::new(cols).unwrap();
        } else {
            *df = DataFrame::new_no_checks(cols)
        }
    }
    Ok(())
}
