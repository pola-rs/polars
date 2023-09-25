use std::path::Path;

use polars_core::prelude::*;
use polars_io::predicates::{BatchStats, ColumnStats};
use polars_io::utils::{BOOLEAN_RE, FLOAT_RE, INTEGER_RE};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
pub struct HivePartitions {
    /// Single value Series that can be used to run the predicate against.
    /// They are to be broadcasted if the predicates don't filter them out.
    stats: BatchStats,
}

impl HivePartitions {
    pub fn get_statistics(&self) -> &BatchStats {
        &self.stats
    }

    /// Parse a url and optionally return HivePartitions
    pub(crate) fn parse_url(url: &Path) -> Option<Self> {
        let partitions = url
            .as_os_str()
            .to_str()?
            .split('/')
            .filter_map(|part| {
                let mut it = part.split('=');
                let name = it.next()?;
                let value = it.next()?;

                // Having multiple '=' doesn't seem like valid hive partition,
                // continue as url.
                if it.next().is_some() {
                    return None;
                }

                let s = if INTEGER_RE.is_match(value) {
                    let value = value.parse::<i64>().ok()?;
                    Series::new(name, &[value])
                } else if BOOLEAN_RE.is_match(value) {
                    let value = value.parse::<bool>().ok()?;
                    Series::new(name, &[value])
                } else if FLOAT_RE.is_match(value) {
                    let value = value.parse::<f64>().ok()?;
                    Series::new(name, &[value])
                } else {
                    Series::new(name, &[value])
                };
                Some(s)
            })
            .collect::<Vec<_>>();

        if partitions.is_empty() {
            None
        } else {
            let schema: Schema = partitions.as_slice().into();
            let stats = BatchStats::new(
                schema,
                partitions
                    .into_iter()
                    .map(ColumnStats::from_column_literal)
                    .collect(),
            );

            Some(HivePartitions { stats })
        }
    }

    pub fn materialize_partition_columns(&self) -> Vec<Series> {
        self.get_statistics()
            .column_stats()
            .iter()
            .map(|cs| cs.to_min().unwrap().clone())
            .collect()
    }
}
