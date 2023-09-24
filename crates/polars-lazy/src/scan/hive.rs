use polars_core::prelude::*;
use polars_io::utils::{BOOLEAN_RE, FLOAT_RE, INTEGER_RE};

struct HivePartitions {
    /// Single value Series that can be used to run the predicate against.
    /// They are to be broadcasted if the predicates don't filter them out.
    partitions: Vec<Series>,
}

impl HivePartitions {
    /// Parse a url and optionally return HivePartitions
    fn parse_url(url: &str) -> Option<Self> {
        let partitions = url
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
                    return None;
                };
                Some(s)
            })
            .collect::<Vec<_>>();
        if partitions.is_empty() {
            None
        } else {
            Some(HivePartitions { partitions })
        }
    }
}
