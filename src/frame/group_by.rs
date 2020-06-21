use super::hash_join::prepare_hashed_relation;
use crate::prelude::*;
use crate::series::chunked_array::builder::PrimitiveChunkedBuilder;
use fnv::FnvHashMap;
use std::hash::Hash;

fn group_by<T>(a: impl Iterator<Item = Option<T>>) -> Vec<(usize, Vec<usize>)>
where
    T: Hash + Eq + Copy,
{
    let hash_tbl = prepare_hashed_relation(a);

    hash_tbl
        .into_iter()
        .map(|(_, indexes)| {
            let first = unsafe { *indexes.get_unchecked(0) };
            (first, indexes)
        })
        .collect()
}

impl DataFrame {
    pub fn group_by(&self, by: &str) -> Option<GroupBy> {
        let groups = if let Some(s) = self.select(by) {
            match s {
                Series::UInt32(ca) => group_by(ca.iter()),
                Series::Int32(ca) => group_by(ca.iter()),
                Series::Int64(ca) => group_by(ca.iter()),
                Series::Bool(ca) => group_by(ca.iter()),
                Series::Utf8(ca) => group_by(ca.iter().map(|v| Some(v))),
                Series::Date32(ca) => group_by(ca.iter().map(|v| Some(v))),
                Series::Date64(ca) => group_by(ca.iter().map(|v| Some(v))),
                Series::Time64Ns(ca) => group_by(ca.iter().map(|v| Some(v))),
                Series::DurationNs(ca) => group_by(ca.iter().map(|v| Some(v))),
                _ => unimplemented!(),
            }
        } else {
            return None;
        };

        Some(GroupBy {
            df: self,
            by: by.to_string(),
            groups,
            selection: None,
        })
    }
}

#[derive(Debug, Clone)]
pub struct GroupBy<'a> {
    df: &'a DataFrame,
    by: String,
    // [first idx, [other idx]]
    groups: Vec<(usize, Vec<usize>)>,
    selection: Option<String>,
}

macro_rules! build_ca_agg {
    ($self:ident, $new_name:ident, $agg_col:ident, $variant:ident, $agg_fn:ident) => {{
        let mut builder = PrimitiveChunkedBuilder::new(&$new_name, $self.groups.len());
        for (_first, idx) in &$self.groups {
            let s = $agg_col.take(idx, None)?;
            builder.append_option(s.$agg_fn());
        }
        let ca = builder.finish();
        Series::$variant(ca)
    }};
}

macro_rules! build_ca_agg_variants {
    ($self:ident, $new_name:ident, $agg_col:ident, $agg_fn:ident) => {
        match $agg_col.dtype() {
            ArrowDataType::UInt32 => build_ca_agg!($self, $new_name, $agg_col, UInt32, $agg_fn),
            ArrowDataType::Int32 => build_ca_agg!($self, $new_name, $agg_col, Int32, $agg_fn),
            ArrowDataType::Int64 => build_ca_agg!($self, $new_name, $agg_col, Int64, $agg_fn),
            ArrowDataType::Float32 => build_ca_agg!($self, $new_name, $agg_col, Float32, $agg_fn),
            ArrowDataType::Float64 => build_ca_agg!($self, $new_name, $agg_col, Float64, $agg_fn),
            ArrowDataType::Date32(_) => build_ca_agg!($self, $new_name, $agg_col, Date32, $agg_fn),
            ArrowDataType::Date64(_) => build_ca_agg!($self, $new_name, $agg_col, Date64, $agg_fn),
            ArrowDataType::Time64(TimeUnit::Nanosecond) => {
                build_ca_agg!($self, $new_name, $agg_col, Time64Ns, $agg_fn)
            }
            ArrowDataType::Duration(TimeUnit::Nanosecond) => {
                build_ca_agg!($self, $new_name, $agg_col, DurationNs, $agg_fn)
            }
            _ => return Err(PolarsError::DataTypeMisMatch),
        };
    };
}

impl<'a> GroupBy<'a> {
    pub fn select(mut self, name: &str) -> Self {
        self.selection = Some(name.to_string());
        self
    }

    fn keys(&self) -> Result<Series> {
        self.df.f_select(&self.by).take_iter(
            self.groups.iter().map(|(idx, _)| Some(*idx)),
            None,
            Some(self.groups.len()),
        )
    }

    fn prepare_agg(&self) -> Result<(&String, Series, &Series)> {
        let name = match &self.selection {
            Some(name) => name,
            None => return Err(PolarsError::NoSelection),
        };

        let keys = self.keys()?;
        let agg_col = self.df.select(name).ok_or(PolarsError::NotFound)?;
        Ok((name, keys, agg_col))
    }

    pub fn mean(&self) -> Result<DataFrame> {
        let (name, keys, agg_col) = self.prepare_agg()?;
        let new_name = format!["{}_mean", name];
        let agg = build_ca_agg_variants!(self, new_name, agg_col, mean);
        DataFrame::new_from_columns(vec![keys, agg])
    }

    pub fn sum(&self) -> Result<DataFrame> {
        let (name, keys, agg_col) = self.prepare_agg()?;
        let new_name = format!["{}_sum", name];
        let agg = build_ca_agg_variants!(self, new_name, agg_col, sum);
        DataFrame::new_from_columns(vec![keys, agg])
    }

    pub fn min(&self) -> Result<DataFrame> {
        let (name, keys, agg_col) = self.prepare_agg()?;
        let new_name = format!["{}_min", name];
        let agg = build_ca_agg_variants!(self, new_name, agg_col, min);
        DataFrame::new_from_columns(vec![keys, agg])
    }

    pub fn max(&self) -> Result<DataFrame> {
        let (name, keys, agg_col) = self.prepare_agg()?;
        let new_name = format!["{}_max", name];
        let agg = build_ca_agg_variants!(self, new_name, agg_col, max);
        DataFrame::new_from_columns(vec![keys, agg])
    }

    pub fn count(&self) -> Result<DataFrame> {
        let (name, keys, agg_col) = self.prepare_agg()?;
        let new_name = format!["{}_count", name];

        let mut builder = PrimitiveChunkedBuilder::new(&new_name, self.groups.len());
        for (_first, idx) in &self.groups {
            let s = agg_col.take(idx, None)?;
            builder.append_value(s.len() as u32)?;
        }
        let ca = builder.finish();
        let agg = Series::UInt32(ca);
        DataFrame::new_from_columns(vec![keys, agg])
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn test_group_by() {
        let s0 = Series::init("days", ["mo", "mo", "tue", "wed", "tue"].as_ref());
        let s1 = Series::init("temp", [20, 10, 7, 9, 1].as_ref());
        let s2 = Series::init("rain", [0.2, 0.1, 0.3, 0.1, 0.01].as_ref());
        let df = DataFrame::new_from_columns(vec![s0, s1, s2]).unwrap();

        println!(
            "{:?}",
            df.group_by("days").unwrap().select("temp").count().unwrap()
        );
        println!(
            "{:?}",
            df.group_by("days").unwrap().select("temp").mean().unwrap()
        );
        println!(
            "{:?}",
            df.group_by("days").unwrap().select("temp").sum().unwrap()
        );
        println!(
            "{:?}",
            df.group_by("days").unwrap().select("temp").min().unwrap()
        );
        println!(
            "{:?}",
            df.group_by("days").unwrap().select("temp").max().unwrap()
        );
    }
}
