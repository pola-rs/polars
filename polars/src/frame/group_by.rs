use super::hash_join::{prepare_hashed_relation, prepare_hashed_relation_non_null};
use crate::prelude::*;
use crate::series::chunked_array::builder::{build_primitive_ca_with_opt, PrimitiveChunkedBuilder};
use num::{Num, NumCast, Zero};
use rayon::prelude::*;
use std::hash::Hash;

fn groupby_opt<T>(a: impl Iterator<Item = Option<T>>) -> Vec<(usize, Vec<usize>)>
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

/// No null values
fn groupby_no_null<T>(a: impl Iterator<Item = T>) -> Vec<(usize, Vec<usize>)>
where
    T: Hash + Eq + Copy,
{
    let hash_tbl = prepare_hashed_relation_non_null(a);

    hash_tbl
        .into_iter()
        .map(|(_, indexes)| {
            let first = unsafe { *indexes.get_unchecked(0) };
            (first, indexes)
        })
        .collect()
}

impl DataFrame {
    /// Group DataFrame using a Series column.
    pub fn groupby(&self, by: &str) -> Option<GroupBy> {
        let groups = if let Some(s) = self.select(by) {
            macro_rules! create_iter {
                ($ca:ident) => {{
                    if let Ok(slice) = $ca.cont_slice() {
                        groupby_no_null(slice.iter())
                    } else {
                        groupby_opt($ca.iter())
                    }
                }};
            }

            match s {
                Series::UInt32(ca) => create_iter!(ca),
                Series::Int32(ca) => create_iter!(ca),
                Series::Int64(ca) => create_iter!(ca),
                Series::Bool(ca) => groupby_opt(ca.iter()),
                Series::Utf8(ca) => groupby_opt(ca.iter().map(|v| Some(v))),
                Series::Date32(ca) => create_iter!(ca),
                Series::Date64(ca) => create_iter!(ca),
                Series::Time64Ns(ca) => create_iter!(ca),
                Series::DurationNs(ca) => create_iter!(ca),
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
    pub by: String,
    // [first idx, [other idx]]
    groups: Vec<(usize, Vec<usize>)>,
    selection: Option<String>,
}

macro_rules! build_ca_agg {
    ($self:ident, $new_name:ident, $agg_col:ident, $variant:ident, $agg_fn:ident) => {{
        // First parallelize
        let vec_opts = $self
            .groups
            .par_iter()
            .map(|(_first, idx)| {
                let s = $agg_col.take(idx, None).expect("could not append");
                let opt = s.$agg_fn();
                opt
            })
            .collect::<Vec<_>>();

        let mut builder = PrimitiveChunkedBuilder::new(&$new_name, $self.groups.len());
        for opt in vec_opts {
            builder.append_option(opt).expect("could not append");
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
            ArrowDataType::Date32(DateUnit::Millisecond) => {
                build_ca_agg!($self, $new_name, $agg_col, Date32, $agg_fn)
            }
            ArrowDataType::Date64(DateUnit::Millisecond) => {
                build_ca_agg!($self, $new_name, $agg_col, Date64, $agg_fn)
            }
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

fn agg_sum<T>(
    ca: &ChunkedArray<T>,
    groups: &Vec<(usize, Vec<usize>)>,
    agg_col: &Series,
) -> Vec<Option<T::Native>>
where
    T: PolarNumericType + Sync,
    T::Native: std::ops::Add<Output = T::Native> + Num + NumCast,
{
    groups
        .par_iter()
        .map(|(_first, idx)| {
            if let Ok(slice) = ca.cont_slice() {
                let mut sum = Zero::zero();
                for i in idx {
                    sum = sum + slice[*i]
                }
                Some(sum)
            } else {
                let take = agg_col.take(idx, None).unwrap();
                take.sum()
            }
        })
        .collect()
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

    /// Aggregate grouped series and compute the mean per group.
    pub fn mean(&self) -> Result<DataFrame> {
        let (name, keys, agg_col) = self.prepare_agg()?;
        let new_name = format!["{}_mean", name];
        let agg = build_ca_agg_variants!(self, new_name, agg_col, mean);
        DataFrame::new_from_columns(vec![keys, agg])
    }

    /// Aggregate grouped series and compute the sum per group.
    pub fn sum(&self) -> Result<DataFrame> {
        let (name, keys, agg_col) = self.prepare_agg()?;
        let new_name = format!["{}_sum", name];
        let agg = match agg_col.dtype() {
            ArrowDataType::Int32 => {
                let ca = agg_col.i32().unwrap();
                let vec_opts = agg_sum(ca, &self.groups, agg_col);
                let ca = build_primitive_ca_with_opt(&vec_opts, &new_name);
                Series::Int32(ca)
            }
            ArrowDataType::Int64 => {
                let ca = agg_col.i64().unwrap();
                let vec_opts = agg_sum(ca, &self.groups, agg_col);
                let ca = build_primitive_ca_with_opt(&vec_opts, &new_name);
                Series::Int64(ca)
            }
            ArrowDataType::UInt32 => {
                let ca = agg_col.u32().unwrap();
                let vec_opts = agg_sum(ca, &self.groups, agg_col);
                let ca = build_primitive_ca_with_opt(&vec_opts, &new_name);
                Series::UInt32(ca)
            }
            ArrowDataType::Float32 => {
                let ca = agg_col.f32().unwrap();
                let vec_opts = agg_sum(ca, &self.groups, agg_col);
                let ca = build_primitive_ca_with_opt(&vec_opts, &new_name);
                Series::Float32(ca)
            }
            ArrowDataType::Float64 => {
                let ca = agg_col.f64().unwrap();
                let vec_opts = agg_sum(ca, &self.groups, agg_col);
                let ca = build_primitive_ca_with_opt(&vec_opts, &new_name);
                Series::Float64(ca)
            }
            ArrowDataType::Date32(DateUnit::Millisecond) => {
                let ca = agg_col.date32().unwrap();
                let vec_opts = agg_sum(ca, &self.groups, agg_col);
                let ca = build_primitive_ca_with_opt(&vec_opts, &new_name);
                Series::Date32(ca)
            }
            ArrowDataType::Date64(DateUnit::Millisecond) => {
                let ca = agg_col.date64().unwrap();
                let vec_opts = agg_sum(ca, &self.groups, agg_col);
                let ca = build_primitive_ca_with_opt(&vec_opts, &new_name);
                Series::Date64(ca)
            }
            ArrowDataType::Time64(TimeUnit::Nanosecond) => {
                let ca = agg_col.time64ns().unwrap();
                let vec_opts = agg_sum(ca, &self.groups, agg_col);
                let ca = build_primitive_ca_with_opt(&vec_opts, &new_name);
                Series::Time64Ns(ca)
            }
            ArrowDataType::Duration(TimeUnit::Nanosecond) => {
                let ca = agg_col.duration_ns().unwrap();
                let vec_opts = agg_sum(ca, &self.groups, agg_col);
                let ca = build_primitive_ca_with_opt(&vec_opts, &new_name);
                Series::DurationNs(ca)
            }
            _ => return Err(PolarsError::DataTypeMisMatch),
        };
        // let agg = build_ca_agg_variants!(self, new_name, agg_col, sum);
        DataFrame::new_from_columns(vec![keys, agg])
    }

    /// Aggregate grouped series and compute the minimal value per group.
    pub fn min(&self) -> Result<DataFrame> {
        let (name, keys, agg_col) = self.prepare_agg()?;
        let new_name = format!["{}_min", name];
        let agg = build_ca_agg_variants!(self, new_name, agg_col, min);
        DataFrame::new_from_columns(vec![keys, agg])
    }

    /// Aggregate grouped series and compute the maximum value per group.
    pub fn max(&self) -> Result<DataFrame> {
        let (name, keys, agg_col) = self.prepare_agg()?;
        let new_name = format!["{}_max", name];
        let agg = build_ca_agg_variants!(self, new_name, agg_col, max);
        DataFrame::new_from_columns(vec![keys, agg])
    }

    /// Aggregate grouped series and compute the number of values per group.
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
            df.groupby("days").unwrap().select("temp").count().unwrap()
        );
        println!(
            "{:?}",
            df.groupby("days").unwrap().select("temp").mean().unwrap()
        );
        println!(
            "{:?}",
            df.groupby("days").unwrap().select("temp").sum().unwrap()
        );
        println!(
            "{:?}",
            df.groupby("days").unwrap().select("temp").min().unwrap()
        );
        println!(
            "{:?}",
            df.groupby("days").unwrap().select("temp").max().unwrap()
        );
    }
}
