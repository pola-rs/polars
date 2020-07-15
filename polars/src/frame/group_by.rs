use super::hash_join::prepare_hashed_relation;
use crate::chunked_array::builder::{build_primitive_ca_with_opt, PrimitiveChunkedBuilder};
use crate::prelude::*;
use arrow::datatypes::ArrowNativeType;
use num::{Num, NumCast, ToPrimitive, Zero};
use rayon::prelude::*;
use std::hash::Hash;

fn groupby<T>(a: impl Iterator<Item = T>) -> Vec<(usize, Vec<usize>)>
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
    /// Group DataFrame using a Series column.
    ///
    /// # Example
    ///
    /// ```
    /// use polars::prelude::*;
    /// fn groupby_sum(df: &DataFrame) -> Result<DataFrame> {
    ///     df.groupby("column_name")?
    ///     .select("agg_column_name")
    ///     .sum()
    /// }
    /// ```
    pub fn groupby(&self, by: &str) -> Result<GroupBy> {
        let groups = if let Some(s) = self.column(by) {
            macro_rules! create_iter {
                ($ca:ident) => {{
                    if let Ok(slice) = $ca.cont_slice() {
                        groupby(slice.iter())
                    } else {
                        groupby($ca.into_iter())
                    }
                }};
            }

            match s {
                Series::UInt32(ca) => create_iter!(ca),
                Series::Int32(ca) => create_iter!(ca),
                Series::Int64(ca) => create_iter!(ca),
                Series::Bool(ca) => groupby(ca.into_iter()),
                Series::Utf8(ca) => groupby(ca.into_iter()),
                Series::Date32(ca) => create_iter!(ca),
                Series::Date64(ca) => create_iter!(ca),
                Series::Time64Ns(ca) => create_iter!(ca),
                Series::DurationNs(ca) => create_iter!(ca),
                _ => unimplemented!(),
            }
        } else {
            return Err(PolarsError::NotFound);
        };

        Ok(GroupBy {
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

fn agg_sum<T>(
    ca: &ChunkedArray<T>,
    groups: &Vec<(usize, Vec<usize>)>,
    agg_col: &Series,
) -> Vec<Option<T::Native>>
where
    T: PolarsNumericType + Sync,
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
                let take = agg_col.take(idx).unwrap();
                take.sum()
            }
        })
        .collect()
}

fn agg_mean<T>(
    ca: &ChunkedArray<T>,
    groups: &Vec<(usize, Vec<usize>)>,
    agg_col: &Series,
) -> Vec<Option<f64>>
where
    T: PolarsNumericType + Sync,
    T::Native: std::ops::Add<Output = T::Native> + Num + NumCast + ToPrimitive,
{
    groups
        .par_iter()
        .map(|(_first, idx)| {
            if let Ok(slice) = ca.cont_slice() {
                let mut sum = 0.;
                for i in idx {
                    sum = sum + slice[*i].to_f64().unwrap()
                }
                Some(sum / idx.len() as f64)
            } else {
                let take = agg_col.take(idx).unwrap();
                let opt_sum: Option<T::Native> = take.sum();
                opt_sum.map(|sum| sum.to_f64().unwrap() / idx.len() as f64)
            }
        })
        .collect()
}

fn agg_min<T>(
    ca: &ChunkedArray<T>,
    groups: &Vec<(usize, Vec<usize>)>,
    agg_col: &Series,
) -> Vec<Option<T::Native>>
where
    T: PolarsNumericType + Sync,
    T::Native: std::ops::Add<Output = T::Native> + Num + NumCast,
{
    groups
        .par_iter()
        .map(|(_first, idx)| {
            if let Ok(slice) = ca.cont_slice() {
                let mut min = None;
                for i in idx {
                    let v = slice[*i];

                    min = match min {
                        Some(min) => {
                            if min < v {
                                Some(min)
                            } else {
                                Some(v)
                            }
                        }
                        None => Some(v),
                    };
                }
                min
            } else {
                let take = agg_col.take(idx).unwrap();
                take.min()
            }
        })
        .collect()
}

fn agg_max<T>(
    ca: &ChunkedArray<T>,
    groups: &Vec<(usize, Vec<usize>)>,
    agg_col: &Series,
) -> Vec<Option<T::Native>>
where
    T: PolarsNumericType + Sync,
    T::Native: std::ops::Add<Output = T::Native> + Num + NumCast,
{
    groups
        .par_iter()
        .map(|(_first, idx)| {
            if let Ok(slice) = ca.cont_slice() {
                let mut max = None;
                for i in idx {
                    let v = slice[*i];

                    max = match max {
                        Some(max) => {
                            if max > v {
                                Some(max)
                            } else {
                                Some(v)
                            }
                        }
                        None => Some(v),
                    };
                }
                max
            } else {
                let take = agg_col.take(idx).unwrap();
                take.max()
            }
        })
        .collect()
}

macro_rules! apply_agg_fn {
    // agg_fn
    //     function take as input:
    //     ca: &ChunkedArray<T>,
    //     groups: &Vec<(usize, Vec<usize>)>,
    //     agg_col: &Series,
    ($self:ident, $agg_fn:ident, $agg_col:ident, $new_name:ident) => {
        match $agg_col.dtype() {
            ArrowDataType::Int32 => {
                let ca = $agg_col.i32().unwrap();
                let vec_opts = $agg_fn(ca, &$self.groups, $agg_col);
                let ca = build_primitive_ca_with_opt(&vec_opts, &$new_name);
                Series::Int32(ca)
            }
            ArrowDataType::Int64 => {
                let ca = $agg_col.i64().unwrap();
                let vec_opts = $agg_fn(ca, &$self.groups, $agg_col);
                let ca = build_primitive_ca_with_opt(&vec_opts, &$new_name);
                Series::Int64(ca)
            }
            ArrowDataType::UInt32 => {
                let ca = $agg_col.u32().unwrap();
                let vec_opts = $agg_fn(ca, &$self.groups, $agg_col);
                let ca = build_primitive_ca_with_opt(&vec_opts, &$new_name);
                Series::UInt32(ca)
            }
            ArrowDataType::Float32 => {
                let ca = $agg_col.f32().unwrap();
                let vec_opts = $agg_fn(ca, &$self.groups, $agg_col);
                let ca = build_primitive_ca_with_opt(&vec_opts, &$new_name);
                Series::Float32(ca)
            }
            ArrowDataType::Float64 => {
                let ca = $agg_col.f64().unwrap();
                let vec_opts = $agg_fn(ca, &$self.groups, $agg_col);
                let ca = build_primitive_ca_with_opt(&vec_opts, &$new_name);
                Series::Float64(ca)
            }
            ArrowDataType::Date32(DateUnit::Millisecond) => {
                let ca = $agg_col.date32().unwrap();
                let vec_opts = $agg_fn(ca, &$self.groups, $agg_col);
                let ca = build_primitive_ca_with_opt(&vec_opts, &$new_name);
                Series::Date32(ca)
            }
            ArrowDataType::Date64(DateUnit::Millisecond) => {
                let ca = $agg_col.date64().unwrap();
                let vec_opts = $agg_fn(ca, &$self.groups, $agg_col);
                let ca = build_primitive_ca_with_opt(&vec_opts, &$new_name);
                Series::Date64(ca)
            }
            ArrowDataType::Time64(TimeUnit::Nanosecond) => {
                let ca = $agg_col.time64ns().unwrap();
                let vec_opts = $agg_fn(ca, &$self.groups, $agg_col);
                let ca = build_primitive_ca_with_opt(&vec_opts, &$new_name);
                Series::Time64Ns(ca)
            }
            ArrowDataType::Duration(TimeUnit::Nanosecond) => {
                let ca = $agg_col.duration_ns().unwrap();
                let vec_opts = $agg_fn(ca, &$self.groups, $agg_col);
                let ca = build_primitive_ca_with_opt(&vec_opts, &$new_name);
                Series::DurationNs(ca)
            }
            _ => return Err(PolarsError::DataTypeMisMatch),
        }
    };
}

macro_rules! apply_agg_fn_return_single_variant {
    // agg_fn
    //     function take as input:
    //     ca: &ChunkedArray<T>,
    //     groups: &Vec<(usize, Vec<usize>)>,
    //     agg_col: &Series,
    ($self:ident, $agg_fn:ident, $agg_col:ident, $new_name:ident, $variant:ident) => {
        match $agg_col.dtype() {
            ArrowDataType::Int32 => {
                let ca = $agg_col.i32().unwrap();
                let vec_opts = $agg_fn(ca, &$self.groups, $agg_col);
                let ca = build_primitive_ca_with_opt(&vec_opts, &$new_name);
                Series::$variant(ca)
            }
            ArrowDataType::Int64 => {
                let ca = $agg_col.i64().unwrap();
                let vec_opts = $agg_fn(ca, &$self.groups, $agg_col);
                let ca = build_primitive_ca_with_opt(&vec_opts, &$new_name);
                Series::$variant(ca)
            }
            ArrowDataType::UInt32 => {
                let ca = $agg_col.u32().unwrap();
                let vec_opts = $agg_fn(ca, &$self.groups, $agg_col);
                let ca = build_primitive_ca_with_opt(&vec_opts, &$new_name);
                Series::$variant(ca)
            }
            ArrowDataType::Float32 => {
                let ca = $agg_col.f32().unwrap();
                let vec_opts = $agg_fn(ca, &$self.groups, $agg_col);
                let ca = build_primitive_ca_with_opt(&vec_opts, &$new_name);
                Series::$variant(ca)
            }
            ArrowDataType::Float64 => {
                let ca = $agg_col.f64().unwrap();
                let vec_opts = $agg_fn(ca, &$self.groups, $agg_col);
                let ca = build_primitive_ca_with_opt(&vec_opts, &$new_name);
                Series::$variant(ca)
            }
            ArrowDataType::Date32(DateUnit::Millisecond) => {
                let ca = $agg_col.date32().unwrap();
                let vec_opts = $agg_fn(ca, &$self.groups, $agg_col);
                let ca = build_primitive_ca_with_opt(&vec_opts, &$new_name);
                Series::$variant(ca)
            }
            ArrowDataType::Date64(DateUnit::Millisecond) => {
                let ca = $agg_col.date64().unwrap();
                let vec_opts = $agg_fn(ca, &$self.groups, $agg_col);
                let ca = build_primitive_ca_with_opt(&vec_opts, &$new_name);
                Series::$variant(ca)
            }
            ArrowDataType::Time64(TimeUnit::Nanosecond) => {
                let ca = $agg_col.time64ns().unwrap();
                let vec_opts = $agg_fn(ca, &$self.groups, $agg_col);
                let ca = build_primitive_ca_with_opt(&vec_opts, &$new_name);
                Series::$variant(ca)
            }
            ArrowDataType::Duration(TimeUnit::Nanosecond) => {
                let ca = $agg_col.duration_ns().unwrap();
                let vec_opts = $agg_fn(ca, &$self.groups, $agg_col);
                let ca = build_primitive_ca_with_opt(&vec_opts, &$new_name);
                Series::$variant(ca)
            }
            _ => return Err(PolarsError::DataTypeMisMatch),
        }
    };
}

impl<'a> GroupBy<'a> {
    pub fn select(mut self, name: &str) -> Self {
        self.selection = Some(name.to_string());
        self
    }

    fn keys(&self) -> Result<Series> {
        self.df.f_column(&self.by).take_iter(
            self.groups.iter().map(|(idx, _)| Some(*idx)),
            Some(self.groups.len()),
        )
    }

    fn prepare_agg(&self) -> Result<(&String, Series, &Series)> {
        let name = match &self.selection {
            Some(name) => name,
            None => return Err(PolarsError::NoSelection),
        };

        let keys = self.keys()?;
        let agg_col = self.df.column(name).ok_or(PolarsError::NotFound)?;
        Ok((name, keys, agg_col))
    }

    /// Aggregate grouped series and compute the mean per group.
    pub fn mean(&self) -> Result<DataFrame> {
        let (name, keys, agg_col) = self.prepare_agg()?;
        let new_name = format!["{}_mean", name];
        let agg = apply_agg_fn_return_single_variant!(self, agg_mean, agg_col, new_name, Float64);
        DataFrame::new(vec![keys, agg])
    }

    /// Aggregate grouped series and compute the sum per group.
    pub fn sum(&self) -> Result<DataFrame> {
        let (name, keys, agg_col) = self.prepare_agg()?;
        let new_name = format!["{}_sum", name];
        let agg = apply_agg_fn!(self, agg_sum, agg_col, new_name);
        DataFrame::new(vec![keys, agg])
    }

    /// Aggregate grouped series and compute the minimal value per group.
    pub fn min(&self) -> Result<DataFrame> {
        let (name, keys, agg_col) = self.prepare_agg()?;
        let new_name = format!["{}_min", name];
        let agg = apply_agg_fn!(self, agg_min, agg_col, new_name);
        DataFrame::new(vec![keys, agg])
    }

    /// Aggregate grouped series and compute the maximum value per group.
    pub fn max(&self) -> Result<DataFrame> {
        let (name, keys, agg_col) = self.prepare_agg()?;
        let new_name = format!["{}_max", name];
        let agg = apply_agg_fn!(self, agg_max, agg_col, new_name);
        DataFrame::new(vec![keys, agg])
    }

    /// Aggregate grouped series and compute the number of values per group.
    pub fn count(&self) -> Result<DataFrame> {
        let (name, keys, agg_col) = self.prepare_agg()?;
        let new_name = format!["{}_count", name];

        let mut builder = PrimitiveChunkedBuilder::new(&new_name, self.groups.len());
        for (_first, idx) in &self.groups {
            let s = agg_col.take(idx)?;
            builder.append_value(s.len() as u32)?;
        }
        let ca = builder.finish();
        let agg = Series::UInt32(ca);
        DataFrame::new(vec![keys, agg])
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn test_group_by() {
        let s0 = Series::new("days", ["mo", "mo", "tue", "wed", "tue"].as_ref());
        let s1 = Series::new("temp", [20, 10, 7, 9, 1].as_ref());
        let s2 = Series::new("rain", [0.2, 0.1, 0.3, 0.1, 0.01].as_ref());
        let df = DataFrame::new(vec![s0, s1, s2]).unwrap();

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
