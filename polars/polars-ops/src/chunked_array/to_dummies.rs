use polars_core::frame::groupby::IntoGroupsProxy;
use polars_core::utils::Wrap;

use super::*;

pub trait ToDummies<T> {
    fn to_dummies(&self) -> PolarsResult<DataFrame> {
        Err(PolarsError::InvalidOperation(
            "to_dummies is not implemented for this dtype".into(),
        ))
    }
}

#[cfg(feature = "dtype-u8")]
type DummyType = u8;
#[cfg(feature = "dtype-u8")]
type DummyCa = UInt8Chunked;

#[cfg(not(feature = "dtype-u8"))]
type DummyType = i32;
#[cfg(not(feature = "dtype-u8"))]
type DummyCa = Int32Chunked;

fn dummies_helper(mut groups: Vec<IdxSize>, len: usize, name: &str) -> DummyCa {
    groups.sort_unstable();

    // let mut group_member_iter = groups.into_iter();
    let mut av = vec![0 as DummyType; len];

    for idx in groups {
        let elem = unsafe { av.get_unchecked_mut(idx as usize) };
        *elem = 1;
    }

    ChunkedArray::from_vec(name, av)
}

fn sort_columns(mut columns: Vec<Series>) -> Vec<Series> {
    columns.sort_by(|a, b| a.name().partial_cmp(b.name()).unwrap());
    columns
}

impl ToDummies<Utf8Type> for Wrap<Utf8Chunked> {
    fn to_dummies(&self) -> PolarsResult<DataFrame> {
        let ca = &self.0;
        let groups = ca.group_tuples(true, false)?.into_idx();
        let col_name = ca.name();
        let taker = ca.take_rand();

        let columns = groups
            .into_par_iter()
            .map(|(first, groups)| {
                let name = match unsafe { taker.get_unchecked(first as usize) } {
                    Some(val) => format!("{}_{}", col_name, val),
                    None => format!("{}_null", col_name),
                };
                let ca = dummies_helper(groups, self.len(), &name);
                ca.into_series()
            })
            .collect();

        Ok(DataFrame::new_no_checks(sort_columns(columns)))
    }
}

#[cfg(feature = "dtype-categorical")]
impl ToDummies<Utf8Type> for Wrap<CategoricalChunked> {
    fn to_dummies(&self) -> PolarsResult<DataFrame> {
        let rev_map = self.get_rev_map();

        let groups = self.logical().group_tuples(true, false)?.into_idx();
        let col_name = self.name();
        let taker = self.logical().take_rand();

        let columns = groups
            .into_par_iter()
            .map(|(first, groups)| {
                let name = match unsafe { taker.get_unchecked(first as usize) } {
                    Some(val) => {
                        let name = rev_map.get(val);
                        format!("{}_{}", col_name, name)
                    }
                    None => format!("{}_null", col_name),
                };
                let ca = dummies_helper(groups, self.len(), &name);
                ca.into_series()
            })
            .collect();

        Ok(DataFrame::new_no_checks(sort_columns(columns)))
    }
}

impl<T> ToDummies<T> for ChunkedArray<T>
where
    T: PolarsIntegerType + Sync,
    T::Native: NumericNative,
{
    fn to_dummies(&self) -> PolarsResult<DataFrame> {
        let groups = self.group_tuples(true, false)?.into_idx();
        let col_name = self.name();
        let taker = self.take_rand();

        let columns = groups
            .into_par_iter()
            .map(|(first, groups)| {
                let name = match unsafe { taker.get_unchecked(first as usize) } {
                    Some(val) => format!("{}_{}", col_name, val),
                    None => format!("{}_null", col_name),
                };

                let ca = dummies_helper(groups, self.len(), &name);
                ca.into_series()
            })
            .collect();

        Ok(DataFrame::new_no_checks(sort_columns(columns)))
    }
}

impl<T: PolarsFloatType> ToDummies<Float32Type> for WrapFloat<ChunkedArray<T>> {}
impl ToDummies<Wrap<BooleanType>> for Wrap<BooleanChunked> {
    fn to_dummies(&self) -> PolarsResult<DataFrame> {
        let ca = self.cast(&DataType::Int8)?;
        ca.to_ops().to_dummies()
    }
}
