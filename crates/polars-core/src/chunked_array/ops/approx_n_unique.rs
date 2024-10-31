use std::hash::Hash;

use polars_compute::hyperloglogplus::HyperLogLog;
use polars_utils::total_ord::{ToTotalOrd, TotalEq, TotalHash};
use polars_utils::IdxSize;

use super::{ChunkApproxNUnique, ChunkedArray, PolarsDataType};

impl<T> ChunkApproxNUnique for ChunkedArray<T>
where
    T: PolarsDataType,
    for<'a> T::Physical<'a>: TotalHash + TotalEq + Copy + ToTotalOrd,
    for<'a> <Option<T::Physical<'a>> as ToTotalOrd>::TotalOrdItem: Hash + Eq,
{
    fn approx_n_unique(&self) -> IdxSize {
        let mut hllp = HyperLogLog::new();
        self.iter().for_each(|item| hllp.add(&item.to_total_ord()));
        hllp.count() as IdxSize
    }
}
