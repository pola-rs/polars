use polars_arrow::compute::tile;
use polars_row::ArrayRef;

use crate::datatypes::PolarsNumericType;
use crate::prelude::ChunkedArray;

impl<T: PolarsNumericType> ChunkedArray<T> {
    pub fn tile(&self, n: usize) -> Self {
        let ca = self.rechunk();
        let arr = ca.downcast_iter().next().unwrap();

        let arr = Box::new(tile::tile_primitive(arr, n)) as ArrayRef;
        unsafe { ChunkedArray::from_chunks(self.name(), vec![arr]) }
    }
}
