#[cfg(any(
    feature = "dtype-datetime",
    feature = "dtype-date",
    feature = "dtype-duration",
    feature = "dtype-time"
))]
use arrow::compute::cast::cast_default as cast;

use crate::prelude::*;

impl Series {
    /// Returns a reference to the Arrow ArrayRef
    #[inline]
    pub fn array_ref(&self, chunk_idx: usize) -> &ArrayRef {
        &self.chunks()[chunk_idx] as &ArrayRef
    }

    /// Convert a chunk in the Series to the correct Arrow type.
    /// This conversion is needed because polars doesn't use a
    /// 1 on 1 mapping for logical/ categoricals, etc.
    pub fn to_arrow(&self, chunk_idx: usize, pl_flavor: bool) -> ArrayRef {
        match self.dtype() {
            // make sure that we recursively apply all logical types.
            #[cfg(feature = "dtype-struct")]
            DataType::Struct(_) => self.struct_().unwrap().to_arrow(chunk_idx),
            // special list branch to
            // make sure that we recursively apply all logical types.
            DataType::List(inner) => {
                let ca = self.list().unwrap();
                let arr = ca.chunks[chunk_idx].clone();
                let arr = arr.as_any().downcast_ref::<ListArray<i64>>().unwrap();

                let new_values = if let DataType::Null = &**inner {
                    arr.values().clone()
                } else {
                    // We pass physical arrays and cast to logical before we convert to arrow.
                    let s = unsafe {
                        Series::from_chunks_and_dtype_unchecked(
                            "",
                            vec![arr.values().clone()],
                            &inner.to_physical(),
                        )
                        .cast_unchecked(inner)
                        .unwrap()
                    };

                    s.to_arrow(0, true)
                };

                let data_type = ListArray::<i64>::default_datatype(inner.to_arrow(true));
                let arr = ListArray::<i64>::new(
                    data_type,
                    arr.offsets().clone(),
                    new_values,
                    arr.validity().cloned(),
                );
                Box::new(arr)
            },
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(_, ordering) => {
                let ca = self.categorical().unwrap();
                let arr = ca.physical().chunks()[chunk_idx].clone();
                // SAFETY: categoricals are always u32's.
                let cats = unsafe { UInt32Chunked::from_chunks("", vec![arr]) };

                // SAFETY: we only take a single chunk and change nothing about the index/rev_map mapping.
                let new = unsafe {
                    CategoricalChunked::from_cats_and_rev_map_unchecked(
                        cats,
                        ca.get_rev_map().clone(),
                        *ordering,
                    )
                };

                let arr: DictionaryArray<u32> = (&new).into();
                Box::new(arr) as ArrayRef
            },
            #[cfg(feature = "dtype-date")]
            DataType::Date => {
                cast(&*self.chunks()[chunk_idx], &DataType::Date.to_arrow(true)).unwrap()
            },
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime(_, _) => {
                cast(&*self.chunks()[chunk_idx], &self.dtype().to_arrow(true)).unwrap()
            },
            #[cfg(feature = "dtype-duration")]
            DataType::Duration(_) => {
                cast(&*self.chunks()[chunk_idx], &self.dtype().to_arrow(true)).unwrap()
            },
            #[cfg(feature = "dtype-time")]
            DataType::Time => {
                cast(&*self.chunks()[chunk_idx], &DataType::Time.to_arrow(true)).unwrap()
            },
            #[cfg(feature = "object")]
            DataType::Object(_, None) => {
                use crate::chunked_array::object::builder::object_series_to_arrow_array;
                if self.chunks().len() == 1 && chunk_idx == 0 {
                    object_series_to_arrow_array(self)
                } else {
                    // we slice the series to only that chunk
                    let offset = self.chunks()[..chunk_idx]
                        .iter()
                        .map(|arr| arr.len())
                        .sum::<usize>() as i64;
                    let len = self.chunks()[chunk_idx].len();
                    let s = self.slice(offset, len);
                    object_series_to_arrow_array(&s)
                }
            },
            DataType::String if pl_flavor => {
                // TODO: implement Utf8ViewCast Here
                self.array_ref(chunk_idx).clone()
            },
            DataType::Binary if pl_flavor => {
                // TODO: implement BinViewCast Here
                self.array_ref(chunk_idx).clone()
            },
            _ => self.array_ref(chunk_idx).clone(),
        }
    }
}
