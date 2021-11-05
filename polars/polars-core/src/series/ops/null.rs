use crate::prelude::*;

impl Series {
    pub fn full_null(name: &str, size: usize, dtype: &DataType) -> Self {
        if dtype == &dtype.to_physical() {
            macro_rules! primitive {
                ($type:ty) => {{
                    ChunkedArray::<$type>::full_null(name, size).into_series()
                }};
            }
            macro_rules! bool {
                () => {{
                    ChunkedArray::<BooleanType>::full_null(name, size).into_series()
                }};
            }
            macro_rules! utf8 {
                () => {{
                    ChunkedArray::<Utf8Type>::full_null(name, size).into_series()
                }};
            }
            match_dtype_to_logical_apply_macro!(dtype, primitive, utf8, bool)
        } else {
            // match the logical types and create them
            match dtype {
                #[cfg(feature = "dtype-categorical")]
                DataType::Categorical => CategoricalChunked::full_null(name, size).into_series(),
                #[cfg(feature = "dtype-date")]
                DataType::Date => Int32Chunked::full_null(name, size)
                    .into_date()
                    .into_series(),
                #[cfg(feature = "dtype-datetime")]
                DataType::Datetime => Int64Chunked::full_null(name, size)
                    .into_date()
                    .into_series(),
                #[cfg(feature = "dtype-time")]
                DataType::Time => Int64Chunked::full_null(name, size)
                    .into_time()
                    .into_series(),
                dt => panic!("logical-type not yet implemented: {}", dt),
            }
        }
    }
}
