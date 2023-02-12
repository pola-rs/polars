use crate::prelude::*;

impl Series {
    pub fn full_null(name: &str, size: usize, dtype: &DataType) -> Self {
        // match the logical types and create them
        match dtype {
            DataType::List(inner_dtype) => {
                ListChunked::full_null_with_dtype(name, size, inner_dtype).into_series()
            }
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(_) => CategoricalChunked::full_null(name, size).into_series(),
            #[cfg(feature = "dtype-date")]
            DataType::Date => Int32Chunked::full_null(name, size)
                .into_date()
                .into_series(),
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime(tu, tz) => Int64Chunked::full_null(name, size)
                .into_datetime(*tu, tz.clone())
                .into_series(),
            #[cfg(feature = "dtype-duration")]
            DataType::Duration(tu) => Int64Chunked::full_null(name, size)
                .into_duration(*tu)
                .into_series(),
            #[cfg(feature = "dtype-time")]
            DataType::Time => Int64Chunked::full_null(name, size)
                .into_time()
                .into_series(),
            #[cfg(feature = "dtype-struct")]
            DataType::Struct(fields) => {
                let fields = fields
                    .iter()
                    .map(|fld| Series::full_null(fld.name(), size, fld.data_type()))
                    .collect::<Vec<_>>();
                StructChunked::new(name, &fields).unwrap().into_series()
            }
            DataType::Null => Series::new_null(name, size),
            _ => {
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
                #[cfg(feature = "dtype-binary")]
                macro_rules! binary {
                    () => {{
                        ChunkedArray::<BinaryType>::full_null(name, size).into_series()
                    }};
                }
                match_dtype_to_logical_apply_macro!(dtype, primitive, utf8, binary, bool)
            }
        }
    }
}
