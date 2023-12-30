use crate::prelude::*;

impl Series {
    pub fn full_null(name: &str, size: usize, dtype: &DataType) -> Self {
        // match the logical types and create them
        match dtype {
            DataType::List(inner_dtype) => {
                ListChunked::full_null_with_dtype(name, size, inner_dtype).into_series()
            },
            #[cfg(feature = "dtype-array")]
            DataType::Array(inner_dtype, width) => {
                ArrayChunked::full_null_with_dtype(name, size, inner_dtype, *width).into_series()
            },
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(rev_map, _) => {
                let mut ca = CategoricalChunked::full_null(name, size);
                // ensure we keep the rev-map of a cleared series
                if let Some(rev_map) = rev_map {
                    unsafe { ca.set_rev_map(rev_map.clone(), false) }
                }
                ca.into_series()
            },
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
            #[cfg(feature = "dtype-decimal")]
            DataType::Decimal(precision, scale) => Int128Chunked::full_null(name, size)
                .into_decimal_unchecked(
                    *precision,
                    scale.unwrap_or_else(|| unreachable!("scale should be set")),
                )
                .into_series(),
            #[cfg(feature = "dtype-struct")]
            DataType::Struct(fields) => {
                let fields = fields
                    .iter()
                    .map(|fld| Series::full_null(fld.name(), size, fld.data_type()))
                    .collect::<Vec<_>>();
                StructChunked::new(name, &fields).unwrap().into_series()
            },
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
                macro_rules! string {
                    () => {{
                        ChunkedArray::<StringType>::full_null(name, size).into_series()
                    }};
                }
                macro_rules! binary {
                    () => {{
                        ChunkedArray::<BinaryType>::full_null(name, size).into_series()
                    }};
                }
                match_dtype_to_logical_apply_macro!(dtype, primitive, string, binary, bool)
            },
        }
    }
}
