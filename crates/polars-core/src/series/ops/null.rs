use arrow::bitmap::Bitmap;

#[cfg(feature = "object")]
use crate::chunked_array::object::registry::get_object_builder;
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
            dt @ (DataType::Categorical(rev_map, ord) | DataType::Enum(rev_map, ord)) => {
                let mut ca = CategoricalChunked::full_null(
                    name,
                    matches!(dt, DataType::Enum(_, _)),
                    size,
                    *ord,
                );
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
                .into_decimal_unchecked(*precision, scale.unwrap_or(0))
                .into_series(),
            #[cfg(feature = "dtype-struct")]
            DataType::Struct(fields) => {
                let fields = fields
                    .iter()
                    .map(|fld| Series::full_null(fld.name(), size, fld.data_type()))
                    .collect::<Vec<_>>();
                let ca = StructChunked::from_series(name, &fields).unwrap();

                if !fields.is_empty() {
                    ca.with_outer_validity(Some(Bitmap::new_zeroed(size)))
                        .into_series()
                } else {
                    ca.into_series()
                }
            },
            DataType::Null => Series::new_null(name, size),
            DataType::Unknown(kind) => {
                let dtype = kind.materialize().expect("expected known type");
                Series::full_null(name, size, &dtype)
            },
            #[cfg(feature = "object")]
            DataType::Object(_, _) => {
                let mut builder = get_object_builder(name, size);
                for _ in 0..size {
                    builder.append_null();
                }
                builder.to_series()
            },
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
