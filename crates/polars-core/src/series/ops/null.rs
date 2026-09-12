#[cfg(feature = "object")]
use crate::chunked_array::object::registry::get_object_builder;
use crate::prelude::*;

impl Series {
    /// Create a Series of `size` null values with the requested dtype.
    ///
    /// # Panics
    /// Panics if `dtype` contains an invalid Map dtype.
    pub fn full_null(name: PlSmallStr, size: usize, dtype: &DataType) -> Self {
        // Separate from the `match` below, because it only peels off a single layer of
        // nesting.
        #[cfg(feature = "dtype-map")]
        dtype
            .ensure_valid_map_dtypes()
            .expect("invalid Map dtype in `Series::full_null`");

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
            dt @ (DataType::Categorical(_, _) | DataType::Enum(_, _)) => {
                with_match_categorical_physical_type!(dt.cat_physical().unwrap(), |$C| {
                    CategoricalChunked::<$C>::full_null_with_dtype(
                        name,
                        size,
                        dtype.clone()
                    )
                        .into_series()
                })
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
                .into_decimal_unchecked(*precision, *scale)
                .into_series(),
            #[cfg(feature = "dtype-struct")]
            DataType::Struct(fields) => {
                let fields = fields
                    .iter()
                    .map(|fld| Series::full_null(fld.name().clone(), size, fld.dtype()))
                    .collect::<Vec<_>>();
                let ca = StructChunked::from_series(name, size, fields.iter()).unwrap();

                ca.with_outer_validity(Some(PlBitmap::new_scalar(false, size)))
                    .into_series()
            },
            DataType::BinaryOffset => {
                let array = PlBinaryArray::new_full_null(size);

                unsafe {
                    BinaryOffsetChunked::new_with_dims(
                        Arc::new(Field::new(name, dtype.clone())),
                        vec![Box::new(array)],
                        size,
                        size,
                    )
                }
                .into_series()
            },
            DataType::Null => Series::new_null(name, size),
            DataType::Unknown(kind) => {
                let dtype = kind.materialize().unwrap_or(DataType::Null);
                Series::full_null(name, size, &dtype)
            },
            #[cfg(feature = "object")]
            DataType::Object(_) => {
                let mut builder = get_object_builder(name, size);
                for _ in 0..size {
                    builder.append_null();
                }
                builder.to_series()
            },
            #[cfg(feature = "dtype-map")]
            DataType::Map(_, _) => {
                let storage = Series::full_null(name, size, &dtype.map_storage_dtype().unwrap());
                // SAFETY: the dtype is checked above, and an all-null Map holds no entries.
                unsafe { MapChunked::from_storage_unchecked(dtype.clone(), storage) }.into_series()
            },
            #[cfg(feature = "dtype-extension")]
            DataType::Extension(typ, storage_dtype) => {
                Series::full_null(name, size, storage_dtype).into_extension(typ.clone())
            },
            _ => {
                macro_rules! primitive {
                    ($type:ty) => {{ ChunkedArray::<$type>::full_null(name, size).into_series() }};
                }
                macro_rules! bool {
                    () => {{ ChunkedArray::<BooleanType>::full_null(name, size).into_series() }};
                }
                macro_rules! string {
                    () => {{ ChunkedArray::<StringType>::full_null(name, size).into_series() }};
                }
                macro_rules! binary {
                    () => {{ ChunkedArray::<BinaryType>::full_null(name, size).into_series() }};
                }
                match_dtype_to_logical_apply_macro!(dtype, primitive, string, binary, bool)
            },
        }
    }
}
