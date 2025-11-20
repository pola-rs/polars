#[cfg(any(
    feature = "dtype-datetime",
    feature = "dtype-date",
    feature = "dtype-duration",
    feature = "dtype-time"
))]
use polars_compute::cast::cast_default;
use polars_compute::cast::cast_unchecked;

use crate::prelude::*;

impl Series {
    /// Returns a reference to the Arrow ArrayRef
    #[inline]
    pub fn array_ref(&self, chunk_idx: usize) -> &ArrayRef {
        &self.chunks()[chunk_idx] as &ArrayRef
    }

    /// Convert a chunk in the Series to the correct Arrow type.
    /// This conversion is needed because polars doesn't use a
    /// 1 on 1 mapping for logical/categoricals, etc.
    pub fn to_arrow(&self, chunk_idx: usize, compat_level: CompatLevel) -> ArrayRef {
        ToArrowConverter {
            compat_level,
            #[cfg(feature = "dtype-categorical")]
            categorical_converter: {
                let mut categorical_converter =
                    crate::series::categorical_to_arrow::CategoricalToArrowConverter {
                        converters: Default::default(),
                        persist_remap: false,
                        output_keys_only: false,
                    };

                categorical_converter.initialize(self.dtype());

                categorical_converter
            },
        }
        .array_to_arrow(self.chunks().get(chunk_idx).unwrap().as_ref(), self.dtype())
    }
}

pub struct ToArrowConverter {
    pub compat_level: CompatLevel,
    #[cfg(feature = "dtype-categorical")]
    pub categorical_converter: crate::series::categorical_to_arrow::CategoricalToArrowConverter,
}

impl ToArrowConverter {
    pub fn array_to_arrow(&mut self, array: &dyn Array, dtype: &DataType) -> Box<dyn Array> {
        match dtype {
            // make sure that we recursively apply all logical types.
            #[cfg(feature = "dtype-struct")]
            DataType::Struct(fields) => {
                use arrow::array::StructArray;

                let arr: &StructArray = array.as_any().downcast_ref().unwrap();
                let values = arr
                    .values()
                    .iter()
                    .zip(fields.iter())
                    .map(|(values, field)| self.array_to_arrow(values.as_ref(), field.dtype()))
                    .collect::<Vec<_>>();

                StructArray::new(
                    ArrowDataType::Struct(
                        fields
                            .iter()
                            .map(|x| (x.name().clone(), x.dtype()))
                            .zip(values.iter().map(|x| x.dtype()))
                            .map(|((name, dtype), converted_arrow_dtype)| {
                                create_arrow_field(
                                    name,
                                    dtype,
                                    converted_arrow_dtype,
                                    self.compat_level,
                                )
                            })
                            .collect(),
                    ),
                    arr.len(),
                    values,
                    arr.validity().cloned(),
                )
                .boxed()
            },
            DataType::List(inner) => {
                let arr: &ListArray<i64> = array.as_any().downcast_ref().unwrap();
                let new_values = self.array_to_arrow(arr.values().as_ref(), inner);

                let arr = ListArray::<i64>::new(
                    ArrowDataType::LargeList(Box::new(create_arrow_field(
                        LIST_VALUES_NAME,
                        inner.as_ref(),
                        new_values.dtype(),
                        self.compat_level,
                    ))),
                    arr.offsets().clone(),
                    new_values,
                    arr.validity().cloned(),
                );
                Box::new(arr)
            },
            #[cfg(feature = "dtype-array")]
            DataType::Array(inner, width) => {
                use arrow::array::FixedSizeListArray;

                let arr: &FixedSizeListArray = array.as_any().downcast_ref().unwrap();
                let new_values = self.array_to_arrow(arr.values().as_ref(), inner);

                let arr = FixedSizeListArray::new(
                    ArrowDataType::FixedSizeList(
                        Box::new(create_arrow_field(
                            LIST_VALUES_NAME,
                            inner.as_ref(),
                            new_values.dtype(),
                            self.compat_level,
                        )),
                        *width,
                    ),
                    arr.len(),
                    new_values,
                    arr.validity().cloned(),
                );
                Box::new(arr)
            },
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(_, _) | DataType::Enum(_, _) => self
                .categorical_converter
                .array_to_arrow(array, dtype, self.compat_level),
            #[cfg(feature = "dtype-date")]
            DataType::Date => {
                cast_default(array, &DataType::Date.to_arrow(self.compat_level)).unwrap()
            },
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime(_, _) => {
                cast_default(array, &dtype.to_arrow(self.compat_level)).unwrap()
            },
            #[cfg(feature = "dtype-duration")]
            DataType::Duration(_) => {
                cast_default(array, &dtype.to_arrow(self.compat_level)).unwrap()
            },
            #[cfg(feature = "dtype-time")]
            DataType::Time => {
                cast_default(array, &DataType::Time.to_arrow(self.compat_level)).unwrap()
            },
            #[cfg(feature = "dtype-decimal")]
            DataType::Decimal(_, _) => array
                .as_any()
                .downcast_ref::<arrow::array::PrimitiveArray<i128>>()
                .unwrap()
                .clone()
                .to(dtype.to_arrow(CompatLevel::newest()))
                .to_boxed(),
            #[cfg(feature = "object")]
            DataType::Object(_) => {
                use crate::chunked_array::object::builder::object_series_to_arrow_array;
                object_series_to_arrow_array(&unsafe {
                    Series::from_chunks_and_dtype_unchecked(
                        PlSmallStr::EMPTY,
                        vec![array.to_boxed()],
                        dtype,
                    )
                })
            },
            DataType::String => {
                if self.compat_level.0 >= 1 {
                    array.to_boxed()
                } else {
                    cast_unchecked(array, &ArrowDataType::LargeUtf8).unwrap()
                }
            },
            DataType::Binary => {
                if self.compat_level.0 >= 1 {
                    array.to_boxed()
                } else {
                    cast_unchecked(array, &ArrowDataType::LargeBinary).unwrap()
                }
            },
            _ => {
                assert!(!dtype.is_logical());
                array.to_boxed()
            },
        }
    }
}

fn create_arrow_field(
    name: PlSmallStr,
    dtype: &DataType,
    arrow_dtype: &ArrowDataType,
    compat_level: CompatLevel,
) -> ArrowField {
    match (dtype, arrow_dtype) {
        #[cfg(feature = "dtype-categorical")]
        (DataType::Categorical(..) | DataType::Enum(..), ArrowDataType::Dictionary(_, _, _)) => {
            // Sets _PL_ metadata
            dtype.to_arrow_field(name, compat_level)
        },
        _ => ArrowField::new(name, arrow_dtype.clone(), true),
    }
}
