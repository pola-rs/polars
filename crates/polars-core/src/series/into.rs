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
        self.to_arrow_with_field(chunk_idx, compat_level, None)
            .unwrap()
    }

    pub fn to_arrow_with_field(
        &self,
        chunk_idx: usize,
        compat_level: CompatLevel,
        output_arrow_field: Option<&ArrowField>,
    ) -> PolarsResult<ArrayRef> {
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
        .array_to_arrow(
            self.chunks().get(chunk_idx).unwrap().as_ref(),
            self.dtype(),
            output_arrow_field,
        )
    }
}

pub struct ToArrowConverter {
    pub compat_level: CompatLevel,
    #[cfg(feature = "dtype-categorical")]
    pub categorical_converter: crate::series::categorical_to_arrow::CategoricalToArrowConverter,
}

impl ToArrowConverter {
    /// Returns an error if `output_arrow_field` was provided and does not match the output data type.
    pub fn array_to_arrow(
        &mut self,
        array: &dyn Array,
        dtype: &DataType,
        output_arrow_field: Option<&ArrowField>,
    ) -> PolarsResult<Box<dyn Array>> {
        let out = self.array_to_arrow_impl(array, dtype, output_arrow_field)?;

        if let Some(field) = output_arrow_field {
            polars_ensure!(
                field.is_nullable || !out.has_nulls(),
                SchemaMismatch:
                "to_arrow(): nullable is false but array contained {} NULLs (arrow field: {:?})",
                out.null_count(), field,
            );

            // Don't eq nested types (they will recurse here with the inner types).
            if (!field.dtype().is_nested()
                || matches!(field.dtype(), ArrowDataType::Dictionary(..)))
                && out.dtype() != field.dtype()
            {
                polars_bail!(
                    SchemaMismatch:
                    "to_arrow(): provided dtype ({:?}) does not match output dtype ({:?})",
                    field.dtype(), out.dtype()
                )
            }
        }

        Ok(out)
    }

    fn array_to_arrow_impl(
        &mut self,
        array: &dyn Array,
        dtype: &DataType,
        output_arrow_field: Option<&ArrowField>,
    ) -> PolarsResult<Box<dyn Array>> {
        Ok(match dtype {
            // make sure that we recursively apply all logical types.
            #[cfg(feature = "dtype-struct")]
            DataType::Struct(fields) => {
                use arrow::array::StructArray;
                let arr: &StructArray = array.as_any().downcast_ref().unwrap();

                let expected_output_fields: &[ArrowField] = match output_arrow_field {
                    Some(
                        field @ ArrowField {
                            name: _,
                            dtype: ArrowDataType::Struct(fields),
                            is_nullable: _,
                            metadata: _,
                        },
                    ) if fields.len() == arr.fields().len()
                        && fields
                            .iter()
                            .zip(arr.fields())
                            .all(|(l, r)| l.name() == r.name()) =>
                    {
                        fields.as_slice()
                    },
                    Some(ArrowField { dtype, .. }) => polars_bail!(
                        SchemaMismatch:
                        "to_arrow(): struct dtype mismatch: {:?} != expected: {:?}",
                        dtype, arr.dtype(),
                    ),
                    None => &[],
                };

                let values: Vec<ArrayRef> = arr
                    .values()
                    .iter()
                    .zip(fields.iter())
                    .enumerate()
                    .map(|(i, (values, field))| {
                        self.array_to_arrow(
                            values.as_ref(),
                            field.dtype(),
                            expected_output_fields.get(i),
                        )
                    })
                    .collect::<PolarsResult<_>>()?;

                let converted_arrow_fields: Vec<ArrowField> = fields
                    .iter()
                    .map(|x| (x.name().clone(), x.dtype()))
                    .zip(values.iter().map(|x| x.dtype()))
                    .enumerate()
                    .map(|(i, ((name, dtype), converted_arrow_dtype))| {
                        create_arrow_field(
                            name,
                            dtype,
                            converted_arrow_dtype,
                            self.compat_level,
                            opt_field_is_nullable(expected_output_fields.get(i)),
                        )
                    })
                    .collect();

                StructArray::new(
                    ArrowDataType::Struct(converted_arrow_fields),
                    arr.len(),
                    values,
                    arr.validity().cloned(),
                )
                .boxed()
            },
            DataType::List(inner) => {
                let arr: &ListArray<i64> = array.as_any().downcast_ref().unwrap();

                let expected_inner_output_field: Option<&ArrowField> = match output_arrow_field {
                    Some(ArrowField {
                        name: _,
                        dtype: ArrowDataType::LargeList(inner_field),
                        is_nullable: _,
                        metadata: _,
                    }) if inner_field.name() == &LIST_VALUES_NAME => Some(inner_field),
                    Some(ArrowField { dtype, .. }) => polars_bail!(
                        SchemaMismatch:
                        "to_arrow(): list dtype mismatch: {:?} != expected: {:?}",
                        dtype, arr.dtype(),
                    ),
                    None => None,
                };

                let new_values =
                    self.array_to_arrow(arr.values().as_ref(), inner, expected_inner_output_field)?;

                let arr = ListArray::<i64>::new(
                    ArrowDataType::LargeList(Box::new(create_arrow_field(
                        LIST_VALUES_NAME,
                        inner.as_ref(),
                        new_values.dtype(),
                        self.compat_level,
                        opt_field_is_nullable(expected_inner_output_field),
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

                let expected_inner_output_field: Option<&ArrowField> = match output_arrow_field {
                    Some(
                        field @ ArrowField {
                            name: _,
                            dtype: ArrowDataType::FixedSizeList(inner_field, width),
                            is_nullable: _,
                            metadata: _,
                        },
                    ) if *width == arr.size() && inner_field.name() == &LIST_VALUES_NAME => {
                        Some(inner_field)
                    },
                    Some(ArrowField { dtype, .. }) => polars_bail!(
                        SchemaMismatch:
                        "to_arrow(): fixed-size list dtype mismatch: {:?} != expected: {:?}",
                        dtype, arr.dtype(),
                    ),
                    None => None,
                };

                let new_values =
                    self.array_to_arrow(arr.values().as_ref(), inner, expected_inner_output_field)?;

                let arr = FixedSizeListArray::new(
                    ArrowDataType::FixedSizeList(
                        Box::new(create_arrow_field(
                            LIST_VALUES_NAME,
                            inner.as_ref(),
                            new_values.dtype(),
                            self.compat_level,
                            opt_field_is_nullable(expected_inner_output_field),
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
            #[cfg(feature = "dtype-extension")]
            DataType::Extension(typ, storage_dtype) => {
                use arrow::datatypes::ExtensionType;

                let output_ext_name: PlSmallStr = typ.name().into();
                let output_ext_md: Option<PlSmallStr> =
                    typ.serialize_metadata().map(|md| md.into());

                let expected_inner_output_field: Option<ArrowField> = match output_arrow_field {
                    Some(
                        field @ ArrowField {
                            name: _,
                            dtype: ArrowDataType::Extension(ext_type),
                            is_nullable: _,
                            metadata: _,
                        },
                    ) if {
                        let ExtensionType {
                            name,
                            inner: _,
                            metadata,
                        } = ext_type.as_ref();

                        name == &output_ext_name
                            && metadata.as_ref().filter(|x| !x.is_empty())
                                == output_ext_md.as_ref().filter(|x| !x.is_empty())
                    } =>
                    {
                        let ExtensionType {
                            name,
                            inner,
                            metadata: _,
                        } = ext_type.as_ref();

                        Some(create_arrow_field(
                            name.clone(),
                            storage_dtype.as_ref(),
                            inner,
                            self.compat_level,
                            true,
                        ))
                    },
                    Some(ArrowField { dtype, .. }) => {
                        let expected_inner = self
                            .array_to_arrow(array.sliced(0, 0).as_ref(), storage_dtype, None)
                            .unwrap()
                            .dtype()
                            .clone();

                        let expected = ArrowDataType::Extension(Box::new(ExtensionType {
                            name: output_ext_name,
                            inner: expected_inner,
                            metadata: output_ext_md,
                        }));

                        polars_bail!(
                            SchemaMismatch:
                            "to_arrow(): extension dtype mismatch: {:?} != expected: {:?}",
                            dtype, expected,
                        )
                    },
                    None => None,
                };

                let mut arr = self.array_to_arrow(
                    array,
                    storage_dtype,
                    expected_inner_output_field.as_ref(),
                )?;

                *arr.dtype_mut() = ArrowDataType::Extension(Box::new(ExtensionType {
                    name: output_ext_name,
                    inner: arr.dtype().clone(),
                    metadata: output_ext_md,
                }));
                arr
            },
            _ => {
                assert!(!dtype.is_logical());
                array.to_boxed()
            },
        })
    }
}

fn create_arrow_field(
    name: PlSmallStr,
    dtype: &DataType,
    arrow_dtype: &ArrowDataType,
    compat_level: CompatLevel,
    is_nullable: bool,
) -> ArrowField {
    match (dtype, arrow_dtype) {
        #[cfg(feature = "dtype-categorical")]
        (DataType::Categorical(..) | DataType::Enum(..), ArrowDataType::Dictionary(_, _, _)) => {
            // Sets _PL_ metadata
            let mut out = dtype.to_arrow_field(name, compat_level);
            out.is_nullable = is_nullable;
            out
        },
        _ => ArrowField::new(name, arrow_dtype.clone(), is_nullable),
    }
}

fn opt_field_is_nullable(opt_field: Option<&ArrowField>) -> bool {
    opt_field.is_none_or(|x| x.is_nullable)
}
