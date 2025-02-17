use arrow::array::{PrimitiveArray, StructArray};
use arrow::datatypes::{IntegerType, DTYPE_CATEGORICAL, DTYPE_ENUM_VALUES};
use ethnum::I256;
use polars_compute::cast::CastOptionsImpl;

use self::categorical::CategoricalDecoder;
use self::nested::deserialize::utils::freeze_validity;
use self::nested_utils::{NestedContent, PageNestedDecoder};
use self::primitive::{self};
use super::*;
use crate::parquet::error::ParquetResult;

pub fn columns_to_iter_recursive(
    mut columns: Vec<BasicDecompressor>,
    mut types: Vec<&PrimitiveType>,
    field: Field,
    mut init: Vec<InitNested>,
    filter: Option<Filter>,
) -> ParquetResult<(NestedState, Box<dyn Array>, Bitmap)> {
    use arrow::datatypes::PhysicalType::*;
    use arrow::datatypes::PrimitiveType::*;

    match field.dtype().to_physical_type() {
        Null => {
            // physical type is i32
            init.push(InitNested::Primitive(field.is_nullable));
            types.pop();
            PageNestedDecoder::new(
                columns.pop().unwrap(),
                field.dtype().clone(),
                null::NullDecoder,
                init,
            )?
            .collect_boxed(filter)
        },
        Boolean => {
            init.push(InitNested::Primitive(field.is_nullable));
            types.pop();
            PageNestedDecoder::new(
                columns.pop().unwrap(),
                ArrowDataType::Boolean,
                boolean::BooleanDecoder,
                init,
            )?
            .collect_boxed(filter)
        },
        Primitive(Int8) => {
            init.push(InitNested::Primitive(field.is_nullable));
            types.pop();
            PageNestedDecoder::new(
                columns.pop().unwrap(),
                field.dtype().clone(),
                primitive::IntDecoder::<i32, i8, _>::cast_as(),
                init,
            )?
            .collect_boxed(filter)
        },
        Primitive(Int16) => {
            init.push(InitNested::Primitive(field.is_nullable));
            types.pop();
            PageNestedDecoder::new(
                columns.pop().unwrap(),
                field.dtype().clone(),
                primitive::IntDecoder::<i32, i16, _>::cast_as(),
                init,
            )?
            .collect_boxed(filter)
        },
        Primitive(Int32) => {
            init.push(InitNested::Primitive(field.is_nullable));
            types.pop();
            PageNestedDecoder::new(
                columns.pop().unwrap(),
                field.dtype().clone(),
                primitive::IntDecoder::<i32, _, _>::unit(),
                init,
            )?
            .collect_boxed(filter)
        },
        Primitive(Int64) => {
            init.push(InitNested::Primitive(field.is_nullable));
            types.pop();
            PageNestedDecoder::new(
                columns.pop().unwrap(),
                field.dtype().clone(),
                primitive::IntDecoder::<i64, _, _>::unit(),
                init,
            )?
            .collect_boxed(filter)
        },
        Primitive(UInt8) => {
            init.push(InitNested::Primitive(field.is_nullable));
            types.pop();
            PageNestedDecoder::new(
                columns.pop().unwrap(),
                field.dtype().clone(),
                primitive::IntDecoder::<i32, u8, _>::cast_as(),
                init,
            )?
            .collect_boxed(filter)
        },
        Primitive(UInt16) => {
            init.push(InitNested::Primitive(field.is_nullable));
            types.pop();
            PageNestedDecoder::new(
                columns.pop().unwrap(),
                field.dtype().clone(),
                primitive::IntDecoder::<i32, u16, _>::cast_as(),
                init,
            )?
            .collect_boxed(filter)
        },
        Primitive(UInt32) => {
            init.push(InitNested::Primitive(field.is_nullable));
            let type_ = types.pop().unwrap();
            match type_.physical_type {
                PhysicalType::Int32 => PageNestedDecoder::new(
                    columns.pop().unwrap(),
                    field.dtype().clone(),
                    primitive::IntDecoder::<i32, u32, _>::cast_as(),
                    init,
                )?
                .collect_boxed(filter),
                // some implementations of parquet write arrow's u32 into i64.
                PhysicalType::Int64 => PageNestedDecoder::new(
                    columns.pop().unwrap(),
                    field.dtype().clone(),
                    primitive::IntDecoder::<i64, u32, _>::cast_as(),
                    init,
                )?
                .collect_boxed(filter),
                other => Err(ParquetError::not_supported(format!(
                    "deserializing UInt32 from {other:?}'s parquet"
                ))),
            }
        },
        Primitive(UInt64) => {
            init.push(InitNested::Primitive(field.is_nullable));
            types.pop();
            PageNestedDecoder::new(
                columns.pop().unwrap(),
                field.dtype().clone(),
                primitive::IntDecoder::<i64, u64, _>::cast_as(),
                init,
            )?
            .collect_boxed(filter)
        },
        Primitive(Float32) => {
            init.push(InitNested::Primitive(field.is_nullable));
            types.pop();
            PageNestedDecoder::new(
                columns.pop().unwrap(),
                field.dtype().clone(),
                primitive::FloatDecoder::<f32, _, _>::unit(),
                init,
            )?
            .collect_boxed(filter)
        },
        Primitive(Float64) => {
            init.push(InitNested::Primitive(field.is_nullable));
            types.pop();
            PageNestedDecoder::new(
                columns.pop().unwrap(),
                field.dtype().clone(),
                primitive::FloatDecoder::<f64, _, _>::unit(),
                init,
            )?
            .collect_boxed(filter)
        },
        dtype @ (BinaryView | Utf8View) => {
            init.push(InitNested::Primitive(field.is_nullable));
            types.pop();
            PageNestedDecoder::new(
                columns.pop().unwrap(),
                field.dtype().clone(),
                binview::BinViewDecoder {
                    is_string: matches!(dtype, Utf8View),
                },
                init,
            )?
            .collect_boxed(filter)
        },
        // These are all converted to View variants before.
        LargeBinary | LargeUtf8 | Binary | Utf8 => unreachable!(),
        _ => match field.dtype().to_logical_type() {
            ArrowDataType::Dictionary(key_type, value_type, _) => {
                // @note: this should only hit in two cases:
                // - polars enum's and categorical's
                // - int -> string which can be turned into categoricals
                assert!(matches!(value_type.as_ref(), ArrowDataType::Utf8View));

                init.push(InitNested::Primitive(field.is_nullable));

                if field.metadata.as_ref().is_none_or(|md| {
                    !md.contains_key(DTYPE_ENUM_VALUES) && !md.contains_key(DTYPE_CATEGORICAL)
                }) {
                    let (nested, arr, ptm) = PageNestedDecoder::new(
                        columns.pop().unwrap(),
                        ArrowDataType::Utf8View,
                        binview::BinViewDecoder::new_string(),
                        init,
                    )?
                    .collect(filter)?;

                    let arr = polars_compute::cast::cast(
                        arr.as_ref(),
                        field.dtype(),
                        CastOptionsImpl::default(),
                    )
                    .unwrap();

                    Ok((nested, arr, ptm))
                } else {
                    assert!(matches!(key_type, IntegerType::UInt32));

                    PageNestedDecoder::new(
                        columns.pop().unwrap(),
                        field.dtype().clone(),
                        CategoricalDecoder::new(),
                        init,
                    )?
                    .collect_boxed(filter)
                }
            },
            ArrowDataType::List(inner) | ArrowDataType::LargeList(inner) => {
                init.push(InitNested::List(field.is_nullable));
                let (mut nested, array, ptm) = columns_to_iter_recursive(
                    columns,
                    types,
                    inner.as_ref().clone(),
                    init,
                    filter,
                )?;
                let array = create_list(field.dtype().clone(), &mut nested, array);
                Ok((nested, array, ptm))
            },
            ArrowDataType::FixedSizeList(inner, width) => {
                init.push(InitNested::FixedSizeList(field.is_nullable, *width));
                let (mut nested, array, ptm) = columns_to_iter_recursive(
                    columns,
                    types,
                    inner.as_ref().clone(),
                    init,
                    filter,
                )?;
                let array = create_list(field.dtype().clone(), &mut nested, array);
                Ok((nested, array, ptm))
            },
            ArrowDataType::Decimal(_, _) => {
                init.push(InitNested::Primitive(field.is_nullable));
                let type_ = types.pop().unwrap();
                match type_.physical_type {
                    PhysicalType::Int32 => PageNestedDecoder::new(
                        columns.pop().unwrap(),
                        field.dtype.clone(),
                        primitive::IntDecoder::<i32, i128, _>::cast_into(),
                        init,
                    )?
                    .collect_boxed(filter),
                    PhysicalType::Int64 => PageNestedDecoder::new(
                        columns.pop().unwrap(),
                        field.dtype.clone(),
                        primitive::IntDecoder::<i64, i128, _>::cast_into(),
                        init,
                    )?
                    .collect_boxed(filter),
                    PhysicalType::FixedLenByteArray(n) if n > 16 => {
                        Err(ParquetError::not_supported(format!(
                            "Can't decode Decimal128 type from `FixedLenByteArray` of len {n}"
                        )))
                    },
                    PhysicalType::FixedLenByteArray(size) => {
                        let (nested, array, ptm) = PageNestedDecoder::new(
                            columns.pop().unwrap(),
                            ArrowDataType::FixedSizeBinary(size),
                            fixed_size_binary::BinaryDecoder { size },
                            init,
                        )?
                        .collect(filter)?;

                        // Convert the fixed length byte array to Decimal.
                        let values = array
                            .values()
                            .chunks_exact(size)
                            .map(|value: &[u8]| super::super::convert_i128(value, size))
                            .collect::<Vec<_>>();
                        let validity = array.validity().cloned();

                        let array: Box<dyn Array> = Box::new(PrimitiveArray::<i128>::try_new(
                            field.dtype.clone(),
                            values.into(),
                            validity,
                        )?);

                        Ok((nested, array, ptm))
                    },
                    _ => Err(ParquetError::not_supported(format!(
                        "Deserializing type for Decimal {:?} from parquet",
                        type_.physical_type
                    ))),
                }
            },
            ArrowDataType::Decimal256(_, _) => {
                init.push(InitNested::Primitive(field.is_nullable));
                let type_ = types.pop().unwrap();
                match type_.physical_type {
                    PhysicalType::Int32 => PageNestedDecoder::new(
                        columns.pop().unwrap(),
                        field.dtype.clone(),
                        primitive::IntDecoder::closure(|x: i32| i256(I256::new(x as i128))),
                        init,
                    )?
                    .collect_boxed(filter),
                    PhysicalType::Int64 => PageNestedDecoder::new(
                        columns.pop().unwrap(),
                        field.dtype.clone(),
                        primitive::IntDecoder::closure(|x: i64| i256(I256::new(x as i128))),
                        init,
                    )?
                    .collect_boxed(filter),
                    PhysicalType::FixedLenByteArray(size) if size <= 16 => {
                        let (nested, array, ptm) = PageNestedDecoder::new(
                            columns.pop().unwrap(),
                            ArrowDataType::FixedSizeBinary(size),
                            fixed_size_binary::BinaryDecoder { size },
                            init,
                        )?
                        .collect(filter)?;

                        // Convert the fixed length byte array to Decimal.
                        let values = array
                            .values()
                            .chunks_exact(size)
                            .map(|value| i256(I256::new(super::super::convert_i128(value, size))))
                            .collect::<Vec<_>>();
                        let validity = array.validity().cloned();

                        let array: Box<dyn Array> = Box::new(PrimitiveArray::<i256>::try_new(
                            field.dtype.clone(),
                            values.into(),
                            validity,
                        )?);

                        Ok((nested, array, ptm))
                    },

                    PhysicalType::FixedLenByteArray(size) if size <= 32 => {
                        let (nested, array, ptm) = PageNestedDecoder::new(
                            columns.pop().unwrap(),
                            ArrowDataType::FixedSizeBinary(size),
                            fixed_size_binary::BinaryDecoder { size },
                            init,
                        )?
                        .collect(filter)?;

                        // Convert the fixed length byte array to Decimal.
                        let values = array
                            .values()
                            .chunks_exact(size)
                            .map(super::super::convert_i256)
                            .collect::<Vec<_>>();
                        let validity = array.validity().cloned();

                        let array: Box<dyn Array> = Box::new(PrimitiveArray::<i256>::try_new(
                            field.dtype.clone(),
                            values.into(),
                            validity,
                        )?);

                        Ok((nested, array, ptm))
                    },
                    PhysicalType::FixedLenByteArray(n) => Err(ParquetError::not_supported(
                        format!("Can't decode Decimal256 type from `FixedLenByteArray` of len {n}"),
                    )),
                    _ => Err(ParquetError::not_supported(format!(
                        "Deserializing type for Decimal {:?} from parquet",
                        type_.physical_type
                    ))),
                }
            },
            ArrowDataType::Struct(fields) => {
                // This definitely does not support Filter predicate yet.
                assert!(!matches!(&filter, Some(Filter::Predicate(_))));

                // @NOTE:
                // We go back to front here, because we constantly split off the end of the array
                // to grab the relevant columns and types.
                //
                // Is this inefficient? Yes. Is this how we are going to do it for now? Yes.

                let Some(last_field) = fields.last() else {
                    return Err(ParquetError::not_supported("Struct has zero fields"));
                };

                let field_to_nested_array =
                    |mut init: Vec<InitNested>,
                     columns: &mut Vec<BasicDecompressor>,
                     types: &mut Vec<&PrimitiveType>,
                     struct_field: &Field| {
                        init.push(InitNested::Struct(field.is_nullable));
                        let n = n_columns(&struct_field.dtype);
                        let columns = columns.split_off(columns.len() - n);
                        let types = types.split_off(types.len() - n);

                        columns_to_iter_recursive(
                            columns,
                            types,
                            struct_field.clone(),
                            init,
                            filter.clone(),
                        )
                    };

                let (mut nested, last_array, _) =
                    field_to_nested_array(init.clone(), &mut columns, &mut types, last_field)?;
                debug_assert!(matches!(nested.last().unwrap(), NestedContent::Struct));
                let (length, _, struct_validity) = nested.pop().unwrap();

                let mut field_arrays = Vec::<Box<dyn Array>>::with_capacity(fields.len());
                field_arrays.push(last_array);

                for field in fields.iter().rev().skip(1) {
                    let (mut _nested, array, _) =
                        field_to_nested_array(init.clone(), &mut columns, &mut types, field)?;

                    #[cfg(debug_assertions)]
                    {
                        debug_assert!(matches!(_nested.last().unwrap(), NestedContent::Struct));
                        debug_assert_eq!(
                            _nested.pop().unwrap().2.and_then(freeze_validity),
                            struct_validity.clone().and_then(freeze_validity),
                        );
                    }

                    field_arrays.push(array);
                }

                field_arrays.reverse();
                let struct_validity = struct_validity.and_then(freeze_validity);

                Ok((
                    nested,
                    StructArray::new(
                        ArrowDataType::Struct(fields.clone()),
                        length,
                        field_arrays,
                        struct_validity,
                    )
                    .to_boxed(),
                    Bitmap::new(),
                ))
            },
            ArrowDataType::Map(inner, _) => {
                init.push(InitNested::List(field.is_nullable));
                let (mut nested, array, ptm) = columns_to_iter_recursive(
                    columns,
                    types,
                    inner.as_ref().clone(),
                    init,
                    filter,
                )?;
                let array = create_map(field.dtype().clone(), &mut nested, array);
                Ok((nested, array, ptm))
            },
            other => Err(ParquetError::not_supported(format!(
                "Deserializing type {other:?} from parquet"
            ))),
        },
    }
}
