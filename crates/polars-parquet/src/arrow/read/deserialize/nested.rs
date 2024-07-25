use arrow::array::{FixedSizeBinaryArray, PrimitiveArray, StructArray};
use arrow::match_integer_type;
use ethnum::I256;
use polars_error::polars_bail;

use self::nested_utils::PageNestedDecoder;
use self::primitive::{AsDecoderFunction, IntoDecoderFunction, UnitDecoderFunction};
use super::*;

/// Converts an iterator of arrays to a trait object returning trait objects
#[inline]
fn primitive<'a, A, I>(iter: I) -> NestedArrayIter<'a>
where
    A: Array,
    I: Iterator<Item = PolarsResult<(NestedState, A)>> + Send + Sync + 'a,
{
    Box::new(iter.map(|x| {
        x.map(|(mut nested, array)| {
            let _ = nested.pop().unwrap(); // the primitive
            (nested, Box::new(array) as _)
        })
    }))
}

pub fn columns_to_iter_recursive<I>(
    mut columns: Vec<BasicDecompressor<I>>,
    mut types: Vec<&PrimitiveType>,
    field: Field,
    mut init: Vec<InitNested>,
    num_rows: usize,
) -> PolarsResult<(NestedState, Box<dyn Array>)>
where
    I: CompressedPagesIter,
{
    use arrow::datatypes::PhysicalType::*;
    use arrow::datatypes::PrimitiveType::*;

    let chunk_size = Some(num_rows);

    Ok(match field.data_type().to_physical_type() {
        Null => {
            // physical type is i32
            init.push(InitNested::Primitive(field.is_nullable));
            types.pop();
            PageNestedDecoder::new(
                columns.pop().unwrap(),
                field.data_type().clone(),
                null::NullDecoder,
                init,
            )?
            .collect_n(num_rows)?
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
            .collect_n(num_rows)?
        },
        Primitive(Int8) => {
            init.push(InitNested::Primitive(field.is_nullable));
            types.pop();
            PageNestedDecoder::new(
                columns.pop().unwrap(),
                field.data_type().clone(),
                primitive::PrimitiveDecoder::new(AsDecoderFunction::<i32, i8>::default()),
                init,
            )?
            .collect_n(num_rows)?
        },
        Primitive(Int16) => {
            init.push(InitNested::Primitive(field.is_nullable));
            types.pop();
            PageNestedDecoder::new(
                columns.pop().unwrap(),
                field.data_type().clone(),
                primitive::PrimitiveDecoder::new(AsDecoderFunction::<i32, i16>::default()),
                init,
            )?
            .collect_n(num_rows)?
        },
        Primitive(Int32) => {
            init.push(InitNested::Primitive(field.is_nullable));
            types.pop();
            PageNestedDecoder::new(
                columns.pop().unwrap(),
                field.data_type().clone(),
                primitive::PrimitiveDecoder::new(UnitDecoderFunction::<i32>::default()),
                init,
            )?
            .collect_n(num_rows)?
        },
        Primitive(Int64) => {
            init.push(InitNested::Primitive(field.is_nullable));
            types.pop();
            PageNestedDecoder::new(
                columns.pop().unwrap(),
                field.data_type().clone(),
                primitive::PrimitiveDecoder::new(UnitDecoderFunction::<i64>::default()),
                init,
            )?
            .collect_n(num_rows)?
        },
        Primitive(UInt8) => {
            init.push(InitNested::Primitive(field.is_nullable));
            types.pop();
            PageNestedDecoder::new(
                columns.pop().unwrap(),
                field.data_type().clone(),
                primitive::PrimitiveDecoder::new(AsDecoderFunction::<i32, u8>::default()),
                init,
            )?
            .collect_n(num_rows)?
        },
        Primitive(UInt16) => {
            init.push(InitNested::Primitive(field.is_nullable));
            types.pop();
            PageNestedDecoder::new(
                columns.pop().unwrap(),
                field.data_type().clone(),
                primitive::PrimitiveDecoder::new(AsDecoderFunction::<i32, u16>::default()),
                init,
            )?
            .collect_n(num_rows)?
        },
        Primitive(UInt32) => {
            init.push(InitNested::Primitive(field.is_nullable));
            let type_ = types.pop().unwrap();
            match type_.physical_type {
                PhysicalType::Int32 => PageNestedDecoder::new(
                    columns.pop().unwrap(),
                    field.data_type().clone(),
                    primitive::PrimitiveDecoder::new(AsDecoderFunction::<i32, u32>::default()),
                    init,
                )?
                .collect_n(num_rows)?,
                // some implementations of parquet write arrow's u32 into i64.
                PhysicalType::Int64 => PageNestedDecoder::new(
                    columns.pop().unwrap(),
                    field.data_type().clone(),
                    primitive::PrimitiveDecoder::new(AsDecoderFunction::<i64, u32>::default()),
                    init,
                )?
                .collect_n(num_rows)?,
                other => {
                    polars_bail!(ComputeError:
                        "deserializing UInt32 from {other:?}'s parquet"
                    )
                },
            }
        },
        Primitive(UInt64) => {
            init.push(InitNested::Primitive(field.is_nullable));
            types.pop();
            PageNestedDecoder::new(
                columns.pop().unwrap(),
                field.data_type().clone(),
                primitive::PrimitiveDecoder::new(AsDecoderFunction::<i64, u64>::default()),
                init,
            )?
            .collect_n(num_rows)?
        },
        Primitive(Float32) => {
            init.push(InitNested::Primitive(field.is_nullable));
            types.pop();
            PageNestedDecoder::new(
                columns.pop().unwrap(),
                field.data_type().clone(),
                primitive::PrimitiveDecoder::new(UnitDecoderFunction::<f32>::default()),
                init,
            )?
            .collect_n(num_rows)?
        },
        Primitive(Float64) => {
            init.push(InitNested::Primitive(field.is_nullable));
            types.pop();
            PageNestedDecoder::new(
                columns.pop().unwrap(),
                field.data_type().clone(),
                primitive::PrimitiveDecoder::new(UnitDecoderFunction::<f64>::default()),
                init,
            )?
            .collect_n(num_rows)?
        },
        BinaryView | Utf8View => {
            init.push(InitNested::Primitive(field.is_nullable));
            types.pop();
            PageNestedDecoder::new(
                columns.pop().unwrap(),
                field.data_type().clone(),
                binview::BinViewDecoder::default(),
                init,
            )?
            .collect_n(num_rows)?
        },
        LargeBinary | LargeUtf8 => {
            init.push(InitNested::Primitive(field.is_nullable));
            types.pop();
            PageNestedDecoder::new(
                columns.pop().unwrap(),
                field.data_type().clone(),
                binary::BinaryDecoder::<i64>::default(),
                init,
            )?
            .collect_n(num_rows)?
        },
        _ => match field.data_type().to_logical_type() {
            ArrowDataType::Dictionary(key_type, _, _) => {
                init.push(InitNested::Primitive(field.is_nullable));
                let type_ = types.pop().unwrap();
                let iter = columns.pop().unwrap();
                let data_type = field.data_type().clone();
                match_integer_type!(key_type, |$K| {
                    dict_read::<$K, _>(iter, init, type_, data_type, num_rows, chunk_size)
                })?
                .next()
                .unwrap()?
            },
            ArrowDataType::List(inner) | ArrowDataType::LargeList(inner) => {
                init.push(InitNested::List(field.is_nullable));
                let (mut nested, array) = columns_to_iter_recursive(
                    columns,
                    types,
                    inner.as_ref().clone(),
                    init,
                    num_rows,
                )?;
                let array = create_list(field.data_type().clone(), &mut nested, array);
                (nested, array)
            },
            ArrowDataType::FixedSizeList(inner, width) => {
                init.push(InitNested::FixedSizeList(field.is_nullable, *width));
                let (mut nested, array) = columns_to_iter_recursive(
                    columns,
                    types,
                    inner.as_ref().clone(),
                    init,
                    num_rows,
                )?;
                let array = create_list(field.data_type().clone(), &mut nested, array);
                (nested, array)
            },
            ArrowDataType::Decimal(_, _) => {
                init.push(InitNested::Primitive(field.is_nullable));
                let type_ = types.pop().unwrap();
                match type_.physical_type {
                    PhysicalType::Int32 => PageNestedDecoder::new(
                        columns.pop().unwrap(),
                        field.data_type.clone(),
                        primitive::PrimitiveDecoder::new(
                            IntoDecoderFunction::<i32, i128>::default(),
                        ),
                        init,
                    )?
                    .collect_n(num_rows)?,
                    PhysicalType::Int64 => PageNestedDecoder::new(
                        columns.pop().unwrap(),
                        field.data_type.clone(),
                        primitive::PrimitiveDecoder::new(
                            IntoDecoderFunction::<i64, i128>::default(),
                        ),
                        init,
                    )?
                    .collect_n(num_rows)?,
                    PhysicalType::FixedLenByteArray(n) if n > 16 => {
                        polars_bail!(
                            ComputeError: "Can't decode Decimal128 type from `FixedLenByteArray` of len {n}"
                        )
                    },
                    PhysicalType::FixedLenByteArray(size) => {
                        let (mut nested, array) = PageNestedDecoder::new(
                            columns.pop().unwrap(),
                            field.data_type().clone(),
                            fixed_size_binary::BinaryDecoder { size },
                            init,
                        )?
                        .collect_n(num_rows)?;

                        let array = array
                            .as_any()
                            .downcast_ref::<FixedSizeBinaryArray>()
                            .unwrap();

                        // Convert the fixed length byte array to Decimal.
                        let values = array
                            .values()
                            .chunks_exact(size)
                            .map(|value: &[u8]| super::super::convert_i128(value, size))
                            .collect::<Vec<_>>();
                        let validity = array.validity().cloned();

                        let array: Box<dyn Array> = Box::new(PrimitiveArray::<i128>::try_new(
                            field.data_type.clone(),
                            values.into(),
                            validity,
                        )?);

                        // @TODO: I am pretty sure this does not work
                        let _ = nested.pop().unwrap(); // the primitive

                        (nested, array)
                    },
                    _ => {
                        polars_bail!(ComputeError:
                            "Deserializing type for Decimal {:?} from parquet",
                            type_.physical_type
                        )
                    },
                }
            },
            ArrowDataType::Decimal256(_, _) => {
                init.push(InitNested::Primitive(field.is_nullable));
                let type_ = types.pop().unwrap();
                match type_.physical_type {
                    PhysicalType::Int32 => PageNestedDecoder::new(
                        columns.pop().unwrap(),
                        field.data_type.clone(),
                        primitive::PrimitiveDecoder::new(
                            decoder_fn!((x) => <i32, i256> => i256(I256::new(x as i128))),
                        ),
                        init,
                    )?
                    .collect_n(num_rows)?,
                    PhysicalType::Int64 => PageNestedDecoder::new(
                        columns.pop().unwrap(),
                        field.data_type.clone(),
                        primitive::PrimitiveDecoder::new(
                            decoder_fn!((x) => <i64, i256> => i256(I256::new(x as i128))),
                        ),
                        init,
                    )?
                    .collect_n(num_rows)?,
                    PhysicalType::FixedLenByteArray(size) if size <= 16 => {
                        let (mut nested, array) = PageNestedDecoder::new(
                            columns.pop().unwrap(),
                            field.data_type().clone(),
                            fixed_size_binary::BinaryDecoder { size },
                            init,
                        )?
                        .collect_n(num_rows)?;

                        let array = array
                            .as_any()
                            .downcast_ref::<FixedSizeBinaryArray>()
                            .unwrap();

                        // Convert the fixed length byte array to Decimal.
                        let values = array
                            .values()
                            .chunks_exact(size)
                            .map(|value| i256(I256::new(super::super::convert_i128(value, size))))
                            .collect::<Vec<_>>();
                        let validity = array.validity().cloned();

                        let array: Box<dyn Array> = Box::new(PrimitiveArray::<i256>::try_new(
                            field.data_type.clone(),
                            values.into(),
                            validity,
                        )?);

                        // @TODO: I am pretty sure this is not needed
                        let _ = nested.pop().unwrap(); // the primitive

                        (nested, array)
                    },

                    PhysicalType::FixedLenByteArray(size) if size <= 32 => {
                        let (mut nested, array) = PageNestedDecoder::new(
                            columns.pop().unwrap(),
                            field.data_type().clone(),
                            fixed_size_binary::BinaryDecoder { size },
                            init,
                        )?
                        .collect_n(num_rows)?;

                        let array = array
                            .as_any()
                            .downcast_ref::<FixedSizeBinaryArray>()
                            .unwrap();

                        // Convert the fixed length byte array to Decimal.
                        let values = array
                            .values()
                            .chunks_exact(size)
                            .map(super::super::convert_i256)
                            .collect::<Vec<_>>();
                        let validity = array.validity().cloned();

                        let array: Box<dyn Array> = Box::new(PrimitiveArray::<i256>::try_new(
                            field.data_type.clone(),
                            values.into(),
                            validity,
                        )?);

                        // @TODO: I am pretty sure this is not needed
                        let _ = nested.pop().unwrap(); // the primitive

                        (nested, array)
                    },
                    PhysicalType::FixedLenByteArray(n) => {
                        polars_bail!(ComputeError:
                            "Can't decode Decimal256 type from `FixedLenByteArray` of len {n}"
                        )
                    },
                    _ => {
                        polars_bail!(ComputeError:
                            "Deserializing type for Decimal {:?} from parquet",
                            type_.physical_type
                        )
                    },
                }
            },
            ArrowDataType::Struct(fields) => {
                let columns = fields
                    .iter()
                    .rev()
                    .map(|f| {
                        let mut init = init.clone();
                        init.push(InitNested::Struct(field.is_nullable));
                        let n = n_columns(&f.data_type);
                        let columns = columns.drain(columns.len() - n..).collect();
                        let types = types.drain(types.len() - n..).collect();
                        columns_to_iter_recursive(columns, types, f.clone(), init, num_rows)
                    })
                    .collect::<PolarsResult<Vec<(NestedState, Box<dyn Array>)>>>()?;

                // @TODO: This is overcomplicated

                let mut nested = vec![];
                let mut new_values = vec![];
                for (nest, values) in columns.into_iter().rev() {
                    new_values.push(values);
                    nested.push(nest);
                }

                let mut nested = nested.pop().unwrap();
                let (_, validity) = nested.pop().unwrap();

                (
                    nested,
                    Box::new(StructArray::new(
                        ArrowDataType::Struct(fields.clone()),
                        new_values,
                        validity.and_then(|x| x.into()),
                    )),
                )
            },
            ArrowDataType::Map(inner, _) => {
                init.push(InitNested::List(field.is_nullable));
                let (mut nested, array) = columns_to_iter_recursive(
                    columns,
                    types,
                    inner.as_ref().clone(),
                    init,
                    num_rows,
                )?;
                let array = create_map(field.data_type().clone(), &mut nested, array);
                (nested, array)
            },
            other => {
                polars_bail!(ComputeError:
                    "Deserializing type {other:?} from parquet"
                )
            },
        },
    })
}

fn dict_read<'a, K: DictionaryKey, I: 'a + CompressedPagesIter>(
    iter: BasicDecompressor<I>,
    init: Vec<InitNested>,
    _type_: &PrimitiveType,
    data_type: ArrowDataType,
    num_rows: usize,
    chunk_size: Option<usize>,
) -> PolarsResult<NestedArrayIter<'a>> {
    use ArrowDataType::*;
    let values_data_type = if let Dictionary(_, v, _) = &data_type {
        v.as_ref()
    } else {
        panic!()
    };

    Ok(match values_data_type.to_logical_type() {
        UInt8 => primitive(primitive::NestedDictIter::<K, _, _, _, _>::new(
            iter,
            init,
            data_type,
            num_rows,
            chunk_size,
            AsDecoderFunction::<i32, u8>::default(),
        )),
        UInt16 => primitive(primitive::NestedDictIter::<K, _, _, _, _>::new(
            iter,
            init,
            data_type,
            num_rows,
            chunk_size,
            AsDecoderFunction::<i32, u16>::default(),
        )),
        UInt32 => primitive(primitive::NestedDictIter::<K, _, _, _, _>::new(
            iter,
            init,
            data_type,
            num_rows,
            chunk_size,
            AsDecoderFunction::<i32, u32>::default(),
        )),
        Int8 => primitive(primitive::NestedDictIter::<K, _, _, _, _>::new(
            iter,
            init,
            data_type,
            num_rows,
            chunk_size,
            AsDecoderFunction::<i32, i8>::default(),
        )),
        Int16 => primitive(primitive::NestedDictIter::<K, _, _, _, _>::new(
            iter,
            init,
            data_type,
            num_rows,
            chunk_size,
            AsDecoderFunction::<i32, i16>::default(),
        )),
        Int32 | Date32 | Time32(_) | Interval(IntervalUnit::YearMonth) => {
            primitive(primitive::NestedDictIter::<K, _, _, _, _>::new(
                iter,
                init,
                data_type,
                num_rows,
                chunk_size,
                UnitDecoderFunction::<i32>::default(),
            ))
        },
        Int64 | Date64 | Time64(_) | Duration(_) => {
            primitive(primitive::NestedDictIter::<K, _, _, _, _>::new(
                iter,
                init,
                data_type,
                num_rows,
                chunk_size,
                AsDecoderFunction::<i64, i32>::default(),
            ))
        },
        Float32 => primitive(primitive::NestedDictIter::<K, _, _, _, _>::new(
            iter,
            init,
            data_type,
            num_rows,
            chunk_size,
            UnitDecoderFunction::<f32>::default(),
        )),
        Float64 => primitive(primitive::NestedDictIter::<K, _, _, _, _>::new(
            iter,
            init,
            data_type,
            num_rows,
            chunk_size,
            UnitDecoderFunction::<f64>::default(),
        )),
        LargeUtf8 | LargeBinary => primitive(binary::NestedDictIter::<K, i64, _>::new(
            iter, init, data_type, num_rows, chunk_size,
        )),
        Utf8View | BinaryView => primitive(binview::NestedDictIter::<K, _>::new(
            iter, init, data_type, num_rows, chunk_size,
        )),
        FixedSizeBinary(_) => primitive(fixed_size_binary::NestedDictIter::<K, _>::new(
            iter, init, data_type, num_rows, chunk_size,
        )),
        /*

        Timestamp(time_unit, _) => {
            let time_unit = *time_unit;
            return timestamp_dict::<K, _>(
                iter,
                physical_type,
                logical_type,
                data_type,
                chunk_size,
                time_unit,
            );
        }
         */
        other => {
            polars_bail!(ComputeError:
                "Reading nested dictionaries of type {other:?}"
            )
        },
    })
}
