use std::fmt::Debug;

use parquet2::page::Page;
use parquet2::schema::types::{ParquetType, PrimitiveType as ParquetPrimitiveType};
use parquet2::write::DynIter;
use polars_error::{polars_bail, PolarsResult};

use super::{array_to_pages, Encoding, WriteOptions};
use crate::array::{Array, ListArray, MapArray, StructArray};
use crate::bitmap::Bitmap;
use crate::datatypes::PhysicalType;
use crate::io::parquet::read::schema::is_nullable;
use crate::offset::{Offset, OffsetsBuffer};

#[derive(Debug, Clone, PartialEq)]
pub struct ListNested<O: Offset> {
    pub is_optional: bool,
    pub offsets: OffsetsBuffer<O>,
    pub validity: Option<Bitmap>,
}

impl<O: Offset> ListNested<O> {
    pub fn new(offsets: OffsetsBuffer<O>, validity: Option<Bitmap>, is_optional: bool) -> Self {
        Self {
            is_optional,
            offsets,
            validity,
        }
    }
}

/// Descriptor of nested information of a field
#[derive(Debug, Clone, PartialEq)]
pub enum Nested {
    /// a primitive (leaf or parquet column)
    /// bitmap, _, length
    Primitive(Option<Bitmap>, bool, usize),
    /// a list
    List(ListNested<i32>),
    /// a list
    LargeList(ListNested<i64>),
    /// a struct
    Struct(Option<Bitmap>, bool, usize),
}

impl Nested {
    /// Returns the length (number of rows) of the element
    pub fn len(&self) -> usize {
        match self {
            Nested::Primitive(_, _, length) => *length,
            Nested::List(nested) => nested.offsets.len_proxy(),
            Nested::LargeList(nested) => nested.offsets.len_proxy(),
            Nested::Struct(_, _, len) => *len,
        }
    }
}

/// Constructs the necessary `Vec<Vec<Nested>>` to write the rep and def levels of `array` to parquet
pub fn to_nested(array: &dyn Array, type_: &ParquetType) -> PolarsResult<Vec<Vec<Nested>>> {
    let mut nested = vec![];

    to_nested_recursive(array, type_, &mut nested, vec![])?;
    Ok(nested)
}

fn to_nested_recursive(
    array: &dyn Array,
    type_: &ParquetType,
    nested: &mut Vec<Vec<Nested>>,
    mut parents: Vec<Nested>,
) -> PolarsResult<()> {
    let is_optional = is_nullable(type_.get_field_info());

    use PhysicalType::*;
    match array.data_type().to_physical_type() {
        Struct => {
            let array = array.as_any().downcast_ref::<StructArray>().unwrap();
            let fields = if let ParquetType::GroupType { fields, .. } = type_ {
                fields
            } else {
                polars_bail!(InvalidOperation:
                    "Parquet type must be a group for a struct array".to_string(),
                )
            };

            parents.push(Nested::Struct(
                array.validity().cloned(),
                is_optional,
                array.len(),
            ));

            for (type_, array) in fields.iter().zip(array.values()) {
                to_nested_recursive(array.as_ref(), type_, nested, parents.clone())?;
            }
        },
        List => {
            let array = array.as_any().downcast_ref::<ListArray<i32>>().unwrap();
            let type_ = if let ParquetType::GroupType { fields, .. } = type_ {
                if let ParquetType::GroupType { fields, .. } = &fields[0] {
                    &fields[0]
                } else {
                    polars_bail!(InvalidOperation:
                        "Parquet type must be a group for a list array".to_string(),
                    )
                }
            } else {
                polars_bail!(InvalidOperation:
                    "Parquet type must be a group for a list array".to_string(),
                )
            };

            parents.push(Nested::List(ListNested::new(
                array.offsets().clone(),
                array.validity().cloned(),
                is_optional,
            )));
            to_nested_recursive(array.values().as_ref(), type_, nested, parents)?;
        },
        LargeList => {
            let array = array.as_any().downcast_ref::<ListArray<i64>>().unwrap();
            let type_ = if let ParquetType::GroupType { fields, .. } = type_ {
                if let ParquetType::GroupType { fields, .. } = &fields[0] {
                    &fields[0]
                } else {
                    polars_bail!(InvalidOperation:
                        "Parquet type must be a group for a list array".to_string(),
                    )
                }
            } else {
                polars_bail!(InvalidOperation:
                    "Parquet type must be a group for a list array".to_string(),
                )
            };

            parents.push(Nested::LargeList(ListNested::new(
                array.offsets().clone(),
                array.validity().cloned(),
                is_optional,
            )));
            to_nested_recursive(array.values().as_ref(), type_, nested, parents)?;
        },
        Map => {
            let array = array.as_any().downcast_ref::<MapArray>().unwrap();
            let type_ = if let ParquetType::GroupType { fields, .. } = type_ {
                if let ParquetType::GroupType { fields, .. } = &fields[0] {
                    &fields[0]
                } else {
                    polars_bail!(InvalidOperation:
                        "Parquet type must be a group for a map array".to_string(),
                    )
                }
            } else {
                polars_bail!(InvalidOperation:
                    "Parquet type must be a group for a map array".to_string(),
                )
            };

            parents.push(Nested::List(ListNested::new(
                array.offsets().clone(),
                array.validity().cloned(),
                is_optional,
            )));
            to_nested_recursive(array.field().as_ref(), type_, nested, parents)?;
        },
        _ => {
            parents.push(Nested::Primitive(
                array.validity().cloned(),
                is_optional,
                array.len(),
            ));
            nested.push(parents)
        },
    }
    Ok(())
}

/// Convert [`Array`] to `Vec<&dyn Array>` leaves in DFS order.
pub fn to_leaves(array: &dyn Array) -> Vec<&dyn Array> {
    let mut leaves = vec![];
    to_leaves_recursive(array, &mut leaves);
    leaves
}

fn to_leaves_recursive<'a>(array: &'a dyn Array, leaves: &mut Vec<&'a dyn Array>) {
    use PhysicalType::*;
    match array.data_type().to_physical_type() {
        Struct => {
            let array = array.as_any().downcast_ref::<StructArray>().unwrap();
            array
                .values()
                .iter()
                .for_each(|a| to_leaves_recursive(a.as_ref(), leaves));
        },
        List => {
            let array = array.as_any().downcast_ref::<ListArray<i32>>().unwrap();
            to_leaves_recursive(array.values().as_ref(), leaves);
        },
        LargeList => {
            let array = array.as_any().downcast_ref::<ListArray<i64>>().unwrap();
            to_leaves_recursive(array.values().as_ref(), leaves);
        },
        Map => {
            let array = array.as_any().downcast_ref::<MapArray>().unwrap();
            to_leaves_recursive(array.field().as_ref(), leaves);
        },
        Null | Boolean | Primitive(_) | Binary | FixedSizeBinary | LargeBinary | Utf8
        | LargeUtf8 | Dictionary(_) => leaves.push(array),
        other => todo!("writing {:?} to parquet not yet implemented", other),
    }
}

/// Convert `ParquetType` to `Vec<ParquetPrimitiveType>` leaves in DFS order.
pub fn to_parquet_leaves(type_: ParquetType) -> Vec<ParquetPrimitiveType> {
    let mut leaves = vec![];
    to_parquet_leaves_recursive(type_, &mut leaves);
    leaves
}

fn to_parquet_leaves_recursive(type_: ParquetType, leaves: &mut Vec<ParquetPrimitiveType>) {
    match type_ {
        ParquetType::PrimitiveType(primitive) => leaves.push(primitive),
        ParquetType::GroupType { fields, .. } => {
            fields
                .into_iter()
                .for_each(|type_| to_parquet_leaves_recursive(type_, leaves));
        },
    }
}

/// Returns a vector of iterators of [`Page`], one per leaf column in the array
pub fn array_to_columns<A: AsRef<dyn Array> + Send + Sync>(
    array: A,
    type_: ParquetType,
    options: WriteOptions,
    encoding: &[Encoding],
) -> PolarsResult<Vec<DynIter<'static, PolarsResult<Page>>>> {
    let array = array.as_ref();
    let nested = to_nested(array, &type_)?;

    let types = to_parquet_leaves(type_);

    let values = to_leaves(array);

    assert_eq!(encoding.len(), types.len());

    values
        .iter()
        .zip(nested)
        .zip(types)
        .zip(encoding.iter())
        .map(|(((values, nested), type_), encoding)| {
            array_to_pages(*values, type_, &nested, options, *encoding)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use parquet2::schema::types::{GroupLogicalType, PrimitiveConvertedType, PrimitiveLogicalType};
    use parquet2::schema::Repetition;

    use super::super::{FieldInfo, ParquetPhysicalType, ParquetPrimitiveType};
    use super::*;
    use crate::array::*;
    use crate::bitmap::Bitmap;
    use crate::datatypes::*;

    #[test]
    fn test_struct() {
        let boolean = BooleanArray::from_slice([false, false, true, true]).boxed();
        let int = Int32Array::from_slice([42, 28, 19, 31]).boxed();

        let fields = vec![
            Field::new("b", DataType::Boolean, false),
            Field::new("c", DataType::Int32, false),
        ];

        let array = StructArray::new(
            DataType::Struct(fields),
            vec![boolean.clone(), int.clone()],
            Some(Bitmap::from([true, true, false, true])),
        );

        let type_ = ParquetType::GroupType {
            field_info: FieldInfo {
                name: "a".to_string(),
                repetition: Repetition::Optional,
                id: None,
            },
            logical_type: None,
            converted_type: None,
            fields: vec![
                ParquetType::PrimitiveType(ParquetPrimitiveType {
                    field_info: FieldInfo {
                        name: "b".to_string(),
                        repetition: Repetition::Required,
                        id: None,
                    },
                    logical_type: None,
                    converted_type: None,
                    physical_type: ParquetPhysicalType::Boolean,
                }),
                ParquetType::PrimitiveType(ParquetPrimitiveType {
                    field_info: FieldInfo {
                        name: "c".to_string(),
                        repetition: Repetition::Required,
                        id: None,
                    },
                    logical_type: None,
                    converted_type: None,
                    physical_type: ParquetPhysicalType::Int32,
                }),
            ],
        };
        let a = to_nested(&array, &type_).unwrap();

        assert_eq!(
            a,
            vec![
                vec![
                    Nested::Struct(Some(Bitmap::from([true, true, false, true])), true, 4),
                    Nested::Primitive(None, false, 4),
                ],
                vec![
                    Nested::Struct(Some(Bitmap::from([true, true, false, true])), true, 4),
                    Nested::Primitive(None, false, 4),
                ],
            ]
        );
    }

    #[test]
    fn test_struct_struct() {
        let boolean = BooleanArray::from_slice([false, false, true, true]).boxed();
        let int = Int32Array::from_slice([42, 28, 19, 31]).boxed();

        let fields = vec![
            Field::new("b", DataType::Boolean, false),
            Field::new("c", DataType::Int32, false),
        ];

        let array = StructArray::new(
            DataType::Struct(fields),
            vec![boolean.clone(), int.clone()],
            Some(Bitmap::from([true, true, false, true])),
        );

        let fields = vec![
            Field::new("b", array.data_type().clone(), true),
            Field::new("c", array.data_type().clone(), true),
        ];

        let array = StructArray::new(
            DataType::Struct(fields),
            vec![Box::new(array.clone()), Box::new(array)],
            None,
        );

        let type_ = ParquetType::GroupType {
            field_info: FieldInfo {
                name: "a".to_string(),
                repetition: Repetition::Optional,
                id: None,
            },
            logical_type: None,
            converted_type: None,
            fields: vec![
                ParquetType::PrimitiveType(ParquetPrimitiveType {
                    field_info: FieldInfo {
                        name: "b".to_string(),
                        repetition: Repetition::Required,
                        id: None,
                    },
                    logical_type: None,
                    converted_type: None,
                    physical_type: ParquetPhysicalType::Boolean,
                }),
                ParquetType::PrimitiveType(ParquetPrimitiveType {
                    field_info: FieldInfo {
                        name: "c".to_string(),
                        repetition: Repetition::Required,
                        id: None,
                    },
                    logical_type: None,
                    converted_type: None,
                    physical_type: ParquetPhysicalType::Int32,
                }),
            ],
        };

        let type_ = ParquetType::GroupType {
            field_info: FieldInfo {
                name: "a".to_string(),
                repetition: Repetition::Required,
                id: None,
            },
            logical_type: None,
            converted_type: None,
            fields: vec![type_.clone(), type_],
        };

        let a = to_nested(&array, &type_).unwrap();

        assert_eq!(
            a,
            vec![
                // a.b.b
                vec![
                    Nested::Struct(None, false, 4),
                    Nested::Struct(Some(Bitmap::from([true, true, false, true])), true, 4),
                    Nested::Primitive(None, false, 4),
                ],
                // a.b.c
                vec![
                    Nested::Struct(None, false, 4),
                    Nested::Struct(Some(Bitmap::from([true, true, false, true])), true, 4),
                    Nested::Primitive(None, false, 4),
                ],
                // a.c.b
                vec![
                    Nested::Struct(None, false, 4),
                    Nested::Struct(Some(Bitmap::from([true, true, false, true])), true, 4),
                    Nested::Primitive(None, false, 4),
                ],
                // a.c.c
                vec![
                    Nested::Struct(None, false, 4),
                    Nested::Struct(Some(Bitmap::from([true, true, false, true])), true, 4),
                    Nested::Primitive(None, false, 4),
                ],
            ]
        );
    }

    #[test]
    fn test_list_struct() {
        let boolean = BooleanArray::from_slice([false, false, true, true]).boxed();
        let int = Int32Array::from_slice([42, 28, 19, 31]).boxed();

        let fields = vec![
            Field::new("b", DataType::Boolean, false),
            Field::new("c", DataType::Int32, false),
        ];

        let array = StructArray::new(
            DataType::Struct(fields),
            vec![boolean.clone(), int.clone()],
            Some(Bitmap::from([true, true, false, true])),
        );

        let array = ListArray::new(
            DataType::List(Box::new(Field::new("l", array.data_type().clone(), true))),
            vec![0i32, 2, 4].try_into().unwrap(),
            Box::new(array),
            None,
        );

        let type_ = ParquetType::GroupType {
            field_info: FieldInfo {
                name: "a".to_string(),
                repetition: Repetition::Optional,
                id: None,
            },
            logical_type: None,
            converted_type: None,
            fields: vec![
                ParquetType::PrimitiveType(ParquetPrimitiveType {
                    field_info: FieldInfo {
                        name: "b".to_string(),
                        repetition: Repetition::Required,
                        id: None,
                    },
                    logical_type: None,
                    converted_type: None,
                    physical_type: ParquetPhysicalType::Boolean,
                }),
                ParquetType::PrimitiveType(ParquetPrimitiveType {
                    field_info: FieldInfo {
                        name: "c".to_string(),
                        repetition: Repetition::Required,
                        id: None,
                    },
                    logical_type: None,
                    converted_type: None,
                    physical_type: ParquetPhysicalType::Int32,
                }),
            ],
        };

        let type_ = ParquetType::GroupType {
            field_info: FieldInfo {
                name: "l".to_string(),
                repetition: Repetition::Required,
                id: None,
            },
            logical_type: None,
            converted_type: None,
            fields: vec![ParquetType::GroupType {
                field_info: FieldInfo {
                    name: "list".to_string(),
                    repetition: Repetition::Repeated,
                    id: None,
                },
                logical_type: None,
                converted_type: None,
                fields: vec![type_],
            }],
        };

        let a = to_nested(&array, &type_).unwrap();

        assert_eq!(
            a,
            vec![
                vec![
                    Nested::List(ListNested::<i32> {
                        is_optional: false,
                        offsets: vec![0, 2, 4].try_into().unwrap(),
                        validity: None,
                    }),
                    Nested::Struct(Some(Bitmap::from([true, true, false, true])), true, 4),
                    Nested::Primitive(None, false, 4),
                ],
                vec![
                    Nested::List(ListNested::<i32> {
                        is_optional: false,
                        offsets: vec![0, 2, 4].try_into().unwrap(),
                        validity: None,
                    }),
                    Nested::Struct(Some(Bitmap::from([true, true, false, true])), true, 4),
                    Nested::Primitive(None, false, 4),
                ],
            ]
        );
    }

    #[test]
    fn test_map() {
        let kv_type = DataType::Struct(vec![
            Field::new("k", DataType::Utf8, false),
            Field::new("v", DataType::Int32, false),
        ]);
        let kv_field = Field::new("kv", kv_type.clone(), false);
        let map_type = DataType::Map(Box::new(kv_field), false);

        let key_array = Utf8Array::<i32>::from_slice(["k1", "k2", "k3", "k4", "k5", "k6"]).boxed();
        let val_array = Int32Array::from_slice([42, 28, 19, 31, 21, 17]).boxed();
        let kv_array = StructArray::try_new(kv_type, vec![key_array, val_array], None)
            .unwrap()
            .boxed();
        let offsets = OffsetsBuffer::try_from(vec![0, 2, 3, 4, 6]).unwrap();

        let array = MapArray::try_new(map_type, offsets, kv_array, None).unwrap();

        let type_ = ParquetType::GroupType {
            field_info: FieldInfo {
                name: "kv".to_string(),
                repetition: Repetition::Optional,
                id: None,
            },
            logical_type: None,
            converted_type: None,
            fields: vec![
                ParquetType::PrimitiveType(ParquetPrimitiveType {
                    field_info: FieldInfo {
                        name: "k".to_string(),
                        repetition: Repetition::Required,
                        id: None,
                    },
                    logical_type: Some(PrimitiveLogicalType::String),
                    converted_type: Some(PrimitiveConvertedType::Utf8),
                    physical_type: ParquetPhysicalType::ByteArray,
                }),
                ParquetType::PrimitiveType(ParquetPrimitiveType {
                    field_info: FieldInfo {
                        name: "v".to_string(),
                        repetition: Repetition::Required,
                        id: None,
                    },
                    logical_type: None,
                    converted_type: None,
                    physical_type: ParquetPhysicalType::Int32,
                }),
            ],
        };

        let type_ = ParquetType::GroupType {
            field_info: FieldInfo {
                name: "m".to_string(),
                repetition: Repetition::Required,
                id: None,
            },
            logical_type: Some(GroupLogicalType::Map),
            converted_type: None,
            fields: vec![ParquetType::GroupType {
                field_info: FieldInfo {
                    name: "map".to_string(),
                    repetition: Repetition::Repeated,
                    id: None,
                },
                logical_type: None,
                converted_type: None,
                fields: vec![type_],
            }],
        };

        let a = to_nested(&array, &type_).unwrap();

        assert_eq!(
            a,
            vec![
                vec![
                    Nested::List(ListNested::<i32> {
                        is_optional: false,
                        offsets: vec![0, 2, 3, 4, 6].try_into().unwrap(),
                        validity: None,
                    }),
                    Nested::Struct(None, true, 6),
                    Nested::Primitive(None, false, 6),
                ],
                vec![
                    Nested::List(ListNested::<i32> {
                        is_optional: false,
                        offsets: vec![0, 2, 3, 4, 6].try_into().unwrap(),
                        validity: None,
                    }),
                    Nested::Struct(None, true, 6),
                    Nested::Primitive(None, false, 6),
                ],
            ]
        );
    }
}
