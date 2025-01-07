mod binary;
mod binview;
mod boolean;
mod dictionary;
mod equal;
mod fixed_size_binary;
mod fixed_size_list;
mod growable;
mod list;
mod map;
mod primitive;
mod struct_;
mod union;
mod utf8;

use arrow::array::{clone, new_empty_array, new_null_array, Array, PrimitiveArray};
use arrow::bitmap::Bitmap;
use arrow::datatypes::{ArrowDataType, ExtensionType, Field, UnionMode};
use union::union_type;

#[test]
fn nulls() {
    let datatypes = vec![
        ArrowDataType::Int32,
        ArrowDataType::Float64,
        ArrowDataType::Utf8,
        ArrowDataType::Binary,
        ArrowDataType::List(Box::new(Field::new(
            "a".into(),
            ArrowDataType::Binary,
            true,
        ))),
    ];
    let a = datatypes
        .into_iter()
        .all(|x| new_null_array(x, 10).null_count() == 10);
    assert!(a);

    // unions' null count is always 0
    let datatypes = vec![
        union_type(
            vec![Field::new("a".into(), ArrowDataType::Binary, true)],
            None,
            UnionMode::Dense,
        ),
        union_type(
            vec![Field::new("a".into(), ArrowDataType::Binary, true)],
            None,
            UnionMode::Sparse,
        ),
    ];
    let a = datatypes
        .into_iter()
        .all(|x| new_null_array(x, 10).null_count() == 0);
    assert!(a);
}

#[test]
fn empty() {
    let datatypes = vec![
        ArrowDataType::Int32,
        ArrowDataType::Float64,
        ArrowDataType::Utf8,
        ArrowDataType::Binary,
        ArrowDataType::List(Box::new(Field::new(
            "a".into(),
            ArrowDataType::Binary,
            true,
        ))),
        ArrowDataType::List(Box::new(Field::new(
            "a".into(),
            ArrowDataType::Extension(Box::new(ExtensionType {
                name: "ext".into(),
                inner: ArrowDataType::Int32,
                metadata: None,
            })),
            true,
        ))),
        union_type(
            vec![Field::new("a".into(), ArrowDataType::Binary, true)],
            None,
            UnionMode::Sparse,
        ),
        union_type(
            vec![Field::new("a".into(), ArrowDataType::Binary, true)],
            None,
            UnionMode::Dense,
        ),
        ArrowDataType::Struct(vec![Field::new("a".into(), ArrowDataType::Int32, true)]),
    ];
    let a = datatypes.into_iter().all(|x| new_empty_array(x).len() == 0);
    assert!(a);
}

#[test]
fn empty_extension() {
    let datatypes = vec![
        ArrowDataType::Int32,
        ArrowDataType::Float64,
        ArrowDataType::Utf8,
        ArrowDataType::Binary,
        ArrowDataType::List(Box::new(Field::new(
            "a".into(),
            ArrowDataType::Binary,
            true,
        ))),
        union_type(
            vec![Field::new("a".into(), ArrowDataType::Binary, true)],
            None,
            UnionMode::Sparse,
        ),
        union_type(
            vec![Field::new("a".into(), ArrowDataType::Binary, true)],
            None,
            UnionMode::Dense,
        ),
        ArrowDataType::Struct(vec![Field::new("a".into(), ArrowDataType::Int32, true)]),
    ];
    let a = datatypes
        .into_iter()
        .map(|dt| {
            ArrowDataType::Extension(Box::new(ExtensionType {
                name: "ext".into(),
                inner: dt,
                metadata: None,
            }))
        })
        .all(|x| {
            let a = new_empty_array(x);
            a.len() == 0 && matches!(a.dtype(), ArrowDataType::Extension(_))
        });
    assert!(a);
}

#[test]
fn test_clone() {
    let datatypes = vec![
        ArrowDataType::Int32,
        ArrowDataType::Float64,
        ArrowDataType::Utf8,
        ArrowDataType::Binary,
        ArrowDataType::List(Box::new(Field::new(
            "a".into(),
            ArrowDataType::Binary,
            true,
        ))),
    ];
    let a = datatypes
        .into_iter()
        .all(|x| clone(new_null_array(x.clone(), 10).as_ref()) == new_null_array(x, 10));
    assert!(a);
}

#[test]
fn test_with_validity() {
    let arr = PrimitiveArray::from_slice([1i32, 2, 3]);
    let validity = Bitmap::from(&[true, false, true]);
    let arr = arr.with_validity(Some(validity));
    let arr_ref = arr.as_any().downcast_ref::<PrimitiveArray<i32>>().unwrap();

    let expected = PrimitiveArray::from(&[Some(1i32), None, Some(3)]);
    assert_eq!(arr_ref, &expected);
}

// check that we ca derive stuff
#[allow(dead_code)]
#[derive(PartialEq, Clone, Debug)]
struct A {
    array: Box<dyn Array>,
}
