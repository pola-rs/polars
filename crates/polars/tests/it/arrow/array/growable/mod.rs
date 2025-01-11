mod binary;
mod boolean;
mod dictionary;
mod fixed_binary;
mod fixed_size_list;
mod list;
mod null;
mod primitive;
mod struct_;
mod utf8;

use arrow::array::growable::make_growable;
use arrow::array::*;
use arrow::datatypes::{ArrowDataType, ExtensionType, Field};

#[test]
fn test_make_growable() {
    let array = Int32Array::from_slice([1, 2]);
    make_growable(&[&array], false, 2);

    let array = BinaryArray::<i32>::from_slice([b"a".as_ref(), b"aa".as_ref()]);
    make_growable(&[&array], false, 2);

    let array = BinaryArray::<i64>::from_slice([b"a".as_ref(), b"aa".as_ref()]);
    make_growable(&[&array], false, 2);

    let array = BinaryArray::<i64>::from_slice([b"a".as_ref(), b"aa".as_ref()]);
    make_growable(&[&array], false, 2);

    let array = FixedSizeBinaryArray::new(
        ArrowDataType::FixedSizeBinary(2),
        b"abcd".to_vec().into(),
        None,
    );
    make_growable(&[&array], false, 2);
}

#[test]
fn test_make_growable_extension() {
    let array = DictionaryArray::try_from_keys(
        Int32Array::from_slice([1, 0]),
        Int32Array::from_slice([1, 2]).boxed(),
    )
    .unwrap();
    make_growable(&[&array], false, 2);

    let dtype = ArrowDataType::Extension(Box::new(ExtensionType {
        name: "ext".into(),
        inner: ArrowDataType::Int32,
        metadata: None,
    }));
    let array = Int32Array::from_slice([1, 2]).to(dtype.clone());
    let array_grown = make_growable(&[&array], false, 2).as_box();
    assert_eq!(array_grown.dtype(), &dtype);

    let dtype = ArrowDataType::Extension(Box::new(ExtensionType {
        name: "ext".into(),
        inner: ArrowDataType::Struct(vec![Field::new("a".into(), ArrowDataType::Int32, false)]),
        metadata: None,
    }));
    let array = StructArray::new(
        dtype.clone(),
        2,
        vec![Int32Array::from_slice([1, 2]).boxed()],
        None,
    );
    let array_grown = make_growable(&[&array], false, 2).as_box();
    assert_eq!(array_grown.dtype(), &dtype);
}
