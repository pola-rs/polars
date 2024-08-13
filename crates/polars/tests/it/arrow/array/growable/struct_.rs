use arrow::array::growable::{Growable, GrowableStruct};
use arrow::array::{Array, PrimitiveArray, StructArray, Utf8ViewArray};
use arrow::bitmap::Bitmap;
use arrow::datatypes::{ArrowDataType, Field};

fn some_values() -> (ArrowDataType, Vec<Box<dyn Array>>) {
    let strings: Box<dyn Array> = Box::new(Utf8ViewArray::from_slice([
        Some("a"),
        Some("aa"),
        None,
        Some("mark"),
        Some("doe"),
    ]));
    let ints: Box<dyn Array> = Box::new(PrimitiveArray::<i32>::from(&[
        Some(1),
        Some(2),
        Some(3),
        Some(4),
        Some(5),
    ]));
    let fields = vec![
        Field::new("f1", ArrowDataType::Utf8View, true),
        Field::new("f2", ArrowDataType::Int32, true),
    ];
    (ArrowDataType::Struct(fields), vec![strings, ints])
}

#[test]
fn basic() {
    let (fields, values) = some_values();

    let array = StructArray::new(fields.clone(), values.clone(), None);

    let mut a = GrowableStruct::new(vec![&array], false, 0);

    unsafe {
        a.extend(0, 1, 2);
    }
    assert_eq!(a.len(), 2);
    let result: StructArray = a.into();

    let expected = StructArray::new(
        fields,
        vec![values[0].sliced(1, 2), values[1].sliced(1, 2)],
        None,
    );
    assert_eq!(result, expected)
}

#[test]
fn offset() {
    let (fields, values) = some_values();

    let array = StructArray::new(fields.clone(), values.clone(), None).sliced(1, 3);

    let mut a = GrowableStruct::new(vec![&array], false, 0);

    unsafe {
        a.extend(0, 1, 2);
    }
    assert_eq!(a.len(), 2);
    let result: StructArray = a.into();

    let expected = StructArray::new(
        fields,
        vec![values[0].sliced(2, 2), values[1].sliced(2, 2)],
        None,
    );

    assert_eq!(result, expected);
}

#[test]
fn nulls() {
    let (fields, values) = some_values();

    let array = StructArray::new(
        fields.clone(),
        values.clone(),
        Some(Bitmap::from_u8_slice([0b00000010], 5)),
    );

    let mut a = GrowableStruct::new(vec![&array], false, 0);

    unsafe {
        a.extend(0, 1, 2);
    }
    assert_eq!(a.len(), 2);
    let result: StructArray = a.into();

    let expected = StructArray::new(
        fields,
        vec![values[0].sliced(1, 2), values[1].sliced(1, 2)],
        Some(Bitmap::from_u8_slice([0b00000010], 5).sliced(1, 2)),
    );

    assert_eq!(result, expected)
}

#[test]
fn many() {
    let (fields, values) = some_values();

    let array = StructArray::new(fields.clone(), values.clone(), None);

    let mut mutable = GrowableStruct::new(vec![&array, &array], true, 0);

    unsafe {
        mutable.extend(0, 1, 2);
    }
    unsafe {
        mutable.extend(1, 0, 2);
    }
    mutable.extend_validity(1);
    assert_eq!(mutable.len(), 5);
    let result = mutable.as_box();

    let expected_string: Box<dyn Array> = Box::new(Utf8ViewArray::from_slice([
        Some("aa"),
        None,
        Some("a"),
        Some("aa"),
        None,
    ]));
    let expected_int: Box<dyn Array> = Box::new(PrimitiveArray::<i32>::from(vec![
        Some(2),
        Some(3),
        Some(1),
        Some(2),
        None,
    ]));

    let expected = StructArray::new(
        fields,
        vec![expected_string, expected_int],
        Some(Bitmap::from([true, true, true, true, false])),
    );
    assert_eq!(expected, result.as_ref())
}
