use arrow::array::*;
use arrow::datatypes::{ArrowDataType, Field};

#[test]
fn primitive() {
    let data = vec![
        Some(vec![Some(1i32), Some(2), Some(3)]),
        Some(vec![None, None, None]),
        Some(vec![Some(4), None, Some(6)]),
    ];

    let mut list = MutableFixedSizeListArray::new(MutablePrimitiveArray::<i32>::new(), 3);
    list.try_extend(data).unwrap();
    let list: FixedSizeListArray = list.into();

    let a = list.value(0);
    let a = a.as_any().downcast_ref::<Int32Array>().unwrap();

    let expected = Int32Array::from(vec![Some(1i32), Some(2), Some(3)]);
    assert_eq!(a, &expected);

    let a = list.value(1);
    let a = a.as_any().downcast_ref::<Int32Array>().unwrap();

    let expected = Int32Array::from(vec![None, None, None]);
    assert_eq!(a, &expected)
}

#[test]
fn new_with_field() {
    let data = vec![
        Some(vec![Some(1i32), Some(2), Some(3)]),
        Some(vec![None, None, None]),
        Some(vec![Some(4), None, Some(6)]),
    ];

    let mut list = MutableFixedSizeListArray::new_with_field(
        MutablePrimitiveArray::<i32>::new(),
        "custom_items",
        false,
        3,
    );
    list.try_extend(data).unwrap();
    let list: FixedSizeListArray = list.into();

    assert_eq!(
        list.data_type(),
        &ArrowDataType::FixedSizeList(
            Box::new(Field::new("custom_items", ArrowDataType::Int32, false)),
            3
        )
    );

    let a = list.value(0);
    let a = a.as_any().downcast_ref::<Int32Array>().unwrap();

    let expected = Int32Array::from(vec![Some(1i32), Some(2), Some(3)]);
    assert_eq!(a, &expected);

    let a = list.value(1);
    let a = a.as_any().downcast_ref::<Int32Array>().unwrap();

    let expected = Int32Array::from(vec![None, None, None]);
    assert_eq!(a, &expected)
}

#[test]
fn extend_from_self() {
    let data = vec![
        Some(vec![Some(1i32), Some(2), Some(3)]),
        None,
        Some(vec![Some(4), None, Some(6)]),
    ];
    let mut a = MutableFixedSizeListArray::new(MutablePrimitiveArray::<i32>::new(), 3);
    a.try_extend(data.clone()).unwrap();

    a.try_extend_from_self(&a.clone()).unwrap();
    let a: FixedSizeListArray = a.into();

    let mut expected = data.clone();
    expected.extend(data);

    let mut b = MutableFixedSizeListArray::new(MutablePrimitiveArray::<i32>::new(), 3);
    b.try_extend(expected).unwrap();
    let b: FixedSizeListArray = b.into();

    assert_eq!(a, b);
}
