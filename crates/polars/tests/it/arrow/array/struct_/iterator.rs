use arrow::array::*;
use arrow::datatypes::*;
use arrow::scalar::new_scalar;

#[test]
fn test_simple_iter() {
    let boolean = BooleanArray::from_slice([false, false, true, true]).boxed();
    let int = Int32Array::from_slice([42, 28, 19, 31]).boxed();

    let fields = vec![
        Field::new("b", ArrowDataType::Boolean, false),
        Field::new("c", ArrowDataType::Int32, false),
    ];

    let array = StructArray::new(
        ArrowDataType::Struct(fields),
        vec![boolean.clone(), int.clone()],
        None,
    );

    for (i, item) in array.iter().enumerate() {
        let expected = Some(vec![
            new_scalar(boolean.as_ref(), i),
            new_scalar(int.as_ref(), i),
        ]);
        assert_eq!(expected, item);
    }
}
