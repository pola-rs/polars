use arrow::array::*;
use arrow::buffer::Buffer;
use arrow::datatypes::*;
use arrow::scalar::{new_scalar, PrimitiveScalar, Scalar, UnionScalar, Utf8Scalar};
use polars_error::PolarsResult;

fn next_unwrap<T, I>(iter: &mut I) -> T
where
    I: Iterator<Item = Box<dyn Scalar>>,
    T: Clone + 'static,
{
    iter.next()
        .unwrap()
        .as_any()
        .downcast_ref::<T>()
        .unwrap()
        .clone()
}

#[test]
fn sparse_debug() -> PolarsResult<()> {
    let fields = vec![
        Field::new("a", ArrowDataType::Int32, true),
        Field::new("b", ArrowDataType::Utf8, true),
    ];
    let data_type = ArrowDataType::Union(fields, None, UnionMode::Sparse);
    let types = vec![0, 0, 1].into();
    let fields = vec![
        Int32Array::from(&[Some(1), None, Some(2)]).boxed(),
        Utf8Array::<i32>::from([Some("a"), Some("b"), Some("c")]).boxed(),
    ];

    let array = UnionArray::new(data_type, types, fields, None);

    assert_eq!(format!("{array:?}"), "UnionArray[1, None, c]");

    Ok(())
}

#[test]
fn dense_debug() -> PolarsResult<()> {
    let fields = vec![
        Field::new("a", ArrowDataType::Int32, true),
        Field::new("b", ArrowDataType::Utf8, true),
    ];
    let data_type = ArrowDataType::Union(fields, None, UnionMode::Dense);
    let types = vec![0, 0, 1].into();
    let fields = vec![
        Int32Array::from(&[Some(1), None, Some(2)]).boxed(),
        Utf8Array::<i32>::from([Some("c")]).boxed(),
    ];
    let offsets = Some(vec![0, 1, 0].into());

    let array = UnionArray::new(data_type, types, fields, offsets);

    assert_eq!(format!("{array:?}"), "UnionArray[1, None, c]");

    Ok(())
}

#[test]
fn slice() -> PolarsResult<()> {
    let fields = vec![
        Field::new("a", ArrowDataType::Int32, true),
        Field::new("b", ArrowDataType::LargeUtf8, true),
    ];
    let data_type = ArrowDataType::Union(fields, None, UnionMode::Sparse);
    let types = Buffer::from(vec![0, 0, 1]);
    let fields = vec![
        Int32Array::from(&[Some(1), None, Some(2)]).boxed(),
        Utf8Array::<i64>::from([Some("a"), Some("b"), Some("c")]).boxed(),
    ];

    let array = UnionArray::new(data_type.clone(), types, fields.clone(), None);

    let result = array.sliced(1, 2);

    let sliced_types = Buffer::from(vec![0, 1]);
    let sliced_fields = vec![
        Int32Array::from(&[None, Some(2)]).boxed(),
        Utf8Array::<i64>::from([Some("b"), Some("c")]).boxed(),
    ];
    let expected = UnionArray::new(data_type, sliced_types, sliced_fields, None);

    assert_eq!(expected, result);
    Ok(())
}

#[test]
fn iter_sparse() -> PolarsResult<()> {
    let fields = vec![
        Field::new("a", ArrowDataType::Int32, true),
        Field::new("b", ArrowDataType::Utf8, true),
    ];
    let data_type = ArrowDataType::Union(fields, None, UnionMode::Sparse);
    let types = Buffer::from(vec![0, 0, 1]);
    let fields = vec![
        Int32Array::from(&[Some(1), None, Some(2)]).boxed(),
        Utf8Array::<i32>::from([Some("a"), Some("b"), Some("c")]).boxed(),
    ];

    let array = UnionArray::new(data_type, types, fields.clone(), None);
    let mut iter = array.iter();

    assert_eq!(
        next_unwrap::<PrimitiveScalar<i32>, _>(&mut iter).value(),
        &Some(1)
    );
    assert_eq!(
        next_unwrap::<PrimitiveScalar<i32>, _>(&mut iter).value(),
        &None
    );
    assert_eq!(
        next_unwrap::<Utf8Scalar<i32>, _>(&mut iter).value(),
        Some("c")
    );
    assert_eq!(iter.next(), None);

    Ok(())
}

#[test]
fn iter_dense() -> PolarsResult<()> {
    let fields = vec![
        Field::new("a", ArrowDataType::Int32, true),
        Field::new("b", ArrowDataType::Utf8, true),
    ];
    let data_type = ArrowDataType::Union(fields, None, UnionMode::Dense);
    let types = Buffer::from(vec![0, 0, 1]);
    let offsets = Buffer::<i32>::from(vec![0, 1, 0]);
    let fields = vec![
        Int32Array::from(&[Some(1), None]).boxed(),
        Utf8Array::<i32>::from([Some("c")]).boxed(),
    ];

    let array = UnionArray::new(data_type, types, fields.clone(), Some(offsets));
    let mut iter = array.iter();

    assert_eq!(
        next_unwrap::<PrimitiveScalar<i32>, _>(&mut iter).value(),
        &Some(1)
    );
    assert_eq!(
        next_unwrap::<PrimitiveScalar<i32>, _>(&mut iter).value(),
        &None
    );
    assert_eq!(
        next_unwrap::<Utf8Scalar<i32>, _>(&mut iter).value(),
        Some("c")
    );
    assert_eq!(iter.next(), None);

    Ok(())
}

#[test]
fn iter_sparse_slice() -> PolarsResult<()> {
    let fields = vec![
        Field::new("a", ArrowDataType::Int32, true),
        Field::new("b", ArrowDataType::Utf8, true),
    ];
    let data_type = ArrowDataType::Union(fields, None, UnionMode::Sparse);
    let types = Buffer::from(vec![0, 0, 1]);
    let fields = vec![
        Int32Array::from(&[Some(1), Some(3), Some(2)]).boxed(),
        Utf8Array::<i32>::from([Some("a"), Some("b"), Some("c")]).boxed(),
    ];

    let array = UnionArray::new(data_type, types, fields.clone(), None);
    let array_slice = array.sliced(1, 1);
    let mut iter = array_slice.iter();

    assert_eq!(
        next_unwrap::<PrimitiveScalar<i32>, _>(&mut iter).value(),
        &Some(3)
    );
    assert_eq!(iter.next(), None);

    Ok(())
}

#[test]
fn iter_dense_slice() -> PolarsResult<()> {
    let fields = vec![
        Field::new("a", ArrowDataType::Int32, true),
        Field::new("b", ArrowDataType::Utf8, true),
    ];
    let data_type = ArrowDataType::Union(fields, None, UnionMode::Dense);
    let types = Buffer::from(vec![0, 0, 1]);
    let offsets = Buffer::<i32>::from(vec![0, 1, 0]);
    let fields = vec![
        Int32Array::from(&[Some(1), Some(3)]).boxed(),
        Utf8Array::<i32>::from([Some("c")]).boxed(),
    ];

    let array = UnionArray::new(data_type, types, fields.clone(), Some(offsets));
    let array_slice = array.sliced(1, 1);
    let mut iter = array_slice.iter();

    assert_eq!(
        next_unwrap::<PrimitiveScalar<i32>, _>(&mut iter).value(),
        &Some(3)
    );
    assert_eq!(iter.next(), None);

    Ok(())
}

#[test]
fn scalar() -> PolarsResult<()> {
    let fields = vec![
        Field::new("a", ArrowDataType::Int32, true),
        Field::new("b", ArrowDataType::Utf8, true),
    ];
    let data_type = ArrowDataType::Union(fields, None, UnionMode::Dense);
    let types = Buffer::from(vec![0, 0, 1]);
    let offsets = Buffer::<i32>::from(vec![0, 1, 0]);
    let fields = vec![
        Int32Array::from(&[Some(1), None]).boxed(),
        Utf8Array::<i32>::from([Some("c")]).boxed(),
    ];

    let array = UnionArray::new(data_type, types, fields.clone(), Some(offsets));

    let scalar = new_scalar(&array, 0);
    let union_scalar = scalar.as_any().downcast_ref::<UnionScalar>().unwrap();
    assert_eq!(
        union_scalar
            .value()
            .as_any()
            .downcast_ref::<PrimitiveScalar<i32>>()
            .unwrap()
            .value(),
        &Some(1)
    );
    assert_eq!(union_scalar.type_(), 0);
    let scalar = new_scalar(&array, 1);
    let union_scalar = scalar.as_any().downcast_ref::<UnionScalar>().unwrap();
    assert_eq!(
        union_scalar
            .value()
            .as_any()
            .downcast_ref::<PrimitiveScalar<i32>>()
            .unwrap()
            .value(),
        &None
    );
    assert_eq!(union_scalar.type_(), 0);

    let scalar = new_scalar(&array, 2);
    let union_scalar = scalar.as_any().downcast_ref::<UnionScalar>().unwrap();
    assert_eq!(
        union_scalar
            .value()
            .as_any()
            .downcast_ref::<Utf8Scalar<i32>>()
            .unwrap()
            .value(),
        Some("c")
    );
    assert_eq!(union_scalar.type_(), 1);

    Ok(())
}

#[test]
fn dense_without_offsets_is_error() {
    let fields = vec![
        Field::new("a", ArrowDataType::Int32, true),
        Field::new("b", ArrowDataType::Utf8, true),
    ];
    let data_type = ArrowDataType::Union(fields, None, UnionMode::Dense);
    let types = vec![0, 0, 1].into();
    let fields = vec![
        Int32Array::from([Some(1), Some(3), Some(2)]).boxed(),
        Utf8Array::<i32>::from([Some("a"), Some("b"), Some("c")]).boxed(),
    ];

    assert!(UnionArray::try_new(data_type, types, fields.clone(), None).is_err());
}

#[test]
fn fields_must_match() {
    let fields = vec![
        Field::new("a", ArrowDataType::Int64, true),
        Field::new("b", ArrowDataType::Utf8, true),
    ];
    let data_type = ArrowDataType::Union(fields, None, UnionMode::Sparse);
    let types = vec![0, 0, 1].into();
    let fields = vec![
        Int32Array::from([Some(1), Some(3), Some(2)]).boxed(),
        Utf8Array::<i32>::from([Some("a"), Some("b"), Some("c")]).boxed(),
    ];

    assert!(UnionArray::try_new(data_type, types, fields.clone(), None).is_err());
}

#[test]
fn sparse_with_offsets_is_error() {
    let fields = vec![
        Field::new("a", ArrowDataType::Int32, true),
        Field::new("b", ArrowDataType::Utf8, true),
    ];
    let data_type = ArrowDataType::Union(fields, None, UnionMode::Sparse);
    let fields = vec![
        Int32Array::from([Some(1), Some(3), Some(2)]).boxed(),
        Utf8Array::<i32>::from([Some("a"), Some("b"), Some("c")]).boxed(),
    ];

    let types = vec![0, 0, 1].into();
    let offsets = vec![0, 1, 0].into();

    assert!(UnionArray::try_new(data_type, types, fields.clone(), Some(offsets)).is_err());
}

#[test]
fn offsets_must_be_in_bounds() {
    let fields = vec![
        Field::new("a", ArrowDataType::Int32, true),
        Field::new("b", ArrowDataType::Utf8, true),
    ];
    let data_type = ArrowDataType::Union(fields, None, UnionMode::Sparse);
    let fields = vec![
        Int32Array::from([Some(1), Some(3), Some(2)]).boxed(),
        Utf8Array::<i32>::from([Some("a"), Some("b"), Some("c")]).boxed(),
    ];

    let types = vec![0, 0, 1].into();
    // it must be equal to length og types
    let offsets = vec![0, 1].into();

    assert!(UnionArray::try_new(data_type, types, fields.clone(), Some(offsets)).is_err());
}

#[test]
fn sparse_with_wrong_offsets1_is_error() {
    let fields = vec![
        Field::new("a", ArrowDataType::Int32, true),
        Field::new("b", ArrowDataType::Utf8, true),
    ];
    let data_type = ArrowDataType::Union(fields, None, UnionMode::Sparse);
    let fields = vec![
        Int32Array::from([Some(1), Some(3), Some(2)]).boxed(),
        Utf8Array::<i32>::from([Some("a"), Some("b"), Some("c")]).boxed(),
    ];

    let types = vec![0, 0, 1].into();
    // it must be equal to length of types
    let offsets = vec![0, 1, 10].into();

    assert!(UnionArray::try_new(data_type, types, fields.clone(), Some(offsets)).is_err());
}

#[test]
fn types_must_be_in_bounds() -> PolarsResult<()> {
    let fields = vec![
        Field::new("a", ArrowDataType::Int32, true),
        Field::new("b", ArrowDataType::Utf8, true),
    ];
    let data_type = ArrowDataType::Union(fields, None, UnionMode::Sparse);
    let fields = vec![
        Int32Array::from([Some(1), Some(3), Some(2)]).boxed(),
        Utf8Array::<i32>::from([Some("a"), Some("b"), Some("c")]).boxed(),
    ];

    // 10 > num fields
    let types = vec![0, 10].into();

    assert!(UnionArray::try_new(data_type, types, fields.clone(), None).is_err());
    Ok(())
}
