use polars_arrow::array::{
    Array, BinaryArray, BooleanArray, DictionaryArray, Float32Array, Int128Array, Int16Array,
    Int256Array, Int32Array, MutableDictionaryArray, MutablePrimitiveArray, MutableUtf8Array,
    NullArray, TryExtend, TryPush, Utf8Array,
};
use polars_arrow::compute::sort::row::{RowConverter, SortField};
use polars_arrow::compute::sort::SortOptions;
use polars_arrow::datatypes::{DataType, IntegerType};
use polars_arrow::types::i256;

#[test]
fn test_fixed_width() {
    let cols = [
        Int16Array::from_iter([Some(1), Some(2), None, Some(-5), Some(2), Some(2), Some(0)])
            .to_boxed(),
        Float32Array::from_iter([
            Some(1.3),
            Some(2.5),
            None,
            Some(4.),
            Some(0.1),
            Some(-4.),
            Some(-0.),
        ])
        .to_boxed(),
    ];

    let mut converter = RowConverter::new(vec![
        SortField::new(DataType::Int16),
        SortField::new(DataType::Float32),
    ]);
    let rows = converter.convert_columns(&cols).unwrap();

    assert!(rows.row(3) < rows.row(6));
    assert!(rows.row(0) < rows.row(1));
    assert!(rows.row(3) < rows.row(0));
    assert!(rows.row(4) < rows.row(1));
    assert!(rows.row(5) < rows.row(4));
}

#[test]
fn test_decimal128() {
    let mut converter = RowConverter::new(vec![SortField::new(DataType::Decimal(38, 7))]);
    let col = Int128Array::from_iter([
        None,
        Some(i128::MIN),
        Some(-13),
        Some(46_i128),
        Some(5456_i128),
        Some(i128::MAX),
    ])
    .to(DataType::Decimal(38, 7))
    .to_boxed();

    let rows = converter.convert_columns(&[col]).unwrap();
    for i in 0..rows.len() - 1 {
        assert!(rows.row(i) < rows.row(i + 1));
    }
}

#[test]
fn test_decimal256() {
    let mut converter = RowConverter::new(vec![SortField::new(DataType::Decimal256(76, 7))]);
    let col = Int256Array::from_iter([
        None,
        Some(i256::from_words(i128::MIN, i128::MIN)),
        Some(i256::from_words(0, 46_i128)),
        Some(i256::from_words(0, -1)),
        Some(i256::from_words(5, 46_i128)),
        Some(i256::from_words(i128::MAX, 0)),
        Some(i256::from_words(i128::MAX, i128::MAX)),
        Some(i256::from_words(i128::MAX, -1)),
    ])
    .to(DataType::Decimal256(76, 7))
    .to_boxed();

    let rows = converter.convert_columns(&[col]).unwrap();
    for i in 0..rows.len() - 1 {
        assert!(rows.row(i) < rows.row(i + 1));
    }
}

#[test]
fn test_bool() {
    let mut converter = RowConverter::new(vec![SortField::new(DataType::Boolean)]);

    let col = BooleanArray::from_iter([None, Some(false), Some(true)]).to_boxed();

    let rows = converter.convert_columns(&[Box::clone(&col)]).unwrap();
    assert!(rows.row(2) > rows.row(1));
    assert!(rows.row(2) > rows.row(0));
    assert!(rows.row(1) > rows.row(0));

    let mut converter = RowConverter::new(vec![SortField::new_with_options(
        DataType::Boolean,
        SortOptions {
            descending: true,
            nulls_first: false,
        },
    )]);

    let rows = converter.convert_columns(&[Box::clone(&col)]).unwrap();
    assert!(rows.row(2) < rows.row(1));
    assert!(rows.row(2) < rows.row(0));
    assert!(rows.row(1) < rows.row(0));
}

#[test]
fn test_null_encoding() {
    let col = NullArray::new(DataType::Null, 10).to_boxed();
    let mut converter = RowConverter::new(vec![SortField::new(DataType::Null)]);
    let rows = converter.convert_columns(&[col]).unwrap();
    assert_eq!(rows.len(), 10);
}

#[test]
fn test_variable_width() {
    let col =
        Utf8Array::<i32>::from([Some("hello"), Some("he"), None, Some("foo"), Some("")]).to_boxed();

    let mut converter = RowConverter::new(vec![SortField::new(DataType::Utf8)]);
    let rows = converter.convert_columns(&[Box::clone(&col)]).unwrap();

    assert!(rows.row(1) < rows.row(0));
    assert!(rows.row(2) < rows.row(4));
    assert!(rows.row(3) < rows.row(0));
    assert!(rows.row(3) < rows.row(1));

    let col = BinaryArray::<i32>::from_iter([
        None,
        Some(vec![0_u8; 0]),
        Some(vec![0_u8; 6]),
        Some(vec![0_u8; 32]),
        Some(vec![0_u8; 33]),
        Some(vec![1_u8; 6]),
        Some(vec![1_u8; 32]),
        Some(vec![1_u8; 33]),
        Some(vec![0xFF_u8; 6]),
        Some(vec![0xFF_u8; 32]),
        Some(vec![0xFF_u8; 33]),
    ])
    .to_boxed();

    let mut converter = RowConverter::new(vec![SortField::new(DataType::Binary)]);
    let rows = converter.convert_columns(&[Box::clone(&col)]).unwrap();

    for i in 0..rows.len() {
        for j in i + 1..rows.len() {
            assert!(
                rows.row(i) < rows.row(j),
                "{} < {} - {:?} < {:?}",
                i,
                j,
                rows.row(i),
                rows.row(j)
            );
        }
    }

    let mut converter = RowConverter::new(vec![SortField::new_with_options(
        DataType::Binary,
        SortOptions {
            descending: true,
            nulls_first: false,
        },
    )]);
    let rows = converter.convert_columns(&[Box::clone(&col)]).unwrap();

    for i in 0..rows.len() {
        for j in i + 1..rows.len() {
            assert!(
                rows.row(i) > rows.row(j),
                "{} > {} - {:?} > {:?}",
                i,
                j,
                rows.row(i),
                rows.row(j)
            );
        }
    }
}

#[test]
fn test_string_dictionary() {
    let data = vec![
        Some("foo"),
        Some("hello"),
        Some("he"),
        None,
        Some("hello"),
        Some(""),
        Some("hello"),
        Some("hello"),
    ];
    let mut array = MutableDictionaryArray::<i32, MutableUtf8Array<i32>>::new();
    array.try_extend(data).unwrap();
    let a: DictionaryArray<i32> = array.into();
    let a = a.to_boxed();

    let mut converter = RowConverter::new(vec![SortField::new(a.data_type().clone())]);
    let rows_a = converter.convert_columns(&[Box::clone(&a)]).unwrap();

    assert!(rows_a.row(3) < rows_a.row(5));
    assert!(rows_a.row(2) < rows_a.row(1));
    assert!(rows_a.row(0) < rows_a.row(1));
    assert!(rows_a.row(3) < rows_a.row(0));

    assert_eq!(rows_a.row(1), rows_a.row(4));
    assert_eq!(rows_a.row(1), rows_a.row(6));
    assert_eq!(rows_a.row(1), rows_a.row(7));

    let data = vec![Some("hello"), None, Some("cupcakes")];
    let mut array = MutableDictionaryArray::<i32, MutableUtf8Array<i32>>::new();
    array.try_extend(data).unwrap();
    let b: DictionaryArray<i32> = array.into();

    let rows_b = converter.convert_columns(&[b.to_boxed()]).unwrap();
    assert_eq!(rows_a.row(1), rows_b.row(0));
    assert_eq!(rows_a.row(3), rows_b.row(1));
    assert!(rows_b.row(2) < rows_a.row(0));

    let mut converter = RowConverter::new(vec![SortField::new_with_options(
        a.data_type().clone(),
        SortOptions {
            descending: true,
            nulls_first: false,
        },
    )]);

    let rows_c = converter.convert_columns(&[Box::clone(&a)]).unwrap();
    assert!(rows_c.row(3) > rows_c.row(5));
    assert!(rows_c.row(2) > rows_c.row(1));
    assert!(rows_c.row(0) > rows_c.row(1));
    assert!(rows_c.row(3) > rows_c.row(0));
}

#[test]
fn test_primitive_dictionary() {
    let mut builder = MutableDictionaryArray::<i32, MutablePrimitiveArray<i32>>::new();
    builder.try_push(Some(2)).unwrap();
    builder.try_push(Some(3)).unwrap();
    builder.try_push(Some(0)).unwrap();
    builder.push_null();
    builder.try_push(Some(5)).unwrap();
    builder.try_push(Some(3)).unwrap();
    builder.try_push(Some(-1)).unwrap();

    let a: DictionaryArray<i32> = builder.into();

    let mut converter = RowConverter::new(vec![SortField::new(a.data_type().clone())]);
    let rows = converter.convert_columns(&[a.to_boxed()]).unwrap();
    assert!(rows.row(0) < rows.row(1));
    assert!(rows.row(2) < rows.row(0));
    assert!(rows.row(3) < rows.row(2));
    assert!(rows.row(6) < rows.row(2));
    assert!(rows.row(3) < rows.row(6));
}

#[test]
fn test_dictionary_nulls() {
    let values = Int32Array::from_iter([Some(1), Some(-1), None, Some(4), None]);
    let keys = Int32Array::from_iter([Some(0), Some(0), Some(1), Some(2), Some(4), None]);

    let data_type = DataType::Dictionary(IntegerType::Int32, Box::new(DataType::Int32), false);
    let data = DictionaryArray::try_from_keys(keys, values.to_boxed()).unwrap();

    let mut converter = RowConverter::new(vec![SortField::new(data_type)]);
    let rows = converter.convert_columns(&[data.to_boxed()]).unwrap();

    assert_eq!(rows.row(0), rows.row(1));
    assert_eq!(rows.row(3), rows.row(4));
    assert_eq!(rows.row(4), rows.row(5));
    assert!(rows.row(3) < rows.row(0));
}
