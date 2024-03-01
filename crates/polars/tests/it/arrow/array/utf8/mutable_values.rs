use arrow::array::{MutableArray, MutableUtf8ValuesArray};
use arrow::datatypes::ArrowDataType;

#[test]
fn capacity() {
    let mut b = MutableUtf8ValuesArray::<i32>::with_capacity(100);

    assert_eq!(b.values().capacity(), 0);
    assert!(b.offsets().capacity() >= 100);
    b.shrink_to_fit();
    assert!(b.offsets().capacity() < 100);
}

#[test]
fn offsets_must_be_in_bounds() {
    let offsets = vec![0, 10].try_into().unwrap();
    let values = b"abbbbb".to_vec();
    assert!(MutableUtf8ValuesArray::<i32>::try_new(ArrowDataType::Utf8, offsets, values).is_err());
}

#[test]
fn data_type_must_be_consistent() {
    let offsets = vec![0, 4].try_into().unwrap();
    let values = b"abbb".to_vec();
    assert!(MutableUtf8ValuesArray::<i32>::try_new(ArrowDataType::Int32, offsets, values).is_err());
}

#[test]
fn must_be_utf8() {
    let offsets = vec![0, 4].try_into().unwrap();
    let values = vec![0, 159, 146, 150];
    assert!(std::str::from_utf8(&values).is_err());
    assert!(MutableUtf8ValuesArray::<i32>::try_new(ArrowDataType::Utf8, offsets, values).is_err());
}

#[test]
fn as_box() {
    let offsets = vec![0, 2].try_into().unwrap();
    let values = b"ab".to_vec();
    let mut b =
        MutableUtf8ValuesArray::<i32>::try_new(ArrowDataType::Utf8, offsets, values).unwrap();
    let _ = b.as_box();
}

#[test]
fn as_arc() {
    let offsets = vec![0, 2].try_into().unwrap();
    let values = b"ab".to_vec();
    let mut b =
        MutableUtf8ValuesArray::<i32>::try_new(ArrowDataType::Utf8, offsets, values).unwrap();
    let _ = b.as_arc();
}

#[test]
fn extend_trusted_len() {
    let offsets = vec![0, 2].try_into().unwrap();
    let values = b"ab".to_vec();
    let mut b =
        MutableUtf8ValuesArray::<i32>::try_new(ArrowDataType::Utf8, offsets, values).unwrap();
    b.extend_trusted_len(vec!["a", "b"].into_iter());

    let offsets = vec![0, 2, 3, 4].try_into().unwrap();
    let values = b"abab".to_vec();
    assert_eq!(
        b.as_box(),
        MutableUtf8ValuesArray::<i32>::try_new(ArrowDataType::Utf8, offsets, values)
            .unwrap()
            .as_box()
    )
}

#[test]
fn from_trusted_len() {
    let mut b = MutableUtf8ValuesArray::<i32>::from_trusted_len_iter(vec!["a", "b"].into_iter());

    let offsets = vec![0, 1, 2].try_into().unwrap();
    let values = b"ab".to_vec();
    assert_eq!(
        b.as_box(),
        MutableUtf8ValuesArray::<i32>::try_new(ArrowDataType::Utf8, offsets, values)
            .unwrap()
            .as_box()
    )
}

#[test]
fn extend_from_iter() {
    let offsets = vec![0, 2].try_into().unwrap();
    let values = b"ab".to_vec();
    let mut b =
        MutableUtf8ValuesArray::<i32>::try_new(ArrowDataType::Utf8, offsets, values).unwrap();
    b.extend_trusted_len(vec!["a", "b"].into_iter());

    let a = b.clone();
    b.extend_trusted_len(a.iter());

    let offsets = vec![0, 2, 3, 4, 6, 7, 8].try_into().unwrap();
    let values = b"abababab".to_vec();
    assert_eq!(
        b.as_box(),
        MutableUtf8ValuesArray::<i32>::try_new(ArrowDataType::Utf8, offsets, values)
            .unwrap()
            .as_box()
    )
}
