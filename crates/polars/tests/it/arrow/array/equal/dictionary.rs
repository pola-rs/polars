use arrow::array::*;

use super::test_equal;

fn create_dictionary_array(values: &[Option<&str>], keys: &[Option<i16>]) -> DictionaryArray<i16> {
    let keys = Int16Array::from(keys);
    let values = Utf8Array::<i64>::from(values);

    DictionaryArray::try_from_keys(keys, values.boxed()).unwrap()
}

#[test]
fn dictionary_equal() {
    // (a, b, c), (0, 1, 0, 2) => (a, b, a, c)
    let a = create_dictionary_array(
        &[Some("a"), Some("b"), Some("c")],
        &[Some(0), Some(1), Some(0), Some(2)],
    );
    // different representation (values and keys are swapped), same result
    let b = create_dictionary_array(
        &[Some("a"), Some("c"), Some("b")],
        &[Some(0), Some(2), Some(0), Some(1)],
    );
    test_equal(&a, &b, true);

    // different len
    let b = create_dictionary_array(
        &[Some("a"), Some("c"), Some("b")],
        &[Some(0), Some(2), Some(1)],
    );
    test_equal(&a, &b, false);

    // different key
    let b = create_dictionary_array(
        &[Some("a"), Some("c"), Some("b")],
        &[Some(0), Some(2), Some(0), Some(0)],
    );
    test_equal(&a, &b, false);

    // different values, same keys
    let b = create_dictionary_array(
        &[Some("a"), Some("b"), Some("d")],
        &[Some(0), Some(1), Some(0), Some(2)],
    );
    test_equal(&a, &b, false);
}

#[test]
fn dictionary_equal_null() {
    // (a, b, c), (1, 2, 1, 3) => (a, b, a, c)
    let a = create_dictionary_array(
        &[Some("a"), Some("b"), Some("c")],
        &[Some(0), None, Some(0), Some(2)],
    );

    // equal to self
    test_equal(&a, &a, true);

    // different representation (values and keys are swapped), same result
    let b = create_dictionary_array(
        &[Some("a"), Some("c"), Some("b")],
        &[Some(0), None, Some(0), Some(1)],
    );
    test_equal(&a, &b, true);

    // different null position
    let b = create_dictionary_array(
        &[Some("a"), Some("c"), Some("b")],
        &[Some(0), Some(2), Some(0), None],
    );
    test_equal(&a, &b, false);

    // different key
    let b = create_dictionary_array(
        &[Some("a"), Some("c"), Some("b")],
        &[Some(0), None, Some(0), Some(0)],
    );
    test_equal(&a, &b, false);

    // different values, same keys
    let b = create_dictionary_array(
        &[Some("a"), Some("b"), Some("d")],
        &[Some(0), None, Some(0), Some(2)],
    );
    test_equal(&a, &b, false);

    // different nulls in keys and values
    let a = create_dictionary_array(
        &[Some("a"), Some("b"), None],
        &[Some(0), None, Some(0), Some(2)],
    );
    let b = create_dictionary_array(
        &[Some("a"), Some("b"), Some("c")],
        &[Some(0), None, Some(0), None],
    );
    test_equal(&a, &b, true);
}
