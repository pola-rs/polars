use arrow::array::growable::{Growable, GrowableDictionary};
use arrow::array::*;
use polars_error::PolarsResult;

#[test]
fn test_single() -> PolarsResult<()> {
    let original_data = vec![Some("a"), Some("b"), Some("a")];

    let data = original_data.clone();
    let mut array = MutableDictionaryArray::<i32, MutableBinaryViewArray<str>>::new();
    array.try_extend(data)?;
    let array = array.into();

    // same values, less keys
    let expected = DictionaryArray::try_from_keys(
        PrimitiveArray::from_vec(vec![1, 0]),
        Box::new(Utf8ViewArray::from_slice(&original_data)),
    )
    .unwrap();

    let mut growable = GrowableDictionary::new(&[&array], false, 0);

    unsafe {
        growable.extend(0, 1, 2);
    }
    assert_eq!(growable.len(), 2);

    let result: DictionaryArray<i32> = growable.into();

    assert_eq!(result, expected);
    Ok(())
}

#[test]
fn test_multi() -> PolarsResult<()> {
    let mut original_data1 = vec![Some("a"), Some("b"), None, Some("a")];
    let original_data2 = vec![Some("c"), Some("b"), None, Some("a")];

    let data1 = original_data1.clone();
    let data2 = original_data2.clone();

    let mut array1 = MutableDictionaryArray::<i32, MutableBinaryViewArray<str>>::new();
    array1.try_extend(data1)?;
    let array1: DictionaryArray<i32> = array1.into();

    let mut array2 = MutableDictionaryArray::<i32, MutableBinaryViewArray<str>>::new();
    array2.try_extend(data2)?;
    let array2: DictionaryArray<i32> = array2.into();

    // same values, less keys
    original_data1.extend(original_data2.iter().cloned());
    let expected = DictionaryArray::try_from_keys(
        PrimitiveArray::from(&[Some(1), None, Some(3), None]),
        Utf8ViewArray::from_slice_values(["a", "b", "c", "b", "a"]).boxed(),
    )
    .unwrap();

    let mut growable = GrowableDictionary::new(&[&array1, &array2], false, 0);

    unsafe {
        growable.extend(0, 1, 2);
    }
    unsafe {
        growable.extend(1, 1, 2);
    }
    assert_eq!(growable.len(), 4);

    let result: DictionaryArray<i32> = growable.into();

    assert_eq!(result, expected);
    Ok(())
}
