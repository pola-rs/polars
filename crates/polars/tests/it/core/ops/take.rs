use super::*;

#[test]
fn test_list_gather_nulls_and_empty() {
    let a: &[i32] = &[];
    let a = Series::new("".into(), a);
    let b = Series::new("".into(), &[None, Some(a.clone())]);
    let indices = [Some(0 as IdxSize), Some(1), None]
        .into_iter()
        .collect_ca("".into());
    let out = b.take(&indices).unwrap();
    let expected = Series::new("".into(), &[None, Some(a), None]);
    assert!(out.equals_missing(&expected))
}

#[test]
#[cfg(feature = "dtype-categorical")]
fn test_list_categorical_dtype_preserved_after_take() {
    // Create List(String) and convert to List(Categorical)
    let mut builder = ListStringChunkedBuilder::new("a".into(), 2, 3);
    builder.append_values_iter(["a", "b"].iter().copied());
    builder.append_values_iter(["c", "d"].iter().copied());
    let list_str = builder.finish().into_series();

    let list_cat = list_str
        .list()
        .unwrap()
        .apply_to_inner(&|s| s.cast(&DataType::Categorical(None, Default::default())))
        .unwrap()
        .into_series();

    // Append to create chunked series
    let mut chunked = list_cat.clone();
    chunked.append(&list_cat).unwrap();
    assert_eq!(chunked.n_chunks(), 2);

    // Take operation
    let indices = [0u32, 2].into_iter().collect_ca("".into());
    let out = chunked.take(&indices).unwrap();

    // Verify dtype is preserved
    assert_eq!(
        out.dtype(),
        &DataType::List(Box::new(DataType::Categorical(None, Default::default()))),
        "List(Categorical) dtype should be preserved after take"
    );
}
