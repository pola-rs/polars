use super::*;

#[test]
fn test_list_gather_nulls_and_empty() {
    let a: &[i32] = &[];
    let a = Series::new("", a);
    let b = Series::new("", &[None, Some(a.clone())]);
    let indices = [Some(0 as IdxSize), Some(1), None]
        .into_iter()
        .collect_ca("");
    let out = b.take(&indices).unwrap();
    let expected = Series::new("", &[None, Some(a), None]);
    assert!(out.equals_missing(&expected))
}
