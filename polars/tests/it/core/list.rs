use polars::prelude::*;

#[test]
fn test_to_list_logical() -> PolarsResult<()> {
    let ca = Utf8Chunked::new("a", &["2021-01-01", "2021-01-02", "2021-01-03"]);
    let out = ca.as_date(None, false)?.into_series();
    let out = out.to_list().unwrap();
    assert_eq!(out.len(), 1);
    let s = format!("{:?}", out);
    // check if dtype is maintained all the way to formatting
    assert!(s.contains("[2021-01-01, 2021-01-02, 2021-01-03]"));

    let expl = out.explode().unwrap();
    assert_eq!(expl.dtype(), &DataType::Date);
    Ok(())
}
