use polars::prelude::*;

#[test]
fn test_schema_rename() {
    use DataType::*;
    let mut schema = Schema::from_iter([
        Field::new("a", UInt64),
        Field::new("b", Int32),
        Field::new("c", Int8),
    ]);
    schema.rename("a", "anton".into()).unwrap();
    let expected = Schema::from_iter([
        Field::new("anton", UInt64),
        Field::new("b", Int32),
        Field::new("c", Int8),
    ]);

    assert_eq!(schema, expected);
}
