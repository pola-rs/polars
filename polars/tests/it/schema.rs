use polars::prelude::*;

#[test]
fn test_schema_rename() {
    use DataType::*;
    let mut schema = Schema::from([
        Field::new("a", UInt64),
        Field::new("b", Int32),
        Field::new("c", Int8),
    ]);
    schema.rename("a", "anton".to_string()).unwrap();
    let mut expected = Schema::from([
        Field::new("anton", UInt64),
        Field::new("b", Int32),
        Field::new("c", Int8),
    ]);

    assert_eq!(schema, expected);
}
