use polars::prelude::*;

#[test]
fn test_schema_rename() {
    use DataType::*;

    #[track_caller]
    fn test_case(old: &str, new: &str, expected: Option<(&str, Vec<Field>)>) {
        fn make_schema() -> Schema {
            Schema::from_iter([
                Field::new("a", UInt64),
                Field::new("b", Int32),
                Field::new("c", Int8),
            ])
        }
        let mut schema = make_schema();
        let res = schema.rename(old, new.into());
        if let Some((old_name, expected_fields)) = expected {
            assert_eq!(res.unwrap(), old_name);
            assert_eq!(schema, Schema::from_iter(expected_fields));
        } else {
            assert!(res.is_none());
            assert_eq!(schema, make_schema());
        }
    }

    test_case(
        "a",
        "anton",
        Some((
            "a",
            vec![
                Field::new("anton", UInt64),
                Field::new("b", Int32),
                Field::new("c", Int8),
            ],
        )),
    );

    test_case(
        "b",
        "bantam",
        Some((
            "b",
            vec![
                Field::new("a", UInt64),
                Field::new("bantam", Int32),
                Field::new("c", Int8),
            ],
        )),
    );

    test_case("d", "dan", None);
}

#[test]
fn test_schema_insert_at_index() {
    use DataType::*;

    #[track_caller]
    fn test_case(
        schema: &Schema,
        index: usize,
        name: &str,
        expected: (Option<DataType>, Vec<Field>),
    ) {
        println!("{index:?} -- {name:?} -- {expected:?}");
        let new = schema
            .new_inserting_at_index(index, name.into(), String)
            .unwrap();

        let mut new_mut = schema.clone();
        let old_dtype = new_mut.insert_at_index(index, name.into(), String).unwrap();

        let (expected_dtype, expected_fields) = expected;
        let expected = Schema::from_iter(expected_fields);

        assert_eq!(expected, new);

        assert_eq!(expected, new_mut);
        assert_eq!(expected_dtype, old_dtype);
    }

    let schema = Schema::from_iter([
        Field::new("a", UInt64),
        Field::new("b", Int32),
        Field::new("c", Int8),
    ]);

    test_case(
        &schema,
        0,
        "new",
        (
            None,
            vec![
                Field::new("new", String),
                Field::new("a", UInt64),
                Field::new("b", Int32),
                Field::new("c", Int8),
            ],
        ),
    );

    test_case(
        &schema,
        0,
        "a",
        (
            Some(UInt64),
            vec![
                Field::new("a", String),
                Field::new("b", Int32),
                Field::new("c", Int8),
            ],
        ),
    );

    test_case(
        &schema,
        0,
        "b",
        (
            Some(Int32),
            vec![
                Field::new("b", String),
                Field::new("a", UInt64),
                Field::new("c", Int8),
            ],
        ),
    );

    test_case(
        &schema,
        1,
        "a",
        (
            Some(UInt64),
            vec![
                Field::new("b", Int32),
                Field::new("a", String),
                Field::new("c", Int8),
            ],
        ),
    );

    test_case(
        &schema,
        2,
        "a",
        (
            Some(UInt64),
            vec![
                Field::new("b", Int32),
                Field::new("c", Int8),
                Field::new("a", String),
            ],
        ),
    );

    test_case(
        &schema,
        3,
        "a",
        (
            Some(UInt64),
            vec![
                Field::new("b", Int32),
                Field::new("c", Int8),
                Field::new("a", String),
            ],
        ),
    );

    test_case(
        &schema,
        3,
        "new",
        (
            None,
            vec![
                Field::new("a", UInt64),
                Field::new("b", Int32),
                Field::new("c", Int8),
                Field::new("new", String),
            ],
        ),
    );

    test_case(
        &schema,
        2,
        "c",
        (
            Some(Int8),
            vec![
                Field::new("a", UInt64),
                Field::new("b", Int32),
                Field::new("c", String),
            ],
        ),
    );

    test_case(
        &schema,
        3,
        "c",
        (
            Some(Int8),
            vec![
                Field::new("a", UInt64),
                Field::new("b", Int32),
                Field::new("c", String),
            ],
        ),
    );

    assert!(schema
        .new_inserting_at_index(4, "oob".into(), String)
        .is_err());
}

#[test]
fn test_with_column() {
    use DataType::*;

    #[track_caller]
    fn test_case(
        schema: &Schema,
        new_name: &str,
        new_dtype: DataType,
        expected: (Option<DataType>, Vec<Field>),
    ) {
        let mut schema = schema.clone();
        let old_dtype = schema.with_column(new_name.into(), new_dtype);
        let (exp_dtype, exp_fields) = expected;
        assert_eq!(exp_dtype, old_dtype);
        assert_eq!(Schema::from_iter(exp_fields), schema);
    }

    let schema = Schema::from_iter([
        Field::new("a", UInt64),
        Field::new("b", Int32),
        Field::new("c", Int8),
    ]);

    test_case(
        &schema,
        "a",
        String,
        (
            Some(UInt64),
            vec![
                Field::new("a", String),
                Field::new("b", Int32),
                Field::new("c", Int8),
            ],
        ),
    );

    test_case(
        &schema,
        "b",
        String,
        (
            Some(Int32),
            vec![
                Field::new("a", UInt64),
                Field::new("b", String),
                Field::new("c", Int8),
            ],
        ),
    );

    test_case(
        &schema,
        "c",
        String,
        (
            Some(Int8),
            vec![
                Field::new("a", UInt64),
                Field::new("b", Int32),
                Field::new("c", String),
            ],
        ),
    );

    test_case(
        &schema,
        "d",
        String,
        (
            None,
            vec![
                Field::new("a", UInt64),
                Field::new("b", Int32),
                Field::new("c", Int8),
                Field::new("d", String),
            ],
        ),
    );
}

#[test]
fn test_getters() {
    use DataType::*;

    macro_rules! test_case {
        ($schema:expr, $method:ident, name: $name:expr, $expected:expr) => {{
            assert_eq!($expected, $schema.$method($name).unwrap());
            assert!($schema.$method("NOT_FOUND").is_none());
        }};
        ($schema:expr, $method:ident, index: $index:expr, $expected:expr) => {{
            assert_eq!($expected, $schema.$method($index).unwrap());
            assert!($schema.$method(usize::MAX).is_none());
        }};
    }

    let mut schema = Schema::from_iter([
        Field::new("a", UInt64),
        Field::new("b", Int32),
        Field::new("c", Int8),
    ]);

    test_case!(schema, get, name: "a", &UInt64);
    test_case!(schema, get_full, name: "a", (0, &"a".into(), &UInt64));
    test_case!(schema, get_field, name: "a", Field::new("a", UInt64));
    test_case!(schema, get_at_index, index: 1, (&"b".into(), &Int32));
    test_case!(schema, get_at_index_mut, index: 1, (&mut "b".into(), &mut Int32));

    assert!(schema.contains("a"));
    assert!(!schema.contains("NOT_FOUND"));
}

#[test]
fn test_removal() {
    use DataType::*;

    #[track_caller]
    fn test_case(
        schema: &Schema,
        to_remove: &str,
        dtype: Option<DataType>,
        swapped_expected: Vec<Field>,
        shifted_expected: Vec<Field>,
    ) {
        #[track_caller]
        fn test_it(expected: (Option<DataType>, Vec<Field>), actual: (Option<DataType>, Schema)) {
            let (exp_dtype, exp_fields) = expected;
            let (act_dtype, act_schema) = actual;

            assert_eq!(Schema::from_iter(exp_fields), act_schema);
            assert_eq!(exp_dtype, act_dtype);
        }

        let mut swapped = schema.clone();
        let swapped_res = swapped.remove(to_remove);

        test_it((dtype.clone(), swapped_expected), (swapped_res, swapped));

        let mut shifted = schema.clone();
        let shifted_res = shifted.shift_remove(to_remove);

        test_it((dtype, shifted_expected), (shifted_res, shifted));
    }

    let schema = Schema::from_iter([
        Field::new("a", UInt64),
        Field::new("b", Int32),
        Field::new("c", Int8),
        Field::new("d", Float64),
    ]);

    test_case(
        &schema,
        "a",
        Some(UInt64),
        vec![
            Field::new("d", Float64),
            Field::new("b", Int32),
            Field::new("c", Int8),
        ],
        vec![
            Field::new("b", Int32),
            Field::new("c", Int8),
            Field::new("d", Float64),
        ],
    );

    test_case(
        &schema,
        "b",
        Some(Int32),
        vec![
            Field::new("a", UInt64),
            Field::new("d", Float64),
            Field::new("c", Int8),
        ],
        vec![
            Field::new("a", UInt64),
            Field::new("c", Int8),
            Field::new("d", Float64),
        ],
    );

    test_case(
        &schema,
        "c",
        Some(Int8),
        vec![
            Field::new("a", UInt64),
            Field::new("b", Int32),
            Field::new("d", Float64),
        ],
        vec![
            Field::new("a", UInt64),
            Field::new("b", Int32),
            Field::new("d", Float64),
        ],
    );

    test_case(
        &schema,
        "d",
        Some(Float64),
        vec![
            Field::new("a", UInt64),
            Field::new("b", Int32),
            Field::new("c", Int8),
        ],
        vec![
            Field::new("a", UInt64),
            Field::new("b", Int32),
            Field::new("c", Int8),
        ],
    );

    test_case(
        &schema,
        "NOT_FOUND",
        None,
        vec![
            Field::new("a", UInt64),
            Field::new("b", Int32),
            Field::new("c", Int8),
            Field::new("d", Float64),
        ],
        vec![
            Field::new("a", UInt64),
            Field::new("b", Int32),
            Field::new("c", Int8),
            Field::new("d", Float64),
        ],
    );
}

#[test]
fn test_set_dtype() {
    use DataType::*;

    #[track_caller]
    fn test_case(
        schema: &Schema,
        name: &str,
        index: usize,
        expected: (Option<DataType>, Vec<Field>),
    ) {
        // test set_dtype
        {
            let mut schema = schema.clone();
            let old_dtype = schema.set_dtype(name, String);
            let (exp_dtype, exp_fields) = &expected;
            assert_eq!(&old_dtype, exp_dtype);
            assert_eq!(Schema::from_iter(exp_fields.clone()), schema);
        }

        // test set_dtype_at_index
        {
            let mut schema = schema.clone();
            let old_dtype = schema.set_dtype_at_index(index, String);
            let (exp_dtype, exp_fields) = &expected;
            assert_eq!(&old_dtype, exp_dtype);
            assert_eq!(Schema::from_iter(exp_fields.clone()), schema);
        }
    }

    let schema = Schema::from_iter([
        Field::new("a", UInt64),
        Field::new("b", Int32),
        Field::new("c", Int8),
    ]);

    test_case(
        &schema,
        "a",
        0,
        (
            Some(UInt64),
            vec![
                Field::new("a", String),
                Field::new("b", Int32),
                Field::new("c", Int8),
            ],
        ),
    );
    test_case(
        &schema,
        "b",
        1,
        (
            Some(Int32),
            vec![
                Field::new("a", UInt64),
                Field::new("b", String),
                Field::new("c", Int8),
            ],
        ),
    );
    test_case(
        &schema,
        "c",
        2,
        (
            Some(Int8),
            vec![
                Field::new("a", UInt64),
                Field::new("b", Int32),
                Field::new("c", String),
            ],
        ),
    );
    test_case(
        &schema,
        "d",
        3,
        (
            None,
            vec![
                Field::new("a", UInt64),
                Field::new("b", Int32),
                Field::new("c", Int8),
            ],
        ),
    );
}
