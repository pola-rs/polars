use polars_core::prelude::*;
use polars_lazy::prelude::*;
use polars_sql::*;

#[test]
fn test_describe() {
    let lf = df! {
      "year"=> [2018],
      "country"=> ["US"],
      "sales"=> [1000.0]
    }
    .unwrap()
    .lazy();
    let mut context = SQLContext::new();
    context.register("df", lf.clone());
    let sql = r#"EXPLAIN SELECT MAX(year) as year_max, MIN(country) AS country FROM df"#;
    let res = context.execute(sql).unwrap();
    let df = res.collect().unwrap();
    let lf = lf.select([
        col("year").max().alias("year_max"),
        col("country").min().alias("country"),
    ]);
    let expected = lf.describe_optimized_plan().unwrap();

    let expected = expected.split('\n').map(Some).collect::<Vec<_>>();
    let actual = df
        .column("Logical Plan")
        .unwrap()
        .str()
        .unwrap()
        .iter()
        .collect::<Vec<_>>();

    assert_eq!(actual, expected);
}

#[test]
fn test_explain_does_not_run_other_statements() {
    let mut context = SQLContext::new();
    context.register("df", df! { "a" => [1, 2] }.unwrap().lazy());

    for sql in [
        "EXPLAIN DROP TABLE df",
        "EXPLAIN CREATE TABLE other AS SELECT * FROM df",
        "EXPLAIN INSERT INTO df VALUES (3)",
        "EXPLAIN WITH q AS (SELECT 3 AS a) INSERT INTO df SELECT a FROM q",
        "EXPLAIN TRUNCATE TABLE df",
    ] {
        let err = context.execute(sql).err().unwrap();
        assert!(err.to_string().contains("EXPLAIN supports queries only"));
    }

    assert_eq!(context.get_tables(), ["df"]);
    let df = context
        .execute("SELECT * FROM df")
        .unwrap()
        .collect()
        .unwrap();
    assert_eq!(df.height(), 2);
}

/// The column types that `DESCRIBE` gives for a table with one column per type of `dtypes`.
fn describe_types(dtypes: &[DataType]) -> Vec<String> {
    let fields = dtypes
        .iter()
        .enumerate()
        .map(|(i, dtype)| Field::new(format!("c{i}").into(), dtype.clone()));
    let mut context = SQLContext::new();
    let table = DataFrame::empty_with_schema(&Schema::from_iter(fields));
    context.register("t", table.lazy());
    let described = context.execute("DESCRIBE t").unwrap().collect().unwrap();
    let types = described.column("column_type").unwrap().str().unwrap();
    types.iter().map(|name| name.unwrap().to_string()).collect()
}

#[test]
fn test_describe_type_names_cast_back() {
    let time_zone = |name: &'static str| TimeZone::opt_try_new(Some(name)).unwrap();
    let struct_of = |fields: Vec<Field>| DataType::Struct(StructFields::try_new(fields).unwrap());
    let geo = struct_of(vec![
        Field::new("lat".into(), DataType::Float64),
        Field::new("lon".into(), DataType::Float64),
    ]);
    let nested = struct_of(vec![Field::new(
        "a".into(),
        struct_of(vec![Field::new("b".into(), DataType::Int32)]),
    )]);
    let odd_names = struct_of(vec![
        Field::new("my \"col\"".into(), DataType::Int32),
        Field::new("select".into(), DataType::String),
        Field::new("tags".into(), DataType::List(Box::new(DataType::String))),
    ]);
    #[allow(unused_mut)]
    let mut cases = vec![
        (DataType::Boolean, "BOOLEAN"),
        (DataType::Int8, "TINYINT"),
        (DataType::Int16, "SMALLINT"),
        (DataType::Int32, "INTEGER"),
        (DataType::Int64, "BIGINT"),
        (DataType::Int128, "HUGEINT"),
        (DataType::UInt8, "UTINYINT"),
        (DataType::UInt16, "USMALLINT"),
        (DataType::UInt32, "UINTEGER"),
        (DataType::UInt64, "UBIGINT"),
        (DataType::UInt128, "UHUGEINT"),
        (DataType::Float32, "REAL"),
        (DataType::Float64, "DOUBLE PRECISION"),
        (DataType::String, "VARCHAR"),
        (DataType::Binary, "BYTEA"),
        (DataType::Date, "DATE"),
        (DataType::Time, "TIME"),
        (
            DataType::Datetime(TimeUnit::Microseconds, None),
            "TIMESTAMP",
        ),
        (
            DataType::Datetime(TimeUnit::Milliseconds, None),
            "TIMESTAMP(3)",
        ),
        (
            DataType::Datetime(TimeUnit::Nanoseconds, None),
            "TIMESTAMP(9)",
        ),
        (
            DataType::Datetime(TimeUnit::Microseconds, Some(TimeZone::UTC)),
            "TIMESTAMP WITH TIME ZONE",
        ),
        (
            DataType::Datetime(TimeUnit::Milliseconds, Some(TimeZone::UTC)),
            "TIMESTAMP(3) WITH TIME ZONE",
        ),
        (
            DataType::Datetime(TimeUnit::Milliseconds, time_zone("Europe/Amsterdam")),
            "DATETIME64(3, 'Europe/Amsterdam')",
        ),
        (DataType::Duration(TimeUnit::Microseconds), "INTERVAL"),
        (DataType::Duration(TimeUnit::Milliseconds), "INTERVAL(3)"),
        (DataType::Duration(TimeUnit::Nanoseconds), "INTERVAL(9)"),
        (DataType::List(Box::new(DataType::String)), "VARCHAR[]"),
        (
            DataType::List(Box::new(DataType::List(Box::new(DataType::Int32)))),
            "INTEGER[][]",
        ),
        (geo, "STRUCT<lat DOUBLE PRECISION, lon DOUBLE PRECISION>"),
        (nested.clone(), "STRUCT<a STRUCT<b INTEGER> >"),
        (
            DataType::List(Box::new(nested)),
            "STRUCT<a STRUCT<b INTEGER> >[]",
        ),
        (
            odd_names,
            r#"STRUCT<"my ""col""" INTEGER, select VARCHAR, tags VARCHAR[]>"#,
        ),
    ];
    #[cfg(feature = "dtype-decimal")]
    cases.push((DataType::Decimal(12, 2), "DECIMAL(12,2)"));
    #[cfg(feature = "dtype-map")]
    cases.push((
        DataType::Map(MapDtype::try_new(DataType::String, DataType::Int32).unwrap()),
        "MAP(VARCHAR, INTEGER)",
    ));
    #[cfg(feature = "dtype-categorical")]
    {
        let named = Categories::new("status".into(), PlSmallStr::EMPTY, CategoricalPhysical::U32);
        let values = FrozenCategories::new(["new", "it's done"]).unwrap();
        cases.push((
            DataType::from_categories(Categories::global()),
            "CATEGORICAL",
        ));
        cases.push((DataType::from_categories(named), "CATEGORICAL('status')"));
        cases.push((
            DataType::from_frozen_categories(values),
            "ENUM('new', 'it''s done')",
        ));
    }

    let dtypes: Vec<DataType> = cases.iter().map(|(dtype, _)| dtype.clone()).collect();
    let names = describe_types(&dtypes);
    let expected: Vec<&str> = cases.iter().map(|(_, name)| *name).collect();
    assert_eq!(names, expected);

    // `CAST` reads each name back as the type it was printed for.
    let casts: Vec<String> = names
        .iter()
        .enumerate()
        .map(|(i, name)| format!("CAST(NULL AS {name}) AS c{i}"))
        .collect();
    let sql = format!("SELECT {}", casts.join(", "));
    let schema = SQLContext::new()
        .execute(&sql)
        .unwrap()
        .collect_schema()
        .unwrap();
    let cast_dtypes: Vec<DataType> = schema.iter_values().cloned().collect();
    assert_eq!(cast_dtypes, dtypes);
}

#[test]
fn test_describe_type_without_sql_name() {
    // A type that has no SQL name is given by its Polars name, also inside a nested type.
    let array = DataType::Array(Box::new(DataType::Int32), 3);
    let nested = DataType::Struct(
        StructFields::try_new(vec![
            Field::new("xyz".into(), array.clone()),
            Field::new("label".into(), DataType::String),
        ])
        .unwrap(),
    );
    let empty = DataType::Struct(StructFields::default());
    assert_eq!(
        describe_types(&[array, nested, empty, DataType::Null]),
        [
            "array[i32, 3]",
            "STRUCT<xyz array[i32, 3], label VARCHAR>",
            "struct[0]",
            "null"
        ]
    );
    #[cfg(feature = "dtype-categorical")]
    {
        let no_values = FrozenCategories::new([]).unwrap();
        let dtype = DataType::from_frozen_categories(no_values);
        assert_eq!(describe_types(&[dtype]), ["enum"]);
    }
}

#[test]
fn test_describe_query() {
    let mut context = SQLContext::new();
    context.register(
        "df",
        df! { "a" => [1i64, 2], "s" => ["x", "y"] }.unwrap().lazy(),
    );

    for describe in ["DESCRIBE", "DESC"] {
        let sql = format!(
            "{describe} SELECT a::DOUBLE PRECISION / 2 AS b, s, a > 1 AS big FROM df WHERE a > 1"
        );
        let described = context.execute(&sql).unwrap().collect().unwrap();
        let expected = df! {
            "column_name" => ["b", "s", "big"],
            "column_type" => ["DOUBLE PRECISION", "VARCHAR", "BOOLEAN"],
        }
        .unwrap();
        assert!(described.equals(&expected), "{described}");
    }

    // The query is not run: this one fails on its data, and is described all the same.
    let query = "SELECT CAST(s AS INTEGER) AS n FROM df";
    assert!(context.execute(query).unwrap().collect().is_err());
    let described = context.execute(&format!("DESCRIBE {query}")).unwrap();
    let expected = df! { "column_name" => ["n"], "column_type" => ["INTEGER"] }.unwrap();
    assert!(described.collect().unwrap().equals(&expected));

    // Nothing but a query is described: the table stays as it is.
    for sql in [
        "DESCRIBE DROP TABLE df",
        "DESCRIBE WITH q AS (SELECT 3 AS a, 'z' AS s) INSERT INTO df SELECT a, s FROM q",
    ] {
        let err = context.execute(sql).err().unwrap();
        assert!(err.to_string().contains("tables and queries only"), "{err}");
    }
    assert_eq!(context.get_tables(), ["df"]);
    let df = context
        .execute("SELECT * FROM df")
        .unwrap()
        .collect()
        .unwrap();
    assert_eq!(df.height(), 2);
}

#[test]
fn test_typeof_gives_the_names_of_describe() {
    let mut context = SQLContext::new();
    let table = df! {
        "a" => [1i64],
        "s" => ["x"],
        "f" => [1.5f64],
        "l" => [Series::new("".into(), [1i32, 2])],
    };
    context.register("df", table.unwrap().lazy());

    let described = context.execute("DESCRIBE df").unwrap().collect().unwrap();
    let described = described.column("column_type").unwrap().str().unwrap();
    let sql = "SELECT TYPEOF(a) AS a, TYPEOF(s) AS s, TYPEOF(f) AS f, TYPEOF(l) AS l FROM df";
    let types = context.execute(sql).unwrap().collect().unwrap();
    for (column, described) in types.columns().iter().zip(described.iter()) {
        assert_eq!(column.str().unwrap().get(0), described);
    }
    let names: Vec<_> = described.iter().map(|name| name.unwrap()).collect();
    assert_eq!(
        names,
        ["BIGINT", "VARCHAR", "DOUBLE PRECISION", "INTEGER[]"]
    );
}
