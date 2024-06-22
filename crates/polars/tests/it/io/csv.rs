use std::io::Cursor;
use std::num::NonZeroUsize;

use polars::io::RowIndex;
use polars_core::utils::concat_df;

use super::*;

const FOODS_CSV: &str = "../../examples/datasets/foods1.csv";

#[test]
fn write_csv() {
    let mut buf: Vec<u8> = Vec::new();
    let mut df = create_df();

    CsvWriter::new(&mut buf)
        .include_header(true)
        .with_batch_size(NonZeroUsize::new(1).unwrap())
        .finish(&mut df)
        .expect("csv written");
    let csv = std::str::from_utf8(&buf).unwrap();
    assert_eq!("days,temp\n0,22.1\n1,19.9\n2,7.0\n3,2.0\n4,3.0\n", csv);

    let mut buf: Vec<u8> = Vec::new();
    CsvWriter::new(&mut buf)
        .include_header(false)
        .finish(&mut df)
        .expect("csv written");
    let csv = std::str::from_utf8(&buf).unwrap();
    assert_eq!("0,22.1\n1,19.9\n2,7.0\n3,2.0\n4,3.0\n", csv);

    let mut buf: Vec<u8> = Vec::new();
    CsvWriter::new(&mut buf)
        .include_header(false)
        .with_line_terminator("\r\n".into())
        .finish(&mut df)
        .expect("csv written");
    let csv = std::str::from_utf8(&buf).unwrap();
    assert_eq!("0,22.1\r\n1,19.9\r\n2,7.0\r\n3,2.0\r\n4,3.0\r\n", csv);
}

#[test]
#[cfg(feature = "timezones")]
fn write_dates() {
    use polars_core::export::chrono;

    let s0 = Series::new("date", [chrono::NaiveDate::from_yo_opt(2024, 33), None]);
    let s1 = Series::new("time", [None, chrono::NaiveTime::from_hms_opt(19, 50, 0)]);
    let s2 = Series::new(
        "datetime",
        [
            Some(chrono::NaiveDateTime::new(
                chrono::NaiveDate::from_ymd_opt(2000, 12, 1).unwrap(),
                chrono::NaiveTime::from_num_seconds_from_midnight_opt(99, 49575634).unwrap(),
            )),
            None,
        ],
    );
    let mut df = DataFrame::new(vec![s0, s1, s2.clone()]).unwrap();

    let mut buf: Vec<u8> = Vec::new();
    CsvWriter::new(&mut buf)
        .include_header(true)
        .with_batch_size(NonZeroUsize::new(1).unwrap())
        .finish(&mut df)
        .expect("csv written");
    let csv = std::str::from_utf8(&buf).unwrap();
    assert_eq!(
        "date,time,datetime\n2024-02-02,,2000-12-01T00:01:39.049\n,19:50:00.000000000,\n",
        csv,
    );

    buf.clear();
    CsvWriter::new(&mut buf)
        .include_header(true)
        .with_batch_size(NonZeroUsize::new(1).unwrap())
        .with_date_format(Some("%d/%m/%Y".into()))
        .with_time_format(Some("%H%M%S".into()))
        .with_datetime_format(Some("%Y-%m-%d %H:%M:%S".into()))
        .finish(&mut df)
        .expect("csv written");
    let csv = std::str::from_utf8(&buf).unwrap();
    assert_eq!(
        "date,time,datetime\n02/02/2024,,2000-12-01 00:01:39\n,195000,\n",
        csv,
    );

    buf.clear();
    CsvWriter::new(&mut buf)
        .include_header(true)
        .with_batch_size(NonZeroUsize::new(1).unwrap())
        .with_date_format(Some("%<invalid format>".into()))
        .finish(&mut df)
        .expect_err("invalid date/time format should err");

    buf.clear();
    CsvWriter::new(&mut buf)
        .include_header(true)
        .with_batch_size(NonZeroUsize::new(1).unwrap())
        .with_date_format(Some("%H".into()))
        .finish(&mut df)
        .expect_err("invalid date/time format should err");

    buf.clear();
    CsvWriter::new(&mut buf)
        .include_header(true)
        .with_batch_size(NonZeroUsize::new(1).unwrap())
        .with_datetime_format(Some("%Z".into()))
        .finish(&mut df)
        .expect_err("invalid date/time format should err");

    let with_timezone = polars_ops::chunked_array::replace_time_zone(
        s2.slice(0, 1).datetime().unwrap(),
        Some("America/New_York"),
        &StringChunked::new("", ["raise"]),
        NonExistent::Raise,
    )
    .unwrap()
    .into_series();
    let mut with_timezone_df = DataFrame::new(vec![with_timezone]).unwrap();
    buf.clear();
    CsvWriter::new(&mut buf)
        .include_header(false)
        .finish(&mut with_timezone_df)
        .expect("csv written");
    let csv = std::str::from_utf8(&buf).unwrap();
    assert_eq!("2000-12-01T00:01:39.049-0500\n", csv);
}

#[test]
fn test_read_csv_file() {
    let file = std::fs::File::open(FOODS_CSV).unwrap();
    let df = CsvReadOptions::default()
        .into_reader_with_file_handle(file)
        .finish()
        .unwrap();

    assert_eq!(df.shape(), (27, 4));
}

#[test]
fn test_read_csv_filter() -> PolarsResult<()> {
    let df = CsvReadOptions::default()
        .try_into_reader_with_file_path(Some(FOODS_CSV.into()))?
        .finish()?;

    let out = df.filter(&df.column("fats_g")?.gt(4)?)?;

    // This fails if all columns are not equal.
    println!("{out}");

    Ok(())
}

#[test]
fn test_parser() -> PolarsResult<()> {
    let s = r#"
 "sepal_length","sepal_width","petal_length","petal_width","variety"
 5.1,3.5,1.4,.2,"Setosa"
 4.9,3,1.4,.2,"Setosa"
 4.7,3.2,1.3,.2,"Setosa"
 4.6,3.1,1.5,.2,"Setosa"
 5,3.6,1.4,.2,"Setosa"
 5.4,3.9,1.7,.4,"Setosa"
 4.6,3.4,1.4,.3,"Setosa"
"#;

    let file = Cursor::new(s);
    CsvReadOptions::default()
        .with_infer_schema_length(Some(100))
        .with_has_header(true)
        .with_ignore_errors(true)
        .into_reader_with_file_handle(file)
        .finish()
        .unwrap();

    let s = r#"
         "sepal_length","sepal_width","petal_length","petal_width","variety"
         5.1,3.5,1.4,.2,"Setosa"
         5.1,3.5,1.4,.2,"Setosa"
 "#;

    let file = Cursor::new(s);

    // just checks if unwrap doesn't panic
    CsvReadOptions::default()
        // we also check if infer schema ignores errors
        .with_infer_schema_length(Some(10))
        .with_has_header(true)
        .with_ignore_errors(true)
        .into_reader_with_file_handle(file)
        .finish()
        .unwrap();

    let s = r#""sepal_length","sepal_width","petal_length","petal_width","variety"
        5.1,3.5,1.4,.2,"Setosa"
        4.9,3,1.4,.2,"Setosa"
        4.7,3.2,1.3,.2,"Setosa"
        4.6,3.1,1.5,.2,"Setosa"
        5,3.6,1.4,.2,"Setosa"
        5.4,3.9,1.7,.4,"Setosa"
        4.6,3.4,1.4,.3,"Setosa"
"#;

    let file = Cursor::new(s);
    let df = CsvReadOptions::default()
        .with_infer_schema_length(Some(100))
        .with_has_header(true)
        .into_reader_with_file_handle(file)
        .finish()
        .unwrap();

    let col = df.column("variety").unwrap();
    assert_eq!(col.get(0)?, AnyValue::String("Setosa"));
    assert_eq!(col.get(2)?, AnyValue::String("Setosa"));

    assert_eq!("sepal_length", df.get_columns()[0].name());
    assert_eq!(1, df.column("sepal_length").unwrap().chunks().len());
    assert_eq!(df.height(), 7);

    // test windows line endings
    let s = "head_1,head_2\r\n1,2\r\n1,2\r\n1,2\r\n";

    let file = Cursor::new(s);
    let df = CsvReadOptions::default()
        .with_infer_schema_length(Some(100))
        .with_has_header(true)
        .into_reader_with_file_handle(file)
        .finish()
        .unwrap();

    assert_eq!("head_1", df.get_columns()[0].name());
    assert_eq!(df.shape(), (3, 2));

    // test windows line ending with 1 byte char column and no line endings for last line.
    let s = "head_1\r\n1\r\n2\r\n3";

    let file = Cursor::new(s);
    let df = CsvReadOptions::default()
        .with_infer_schema_length(Some(100))
        .with_has_header(true)
        .into_reader_with_file_handle(file)
        .finish()
        .unwrap();

    assert_eq!("head_1", df.get_columns()[0].name());
    assert_eq!(df.shape(), (3, 1));
    Ok(())
}

#[test]
fn test_tab_sep() {
    let csv = br#"1003000126	ENKESHAFI	ARDALAN		M.D.	M	I	900 SETON DR		CUMBERLAND	21502	MD	US	Internal Medicine	Y	F	99217	Hospital observation care on day of discharge	N	68	67	68	73.821029412	381.30882353	57.880294118	58.2125
1003000126	ENKESHAFI	ARDALAN		M.D.	M	I	900 SETON DR		CUMBERLAND	21502	MD	US	Internal Medicine	Y	F	99218	Hospital observation care, typically 30 minutes	N	19	19	19	100.88315789	476.94736842	76.795263158	77.469473684
1003000126	ENKESHAFI	ARDALAN		M.D.	M	I	900 SETON DR		CUMBERLAND	21502	MD	US	Internal Medicine	Y	F	99220	Hospital observation care, typically 70 minutes	N	26	26	26	188.11076923	1086.9230769	147.47923077	147.79346154
1003000126	ENKESHAFI	ARDALAN		M.D.	M	I	900 SETON DR		CUMBERLAND	21502	MD	US	Internal Medicine	Y	F	99221	Initial hospital inpatient care, typically 30 minutes per day	N	24	24	24	102.24	474.58333333	80.155	80.943333333
1003000126	ENKESHAFI	ARDALAN		M.D.	M	I	900 SETON DR		CUMBERLAND	21502	MD	US	Internal Medicine	Y	F	99222	Initial hospital inpatient care, typically 50 minutes per day	N	17	17	17	138.04588235	625	108.22529412	109.22
1003000126	ENKESHAFI	ARDALAN		M.D.	M	I	900 SETON DR		CUMBERLAND	21502	MD	US	Internal Medicine	Y	F	99223	Initial hospital inpatient care, typically 70 minutes per day	N	86	82	86	204.85395349	1093.5	159.25906977	161.78093023
1003000126	ENKESHAFI	ARDALAN		M.D.	M	I	900 SETON DR		CUMBERLAND	21502	MD	US	Internal Medicine	Y	F	99232	Subsequent hospital inpatient care, typically 25 minutes per day	N	360	206	360	73.565666667	360.57222222	57.670305556	58.038833333
1003000126	ENKESHAFI	ARDALAN		M.D.	M	I	900 SETON DR		CUMBERLAND	21502	MD	US	Internal Medicine	Y	F	99233	Subsequent hospital inpatient care, typically 35 minutes per day	N	284	148	284	105.34971831	576.98943662	82.512992958	82.805774648
"#.as_ref();

    let file = Cursor::new(csv);
    let df = CsvReadOptions::default()
        .with_infer_schema_length(Some(100))
        .with_has_header(false)
        .with_ignore_errors(true)
        .map_parse_options(|parse_options| parse_options.with_separator(b'\t'))
        .into_reader_with_file_handle(file)
        .finish()
        .unwrap();
    assert_eq!(df.shape(), (8, 26))
}

#[test]
fn test_projection() -> PolarsResult<()> {
    let df = CsvReadOptions::default()
        .with_projection(Some(vec![0, 2].into()))
        .try_into_reader_with_file_path(Some(FOODS_CSV.into()))?
        .finish()?;
    let col_1 = df.select_at_idx(0).unwrap();
    assert_eq!(col_1.get(0)?, AnyValue::String("vegetables"));
    assert_eq!(col_1.get(1)?, AnyValue::String("seafood"));
    assert_eq!(col_1.get(2)?, AnyValue::String("meat"));

    let col_2 = df.select_at_idx(1).unwrap();
    assert_eq!(col_2.get(0)?, AnyValue::Float64(0.5));
    assert_eq!(col_2.get(1)?, AnyValue::Float64(5.0));
    assert_eq!(col_2.get(2)?, AnyValue::Float64(5.0));
    Ok(())
}

#[test]
fn test_missing_data() {
    // missing data should not lead to parser error.
    let csv = r#"column_1,column_2,column_3
1,2,3
1,,3
"#;

    let file = Cursor::new(csv);
    let df = CsvReader::new(file).finish().unwrap();
    assert!(df
        .column("column_1")
        .unwrap()
        .equals(&Series::new("column_1", &[1_i64, 1])));
    assert!(df
        .column("column_2")
        .unwrap()
        .equals_missing(&Series::new("column_2", &[Some(2_i64), None])));
    assert!(df
        .column("column_3")
        .unwrap()
        .equals(&Series::new("column_3", &[3_i64, 3])));
}

#[test]
fn test_escape_comma() {
    let csv = r#"column_1,column_2,column_3
-86.64408227,"Autauga, Alabama, US",11
-86.64408227,"Autauga, Alabama, US",12
"#;
    let file = Cursor::new(csv);
    let df = CsvReader::new(file).finish().unwrap();
    assert_eq!(df.shape(), (2, 3));
    assert!(df
        .column("column_3")
        .unwrap()
        .equals(&Series::new("column_3", &[11_i64, 12])));
}

#[test]
fn test_escape_double_quotes() {
    let csv = r#"column_1,column_2,column_3
-86.64408227,"with ""double quotes"" US",11
-86.64408227,"with ""double quotes followed"", by comma",12
"#;
    let file = Cursor::new(csv);
    let df = CsvReader::new(file).finish().unwrap();
    assert_eq!(df.shape(), (2, 3));
    assert!(df.column("column_2").unwrap().equals(&Series::new(
        "column_2",
        &[
            r#"with "double quotes" US"#,
            r#"with "double quotes followed", by comma"#
        ]
    )));
}

#[test]
fn test_newline_in_custom_quote_char() {
    // newline inside custom quote char (default is ") should parse correctly
    let csv = r#"column_1,column_2
        1,'foo
        bar'
        2,'bar'
"#;

    let file = Cursor::new(csv);
    let df = CsvReadOptions::default()
        .map_parse_options(|parse_options| parse_options.with_quote_char(Some(b'\'')))
        .into_reader_with_file_handle(file)
        .finish()
        .unwrap();
    assert_eq!(df.shape(), (2, 2));
}

#[test]
fn test_escape_2() {
    // this is harder than it looks.
    // Fields:
    // * hello
    // * ","
    // * " "
    // * world
    // * "!"
    let csv = r#"hello,","," ",world,"!"
hello,","," ",world,"!"
hello,","," ",world,"!"
hello,","," ",world,"!"
"#;
    let file = Cursor::new(csv);
    let df = CsvReadOptions::default()
        .with_has_header(false)
        .with_n_threads(Some(1))
        .into_reader_with_file_handle(file)
        .finish()
        .unwrap();

    for (col, val) in &[
        ("column_1", "hello"),
        ("column_2", ","),
        ("column_3", " "),
        ("column_4", "world"),
        ("column_5", "!"),
    ] {
        assert!(df
            .column(col)
            .unwrap()
            .equals(&Series::new(col, &[&**val; 4])));
    }
}

#[test]
fn test_very_long_utf8() {
    let csv = r#"column_1,column_2,column_3
-86.64408227,"Lorem Ipsum is simply dummy text of the printing and typesetting
industry. Lorem Ipsum has been the industry's standard dummy text ever since th
e 1500s, when an unknown printer took a galley of type and scrambled it to make
a type specimen book. It has survived not only five centuries, but also the leap
into electronic typesetting, remaining essentially unchanged. It was popularised
in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages,
and more recently with desktop publishing software like Aldus PageMaker including
versions of Lorem Ipsum.",11
"#;
    let file = Cursor::new(csv);
    let df = CsvReadOptions::default()
        .into_reader_with_file_handle(file)
        .finish()
        .unwrap();

    assert!(df.column("column_2").unwrap().equals(&Series::new(
        "column_2",
        &[
            r#"Lorem Ipsum is simply dummy text of the printing and typesetting
industry. Lorem Ipsum has been the industry's standard dummy text ever since th
e 1500s, when an unknown printer took a galley of type and scrambled it to make
a type specimen book. It has survived not only five centuries, but also the leap
into electronic typesetting, remaining essentially unchanged. It was popularised
in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages,
and more recently with desktop publishing software like Aldus PageMaker including
versions of Lorem Ipsum."#,
        ]
    )));
}

#[test]
fn test_nulls_parser() {
    // test it does not fail on the leading comma.
    let csv = r#"id1,id2,id3,id4,id5,id6,v1,v2,v3
id047,id023,id0000084849,90,96,35790,2,9,93.348148
,id022,id0000031441,50,44,71525,3,11,81.013682
id090,id048,id0000067778,24,2,51862,4,9,
"#;

    let file = Cursor::new(csv);
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .with_n_threads(Some(1))
        .into_reader_with_file_handle(file)
        .finish()
        .unwrap();
    assert_eq!(df.shape(), (3, 9));
}

#[test]
fn test_new_line_escape() {
    let s = r#""sepal_length","sepal_width","petal_length","petal_width","variety"
 5.1,3.5,1.4,.2,"Setosa
 texts after new line character"
 4.9,3,1.4,.2,"Setosa"
 "#;

    let file = Cursor::new(s);
    CsvReadOptions::default()
        .with_has_header(true)
        .into_reader_with_file_handle(file)
        .finish()
        .unwrap();
}

#[test]
fn test_new_line_escape_on_header() {
    let s = r#""length","header with
new line character","width"
5.1,3.5,1.4
"#;
    let file: Cursor<&str> = Cursor::new(s);
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .into_reader_with_file_handle(file)
        .finish()
        .unwrap();
    assert_eq!(df.shape(), (1, 3));
    assert_eq!(
        df.get_column_names(),
        &["length", "header with\nnew line character", "width"]
    );
}

#[test]
fn test_quoted_numeric() {
    // CSV fields may be quoted
    let s = r#""foo","bar"
"4.9","3"
"1.4","2"
"#;

    let file = Cursor::new(s);
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .into_reader_with_file_handle(file)
        .finish()
        .unwrap();
    assert_eq!(df.column("bar").unwrap().dtype(), &DataType::Int64);
    assert_eq!(df.column("foo").unwrap().dtype(), &DataType::Float64);
}

#[test]
fn test_empty_bytes_to_dataframe() {
    let fields = vec![Field::new("test_field", DataType::String)];
    let schema = Schema::from_iter(fields);
    let file = Cursor::new(vec![]);

    let result = CsvReadOptions::default()
        .with_has_header(false)
        .with_columns(Some(schema.iter_names().map(|s| s.to_string()).collect()))
        .with_schema(Some(Arc::new(schema)))
        .into_reader_with_file_handle(file)
        .finish();

    assert!(result.is_ok())
}

#[test]
fn test_carriage_return() {
    let csv = "\"foo\",\"bar\"\r\n\"158252579.00\",\"7.5800\"\r\n\"158252579.00\",\"7.5800\"\r\n";

    let file = Cursor::new(csv);
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .with_n_threads(Some(1))
        .into_reader_with_file_handle(file)
        .finish()
        .unwrap();
    assert_eq!(df.shape(), (2, 2));
}

#[test]
fn test_missing_value() {
    let csv = r#"foo,bar,ham
1,2,3
1,2,3
1,2
"#;

    let file = Cursor::new(csv);
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .with_schema(Some(Arc::new(Schema::from_iter([
            Field::new("foo", DataType::UInt32),
            Field::new("bar", DataType::UInt32),
            Field::new("ham", DataType::UInt32),
        ]))))
        .into_reader_with_file_handle(file)
        .finish()
        .unwrap();
    assert_eq!(df.column("ham").unwrap().len(), 3)
}

#[test]
#[cfg(feature = "temporal")]
fn test_with_dtype() -> PolarsResult<()> {
    // test if timestamps can be parsed as Datetime
    let csv = r#"a,b,c,d,e
AUDCAD,1616455919,0.91212,0.95556,1
AUDCAD,1616455920,0.92212,0.95556,1
AUDCAD,1616455921,0.96212,0.95666,1
"#;
    let file = Cursor::new(csv);
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .with_schema_overwrite(Some(Arc::new(Schema::from_iter([Field::new(
            "b",
            DataType::Datetime(TimeUnit::Nanoseconds, None),
        )]))))
        .with_ignore_errors(true)
        .into_reader_with_file_handle(file)
        .finish()?;

    assert_eq!(
        df.dtypes(),
        &[
            DataType::String,
            DataType::Datetime(TimeUnit::Nanoseconds, None),
            DataType::Float64,
            DataType::Float64,
            DataType::Int64
        ]
    );
    Ok(())
}

#[test]
fn test_skip_rows() -> PolarsResult<()> {
    let csv = r"#doc source pos typeindex type topic
#alpha : 25.0 25.0
#beta : 0.1
0 NA 0 0 57 0
0 NA 0 0 57 0
0 NA 5 5 513 0
";

    let file = Cursor::new(csv);
    let df = CsvReadOptions::default()
        .with_has_header(false)
        .with_skip_rows(3)
        .map_parse_options(|parse_options| parse_options.with_separator(b' '))
        .into_reader_with_file_handle(file)
        .finish()?;

    assert_eq!(df.height(), 3);
    Ok(())
}

#[test]
fn test_projection_idx() -> PolarsResult<()> {
    let csv = r"#0 NA 0 0 57 0
0 NA 0 0 57 0
0 NA 5 5 513 0
";

    let file = Cursor::new(csv);
    let df = CsvReadOptions::default()
        .with_has_header(false)
        .with_projection(Some(Arc::new(vec![4, 5])))
        .map_parse_options(|parse_options| parse_options.with_separator(b' '))
        .into_reader_with_file_handle(file)
        .finish()?;

    assert_eq!(df.width(), 2);

    // this should give out of bounds error
    let file = Cursor::new(csv);
    let out = CsvReadOptions::default()
        .with_has_header(false)
        .with_projection(Some(Arc::new(vec![4, 6])))
        .map_parse_options(|parse_options| parse_options.with_separator(b' '))
        .into_reader_with_file_handle(file)
        .finish();

    assert!(out.is_err());
    Ok(())
}

#[test]
fn test_missing_fields() -> PolarsResult<()> {
    let csv = r"1,2,3,4,5
1,2,3
1,2,3,4,5
1,3,5
";

    let file = Cursor::new(csv);
    let df = CsvReadOptions::default()
        .with_has_header(false)
        .into_reader_with_file_handle(file)
        .finish()?;

    use polars_core::df;
    let expect = df![
        "column_1" => [1, 1, 1, 1],
        "column_2" => [2, 2, 2, 3],
        "column_3" => [3, 3, 3, 5],
        "column_4" => [Some(4), None, Some(4), None],
        "column_5" => [Some(5), None, Some(5), None]
    ]?;
    assert!(df.equals_missing(&expect));
    Ok(())
}

#[test]
fn test_comment_lines() -> PolarsResult<()> {
    let csv = r"1,2,3,4,5
# this is a comment
1,2,3,4,5
# this is also a comment
1,2,3,4,5
";

    let file = Cursor::new(csv);
    let df = CsvReadOptions::default()
        .with_has_header(false)
        .map_parse_options(|parse_options| parse_options.with_comment_prefix(Some("#")))
        .into_reader_with_file_handle(file)
        .finish()?;
    assert_eq!(df.shape(), (3, 5));

    let csv = r"!str,2,3,4,5
!#& this is a comment
!str,2,3,4,5
!#& this is also a comment
!str,2,3,4,5
";

    let file = Cursor::new(csv);
    let df = CsvReadOptions::default()
        .with_has_header(false)
        .map_parse_options(|parse_options| parse_options.with_comment_prefix(Some("!#&")))
        .into_reader_with_file_handle(file)
        .finish()?;
    assert_eq!(df.shape(), (3, 5));

    let csv = r"a,b,c,d,e
1,2,3,4,5
% this is a comment
1,2,3,4,5
% this is also a comment
1,2,3,4,5
";

    let file = Cursor::new(csv);
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .map_parse_options(|parse_options| parse_options.with_comment_prefix(Some("%")))
        .into_reader_with_file_handle(file)
        .finish()?;
    assert_eq!(df.shape(), (3, 5));

    Ok(())
}

#[test]
fn test_null_values_argument() -> PolarsResult<()> {
    let csv = r"1,a,foo
null-value,b,bar
3,null-value,ham
";

    let file = Cursor::new(csv);
    let df = CsvReadOptions::default()
        .map_parse_options(|parse_options| {
            parse_options
                .with_null_values(Some(NullValues::AllColumnsSingle("null-value".to_string())))
        })
        .into_reader_with_file_handle(file)
        .finish()?;
    assert!(df.get_columns()[0].null_count() > 0);
    Ok(())
}

#[test]
fn test_no_newline_at_end() -> PolarsResult<()> {
    let csv = r"a,b
foo,foo
bar,bar";
    let file = Cursor::new(csv);
    let df = CsvReader::new(file).finish()?;

    use polars_core::df;
    let expect = df![
        "a" => ["foo", "bar"],
        "b" => ["foo", "bar"]
    ]?;
    assert!(df.equals(&expect));
    Ok(())
}

#[test]
#[cfg(feature = "temporal")]
fn test_automatic_datetime_parsing() -> PolarsResult<()> {
    let csv = r"timestamp,open,high
2021-01-01 00:00:00,0.00305500,0.00306000
2021-01-01 00:15:00,0.00298800,0.00300400
2021-01-01 00:30:00,0.00298300,0.00300100
2021-01-01 00:45:00,0.00299400,0.00304000
";

    let file = Cursor::new(csv);
    let df = CsvReadOptions::default()
        .map_parse_options(|parse_options| parse_options.with_try_parse_dates(true))
        .into_reader_with_file_handle(file)
        .finish()?;

    let ts = df.column("timestamp")?;
    assert_eq!(
        ts.dtype(),
        &DataType::Datetime(TimeUnit::Microseconds, None)
    );
    assert_eq!(ts.null_count(), 0);

    Ok(())
}

#[test]
#[cfg(feature = "temporal")]
fn test_automatic_datetime_parsing_default_formats() -> PolarsResult<()> {
    let csv = r"ts_dmy,ts_dmy_f,ts_dmy_p
01/01/2021 00:00:00,31-01-2021T00:00:00.123,31-01-2021 11:00
01/01/2021 00:15:00,31-01-2021T00:15:00.123,31-01-2021 01:00
01/01/2021 00:30:00,31-01-2021T00:30:00.123,31-01-2021 01:15
01/01/2021 00:45:00,31-01-2021T00:45:00.123,31-01-2021 01:30
";

    let file = Cursor::new(csv);
    let df = CsvReadOptions::default()
        .map_parse_options(|parse_options| parse_options.with_try_parse_dates(true))
        .into_reader_with_file_handle(file)
        .finish()?;

    for col in df.get_column_names() {
        let ts = df.column(col)?;
        assert_eq!(
            ts.dtype(),
            &DataType::Datetime(TimeUnit::Microseconds, None)
        );
        assert_eq!(ts.null_count(), 0);
    }

    Ok(())
}

#[test]
fn test_no_quotes() -> PolarsResult<()> {
    let rolling_stones = r#"linenum,last_name,first_name
1,Jagger,Mick
2,O"Brian,Mary
3,Richards,Keith
4,L"Etoile,Bennet
5,Watts,Charlie
6,Smith,D"Shawn
7,Wyman,Bill
8,Woods,Ron
9,Jones,Brian
"#;

    let file = Cursor::new(rolling_stones);
    let df = CsvReadOptions::default()
        .map_parse_options(|parse_options| parse_options.with_quote_char(None))
        .into_reader_with_file_handle(file)
        .finish()?;
    assert_eq!(df.shape(), (9, 3));

    Ok(())
}

#[test]
fn test_utf8() -> PolarsResult<()> {
    // first part is valid ascii. later we have removed some bytes from the emoji.
    let invalid_utf8 = [
        111, 10, 98, 97, 114, 10, 104, 97, 109, 10, 115, 112, 97, 109, 10, 106, 97, 109, 10, 107,
        97, 109, 10, 108, 97, 109, 10, 207, 128, 10, 112, 97, 109, 10, 115, 116, 97, 109, 112, 10,
        240, 159, 137, 10, 97, 115, 99, 105, 105, 10, 240, 159, 144, 172, 10, 99, 105, 97, 111,
    ];
    let file = Cursor::new(invalid_utf8);
    assert!(CsvReader::new(file).finish().is_err());

    Ok(())
}

#[test]
fn test_header_inference() -> PolarsResult<()> {
    let csv = r#"not_a_header,really,even_if,it_looks_like_one
1,2,3,4
4,3,2,1
"#;
    let file = Cursor::new(csv);
    let df = CsvReadOptions::default()
        .with_has_header(false)
        .into_reader_with_file_handle(file)
        .finish()?;
    assert_eq!(df.dtypes(), vec![DataType::String; 4]);
    Ok(())
}

#[test]
fn test_header_with_comments() -> PolarsResult<()> {
    let csv = "# ignore me\na,b,c\nd,e,f";

    let file = Cursor::new(csv);
    let df = CsvReadOptions::default()
        .map_parse_options(|parse_options| parse_options.with_comment_prefix(Some("#")))
        .into_reader_with_file_handle(file)
        .finish()?;
    // 1 row.
    assert_eq!(df.shape(), (1, 3));

    Ok(())
}

#[test]
#[cfg(feature = "temporal")]
fn test_ignore_parse_dates() -> PolarsResult<()> {
    // if parse dates is set, a given schema should still prevail above date parsing.
    let csv = r#"a,b,c
1,i,16200126
2,j,16250130
3,k,17220012
4,l,17290009"#;

    use DataType::*;
    let file = Cursor::new(csv);
    let df = CsvReadOptions::default()
        .with_dtype_overwrite(Some(vec![String, String, String].into()))
        .map_parse_options(|parse_options| parse_options.with_try_parse_dates(true))
        .into_reader_with_file_handle(file)
        .finish()?;

    assert_eq!(df.dtypes(), &[String, String, String]);
    Ok(())
}

#[test]
fn test_projection_and_quoting() -> PolarsResult<()> {
    let csv = "a,b,c,d
A1,'B1',C1,1
A2,\"B2\",C2,2
A3,\"B3\",C3,3
A3,\"B4_\"\"with_embedded_double_quotes\"\"\",C4,4";

    let file = Cursor::new(csv);
    let df = CsvReader::new(file).finish()?;
    assert_eq!(df.shape(), (4, 4));

    let file = Cursor::new(csv);
    let df = CsvReadOptions::default()
        .with_n_threads(Some(1))
        .with_projection(Some(vec![0, 2].into()))
        .into_reader_with_file_handle(file)
        .finish()?;
    assert_eq!(df.shape(), (4, 2));

    let file = Cursor::new(csv);
    let df = CsvReadOptions::default()
        .with_n_threads(Some(1))
        .with_projection(Some(vec![1].into()))
        .into_reader_with_file_handle(file)
        .finish()?;
    assert_eq!(df.shape(), (4, 1));

    Ok(())
}

#[test]
fn test_infer_schema_0_rows() -> PolarsResult<()> {
    let csv = r#"a,b,c,d
1,a,1.0,true
1,a,1.0,false
"#;
    let file = Cursor::new(csv);
    let df = CsvReadOptions::default()
        .with_infer_schema_length(Some(0))
        .into_reader_with_file_handle(file)
        .finish()?;
    assert_eq!(
        df.dtypes(),
        &[
            DataType::String,
            DataType::String,
            DataType::String,
            DataType::String
        ]
    );
    Ok(())
}

#[test]
fn test_infer_schema_eol() -> PolarsResult<()> {
    // no eol after header
    let no_eol = "colx,coly\nabcdef,1234";
    let file = Cursor::new(no_eol);
    let df = CsvReader::new(file).finish()?;
    assert_eq!(df.dtypes(), &[DataType::String, DataType::Int64,]);
    Ok(())
}

#[test]
fn test_whitespace_separators() -> PolarsResult<()> {
    let tsv = "\ta\tb\tc\n1\ta1\tb1\tc1\n2\ta2\tb2\tc2\n".to_string();

    let contents = vec![
        (tsv.replace('\t', " "), b' '),
        (tsv.replace('\t', "-"), b'-'),
        (tsv, b'\t'),
    ];

    for (content, sep) in contents {
        let file = Cursor::new(&content);
        let df = CsvReadOptions::default()
            .map_parse_options(|parse_options| parse_options.with_separator(sep))
            .into_reader_with_file_handle(file)
            .finish()?;

        assert_eq!(df.shape(), (2, 4));
        assert_eq!(df.get_column_names(), &["", "a", "b", "c"]);
    }

    Ok(())
}

#[test]
fn test_scientific_floats() -> PolarsResult<()> {
    let csv = r#"foo,bar
10000001,1e-5
10000002,.04
"#;
    let file = Cursor::new(csv);
    let df = CsvReader::new(file).finish()?;
    assert_eq!(df.shape(), (2, 2));
    assert_eq!(df.dtypes(), &[DataType::Int64, DataType::Float64]);

    Ok(())
}

#[test]
fn test_tsv_header_offset() -> PolarsResult<()> {
    let csv = "foo\tbar\n\t1000011\t1\n\t1000026\t2\n\t1000949\t2";
    let file = Cursor::new(csv);
    let df = CsvReadOptions::default()
        .map_parse_options(|parse_options| {
            parse_options
                .with_truncate_ragged_lines(true)
                .with_separator(b'\t')
        })
        .into_reader_with_file_handle(file)
        .finish()?;

    assert_eq!(df.shape(), (3, 2));
    assert_eq!(df.dtypes(), &[DataType::String, DataType::Int64]);
    let a = df.column("foo")?;
    let a = a.str()?;
    assert_eq!(a.get(0), None);

    Ok(())
}

#[test]
fn test_null_values_infer_schema() -> PolarsResult<()> {
    let csv = r#"a,b
1,2
3,NA
5,6"#;
    let file = Cursor::new(csv);
    let df = CsvReadOptions::default()
        .map_parse_options(|parse_options| {
            parse_options.with_null_values(Some(NullValues::AllColumnsSingle("NA".into())))
        })
        .into_reader_with_file_handle(file)
        .finish()?;
    let expected = &[DataType::Int64, DataType::Int64];
    assert_eq!(df.dtypes(), expected);
    Ok(())
}

#[test]
fn test_comma_separated_field_in_tsv() -> PolarsResult<()> {
    let csv = "first\tsecond\n1\t2.3,2.4\n3\t4.5,4.6\n";
    let file = Cursor::new(csv);
    let df = CsvReadOptions::default()
        .map_parse_options(|parse_options| parse_options.with_separator(b'\t'))
        .into_reader_with_file_handle(file)
        .finish()?;
    assert_eq!(df.dtypes(), &[DataType::Int64, DataType::String]);
    Ok(())
}

#[test]
fn test_quoted_projection() -> PolarsResult<()> {
    let csv = r#"c1,c2,c3,c4,c5
a,"b",c,d,1
a,"b",c,d,1
a,b,c,d,1"#;
    let file = Cursor::new(csv);
    let df = CsvReadOptions::default()
        .with_projection(Some(Arc::new(vec![1, 4])))
        .into_reader_with_file_handle(file)
        .finish()?;
    assert_eq!(df.shape(), (3, 2));

    Ok(())
}

#[test]
fn test_last_line_incomplete() -> PolarsResult<()> {
    // test a last line that is incomplete and not finishes with a new line char
    let csv = "b5bbf310dffe3372fd5d37a18339fea5,6a2752ffad059badb5f1f3c7b9e4905d,-2,0.033191,811.619 0.487341,16,GGTGTGAAATTTCACACC,TTTAATTATAATTAAG,+
b5bbf310dffe3372fd5d37a18339fea5,e3fd7b95be3453a34361da84f815687d,-2,0.0335936,821.465 0.490834,1";
    let file = Cursor::new(csv);
    let df = CsvReadOptions::default()
        .with_has_header(false)
        .into_reader_with_file_handle(file)
        .finish()?;
    assert_eq!(df.shape(), (2, 9));
    Ok(())
}

#[test]
fn test_quoted_bool_ints() -> PolarsResult<()> {
    let csv = r#"foo,bar,baz
1,"4","false"
3,"5","false"
5,"6","true"
"#;
    let file = Cursor::new(csv);
    let df = CsvReader::new(file).finish()?;
    let expected = df![
        "foo" => [1, 3, 5],
        "bar" => [4, 5, 6],
        "baz" => [false, false, true],
    ]?;
    assert!(df.equals_missing(&expected));

    Ok(())
}

#[test]
fn test_skip_inference() -> PolarsResult<()> {
    let csv = r#"metadata
line
foo,bar
1,2
3,4
5,6
"#;
    let file = Cursor::new(csv);
    let df = CsvReadOptions::default()
        .with_skip_rows(2)
        .into_reader_with_file_handle(file.clone())
        .finish()?;
    assert_eq!(df.get_column_names(), &["foo", "bar"]);
    assert_eq!(df.shape(), (3, 2));
    let df = CsvReadOptions::default()
        .with_skip_rows(2)
        .with_skip_rows_after_header(2)
        .into_reader_with_file_handle(file.clone())
        .finish()?;
    assert_eq!(df.get_column_names(), &["foo", "bar"]);
    assert_eq!(df.shape(), (1, 2));
    let df = CsvReadOptions::default()
        .map_parse_options(|parse_options| parse_options.with_truncate_ragged_lines(true))
        .into_reader_with_file_handle(file)
        .finish()?;
    assert_eq!(df.shape(), (5, 1));

    Ok(())
}

#[test]
fn test_with_row_index() -> PolarsResult<()> {
    let df = CsvReadOptions::default()
        .with_row_index(Some(RowIndex {
            name: "rc".into(),
            offset: 0,
        }))
        .try_into_reader_with_file_path(Some(FOODS_CSV.into()))?
        .finish()?;
    let rc = df.column("rc")?;
    assert_eq!(
        rc.idx()?.into_no_null_iter().collect::<Vec<_>>(),
        (0 as IdxSize..27).collect::<Vec<_>>()
    );
    let df = CsvReadOptions::default()
        .with_row_index(Some(RowIndex {
            name: "rc_2".into(),
            offset: 10,
        }))
        .try_into_reader_with_file_path(Some(FOODS_CSV.into()))?
        .finish()?;
    let rc = df.column("rc_2")?;
    assert_eq!(
        rc.idx()?.into_no_null_iter().collect::<Vec<_>>(),
        (10 as IdxSize..37).collect::<Vec<_>>()
    );
    Ok(())
}

#[test]
fn test_empty_string_cols() -> PolarsResult<()> {
    let csv = "\nabc\n\nxyz\n";
    let file = Cursor::new(csv);
    let df = CsvReadOptions::default()
        .with_has_header(false)
        .into_reader_with_file_handle(file)
        .finish()?;
    let s = df.column("column_1")?;
    let ca = s.str()?;
    assert_eq!(
        ca.iter().collect::<Vec<_>>(),
        &[None, Some("abc"), None, Some("xyz")]
    );

    let csv = ",\nabc,333\n,666\nxyz,999";
    let file = Cursor::new(csv);
    let df = CsvReadOptions::default()
        .with_has_header(false)
        .into_reader_with_file_handle(file)
        .finish()?;
    let expected = df![
        "column_1" => [None, Some("abc"), None, Some("xyz")],
        "column_2" => [None, Some(333i64), Some(666), Some(999)]
    ]?;
    assert!(df.equals_missing(&expected));
    Ok(())
}

#[test]
fn test_empty_col_names() -> PolarsResult<()> {
    let csv = "a,b,c\n1,2,3";
    let file = Cursor::new(csv);
    let df = CsvReader::new(file).finish()?;
    let expected = df![
        "a" => [1i64],
        "b" => [2i64],
        "c" => [3i64]
    ]?;
    assert!(df.equals(&expected));

    let csv = "a,,c\n1,2,3";
    let file = Cursor::new(csv);
    let df = CsvReader::new(file).finish()?;
    let expected = df![
        "a" => [1i64],
        "" => [2i64],
        "c" => [3i64]
    ]?;
    assert!(df.equals(&expected));

    let csv = "a,b,\n1,2,3";
    let file = Cursor::new(csv);
    let df = CsvReader::new(file).finish()?;
    let expected = df![
        "a" => [1i64],
        "b" => [2i64],
        "" => [3i64]
    ]?;
    assert!(df.equals(&expected));

    let csv = "a,b,,\n1,2,3";
    let file = Cursor::new(csv);
    let df_result = CsvReader::new(file).finish()?;
    assert_eq!(df_result.shape(), (1, 4));

    let csv = "a,b\n1,2,3";
    let file = Cursor::new(csv);
    let df_result = CsvReader::new(file).finish();
    assert!(df_result.is_err());
    Ok(())
}

#[test]
fn test_trailing_empty_string_cols() -> PolarsResult<()> {
    let csv = "colx\nabc\nxyz\n\"\"";
    let file = Cursor::new(csv);
    let df = CsvReader::new(file).finish()?;
    let col = df.column("colx")?;
    let col = col.str()?;
    assert_eq!(
        col.into_no_null_iter().collect::<Vec<_>>(),
        &["abc", "xyz", ""]
    );

    let csv = "colx,coly\nabc,def\nxyz,mno\n,";
    let file = Cursor::new(csv);
    let df = CsvReader::new(file).finish()?;

    assert_eq!(
        df.get(1).unwrap(),
        &[AnyValue::String("xyz"), AnyValue::String("mno")]
    );
    assert_eq!(df.get(2).unwrap(), &[AnyValue::Null, AnyValue::Null]);

    Ok(())
}

#[test]
fn test_escaping_quotes() -> PolarsResult<()> {
    let csv = "a\n\"\"\"\"";
    let file = Cursor::new(csv);
    let df = CsvReader::new(file).finish()?;
    let col = df.column("a")?;
    let col = col.str()?;
    assert_eq!(col.into_no_null_iter().collect::<Vec<_>>(), &["\""]);
    Ok(())
}

#[test]
fn test_header_only() -> PolarsResult<()> {
    let csv = "a,b,c";
    let file = Cursor::new(csv);

    // no header
    let df = CsvReadOptions::default()
        .with_has_header(false)
        .into_reader_with_file_handle(file)
        .finish()?;
    assert_eq!(df.shape(), (1, 3));

    // has header
    for csv in &["x,y,z", "x,y,z\n"] {
        let file = Cursor::new(csv);
        let df = CsvReadOptions::default()
            .with_has_header(true)
            .into_reader_with_file_handle(file)
            .finish()?;

        assert_eq!(df.shape(), (0, 3));
        assert_eq!(
            df.dtypes(),
            &[DataType::String, DataType::String, DataType::String]
        );
    }

    Ok(())
}

#[test]
fn test_empty_csv() {
    let csv = "";
    let file = Cursor::new(csv);
    for h in [true, false] {
        assert!(matches!(
            CsvReadOptions::default()
                .with_has_header(h)
                .into_reader_with_file_handle(file.clone())
                .finish(),
            Err(PolarsError::NoData(_))
        ))
    }
}

#[test]
fn test_try_parse_dates() -> PolarsResult<()> {
    let csv = "date
1745-04-02
1742-03-21
1743-06-16
1730-07-22

1739-03-16
";
    let file = Cursor::new(csv);

    let df = CsvReadOptions::default()
        .map_parse_options(|parse_options| parse_options.with_try_parse_dates(true))
        .into_reader_with_file_handle(file)
        .finish()?;

    assert_eq!(df.dtypes(), &[DataType::Date]);
    assert_eq!(df.column("date")?.null_count(), 1);
    Ok(())
}

#[test]
fn test_try_parse_dates_3380() -> PolarsResult<()> {
    let csv = "lat;lon;validdate;t_2m:C;precip_1h:mm
46.685;7.953;2022-05-10T07:07:12Z;6.1;0.00
46.685;7.953;2022-05-10T08:07:12Z;8.8;0.00";
    let file = Cursor::new(csv);
    let df = CsvReadOptions::default()
        .map_parse_options(|parse_options| {
            parse_options
                .with_separator(b';')
                .with_try_parse_dates(true)
        })
        .into_reader_with_file_handle(file)
        .finish()?;

    assert_eq!(df.column("validdate")?.null_count(), 0);
    Ok(())
}

#[test]
fn test_leading_whitespace_with_quote() -> PolarsResult<()> {
    let csv = r#"
"ABC","DEF",
"24.5","  4.1"
"#;
    let file = Cursor::new(csv);
    let df = CsvReader::new(file).finish()?;
    let col_1 = df.column("ABC").unwrap();
    let col_2 = df.column("DEF").unwrap();
    assert_eq!(col_1.get(0)?, AnyValue::Float64(24.5));
    assert_eq!(col_2.get(0)?, AnyValue::String("  4.1"));
    Ok(())
}

#[test]
fn test_read_io_reader() {
    let path = "../../examples/datasets/foods1.csv";
    let file = std::fs::File::open(path).unwrap();
    let mut reader = CsvReadOptions::default()
        .with_chunk_size(5)
        .try_into_reader_with_file_path(Some(path.into()))
        .unwrap();

    let mut reader = reader.batched_borrowed().unwrap();
    let batches = reader.next_batches(5).unwrap().unwrap();
    // TODO: Fix this
    // assert_eq!(batches.len(), 5);
    let df = concat_df(&batches).unwrap();
    let expected = CsvReader::new(file).finish().unwrap();
    assert!(df.equals(&expected))
}
