use std::io::Cursor;

use polars::io::RowCount;

use super::*;

const FOODS_CSV: &str = "../examples/datasets/foods1.csv";

#[test]
fn write_csv() {
    let mut buf: Vec<u8> = Vec::new();
    let mut df = create_df();

    CsvWriter::new(&mut buf)
        .has_header(true)
        .finish(&mut df)
        .expect("csv written");
    let csv = std::str::from_utf8(&buf).unwrap();
    assert_eq!("days,temp\n0,22.1\n1,19.9\n2,7.0\n3,2.0\n4,3.0\n", csv);

    let mut buf: Vec<u8> = Vec::new();
    CsvWriter::new(&mut buf)
        .has_header(false)
        .finish(&mut df)
        .expect("csv written");
    let csv = std::str::from_utf8(&buf).unwrap();
    assert_eq!("0,22.1\n1,19.9\n2,7.0\n3,2.0\n4,3.0\n", csv);
}

#[test]
fn test_read_csv_file() {
    let file = std::fs::File::open(FOODS_CSV).unwrap();
    let df = CsvReader::new(file)
        .with_path(Some(FOODS_CSV.to_string()))
        .finish()
        .unwrap();

    assert_eq!(df.shape(), (27, 4));
}

#[test]
fn test_parser() -> PolarsResult<()> {
    let s = r#"
 "sepal.length","sepal.width","petal.length","petal.width","variety"
 5.1,3.5,1.4,.2,"Setosa"
 4.9,3,1.4,.2,"Setosa"
 4.7,3.2,1.3,.2,"Setosa"
 4.6,3.1,1.5,.2,"Setosa"
 5,3.6,1.4,.2,"Setosa"
 5.4,3.9,1.7,.4,"Setosa"
 4.6,3.4,1.4,.3,"Setosa"
"#;

    let file = Cursor::new(s);
    CsvReader::new(file)
        .infer_schema(Some(100))
        .has_header(true)
        .with_ignore_errors(true)
        .finish()
        .unwrap();

    let s = r#"
         "sepal.length","sepal.width","petal.length","petal.width","variety"
         5.1,3.5,1.4,.2,"Setosa"
         5.1,3.5,1.4,.2,"Setosa"
 "#;

    let file = Cursor::new(s);

    // just checks if unwrap doesn't panic
    CsvReader::new(file)
        // we also check if infer schema ignores errors
        .infer_schema(Some(10))
        .has_header(true)
        .with_ignore_errors(true)
        .finish()
        .unwrap();

    let s = r#""sepal.length","sepal.width","petal.length","petal.width","variety"
        5.1,3.5,1.4,.2,"Setosa"
        4.9,3,1.4,.2,"Setosa"
        4.7,3.2,1.3,.2,"Setosa"
        4.6,3.1,1.5,.2,"Setosa"
        5,3.6,1.4,.2,"Setosa"
        5.4,3.9,1.7,.4,"Setosa"
        4.6,3.4,1.4,.3,"Setosa"
"#;

    let file = Cursor::new(s);
    let df = CsvReader::new(file)
        .infer_schema(Some(100))
        .has_header(true)
        .finish()
        .unwrap();

    let col = df.column("variety").unwrap();
    assert_eq!(col.get(0)?, AnyValue::Utf8("Setosa"));
    assert_eq!(col.get(2)?, AnyValue::Utf8("Setosa"));

    assert_eq!("sepal.length", df.get_columns()[0].name());
    assert_eq!(1, df.column("sepal.length").unwrap().chunks().len());
    assert_eq!(df.height(), 7);

    // test windows line endings
    let s = "head_1,head_2\r\n1,2\r\n1,2\r\n1,2\r\n";

    let file = Cursor::new(s);
    let df = CsvReader::new(file)
        .infer_schema(Some(100))
        .has_header(true)
        .finish()
        .unwrap();

    assert_eq!("head_1", df.get_columns()[0].name());
    assert_eq!(df.shape(), (3, 2));

    // test windows line ending with 1 byte char column and no line endings for last line.
    let s = "head_1\r\n1\r\n2\r\n3";

    let file = Cursor::new(s);
    let df = CsvReader::new(file)
        .infer_schema(Some(100))
        .has_header(true)
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
    let df = CsvReader::new(file)
        .infer_schema(Some(100))
        .with_delimiter(b'\t')
        .has_header(false)
        .with_ignore_errors(true)
        .finish()
        .unwrap();
    assert_eq!(df.shape(), (8, 26))
}

#[test]
fn test_projection() -> PolarsResult<()> {
    let df = CsvReader::from_path(FOODS_CSV)
        .unwrap()
        .with_projection(Some(vec![0, 2]))
        .finish()
        .unwrap();
    let col_1 = df.select_at_idx(0).unwrap();
    assert_eq!(col_1.get(0)?, AnyValue::Utf8("vegetables"));
    assert_eq!(col_1.get(1)?, AnyValue::Utf8("seafood"));
    assert_eq!(col_1.get(2)?, AnyValue::Utf8("meat"));

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
        .series_equal(&Series::new("column_1", &[1_i64, 1])));
    assert!(df
        .column("column_2")
        .unwrap()
        .series_equal_missing(&Series::new("column_2", &[Some(2_i64), None])));
    assert!(df
        .column("column_3")
        .unwrap()
        .series_equal(&Series::new("column_3", &[3_i64, 3])));
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
        .series_equal(&Series::new("column_3", &[11_i64, 12])));
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
    assert!(df.column("column_2").unwrap().series_equal(&Series::new(
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
    let df = CsvReader::new(file)
        .with_quote_char(Some(b'\''))
        .finish()
        .unwrap();
    assert_eq!(df.shape(), (2, 2));
}

#[test]
fn test_escape_2() {
    // this is is harder than it looks.
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
    let df = CsvReader::new(file)
        .has_header(false)
        .with_n_threads(Some(1))
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
            .series_equal(&Series::new(col, &[&**val; 4])));
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
    let df = CsvReader::new(file).finish().unwrap();

    assert!(df.column("column_2").unwrap().series_equal(&Series::new(
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
    let df = CsvReader::new(file)
        .has_header(true)
        .with_n_threads(Some(1))
        .finish()
        .unwrap();
    assert_eq!(df.shape(), (3, 9));
}

#[test]
fn test_new_line_escape() {
    let s = r#""sepal.length","sepal.width","petal.length","petal.width","variety"
 5.1,3.5,1.4,.2,"Setosa
 texts after new line character"
 4.9,3,1.4,.2,"Setosa"
 "#;

    let file = Cursor::new(s);
    let _df = CsvReader::new(file).has_header(true).finish().unwrap();
}

#[test]
fn test_quoted_numeric() {
    // CSV fields may be quoted
    let s = r#""foo","bar"
"4.9","3"
"1.4","2"
"#;

    let file = Cursor::new(s);
    let df = CsvReader::new(file).has_header(true).finish().unwrap();
    assert_eq!(df.column("bar").unwrap().dtype(), &DataType::Int64);
    assert_eq!(df.column("foo").unwrap().dtype(), &DataType::Float64);
}

#[test]
fn test_empty_bytes_to_dataframe() {
    let fields = vec![Field::new("test_field", DataType::Utf8)];
    let schema = Schema::from(fields.into_iter());
    let file = Cursor::new(vec![]);

    let result = CsvReader::new(file)
        .has_header(false)
        .with_columns(Some(schema.iter_names().cloned().collect()))
        .with_schema(&schema)
        .finish();
    assert!(result.is_ok())
}

#[test]
fn test_carriage_return() {
    let csv = "\"foo\",\"bar\"\r\n\"158252579.00\",\"7.5800\"\r\n\"158252579.00\",\"7.5800\"\r\n";

    let file = Cursor::new(csv);
    let df = CsvReader::new(file)
        .has_header(true)
        .with_n_threads(Some(1))
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
    let df = CsvReader::new(file)
        .has_header(true)
        .with_schema(&Schema::from(
            vec![
                Field::new("foo", DataType::UInt32),
                Field::new("bar", DataType::UInt32),
                Field::new("ham", DataType::UInt32),
            ]
            .into_iter(),
        ))
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
    let df = CsvReader::new(file)
        .has_header(true)
        .with_dtypes(Some(&Schema::from(
            vec![Field::new(
                "b",
                DataType::Datetime(TimeUnit::Nanoseconds, None),
            )]
            .into_iter(),
        )))
        .finish()?;

    assert_eq!(
        df.dtypes(),
        &[
            DataType::Utf8,
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
    let df = CsvReader::new(file)
        .has_header(false)
        .with_skip_rows(3)
        .with_delimiter(b' ')
        .finish()?;

    dbg!(&df);
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
    let df = CsvReader::new(file)
        .has_header(false)
        .with_projection(Some(vec![4, 5]))
        .with_delimiter(b' ')
        .finish()?;

    assert_eq!(df.width(), 2);

    // this should give out of bounds error
    let file = Cursor::new(csv);
    let out = CsvReader::new(file)
        .has_header(false)
        .with_projection(Some(vec![4, 6]))
        .with_delimiter(b' ')
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
    let df = CsvReader::new(file).has_header(false).finish()?;

    use polars_core::df;
    let expect = df![
        "column_1" => [1, 1, 1, 1],
        "column_2" => [2, 2, 2, 3],
        "column_3" => [3, 3, 3, 5],
        "column_4" => [Some(4), None, Some(4), None],
        "column_5" => [Some(5), None, Some(5), None]
    ]?;
    assert!(df.frame_equal_missing(&expect));
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
    let df = CsvReader::new(file)
        .has_header(false)
        .with_comment_char(Some(b'#'))
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
    let df = CsvReader::new(file)
        .has_header(true)
        .with_comment_char(Some(b'%'))
        .finish()?;
    assert_eq!(df.shape(), (3, 5));

    Ok(())
}

#[test]
fn test_null_values_argument() -> PolarsResult<()> {
    let csv = r"1,a,foo
null-value,b,bar,
3,null-value,ham
";

    let file = Cursor::new(csv);
    let df = CsvReader::new(file)
        .has_header(false)
        .with_null_values(NullValues::AllColumnsSingle("null-value".to_string()).into())
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
    assert!(df.frame_equal(&expect));
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
    let df = CsvReader::new(file).with_parse_dates(true).finish()?;

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
01/01/21 00:00:00,31-01-2021T00:00:00.123,31-01-2021 11:00 AM
01/01/21 00:15:00,31-01-2021T00:15:00.123,31-01-2021 01:00 PM
01/01/21 00:30:00,31-01-2021T00:30:00.123,31-01-2021 01:15 PM
01/01/21 00:45:00,31-01-2021T00:45:00.123,31-01-2021 01:30 PM
";

    let file = Cursor::new(csv);
    let df = CsvReader::new(file).with_parse_dates(true).finish()?;

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
    let df = CsvReader::new(file).with_quote_char(None).finish()?;
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
    let df = CsvReader::new(file).has_header(false).finish()?;
    assert_eq!(df.dtypes(), vec![DataType::Utf8; 4]);
    Ok(())
}

#[test]
fn test_header_with_comments() -> PolarsResult<()> {
    let csv = "# ignore me\na,b,c\nd,e,f";

    let file = Cursor::new(csv);
    let df = CsvReader::new(file)
        .with_comment_char(Some(b'#'))
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
    let df = CsvReader::new(file)
        .with_parse_dates(true)
        .with_dtypes_slice(Some(&[Utf8, Utf8, Utf8]))
        .finish()?;

    assert_eq!(df.dtypes(), &[Utf8, Utf8, Utf8]);
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
    let df = CsvReader::new(file)
        .with_n_threads(Some(1))
        .with_projection(Some(vec![0, 2]))
        .finish()?;
    assert_eq!(df.shape(), (4, 2));

    let file = Cursor::new(csv);
    let df = CsvReader::new(file)
        .with_n_threads(Some(1))
        .with_projection(Some(vec![1]))
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
    let df = CsvReader::new(file).infer_schema(Some(0)).finish()?;
    assert_eq!(
        df.dtypes(),
        &[
            DataType::Utf8,
            DataType::Utf8,
            DataType::Utf8,
            DataType::Utf8
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
    assert_eq!(df.dtypes(), &[DataType::Utf8, DataType::Int64,]);
    Ok(())
}

#[test]
fn test_whitespace_delimiters() -> PolarsResult<()> {
    let tsv = "\ta\tb\tc\n1\ta1\tb1\tc1\n2\ta2\tb2\tc2\n".to_string();
    let mut contents = Vec::with_capacity(3);
    contents.push((tsv.replace('\t', " "), b' '));
    contents.push((tsv.replace('\t', "-"), b'-'));
    contents.push((tsv, b'\t'));

    for (content, sep) in contents {
        let file = Cursor::new(&content);
        let df = CsvReader::new(file).with_delimiter(sep).finish()?;

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
    let df = CsvReader::new(file).with_delimiter(b'\t').finish()?;

    assert_eq!(df.shape(), (3, 2));
    assert_eq!(df.dtypes(), &[DataType::Utf8, DataType::Int64]);
    let a = df.column("foo")?;
    let a = a.utf8()?;
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
    let df = CsvReader::new(file)
        .with_null_values(Some(NullValues::AllColumnsSingle("NA".into())))
        .finish()?;
    let expected = &[DataType::Int64, DataType::Int64];
    assert_eq!(df.dtypes(), expected);
    Ok(())
}

#[test]
fn test_comma_separated_field_in_tsv() -> PolarsResult<()> {
    let csv = "first\tsecond\n1\t2.3,2.4\n3\t4.5,4.6\n";
    let file = Cursor::new(csv);
    let df = CsvReader::new(file).with_delimiter(b'\t').finish()?;
    assert_eq!(df.dtypes(), &[DataType::Int64, DataType::Utf8]);
    Ok(())
}

#[test]
fn test_quoted_projection() -> PolarsResult<()> {
    let csv = r#"c1,c2,c3,c4,c5
a,"b",c,d,1
a,"b",c,d,1
a,b,c,d,1"#;
    let file = Cursor::new(csv);
    let df = CsvReader::new(file)
        .with_projection(Some(vec![1, 4]))
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
    let df = CsvReader::new(file).has_header(false).finish()?;
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
    assert!(df.frame_equal_missing(&expected));

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
    let df = CsvReader::new(file.clone()).with_skip_rows(2).finish()?;
    assert_eq!(df.get_column_names(), &["foo", "bar"]);
    assert_eq!(df.shape(), (3, 2));
    let df = CsvReader::new(file.clone())
        .with_skip_rows(2)
        .with_skip_rows_after_header(2)
        .finish()?;
    assert_eq!(df.get_column_names(), &["foo", "bar"]);
    assert_eq!(df.shape(), (1, 2));
    let df = CsvReader::new(file).finish()?;
    assert_eq!(df.shape(), (5, 1));

    Ok(())
}

#[test]
fn test_with_row_count() -> PolarsResult<()> {
    let df = CsvReader::from_path(FOODS_CSV)?
        .with_row_count(Some(RowCount {
            name: "rc".into(),
            offset: 0,
        }))
        .finish()?;
    let rc = df.column("rc")?;
    assert_eq!(
        rc.idx()?.into_no_null_iter().collect::<Vec<_>>(),
        (0 as IdxSize..27).collect::<Vec<_>>()
    );
    let df = CsvReader::from_path(FOODS_CSV)?
        .with_row_count(Some(RowCount {
            name: "rc_2".into(),
            offset: 10,
        }))
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
    let df = CsvReader::new(file).has_header(false).finish()?;
    let s = df.column("column_1")?;
    let ca = s.utf8()?;
    assert_eq!(
        ca.into_iter().collect::<Vec<_>>(),
        &[None, Some("abc"), None, Some("xyz")]
    );

    let csv = ",\nabc,333\n,666\nxyz,999";
    let file = Cursor::new(csv);
    let df = CsvReader::new(file).has_header(false).finish()?;
    let expected = df![
        "column_1" => [None, Some("abc"), None, Some("xyz")],
        "column_2" => [None, Some(333i64), Some(666), Some(999)]
    ]?;
    assert!(df.frame_equal_missing(&expected));
    Ok(())
}

#[test]
fn test_trailing_empty_string_cols() -> PolarsResult<()> {
    let csv = "colx\nabc\nxyz\n\"\"";
    let file = Cursor::new(csv);
    let df = CsvReader::new(file).finish()?;
    let col = df.column("colx")?;
    let col = col.utf8()?;
    assert_eq!(
        col.into_no_null_iter().collect::<Vec<_>>(),
        &["abc", "xyz", ""]
    );

    let csv = "colx,coly\nabc,def\nxyz,mno\n,";
    let file = Cursor::new(csv);
    let df = CsvReader::new(file).finish()?;

    assert_eq!(
        df.get(1).unwrap(),
        &[AnyValue::Utf8("xyz"), AnyValue::Utf8("mno")]
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
    let col = col.utf8()?;
    assert_eq!(col.into_no_null_iter().collect::<Vec<_>>(), &["\""]);
    Ok(())
}

#[test]
fn test_header_only() -> PolarsResult<()> {
    let csv = "a,b,c";
    let file = Cursor::new(csv);

    // no header
    let df = CsvReader::new(file).has_header(false).finish()?;
    assert_eq!(df.shape(), (1, 3));

    // has header
    for csv in &["x,y,z", "x,y,z\n"] {
        let file = Cursor::new(csv);
        let df = CsvReader::new(file).has_header(true).finish()?;

        assert_eq!(df.shape(), (0, 3));
        assert_eq!(
            df.dtypes(),
            &[DataType::Utf8, DataType::Utf8, DataType::Utf8]
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
            CsvReader::new(file.clone()).has_header(h).finish(),
            Err(PolarsError::NoData(_))
        ))
    }
}

#[test]
fn test_parse_dates() -> PolarsResult<()> {
    let csv = "date
1745-04-02
1742-03-21
1743-06-16
1730-07-22
''
1739-03-16
";
    let file = Cursor::new(csv);

    let out = CsvReader::new(file).with_parse_dates(true).finish()?;
    assert_eq!(out.dtypes(), &[DataType::Date]);
    assert_eq!(out.column("date")?.null_count(), 1);
    Ok(())
}

#[test]
fn test_whitespace_skipping() -> PolarsResult<()> {
    let csv = "a,b
  12,   1435";
    let file = Cursor::new(csv);
    let out = CsvReader::new(file).finish()?;
    let expected = df![
        "a" => [12i64],
        "b" => [1435i64],
    ]?;
    assert!(out.frame_equal(&expected));

    Ok(())
}

#[test]
fn test_parse_dates_3380() -> PolarsResult<()> {
    let csv = "lat;lon;validdate;t_2m:C;precip_1h:mm
46.685;7.953;2022-05-10T07:07:12Z;6.1;0.00
46.685;7.953;2022-05-10T08:07:12Z;8.8;0.00";
    let file = Cursor::new(csv);
    let df = CsvReader::new(file)
        .with_delimiter(b';')
        .with_parse_dates(true)
        .finish()?;
    assert_eq!(df.column("validdate")?.null_count(), 0);
    Ok(())
}
