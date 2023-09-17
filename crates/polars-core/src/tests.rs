use crate::prelude::*;

#[test]
fn test_initial_empty_sort() -> PolarsResult<()> {
    // https://github.com/pola-rs/polars/issues/1396
    let data = vec![1.3; 42];
    let mut series = Series::new("data", Vec::<f64>::new());
    let series2 = Series::new("data2", data.clone());
    let series3 = Series::new("data3", data);
    let df = DataFrame::new(vec![series2, series3])?;

    for column in df.get_columns().iter() {
        series.append(column)?;
    }
    series.f64()?.sort(false);
    Ok(())
}

#[cfg(feature = "dtype-decimal")]
#[test]
fn test_dataframe_has_the_correct_schema() {
    let to_decimal = |precision, scale, data| {
        Int128Chunked::from_vec("", data)
            .into_decimal_unchecked(Some(precision), scale)
            .into_series()
    };

    let data = df!(
        "i32" => [1_i32],
        "i64" => [1_i64],
        "u32" => [1_u32],
        "u64" => [1_u64],
        "bool" => [true],
        "f32" => [1_f32],
        "f64" => [1_f64],
        "utf8" => ["foo"],
        "decimal" => to_decimal(38, 0, vec![1_i128])
    )
    .unwrap();
    let get_dtype = |name: &str| data.schema().get_field(name).unwrap().data_type().clone();

    assert_eq!(get_dtype("i32"), DataType::Int32);
    assert_eq!(get_dtype("i64"), DataType::Int64);
    assert_eq!(get_dtype("u32"), DataType::UInt32);
    assert_eq!(get_dtype("u64"), DataType::UInt64);
    assert_eq!(get_dtype("f32"), DataType::Float32);
    assert_eq!(get_dtype("f64"), DataType::Float64);
    assert_eq!(get_dtype("bool"), DataType::Boolean);
    assert_eq!(get_dtype("utf8"), DataType::Utf8);

    #[cfg(not(feature = "python"))]
    {
        assert_eq!(get_dtype("decimal"), DataType::Decimal(Some(38), Some(0)));

        for precision in 0..39 {
            for scale in 0..=precision {
                let data = df!(
                    "decimal" => to_decimal(precision, scale, vec![1_i128])
                )
                .unwrap();
                let dtype = data
                    .schema()
                    .get_field("decimal")
                    .unwrap()
                    .data_type()
                    .clone();
                assert_eq!(dtype, DataType::Decimal(Some(precision), Some(scale)));
            }
        }
    }
}
