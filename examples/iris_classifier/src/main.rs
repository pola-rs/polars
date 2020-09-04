use polars::prelude::*;
use std::io::Cursor;

fn read_csv() -> Result<DataFrame> {
    let s = r#""sepal.length","sepal.width","petal.length","petal.width","variety"
5.1,3.5,1.4,.2,"Setosa"
4.9,3,1.4,.2,"Setosa"
4.7,3.2,1.3,.2,"Setosa"
4.6,3.1,1.5,.2,"Setosa"
5,3.6,1.4,.2,"Setosa"
5.4,3.9,1.7,.4,"Setosa"
4.6,3.4,1.4,.3,"Setosa""#;

    let file = Cursor::new(s);
    CsvReader::new(file)
        .infer_schema(Some(100))
        .has_header(true)
        .with_batch_size(100)
        .finish()
}

fn enforce_schema(mut df: DataFrame) -> Result<DataFrame> {
    let dtypes = &[
        ArrowDataType::Float64,
        ArrowDataType::Float64,
        ArrowDataType::Float64,
        ArrowDataType::Float64,
        ArrowDataType::Utf8,
    ];

    df.schema()
        .clone()
        .fields()
        .iter()
        .zip(dtypes)
        .map(|(field, dtype)| {
            if field.data_type() != dtype {
                df.may_apply(field.name(), |col| match dtype {
                    ArrowDataType::Float64 => col.cast::<Float64Type>(),
                    ArrowDataType::Utf8 => col.cast::<Utf8Type>(),
                    _ => return Err(PolarsError::Other("unexpected type".to_string())),
                })?;
            }
            Ok(())
        })
        .collect::<Result<_>>()?;
    Ok(df)
}

fn normalize(mut df: DataFrame) -> Result<DataFrame> {
    let cols = &["sepal.length", "sepal.width", "petal.length", "petal.width"];

    for &col in cols {
        df.may_apply(col, |s| {
            let ca = s.f64().unwrap();

            match ca.sum() {
                Some(sum) => Ok(ca / sum),
                None => Err(PolarsError::Other("Nulls in column".to_string())),
            }
        })?;
    }
    Ok(df)
}

fn pipe() -> Result<DataFrame> {
    read_csv()?.pipe(enforce_schema)?.pipe(normalize)
}

fn main() {
    let df = pipe().unwrap();

    println!("{:?}", df);
}
