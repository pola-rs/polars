//! Running this program outputs:
//!
//! +----------+----------+----------+----------+---------------+
//! | column_1 | column_2 | column_3 | column_4 | column_5      |
//! | ---      | ---      | ---      | ---      | ---           |
//! | f64      | f64      | f64      | f64      | str           |
//! +==========+==========+==========+==========+===============+
//! | 5.1      | 3.5      | 1.4      | 0.2      | "Iris-setosa" |
//! +----------+----------+----------+----------+---------------+
//! | 4.9      | 3        | 1.4      | 0.2      | "Iris-setosa" |
//! +----------+----------+----------+----------+---------------+
//! | 4.7      | 3.2      | 1.3      | 0.2      | "Iris-setosa" |
//! +----------+----------+----------+----------+---------------+
//!
//! +--------------+-------------+-------------+--------------+---------------+
//! | sepal.length | sepal.width | petal.width | petal.length | class         |
//! | ---          | ---         | ---         | ---          | ---           |
//! | f64          | f64         | f64         | f64          | str           |
//! +==============+=============+=============+==============+===============+
//! | 5.1          | 3.5         | 1.4         | 0.2          | "Iris-setosa" |
//! +--------------+-------------+-------------+--------------+---------------+
//! | 4.9          | 3           | 1.4         | 0.2          | "Iris-setosa" |
//! +--------------+-------------+-------------+--------------+---------------+
//! | 4.7          | 3.2         | 1.3         | 0.2          | "Iris-setosa" |
//! +--------------+-------------+-------------+--------------+---------------+
//!
//! +--------------+-------------+-------------+--------------+---------------+
//! | sepal.length | sepal.width | petal.width | petal.length | class         |
//! | ---          | ---         | ---         | ---          | ---           |
//! | f64          | f64         | f64         | f64          | str           |
//! +==============+=============+=============+==============+===============+
//! | 5.1          | 3.5         | 1.4         | 0.2          | "Iris-setosa" |
//! +--------------+-------------+-------------+--------------+---------------+
//! | 4.9          | 3           | 1.4         | 0.2          | "Iris-setosa" |
//! +--------------+-------------+-------------+--------------+---------------+
//! | 4.7          | 3.2         | 1.3         | 0.2          | "Iris-setosa" |
//! +--------------+-------------+-------------+--------------+---------------+
//!
//! +--------------+-------------+-------------+--------------+---------------+
//! | sepal.length | sepal.width | petal.width | petal.length | class         |
//! | ---          | ---         | ---         | ---          | ---           |
//! | f64          | f64         | f64         | f64          | str           |
//! +==============+=============+=============+==============+===============+
//! | 0.006        | 0.008       | 0.002       | 0.001        | "Iris-setosa" |
//! +--------------+-------------+-------------+--------------+---------------+
//! | 0.006        | 0.007       | 0.002       | 0.001        | "Iris-setosa" |
//! +--------------+-------------+-------------+--------------+---------------+
//! | 0.005        | 0.007       | 0.002       | 0.001        | "Iris-setosa" |
//! +--------------+-------------+-------------+--------------+---------------+
//!
//! +--------------+-------------+-------------+--------------+---------------+-------------+
//! | sepal.length | sepal.width | petal.width | petal.length | class         | ohe         |
//! | ---          | ---         | ---         | ---          | ---           | ---         |
//! | f64          | f64         | f64         | f64          | str           | list [u32]  |
//! +==============+=============+=============+==============+===============+=============+
//! | 0.006        | 0.008       | 0.002       | 0.001        | "Iris-setosa" | "[0, 1, 0]" |
//! +--------------+-------------+-------------+--------------+---------------+-------------+
//! | 0.006        | 0.007       | 0.002       | 0.001        | "Iris-setosa" | "[0, 1, 0]" |
//! +--------------+-------------+-------------+--------------+---------------+-------------+
//! | 0.005        | 0.007       | 0.002       | 0.001        | "Iris-setosa" | "[0, 1, 0]" |
//! +--------------+-------------+-------------+--------------+---------------+-------------+
//!
use polars::prelude::*;
use reqwest;
use std::fs::File;
use std::io::Write;
use std::path::Path;

const FEATURES: [&str; 4] = ["sepal.length", "sepal.width", "petal.width", "petal.length"];

fn download_iris() -> std::io::Result<()> {
    let r = reqwest::blocking::get(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
    )
    .expect("could not download iris");
    let mut f = File::create("iris.csv")?;
    f.write_all(r.text().unwrap().as_bytes())
}

fn read_csv() -> Result<DataFrame> {
    let file = File::open("iris.csv").expect("could not read iris file");
    CsvReader::new(file)
        .infer_schema(Some(100))
        .has_header(false)
        .with_batch_size(100)
        .finish()
}

fn rename_cols(mut df: DataFrame) -> Result<DataFrame> {
    (0..5)
        .zip(&[
            "sepal.length",
            "sepal.width",
            "petal.width",
            "petal.length",
            "class",
        ])
        .for_each(|(idx, name)| {
            df[idx].rename(name);
        });

    Ok(df)
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
                    _ => return Err(PolarsError::Other("unexpected type".into())),
                })?;
            }
            Ok(())
        })
        .collect::<Result<_>>()?;
    Ok(df)
}

fn normalize(mut df: DataFrame) -> Result<DataFrame> {
    let cols = &FEATURES;

    for &col in cols {
        df.may_apply(col, |s| {
            let ca = s.f64().unwrap();

            match ca.sum() {
                Some(sum) => Ok(ca / sum),
                None => Err(PolarsError::Other("Nulls in column".into())),
            }
        })?;
    }
    Ok(df)
}

fn one_hot_encode(mut df: DataFrame) -> Result<DataFrame> {
    let y = df["class"].utf8().unwrap();

    let unique = y.unique()?;
    let n_unique = unique.len();

    let mut ohe = y
        .into_iter()
        .map(|opt_s| {
            let mut ohe = vec![0; n_unique];
            let mut idx = 0;
            for i in 0..n_unique {
                if unique.get(i) == opt_s {
                    idx = i;
                    break;
                }
            }
            ohe[idx] = 1;
            match opt_s {
                Some(s) => UInt32Chunked::new_from_slice(s, &ohe).into_series(),
                None => UInt32Chunked::new_from_slice("null", &ohe).into_series(),
            }
        })
        .collect::<Series>();
    ohe.rename("ohe");
    df.add_column(ohe)?;

    Ok(df)
}

fn print_state(df: DataFrame) -> Result<DataFrame> {
    println!("{:?}", df.head(Some(3)));
    Ok(df)
}

fn pipe() -> Result<DataFrame> {
    read_csv()?
        .pipe(print_state)
        .unwrap()
        .pipe(rename_cols)
        .expect("could not rename columns")
        .pipe(print_state)
        .unwrap()
        .pipe(enforce_schema)
        .expect("could not enforce schema")
        .pipe(print_state)
        .unwrap()
        .pipe(normalize)?
        .pipe(print_state)
        .unwrap()
        .pipe(one_hot_encode)
        .expect("could not ohe")
        .pipe(print_state)
}
fn train(df: DataFrame) -> Result<()> {
    let _feat = df.select(&FEATURES)?.to_ndarray::<Float64Type>()?;

    let _target = df.column("ohe")?.list()?.to_ndarray::<Float64Type>()?;
    println!("train loop not implemented");
    Ok(())
}

fn main() {
    if !Path::new("iris.csv").exists() {
        download_iris().expect("could not create file")
    }

    let df = pipe().expect("could not prepare DataFrame");
    train(df).expect("success");
}
