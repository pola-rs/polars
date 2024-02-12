use polars_parquet::parquet::error::Result;

use crate::read::get_column;
use crate::{get_path, Array};

fn verify_column_data(column: &str) -> Array {
    match column {
        "c0" => {
            let expected = vec![1593604800, 1593604800, 1593604801, 1593604801];
            let expected = expected.into_iter().map(Some).collect::<Vec<_>>();
            Array::Int64(expected)
        },
        "c1" => {
            let expected = vec!["abc", "def", "abc", "def"];
            let expected = expected
                .into_iter()
                .map(|v| Some(v.as_bytes().to_vec()))
                .collect::<Vec<_>>();
            Array::Binary(expected)
        },
        "v11" => {
            let expected = vec![42_f64, 7.7, 42.125, 7.7];
            let expected = expected.into_iter().map(Some).collect::<Vec<_>>();
            Array::Double(expected)
        },
        _ => unreachable!(),
    }
}

#[test]
fn test_lz4_inference() -> Result<()> {
    // - file "hadoop_lz4_compressed.parquet" is compressed using the hadoop Lz4Codec
    // - file "non_hadoop_lz4_compressed.parquet" is "the LZ4 block format without the custom Hadoop header".
    //    see https://github.com/apache/parquet-testing/pull/14

    // Those two files, are all marked as compressed as Lz4, the decompressor should
    // be able to distinguish them from each other.

    let files = [
        "hadoop_lz4_compressed.parquet",
        "non_hadoop_lz4_compressed.parquet",
    ];
    let columns = ["c0", "c1", "v11"];
    for file in files {
        let mut path = get_path();
        path.push(file);
        let path = path.to_str().unwrap();
        for column in columns {
            let (result, _statistics) = get_column(path, column)?;
            assert_eq!(result, verify_column_data(column), "of file {}", file);
        }
    }
    Ok(())
}

#[test]
fn test_lz4_large_file() -> Result<()> {
    //File "hadoop_lz4_compressed_larger.parquet" is compressed using the hadoop Lz4Codec,
    //which contains 10000 rows.

    let mut path = get_path();
    let file = "hadoop_lz4_compressed_larger.parquet";
    path.push(file);
    let path = path.to_str().unwrap();
    let (result, _statistics) = get_column(path, "a")?;
    assert_eq!(result.len(), 10000);
    Ok(())
}
