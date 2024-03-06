use std::io::{Seek, SeekFrom};

use polars::prelude::*;

#[test]
fn test_cast_join_14872() {
    let df1 = df![
        "ints" => [1]
    ]
    .unwrap();

    let mut df2 = df![
        "ints" => [0, 1],
        "strings" => vec![Series::new("", ["a"]); 2],
    ]
    .unwrap();

    let mut buf = std::io::Cursor::new(vec![]);
    ParquetWriter::new(&mut buf)
        .with_row_group_size(Some(1))
        .finish(&mut df2)
        .unwrap();

    let _ = buf.seek(SeekFrom::Start(0));
    let df2 = ParquetReader::new(buf).finish().unwrap();

    let out = df1
        .join(&df2, ["ints"], ["ints"], JoinArgs::new(JoinType::Left))
        .unwrap();

    let expected = df![
        "ints" => [1],
        "strings" => vec![Series::new("", ["a"]); 1],
    ]
    .unwrap();

    assert!(expected.equals(&out));
}
