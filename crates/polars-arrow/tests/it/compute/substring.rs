use polars_arrow::array::*;
use polars_arrow::compute::substring::*;
use polars_arrow::error::Result;
use polars_arrow::offset::Offset;

fn with_nulls_utf8<O: Offset>() -> Result<()> {
    let cases = vec![
        // identity
        (
            vec![Some("hello"), None, Some("word")],
            0,
            None,
            vec![Some("hello"), None, Some("word")],
        ),
        // 0 length -> Nothing
        (
            vec![Some("hello"), None, Some("word")],
            0,
            Some(0),
            vec![Some(""), None, Some("")],
        ),
        // high start -> Nothing
        (
            vec![Some("hello"), None, Some("word")],
            1000,
            Some(0),
            vec![Some(""), None, Some("")],
        ),
        // high negative start -> identity
        (
            vec![Some("hello"), None, Some("word")],
            -1000,
            None,
            vec![Some("hello"), None, Some("word")],
        ),
        // high length -> identity
        (
            vec![Some("hello"), None, Some("word")],
            0,
            Some(1000),
            vec![Some("hello"), None, Some("word")],
        ),
    ];

    cases
        .into_iter()
        .try_for_each::<_, Result<()>>(|(array, start, length, expected)| {
            let array = Utf8Array::<O>::from(array);
            let result = substring(&array, start, &length)?;
            assert_eq!(array.len(), result.len());

            let result = result.as_any().downcast_ref::<Utf8Array<O>>().unwrap();
            let expected = Utf8Array::<O>::from(expected);

            assert_eq!(&expected, result);
            Ok(())
        })?;

    Ok(())
}

#[test]
fn with_nulls_string() -> Result<()> {
    with_nulls_utf8::<i32>()
}

#[test]
fn with_nulls_large_string() -> Result<()> {
    with_nulls_utf8::<i64>()
}

fn without_nulls_utf8<O: Offset>() -> Result<()> {
    let cases = vec![
        // increase start
        (
            vec!["hello", "", "word"],
            0,
            None,
            vec!["hello", "", "word"],
        ),
        (vec!["hello", "", "word"], 1, None, vec!["ello", "", "ord"]),
        (vec!["hello", "", "word"], 2, None, vec!["llo", "", "rd"]),
        (vec!["hello", "", "word"], 3, None, vec!["lo", "", "d"]),
        (vec!["hello", "", "word"], 10, None, vec!["", "", ""]),
        // increase start negatively
        (vec!["hello", "", "word"], -1, None, vec!["o", "", "d"]),
        (vec!["hello", "", "word"], -2, None, vec!["lo", "", "rd"]),
        (vec!["hello", "", "word"], -3, None, vec!["llo", "", "ord"]),
        (
            vec!["hello", "", "word"],
            -10,
            None,
            vec!["hello", "", "word"],
        ),
        // increase length
        (vec!["hello", "", "word"], 1, Some(1), vec!["e", "", "o"]),
        (vec!["hello", "", "word"], 1, Some(2), vec!["el", "", "or"]),
        (
            vec!["hello", "", "word"],
            1,
            Some(3),
            vec!["ell", "", "ord"],
        ),
        (
            vec!["hello", "", "word"],
            1,
            Some(4),
            vec!["ello", "", "ord"],
        ),
        (vec!["hello", "", "word"], -3, Some(1), vec!["l", "", "o"]),
        (vec!["hello", "", "word"], -3, Some(2), vec!["ll", "", "or"]),
        (
            vec!["hello", "", "word"],
            -3,
            Some(3),
            vec!["llo", "", "ord"],
        ),
        (
            vec!["hello", "", "word"],
            -3,
            Some(4),
            vec!["llo", "", "ord"],
        ),
        (
            vec!["😇🔥🥺", "", "😇🔥🗺️"],
            0,
            Some(2),
            vec!["😇🔥", "", "😇🔥"],
        ),
        (vec!["π1π", "", "α1απ"], 1, Some(4), vec!["1π", "", "1απ"]),
    ];

    cases
        .into_iter()
        .try_for_each::<_, Result<()>>(|(array, start, length, expected)| {
            let array = Utf8Array::<O>::from_slice(array);
            let result = substring(&array, start, &length)?;
            assert_eq!(array.len(), result.len());
            let result = result.as_any().downcast_ref::<Utf8Array<O>>().unwrap();
            let expected = Utf8Array::<O>::from_slice(expected);
            assert_eq!(&expected, result);
            Ok(())
        })?;

    Ok(())
}

#[test]
fn without_nulls_string() -> Result<()> {
    without_nulls_utf8::<i32>()
}

#[test]
fn without_nulls_large_string() -> Result<()> {
    without_nulls_utf8::<i64>()
}

fn with_null_binarys<O: Offset>() -> Result<()> {
    let cases = vec![
        // identity
        (
            vec![Some(b"hello"), None, Some(b"world")],
            0,
            None,
            vec![Some("hello"), None, Some("world")],
        ),
        // 0 length -> Nothing
        (
            vec![Some(b"hello"), None, Some(b"world")],
            0,
            Some(0),
            vec![Some(""), None, Some("")],
        ),
        // high start -> Nothing
        (
            vec![Some(b"hello"), None, Some(b"world")],
            1000,
            Some(0),
            vec![Some(""), None, Some("")],
        ),
        // high negative start -> identity
        (
            vec![Some(b"hello"), None, Some(b"world")],
            -1000,
            None,
            vec![Some("hello"), None, Some("world")],
        ),
        // high length -> identity
        (
            vec![Some(b"hello"), None, Some(b"world")],
            0,
            Some(1000),
            vec![Some("hello"), None, Some("world")],
        ),
    ];

    cases
        .into_iter()
        .try_for_each::<_, Result<()>>(|(array, start, length, expected)| {
            let array = BinaryArray::<O>::from(array);
            let result = substring(&array, start, &length)?;
            assert_eq!(array.len(), result.len());

            let result = result.as_any().downcast_ref::<BinaryArray<O>>().unwrap();
            let expected = BinaryArray::<O>::from(expected);
            assert_eq!(&expected, result);
            Ok(())
        })?;

    Ok(())
}

#[test]
fn with_nulls_binary() -> Result<()> {
    with_null_binarys::<i32>()
}

#[test]
fn with_nulls_large_binary() -> Result<()> {
    with_null_binarys::<i64>()
}

fn without_null_binarys<O: Offset>() -> Result<()> {
    let cases = vec![
        // increase start
        (
            vec!["hello", "", "word"],
            0,
            None,
            vec!["hello", "", "word"],
        ),
        (vec!["hello", "", "word"], 1, None, vec!["ello", "", "ord"]),
        (vec!["hello", "", "word"], 2, None, vec!["llo", "", "rd"]),
        (vec!["hello", "", "word"], 3, None, vec!["lo", "", "d"]),
        (vec!["hello", "", "word"], 10, None, vec!["", "", ""]),
        // increase start negatively
        (vec!["hello", "", "word"], -1, None, vec!["o", "", "d"]),
        (vec!["hello", "", "word"], -2, None, vec!["lo", "", "rd"]),
        (vec!["hello", "", "word"], -3, None, vec!["llo", "", "ord"]),
        (
            vec!["hello", "", "word"],
            -10,
            None,
            vec!["hello", "", "word"],
        ),
        // increase length
        (vec!["hello", "", "word"], 1, Some(1), vec!["e", "", "o"]),
        (vec!["hello", "", "word"], 1, Some(2), vec!["el", "", "or"]),
        (
            vec!["hello", "", "word"],
            1,
            Some(3),
            vec!["ell", "", "ord"],
        ),
        (
            vec!["hello", "", "word"],
            1,
            Some(4),
            vec!["ello", "", "ord"],
        ),
        (vec!["hello", "", "word"], -3, Some(1), vec!["l", "", "o"]),
        (vec!["hello", "", "word"], -3, Some(2), vec!["ll", "", "or"]),
        (
            vec!["hello", "", "word"],
            -3,
            Some(3),
            vec!["llo", "", "ord"],
        ),
        (
            vec!["hello", "", "word"],
            -3,
            Some(4),
            vec!["llo", "", "ord"],
        ),
    ];

    cases
        .into_iter()
        .try_for_each::<_, Result<()>>(|(array, start, length, expected)| {
            let array = BinaryArray::<O>::from_slice(array);
            let result = substring(&array, start, &length)?;
            assert_eq!(array.len(), result.len());
            let result = result.as_any().downcast_ref::<BinaryArray<O>>().unwrap();
            let expected = BinaryArray::<O>::from_slice(expected);
            assert_eq!(&expected, result);
            Ok(())
        })?;

    Ok(())
}

#[test]
fn without_nulls_binary() -> Result<()> {
    without_null_binarys::<i32>()
}

#[test]
fn without_nulls_large_binary() -> Result<()> {
    without_null_binarys::<i64>()
}

#[test]
fn consistency() {
    use polars_arrow::datatypes::DataType::*;
    use polars_arrow::datatypes::TimeUnit;
    let datatypes = vec![
        Null,
        Boolean,
        UInt8,
        UInt16,
        UInt32,
        UInt64,
        Int8,
        Int16,
        Int32,
        Int64,
        Float32,
        Float64,
        Timestamp(TimeUnit::Second, None),
        Timestamp(TimeUnit::Millisecond, None),
        Timestamp(TimeUnit::Microsecond, None),
        Timestamp(TimeUnit::Nanosecond, None),
        Time64(TimeUnit::Microsecond),
        Time64(TimeUnit::Nanosecond),
        Date32,
        Time32(TimeUnit::Second),
        Time32(TimeUnit::Millisecond),
        Date64,
        Utf8,
        LargeUtf8,
        Binary,
        LargeBinary,
        Duration(TimeUnit::Second),
        Duration(TimeUnit::Millisecond),
        Duration(TimeUnit::Microsecond),
        Duration(TimeUnit::Nanosecond),
    ];

    datatypes.into_iter().for_each(|d1| {
        let array = new_null_array(d1.clone(), 10);
        if can_substring(&d1) {
            assert!(substring(array.as_ref(), 0, &None).is_ok());
        } else {
            assert!(substring(array.as_ref(), 0, &None).is_err());
        }
    });
}
