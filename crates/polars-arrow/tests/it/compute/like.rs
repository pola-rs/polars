use polars_arrow::array::*;
use polars_arrow::compute::like::*;
use polars_arrow::error::Result;

#[test]
fn test_like_binary() -> Result<()> {
    let strings = BinaryArray::<i32>::from_slice(["Arrow", "Arrow", "Arrow", "Arrow", "Ar"]);
    let patterns = BinaryArray::<i32>::from_slice(["A%", "B%", "%r_ow", "A_", "A_"]);
    let result = like_binary(&strings, &patterns).unwrap();
    assert_eq!(
        result,
        BooleanArray::from_slice([true, false, true, false, true])
    );
    Ok(())
}

#[test]
fn test_nlike_binary() -> Result<()> {
    let strings = BinaryArray::<i32>::from_slice(["Arrow", "Arrow", "Arrow", "Arrow", "Ar"]);
    let patterns = BinaryArray::<i32>::from_slice(["A%", "B%", "%r_ow", "A_", "A_"]);
    let result = nlike_binary(&strings, &patterns).unwrap();
    assert_eq!(
        result,
        BooleanArray::from_slice([false, true, false, true, false])
    );
    Ok(())
}

#[test]
fn test_like_binary_scalar() -> Result<()> {
    let array = BinaryArray::<i32>::from_slice(["Arrow", "Arrow", "Arrow", "BA"]);

    let result = like_binary_scalar(&array, b"A%").unwrap();
    assert_eq!(result, BooleanArray::from_slice([true, true, true, false]));

    let result = like_binary_scalar(&array, b"Arrow").unwrap();
    assert_eq!(result, BooleanArray::from_slice([true, true, true, false]));

    Ok(())
}

#[test]
fn test_like_utf8_scalar() -> Result<()> {
    let array = Utf8Array::<i32>::from_slice(["Arrow", "Arrow", "Arrow", "BA"]);

    let result = like_utf8_scalar(&array, "A%").unwrap();
    assert_eq!(result, BooleanArray::from_slice([true, true, true, false]));

    let result = like_utf8_scalar(&array, "Arrow").unwrap();
    assert_eq!(result, BooleanArray::from_slice([true, true, true, false]));

    let array = Utf8Array::<i32>::from_slice(["A%", "Arrow"]);

    let result = like_utf8_scalar(&array, "A\\%").unwrap();
    assert_eq!(result, BooleanArray::from_slice([true, false]));

    let array = Utf8Array::<i32>::from_slice(["A_row", "Arrow"]);
    let result = like_utf8_scalar(&array, "A\\_row").unwrap();
    assert_eq!(result, BooleanArray::from_slice([true, false]));

    Ok(())
}

#[test]
fn test_nlike_binary_scalar() -> Result<()> {
    let array = BinaryArray::<i32>::from_slice(["Arrow", "Arrow", "Arrow", "BA"]);

    let result = nlike_binary_scalar(&array, "A%".as_bytes()).unwrap();
    assert_eq!(
        result,
        BooleanArray::from_slice([false, false, false, true])
    );

    let result = nlike_binary_scalar(&array, "Arrow".as_bytes()).unwrap();
    assert_eq!(
        result,
        BooleanArray::from_slice([false, false, false, true])
    );

    Ok(())
}
