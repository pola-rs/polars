//! Contains "like" operators such as [`like_utf8`] and [`like_utf8_scalar`].

use ahash::AHashMap;
use regex::bytes::Regex as BytesRegex;
use regex::Regex;

use crate::array::{BinaryArray, BooleanArray, Utf8Array};
use crate::bitmap::Bitmap;
use crate::compute::utils::combine_validities;
use crate::datatypes::DataType;
use crate::error::{Error, Result};
use crate::offset::Offset;

#[inline]
fn is_like_pattern(c: char) -> bool {
    c == '%' || c == '_'
}

/// Transforms a like `pattern` to a regex compatible pattern. To achieve that, it does:
///
/// 1. Replace like wildcards for regex expressions as the pattern will be evaluated using regex match: `%` => `.*` and `_` => `.`
/// 2. Escape regex meta characters to match them and not be evaluated as regex special chars. For example: `.` => `\\.`
/// 3. Replace escaped like wildcards removing the escape characters to be able to match it as a regex. For example: `\\%` => `%`
fn replace_pattern(pattern: &str) -> String {
    let mut result = String::new();
    let text = String::from(pattern);
    let mut chars_iter = text.chars().peekable();
    while let Some(c) = chars_iter.next() {
        if c == '\\' {
            let next = chars_iter.peek();
            match next {
                Some(next) if is_like_pattern(*next) => {
                    result.push(*next);
                    // Skipping the next char as it is already appended
                    chars_iter.next();
                },
                _ => {
                    result.push('\\');
                    result.push('\\');
                },
            }
        } else if regex_syntax::is_meta_character(c) {
            result.push('\\');
            result.push(c);
        } else if c == '%' {
            result.push_str(".*");
        } else if c == '_' {
            result.push('.');
        } else {
            result.push(c);
        }
    }
    result
}

#[inline]
fn a_like_utf8<O: Offset, F: Fn(bool) -> bool>(
    lhs: &Utf8Array<O>,
    rhs: &Utf8Array<O>,
    op: F,
) -> Result<BooleanArray> {
    if lhs.len() != rhs.len() {
        return Err(Error::InvalidArgumentError(
            "Cannot perform comparison operation on arrays of different length".to_string(),
        ));
    }

    let validity = combine_validities(lhs.validity(), rhs.validity());

    let mut map = AHashMap::new();

    let values =
        Bitmap::try_from_trusted_len_iter(lhs.iter().zip(rhs.iter()).map(|(lhs, rhs)| {
            match (lhs, rhs) {
                (Some(lhs), Some(pattern)) => {
                    let pattern = if let Some(pattern) = map.get(pattern) {
                        pattern
                    } else {
                        let re_pattern = replace_pattern(pattern);
                        let re = Regex::new(&format!("^{re_pattern}$")).map_err(|e| {
                            Error::InvalidArgumentError(format!(
                                "Unable to build regex from LIKE pattern: {e}"
                            ))
                        })?;
                        map.insert(pattern, re);
                        map.get(pattern).unwrap()
                    };
                    Result::Ok(op(pattern.is_match(lhs)))
                },
                _ => Ok(false),
            }
        }))?;

    Ok(BooleanArray::new(DataType::Boolean, values, validity))
}

/// Returns `lhs LIKE rhs` operation on two [`Utf8Array`].
///
/// There are two wildcards supported:
///
/// * `%` - The percent sign represents zero, one, or multiple characters
/// * `_` - The underscore represents a single character
///
/// # Error
/// Errors iff:
/// * the arrays have a different length
/// * any of the patterns is not valid
/// # Example
/// ```
/// use arrow2::array::{Utf8Array, BooleanArray};
/// use arrow2::compute::like::like_utf8;
///
/// let strings = Utf8Array::<i32>::from_slice(&["Arrow", "Arrow", "Arrow", "Arrow", "Ar"]);
/// let patterns = Utf8Array::<i32>::from_slice(&["A%", "B%", "%r_ow", "A_", "A_"]);
///
/// let result = like_utf8(&strings, &patterns).unwrap();
/// assert_eq!(result, BooleanArray::from_slice(&[true, false, true, false, true]));
/// ```
pub fn like_utf8<O: Offset>(lhs: &Utf8Array<O>, rhs: &Utf8Array<O>) -> Result<BooleanArray> {
    a_like_utf8(lhs, rhs, |x| x)
}

/// Returns `lhs NOT LIKE rhs` operation on two [`Utf8Array`].
///
/// There are two wildcards supported:
///
/// * `%` - The percent sign represents zero, one, or multiple characters
/// * `_` - The underscore represents a single character
pub fn nlike_utf8<O: Offset>(lhs: &Utf8Array<O>, rhs: &Utf8Array<O>) -> Result<BooleanArray> {
    a_like_utf8(lhs, rhs, |x| !x)
}

fn a_like_utf8_scalar<O: Offset, F: Fn(bool) -> bool>(
    lhs: &Utf8Array<O>,
    rhs: &str,
    op: F,
) -> Result<BooleanArray> {
    let validity = lhs.validity();

    let values = if !rhs.contains(is_like_pattern) {
        Bitmap::from_trusted_len_iter(lhs.values_iter().map(|x| op(x == rhs)))
    } else if rhs.ends_with('%')
        && !rhs.ends_with("\\%")
        && !rhs[..rhs.len() - 1].contains(is_like_pattern)
    {
        // fast path, can use starts_with
        let starts_with = &rhs[..rhs.len() - 1];
        Bitmap::from_trusted_len_iter(lhs.values_iter().map(|x| op(x.starts_with(starts_with))))
    } else if rhs.starts_with('%') && !rhs[1..].contains(is_like_pattern) {
        // fast path, can use ends_with
        let ends_with = &rhs[1..];
        Bitmap::from_trusted_len_iter(lhs.values_iter().map(|x| op(x.ends_with(ends_with))))
    } else {
        let re_pattern = replace_pattern(rhs);
        let re = Regex::new(&format!("^{re_pattern}$")).map_err(|e| {
            Error::InvalidArgumentError(format!("Unable to build regex from LIKE pattern: {e}"))
        })?;
        Bitmap::from_trusted_len_iter(lhs.values_iter().map(|x| op(re.is_match(x))))
    };
    Ok(BooleanArray::new(
        DataType::Boolean,
        values,
        validity.cloned(),
    ))
}

/// Returns `lhs LIKE rhs` operation.
///
/// There are two wildcards supported:
///
/// * `%` - The percent sign represents zero, one, or multiple characters
/// * `_` - The underscore represents a single character
///
/// # Error
/// Errors iff:
/// * the arrays have a different length
/// * any of the patterns is not valid
/// # Example
/// ```
/// use arrow2::array::{Utf8Array, BooleanArray};
/// use arrow2::compute::like::like_utf8_scalar;
///
/// let array = Utf8Array::<i32>::from_slice(&["Arrow", "Arrow", "Arrow", "BA"]);
///
/// let result = like_utf8_scalar(&array, &"A%").unwrap();
/// assert_eq!(result, BooleanArray::from_slice(&[true, true, true, false]));
/// ```
pub fn like_utf8_scalar<O: Offset>(lhs: &Utf8Array<O>, rhs: &str) -> Result<BooleanArray> {
    a_like_utf8_scalar(lhs, rhs, |x| x)
}

/// Returns `lhs NOT LIKE rhs` operation.
///
/// There are two wildcards supported:
///
/// * `%` - The percent sign represents zero, one, or multiple characters
/// * `_` - The underscore represents a single character
pub fn nlike_utf8_scalar<O: Offset>(lhs: &Utf8Array<O>, rhs: &str) -> Result<BooleanArray> {
    a_like_utf8_scalar(lhs, rhs, |x| !x)
}

#[inline]
fn a_like_binary<O: Offset, F: Fn(bool) -> bool>(
    lhs: &BinaryArray<O>,
    rhs: &BinaryArray<O>,
    op: F,
) -> Result<BooleanArray> {
    if lhs.len() != rhs.len() {
        return Err(Error::InvalidArgumentError(
            "Cannot perform comparison operation on arrays of different length".to_string(),
        ));
    }

    let validity = combine_validities(lhs.validity(), rhs.validity());

    let mut map = AHashMap::new();

    let values =
        Bitmap::try_from_trusted_len_iter(lhs.iter().zip(rhs.iter()).map(|(lhs, rhs)| {
            match (lhs, rhs) {
                (Some(lhs), Some(pattern)) => {
                    let pattern = if let Some(pattern) = map.get(pattern) {
                        pattern
                    } else {
                        let re_pattern = simdutf8::basic::from_utf8(pattern).unwrap();
                        let re_pattern = replace_pattern(re_pattern);
                        let re = BytesRegex::new(&format!("^{re_pattern}$")).map_err(|e| {
                            Error::InvalidArgumentError(format!(
                                "Unable to build regex from LIKE pattern: {e}"
                            ))
                        })?;
                        map.insert(pattern, re);
                        map.get(pattern).unwrap()
                    };
                    Result::Ok(op(pattern.is_match(lhs)))
                },
                _ => Ok(false),
            }
        }))?;

    Ok(BooleanArray::new(DataType::Boolean, values, validity))
}

/// Returns `lhs LIKE rhs` operation on two [`BinaryArray`].
///
/// There are two wildcards supported:
///
/// * `%` - The percent sign represents zero, one, or multiple characters
/// * `_` - The underscore represents a single character
///
/// # Error
/// Errors iff:
/// * the arrays have a different length
/// * any of the patterns is not valid
/// # Example
/// ```
/// use arrow2::array::{BinaryArray, BooleanArray};
/// use arrow2::compute::like::like_binary;
///
/// let strings = BinaryArray::<i32>::from_slice(&["Arrow", "Arrow", "Arrow", "Arrow", "Ar"]);
/// let patterns = BinaryArray::<i32>::from_slice(&["A%", "B%", "%r_ow", "A_", "A_"]);
///
/// let result = like_binary(&strings, &patterns).unwrap();
/// assert_eq!(result, BooleanArray::from_slice(&[true, false, true, false, true]));
/// ```
pub fn like_binary<O: Offset>(lhs: &BinaryArray<O>, rhs: &BinaryArray<O>) -> Result<BooleanArray> {
    a_like_binary(lhs, rhs, |x| x)
}

/// Returns `lhs NOT LIKE rhs` operation on two [`BinaryArray`]s.
///
/// There are two wildcards supported:
///
/// * `%` - The percent sign represents zero, one, or multiple characters
/// * `_` - The underscore represents a single character
///
pub fn nlike_binary<O: Offset>(lhs: &BinaryArray<O>, rhs: &BinaryArray<O>) -> Result<BooleanArray> {
    a_like_binary(lhs, rhs, |x| !x)
}

fn a_like_binary_scalar<O: Offset, F: Fn(bool) -> bool>(
    lhs: &BinaryArray<O>,
    rhs: &[u8],
    op: F,
) -> Result<BooleanArray> {
    let validity = lhs.validity();
    let pattern = simdutf8::basic::from_utf8(rhs).map_err(|e| {
        Error::InvalidArgumentError(format!("Unable to convert the LIKE pattern to string: {e}"))
    })?;

    let values = if !pattern.contains(is_like_pattern) {
        Bitmap::from_trusted_len_iter(lhs.values_iter().map(|x| op(x == rhs)))
    } else if pattern.ends_with('%')
        && !pattern.ends_with("\\%")
        && !pattern[..pattern.len() - 1].contains(is_like_pattern)
    {
        // fast path, can use starts_with
        let starts_with = &rhs[..rhs.len() - 1];
        Bitmap::from_trusted_len_iter(lhs.values_iter().map(|x| op(x.starts_with(starts_with))))
    } else if pattern.starts_with('%') && !pattern[1..].contains(is_like_pattern) {
        // fast path, can use ends_with
        let ends_with = &rhs[1..];
        Bitmap::from_trusted_len_iter(lhs.values_iter().map(|x| op(x.ends_with(ends_with))))
    } else {
        let re_pattern = replace_pattern(pattern);
        let re = BytesRegex::new(&format!("^{re_pattern}$")).map_err(|e| {
            Error::InvalidArgumentError(format!("Unable to build regex from LIKE pattern: {e}"))
        })?;
        Bitmap::from_trusted_len_iter(lhs.values_iter().map(|x| op(re.is_match(x))))
    };
    Ok(BooleanArray::new(
        DataType::Boolean,
        values,
        validity.cloned(),
    ))
}

/// Returns `lhs LIKE rhs` operation.
///
/// There are two wildcards supported:
///
/// * `%` - The percent sign represents zero, one, or multiple characters
/// * `_` - The underscore represents a single character
///
/// # Error
/// Errors iff:
/// * the arrays have a different length
/// * any of the patterns is not valid
/// # Example
/// ```
/// use arrow2::array::{BinaryArray, BooleanArray};
/// use arrow2::compute::like::like_binary_scalar;
///
/// let array = BinaryArray::<i32>::from_slice(&["Arrow", "Arrow", "Arrow", "BA"]);
///
/// let result = like_binary_scalar(&array, b"A%").unwrap();
/// assert_eq!(result, BooleanArray::from_slice(&[true, true, true, false]));
/// ```
pub fn like_binary_scalar<O: Offset>(lhs: &BinaryArray<O>, rhs: &[u8]) -> Result<BooleanArray> {
    a_like_binary_scalar(lhs, rhs, |x| x)
}

/// Returns `lhs NOT LIKE rhs` operation on two [`BinaryArray`]s.
///
/// There are two wildcards supported:
///
/// * `%` - The percent sign represents zero, one, or multiple characters
/// * `_` - The underscore represents a single character
///
pub fn nlike_binary_scalar<O: Offset>(lhs: &BinaryArray<O>, rhs: &[u8]) -> Result<BooleanArray> {
    a_like_binary_scalar(lhs, rhs, |x| !x)
}
