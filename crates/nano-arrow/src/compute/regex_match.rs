//! Contains regex matching operators [`regex_match`] and [`regex_match_scalar`].

use ahash::AHashMap;
use regex::Regex;

use crate::array::{BooleanArray, Utf8Array};
use crate::bitmap::Bitmap;
use crate::datatypes::DataType;
use crate::error::{Error, Result};
use crate::offset::Offset;

use super::utils::combine_validities;

/// Regex matches
pub fn regex_match<O: Offset>(values: &Utf8Array<O>, regex: &Utf8Array<O>) -> Result<BooleanArray> {
    if values.len() != regex.len() {
        return Err(Error::InvalidArgumentError(
            "Cannot perform comparison operation on arrays of different length".to_string(),
        ));
    }

    let mut map = AHashMap::new();
    let validity = combine_validities(values.validity(), regex.validity());

    let iterator = values.iter().zip(regex.iter()).map(|(haystack, regex)| {
        if haystack.is_none() | regex.is_none() {
            // regex is expensive => short-circuit if null
            return Result::Ok(false);
        };
        let haystack = haystack.unwrap();
        let regex = regex.unwrap();

        let regex = if let Some(regex) = map.get(regex) {
            regex
        } else {
            let re = Regex::new(regex).map_err(|e| {
                Error::InvalidArgumentError(format!("Unable to build regex from LIKE pattern: {e}"))
            })?;
            map.insert(regex, re);
            map.get(regex).unwrap()
        };

        Ok(regex.is_match(haystack))
    });
    let new_values = Bitmap::try_from_trusted_len_iter(iterator)?;

    Ok(BooleanArray::new(DataType::Boolean, new_values, validity))
}

/// Regex matches
/// # Example
/// ```
/// use arrow2::array::{Utf8Array, BooleanArray};
/// use arrow2::compute::regex_match::regex_match_scalar;
///
/// let strings = Utf8Array::<i32>::from_slice(&vec!["ArAow", "A_B", "AAA"]);
///
/// let result = regex_match_scalar(&strings, "^A.A").unwrap();
/// assert_eq!(result, BooleanArray::from_slice(&vec![true, false, true]));
/// ```
pub fn regex_match_scalar<O: Offset>(values: &Utf8Array<O>, regex: &str) -> Result<BooleanArray> {
    let regex = Regex::new(regex)
        .map_err(|e| Error::InvalidArgumentError(format!("Unable to compile regex: {e}")))?;
    Ok(unary_utf8_boolean(values, |x| regex.is_match(x)))
}

fn unary_utf8_boolean<O: Offset, F: Fn(&str) -> bool>(
    values: &Utf8Array<O>,
    op: F,
) -> BooleanArray {
    let validity = values.validity().cloned();

    let iterator = values.iter().map(|value| {
        if value.is_none() {
            return false;
        };
        op(value.unwrap())
    });
    let values = Bitmap::from_trusted_len_iter(iterator);
    BooleanArray::new(DataType::Boolean, values, validity)
}
