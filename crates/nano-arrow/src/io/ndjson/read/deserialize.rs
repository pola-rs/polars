use json_deserializer::parse;

use crate::array::Array;
use crate::datatypes::DataType;
use crate::error::Error;

use super::super::super::json::read::_deserialize;

/// Deserializes rows into an [`Array`] of [`DataType`].
/// # Implementation
/// This function is CPU-bounded.
/// This function is guaranteed to return an array of length equal to `rows.len()`.
/// # Errors
/// This function errors iff any of the rows is not a valid JSON (i.e. the format is not valid NDJSON).
pub fn deserialize(rows: &[String], data_type: DataType) -> Result<Box<dyn Array>, Error> {
    if rows.is_empty() {
        return Err(Error::ExternalFormat(
            "Cannot deserialize 0 NDJSON rows because empty string is not a valid JSON value"
                .to_string(),
        ));
    }

    deserialize_iter(rows.iter().map(|x| x.as_ref()), data_type)
}

/// Deserializes an iterator of rows into an [`Array`] of [`DataType`].
/// # Implementation
/// This function is CPU-bounded.
/// This function is guaranteed to return an array of length equal to the leng
/// # Errors
/// This function errors iff any of the rows is not a valid JSON (i.e. the format is not valid NDJSON).
pub fn deserialize_iter<'a>(
    rows: impl Iterator<Item = &'a str>,
    data_type: DataType,
) -> Result<Box<dyn Array>, Error> {
    // deserialize strings to `Value`s
    let rows = rows
        .map(|row| parse(row.as_bytes()).map_err(Error::from))
        .collect::<Result<Vec<_>, Error>>()?;

    // deserialize &[Value] to Array
    Ok(_deserialize(&rows, data_type))
}
