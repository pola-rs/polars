use crate::datatypes::{DataType, Field, TimeUnit};
use ahash::AHashSet;

pub(super) const RFC3339: &str = "%Y-%m-%dT%H:%M:%S%.f%:z";

fn is_boolean(bytes: &[u8]) -> bool {
    bytes.eq_ignore_ascii_case(b"true") | bytes.eq_ignore_ascii_case(b"false")
}

fn is_float(bytes: &[u8]) -> bool {
    lexical_core::parse::<f64>(bytes).is_ok()
}

fn is_integer(bytes: &[u8]) -> bool {
    lexical_core::parse::<i64>(bytes).is_ok()
}

fn is_date(string: &str) -> bool {
    string.parse::<chrono::NaiveDate>().is_ok()
}

fn is_time(string: &str) -> bool {
    string.parse::<chrono::NaiveTime>().is_ok()
}

fn is_naive_datetime(string: &str) -> bool {
    string.parse::<chrono::NaiveDateTime>().is_ok()
}

fn is_datetime(string: &str) -> Option<String> {
    let mut parsed = chrono::format::Parsed::new();
    let fmt = chrono::format::StrftimeItems::new(RFC3339);
    if chrono::format::parse(&mut parsed, string, fmt).is_ok() {
        parsed.offset.map(|x| {
            let hours = x / 60 / 60;
            let minutes = x / 60 - hours * 60;
            format!("{hours:03}:{minutes:02}")
        })
    } else {
        None
    }
}

/// Infers [`DataType`] from `bytes`
/// # Implementation
/// * case insensitive "true" or "false" are mapped to [`DataType::Boolean`]
/// * parsable to integer is mapped to [`DataType::Int64`]
/// * parsable to float is mapped to [`DataType::Float64`]
/// * parsable to date is mapped to [`DataType::Date32`]
/// * parsable to time is mapped to [`DataType::Time32(TimeUnit::Millisecond)`]
/// * parsable to naive datetime is mapped to [`DataType::Timestamp(TimeUnit::Millisecond, None)`]
/// * parsable to time-aware datetime is mapped to [`DataType::Timestamp`] of milliseconds and parsed offset.
/// * other utf8 is mapped to [`DataType::Utf8`]
/// * invalid utf8 is mapped to [`DataType::Binary`]
pub fn infer(bytes: &[u8]) -> DataType {
    if is_boolean(bytes) {
        DataType::Boolean
    } else if is_integer(bytes) {
        DataType::Int64
    } else if is_float(bytes) {
        DataType::Float64
    } else if let Ok(string) = simdutf8::basic::from_utf8(bytes) {
        if is_date(string) {
            DataType::Date32
        } else if is_time(string) {
            DataType::Time32(TimeUnit::Millisecond)
        } else if is_naive_datetime(string) {
            DataType::Timestamp(TimeUnit::Millisecond, None)
        } else if let Some(offset) = is_datetime(string) {
            DataType::Timestamp(TimeUnit::Millisecond, Some(offset))
        } else {
            DataType::Utf8
        }
    } else {
        // invalid utf8
        DataType::Binary
    }
}

fn merge_fields(field_name: &str, possibilities: &mut AHashSet<DataType>) -> Field {
    // determine data type based on possible types
    // if there are incompatible types, use DataType::Utf8
    let data_type = match possibilities.len() {
        1 => possibilities.drain().next().unwrap(),
        2 => {
            if possibilities.contains(&DataType::Int64)
                && possibilities.contains(&DataType::Float64)
            {
                // we have an integer and double, fall down to double
                DataType::Float64
            } else {
                // default to Utf8 for conflicting datatypes (e.g bool and int)
                DataType::Utf8
            }
        }
        _ => DataType::Utf8,
    };
    Field::new(field_name, data_type, true)
}

pub(crate) fn merge_schema(
    headers: &[String],
    column_types: &mut [AHashSet<DataType>],
) -> Vec<Field> {
    headers
        .iter()
        .zip(column_types.iter_mut())
        .map(|(field_name, possibilities)| merge_fields(field_name, possibilities))
        .collect()
}
