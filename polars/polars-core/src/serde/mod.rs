use serde::{Deserialize, Serialize};

use crate::prelude::*;

pub mod chunked_array;
pub mod series;

// Serde calls this the definition of the remote type. It is just a copy of the
// remote data structure. The `remote` attribute gives the path to the actual
// type we intend to derive code for.
#[derive(Serialize, Deserialize, Debug)]
#[serde(remote = "TimeUnit")]
enum TimeUnitDef {
    /// Time in seconds.
    Second,
    /// Time in milliseconds.
    Millisecond,
    /// Time in microseconds.
    Microsecond,
    /// Time in nanoseconds.
    Nanosecond,
}

/// Intermediate enum. Needed because [crate::datatypes::DataType] has
/// a &static str and thus requires Deserialize<&static>
#[derive(Serialize, Deserialize, Debug)]
enum DeDataType<'a> {
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
    Utf8,
    Date,
    Datetime,
    #[serde(with = "TimeUnitDef")]
    Time64(TimeUnit),
    List,
    Object(&'a str),
    Null,
    Categorical,
}

impl From<&DataType> for DeDataType<'_> {
    fn from(dt: &DataType) -> Self {
        match dt {
            DataType::Int32 => DeDataType::Int32,
            DataType::UInt32 => DeDataType::UInt32,
            DataType::Int64 => DeDataType::Int64,
            DataType::UInt64 => DeDataType::UInt64,
            DataType::Date => DeDataType::Date,
            DataType::Datetime => DeDataType::Datetime,
            DataType::Float32 => DeDataType::Float32,
            DataType::Float64 => DeDataType::Float64,
            DataType::Utf8 => DeDataType::Utf8,
            DataType::Boolean => DeDataType::Boolean,
            DataType::Null => DeDataType::Null,
            DataType::List(_) => DeDataType::List,
            #[cfg(feature = "object")]
            DataType::Object(s) => DeDataType::Object(s),
            _ => unimplemented!(),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_serde() -> Result<()> {
        let ca = UInt32Chunked::new("foo", &[Some(1), None, Some(2)]);

        let json = serde_json::to_string(&ca).unwrap();
        dbg!(&json);

        let out = serde_json::from_str::<Series>(&json).unwrap();
        assert!(ca.into_series().series_equal_missing(&out));

        let ca = Utf8Chunked::new("foo", &[Some("foo"), None, Some("bar")]);

        let json = serde_json::to_string(&ca).unwrap();
        dbg!(&json);

        let out = serde_json::from_str::<Series>(&json).unwrap(); // uses `Deserialize<'de>`
        assert!(ca.into_series().series_equal_missing(&out));

        Ok(())
    }

    /// test using the `DeserializedOwned` trait
    #[test]
    fn test_serde_owned() {
        let ca = UInt32Chunked::new("foo", &[Some(1), None, Some(2)]);

        let json = serde_json::to_string(&ca).unwrap();
        dbg!(&json);

        let out = serde_json::from_reader::<_, Series>(json.as_bytes()).unwrap(); // uses `DeserializeOwned`
        assert!(ca.into_series().series_equal_missing(&out));
    }

    fn sample_dataframe() -> DataFrame {
        let s1 = Series::new("foo", &[1, 2, 3]);
        let s2 = Series::new("bar", &[Some(true), None, Some(false)]);
        let s3 = Series::new("utf8", &["mouse", "elephant", "dog"]);
        let s_list = Series::new("list", &[s1.clone(), s1.clone(), s1.clone()]);

        DataFrame::new(vec![s1, s2, s3, s_list]).unwrap()
    }

    #[test]
    fn test_serde_df_json() {
        let df = sample_dataframe();
        let json = serde_json::to_string(&df).unwrap();
        dbg!(&json);
        let out = serde_json::from_str::<DataFrame>(&json).unwrap(); // uses `Deserialize<'de>`
        assert!(df.frame_equal_missing(&out));
    }

    #[test]
    fn test_serde_df_bincode() {
        let df = sample_dataframe();
        let bytes = bincode::serialize(&df).unwrap();
        let out = bincode::deserialize::<DataFrame>(&bytes).unwrap(); // uses `Deserialize<'de>`
        assert!(df.frame_equal_missing(&out));
    }

    /// test using the `DeserializedOwned` trait
    #[test]
    fn test_serde_df_owned_json() {
        let df = sample_dataframe();
        let json = serde_json::to_string(&df).unwrap();
        dbg!(&json);

        let out = serde_json::from_reader::<_, DataFrame>(json.as_bytes()).unwrap(); // uses `DeserializeOwned`
        assert!(df.frame_equal_missing(&out));
    }

    /// test using the `DeserializedOwned` trait
    #[test]
    fn test_serde_df_owned_bincode() {
        let df = sample_dataframe();
        let bytes = bincode::serialize(&df).unwrap();
        let out = bincode::deserialize_from::<_, DataFrame>(bytes.as_slice()).unwrap(); // uses `DeserializeOwned`
        assert!(df.frame_equal_missing(&out));
    }
}
