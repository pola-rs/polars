use serde::{Deserialize, Serialize};

use crate::prelude::*;

pub mod chunked_array;
pub mod series;

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
    Binary,
    Date,
    Datetime(TimeUnit, Option<TimeZone>),
    Duration(TimeUnit),
    Time,
    List,
    Object(&'a str),
    Null,
    Categorical,
    Struct,
}

impl From<&DataType> for DeDataType<'_> {
    fn from(dt: &DataType) -> Self {
        match dt {
            DataType::Int32 => DeDataType::Int32,
            DataType::UInt32 => DeDataType::UInt32,
            DataType::Int64 => DeDataType::Int64,
            DataType::UInt64 => DeDataType::UInt64,
            DataType::Date => DeDataType::Date,
            DataType::Datetime(tu, tz) => DeDataType::Datetime(*tu, tz.clone()),
            DataType::Duration(tu) => DeDataType::Duration(*tu),
            DataType::Time => DeDataType::Time,
            DataType::Float32 => DeDataType::Float32,
            DataType::Float64 => DeDataType::Float64,
            DataType::Utf8 => DeDataType::Utf8,
            DataType::Boolean => DeDataType::Boolean,
            DataType::Null => DeDataType::Null,
            DataType::List(_) => DeDataType::List,
            #[cfg(feature = "dtype-binary")]
            DataType::Binary => DeDataType::Binary,
            #[cfg(feature = "object")]
            DataType::Object(s) => DeDataType::Object(s),
            #[cfg(feature = "dtype-struct")]
            DataType::Struct(_) => DeDataType::Struct,
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(_) => DeDataType::Categorical,
            _ => unimplemented!(),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_serde() -> PolarsResult<()> {
        let ca = UInt32Chunked::new("foo", &[Some(1), None, Some(2)]);

        let json = serde_json::to_string(&ca).unwrap();

        let out = serde_json::from_str::<Series>(&json).unwrap();
        assert!(ca.into_series().series_equal_missing(&out));

        let ca = Utf8Chunked::new("foo", &[Some("foo"), None, Some("bar")]);

        let json = serde_json::to_string(&ca).unwrap();

        let out = serde_json::from_str::<Series>(&json).unwrap(); // uses `Deserialize<'de>`
        assert!(ca.into_series().series_equal_missing(&out));

        Ok(())
    }

    /// test using the `DeserializedOwned` trait
    #[test]
    fn test_serde_owned() {
        let ca = UInt32Chunked::new("foo", &[Some(1), None, Some(2)]);

        let json = serde_json::to_string(&ca).unwrap();

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

        let out = serde_json::from_reader::<_, DataFrame>(json.as_bytes()).unwrap(); // uses `DeserializeOwned`
        assert!(df.frame_equal_missing(&out));
    }

    #[test]
    #[cfg(feature = "dtype-binary")]
    fn test_serde_binary_series_owned_bincode() {
        let s1 = Series::new(
            "foo",
            &[
                vec![1u8, 2u8, 3u8],
                vec![4u8, 5u8, 6u8, 7u8],
                vec![8u8, 9u8],
            ],
        );
        let df = DataFrame::new(vec![s1]).unwrap();
        let bytes = bincode::serialize(&df).unwrap();
        let out = bincode::deserialize_from::<_, DataFrame>(bytes.as_slice()).unwrap();
        assert!(df.frame_equal_missing(&out));
    }

    #[test]
    #[cfg(feature = "dtype-struct")]
    fn test_serde_struct_series_owned_json() {
        let row_1 = AnyValue::StructOwned(Box::new((
            vec![AnyValue::Utf8("1:1"), AnyValue::Null, AnyValue::Utf8("1:3")],
            vec![
                Field::new("fld_1", DataType::Utf8),
                Field::new("fld_2", DataType::Utf8),
                Field::new("fld_3", DataType::Utf8),
            ],
        )));
        let dtype = DataType::Struct(vec![
            Field::new("fld_1", DataType::Utf8),
            Field::new("fld_2", DataType::Utf8),
            Field::new("fld_3", DataType::Utf8),
        ]);
        let row_2 = AnyValue::StructOwned(Box::new((
            vec![
                AnyValue::Utf8("2:1"),
                AnyValue::Utf8("2:2"),
                AnyValue::Utf8("2:3"),
            ],
            vec![
                Field::new("fld_1", DataType::Utf8),
                Field::new("fld_2", DataType::Utf8),
                Field::new("fld_3", DataType::Utf8),
            ],
        )));
        let row_3 = AnyValue::Null;

        let s =
            Series::from_any_values_and_dtype("item", &vec![row_1, row_2, row_3], &dtype).unwrap();
        let df = DataFrame::new(vec![s]).unwrap();

        let df_str = serde_json::to_string(&df).unwrap();
        let out = serde_json::from_str::<DataFrame>(&df_str).unwrap();
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
