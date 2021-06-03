use serde::{Deserialize, Serialize};
pub mod chunked_array;
pub mod series;
use crate::prelude::*;

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
    Date32,
    Date64,
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
            DataType::Date32 => DeDataType::Date32,
            DataType::Date64 => DeDataType::Date64,
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
    use crate::prelude::*;

    #[test]
    fn test_serde() -> Result<()> {
        let ca = UInt32Chunked::new_from_opt_slice("foo", &[Some(1), None, Some(2)]);

        let json = serde_json::to_string(&ca).unwrap();
        dbg!(&json);

        let out = serde_json::from_str::<Series>(&json).unwrap();
        assert!(ca.into_series().series_equal_missing(&out));

        let ca = Utf8Chunked::new_from_opt_slice("foo", &[Some("foo"), None, Some("bar")]);

        let json = serde_json::to_string(&ca).unwrap();
        dbg!(&json);

        let out = serde_json::from_str::<Series>(&json).unwrap();
        assert!(ca.into_series().series_equal_missing(&out));

        Ok(())
    }

    #[test]
    fn test_serde_df() {
        let s = Series::new("foo", &[1, 2, 3]);
        let s1 = Series::new("bar", &[Some(true), None, Some(false)]);
        let s_list = Series::new("list", &[s.clone(), s.clone(), s.clone()]);

        let df = DataFrame::new(vec![s, s_list, s1]).unwrap();
        let json = serde_json::to_string(&df).unwrap();
        dbg!(&json);
        let out = serde_json::from_str::<DataFrame>(&json).unwrap();
        assert!(df.frame_equal_missing(&out));
    }
}
