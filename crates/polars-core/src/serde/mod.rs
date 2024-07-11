pub mod chunked_array;
mod df;
pub mod series;

#[cfg(test)]
mod test {
    use crate::chunked_array::metadata::MetadataFlags;
    use crate::prelude::*;
    use crate::series::IsSorted;

    #[test]
    fn test_serde() -> PolarsResult<()> {
        let ca = UInt32Chunked::new("foo", &[Some(1), None, Some(2)]);

        let json = serde_json::to_string(&ca).unwrap();

        let out = serde_json::from_str::<Series>(&json).unwrap();
        assert!(ca.into_series().equals_missing(&out));

        let ca = StringChunked::new("foo", &[Some("foo"), None, Some("bar")]);

        let json = serde_json::to_string(&ca).unwrap();

        let out = serde_json::from_str::<Series>(&json).unwrap(); // uses `Deserialize<'de>`
        assert!(ca.into_series().equals_missing(&out));

        Ok(())
    }

    /// test using the `DeserializedOwned` trait
    #[test]
    fn test_serde_owned() {
        let ca = UInt32Chunked::new("foo", &[Some(1), None, Some(2)]);

        let json = serde_json::to_string(&ca).unwrap();

        let out = serde_json::from_reader::<_, Series>(json.as_bytes()).unwrap(); // uses `DeserializeOwned`
        assert!(ca.into_series().equals_missing(&out));
    }

    fn sample_dataframe() -> DataFrame {
        let s1 = Series::new("foo", &[1, 2, 3]);
        let s2 = Series::new("bar", &[Some(true), None, Some(false)]);
        let s3 = Series::new("string", &["mouse", "elephant", "dog"]);
        let s_list = Series::new("list", &[s1.clone(), s1.clone(), s1.clone()]);

        DataFrame::new(vec![s1, s2, s3, s_list]).unwrap()
    }

    #[test]
    fn test_serde_flags() {
        let df = sample_dataframe();

        for mut column in df.columns {
            column.set_sorted_flag(IsSorted::Descending);
            let json = serde_json::to_string(&column).unwrap();
            let out = serde_json::from_reader::<_, Series>(json.as_bytes()).unwrap();
            let f = out.get_flags();
            assert_ne!(f, MetadataFlags::empty());
            assert_eq!(column.get_flags(), out.get_flags());
        }
    }

    #[test]
    fn test_serde_df_json() {
        let df = sample_dataframe();
        let json = serde_json::to_string(&df).unwrap();
        let out = serde_json::from_str::<DataFrame>(&json).unwrap(); // uses `Deserialize<'de>`
        assert!(df.equals_missing(&out));
    }

    #[test]
    fn test_serde_df_bincode() {
        let df = sample_dataframe();
        let bytes = bincode::serialize(&df).unwrap();
        let out = bincode::deserialize::<DataFrame>(&bytes).unwrap(); // uses `Deserialize<'de>`
        assert!(df.equals_missing(&out));
    }

    /// test using the `DeserializedOwned` trait
    #[test]
    fn test_serde_df_owned_json() {
        let df = sample_dataframe();
        let json = serde_json::to_string(&df).unwrap();

        let out = serde_json::from_reader::<_, DataFrame>(json.as_bytes()).unwrap(); // uses `DeserializeOwned`
        assert!(df.equals_missing(&out));
    }

    #[test]
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
        assert!(df.equals_missing(&out));
    }

    // STRUCT REFACTOR
    #[ignore]
    #[test]
    #[cfg(feature = "dtype-struct")]
    fn test_serde_struct_series_owned_json() {
        let row_1 = AnyValue::StructOwned(Box::new((
            vec![
                AnyValue::String("1:1"),
                AnyValue::Null,
                AnyValue::String("1:3"),
            ],
            vec![
                Field::new("fld_1", DataType::String),
                Field::new("fld_2", DataType::String),
                Field::new("fld_3", DataType::String),
            ],
        )));
        let dtype = DataType::Struct(vec![
            Field::new("fld_1", DataType::String),
            Field::new("fld_2", DataType::String),
            Field::new("fld_3", DataType::String),
        ]);
        let row_2 = AnyValue::StructOwned(Box::new((
            vec![
                AnyValue::String("2:1"),
                AnyValue::String("2:2"),
                AnyValue::String("2:3"),
            ],
            vec![
                Field::new("fld_1", DataType::String),
                Field::new("fld_2", DataType::String),
                Field::new("fld_3", DataType::String),
            ],
        )));
        let row_3 = AnyValue::Null;

        let s = Series::from_any_values_and_dtype("item", &[row_1, row_2, row_3], &dtype, false)
            .unwrap();
        let df = DataFrame::new(vec![s]).unwrap();

        let df_str = serde_json::to_string(&df).unwrap();
        let out = serde_json::from_str::<DataFrame>(&df_str).unwrap();
        assert!(df.equals_missing(&out));
    }
    /// test using the `DeserializedOwned` trait
    #[test]
    fn test_serde_df_owned_bincode() {
        let df = sample_dataframe();
        let bytes = bincode::serialize(&df).unwrap();
        let out = bincode::deserialize_from::<_, DataFrame>(bytes.as_slice()).unwrap(); // uses `DeserializeOwned`
        assert!(df.equals_missing(&out));
    }
}
