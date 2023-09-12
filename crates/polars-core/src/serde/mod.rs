pub mod chunked_array;
mod df;
pub mod series;

#[cfg(test)]
mod test {
    use crate::chunked_array::Settings;
    use crate::prelude::*;
    use crate::series::IsSorted;

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
    fn test_serde_flags() {
        let df = sample_dataframe();

        for mut column in df.columns {
            column.set_sorted_flag(IsSorted::Descending);
            let json = serde_json::to_string(&column).unwrap();
            let out = serde_json::from_reader::<_, Series>(json.as_bytes()).unwrap();
            let f = out.get_flags();
            assert_ne!(f, Settings::empty());
            assert_eq!(column.get_flags(), out.get_flags());
        }
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

    #[test]
    fn test_serde_null_series_owned() {
        let s = NullChunked::new(Arc::from("new"), 3).into_series();
        let json = serde_json::to_string(&s).unwrap();
        let out = serde_json::from_reader::<_, Series>(json.as_bytes()).unwrap();
        assert_eq!(out, s);
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
    #[cfg(feature = "dtype-array")]
    fn test_serde_array_owned_json() {
        let input_s = Series::new("test", vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9]).rechunk();
        let dtype = ArrowDataType::FixedSizeList(
            Box::new(ArrowField::new("test", ArrowDataType::UInt8, false)),
            3,
        );
        let arr = FixedSizeListArray::new(dtype, input_s.to_arrow(0), None);
        let s = Series::try_from(("test", arr.to_boxed())).unwrap();
        let json = serde_json::to_string(&s).unwrap();
        let out = serde_json::from_reader::<_, Series>(json.as_bytes()).unwrap();
        assert!(s.series_equal_missing(&out));
    }

    #[test]
    #[cfg(feature = "dtype-array")]
    fn test_serde_empty_array() {
        let empty_s = Series::new_empty("test", &DataType::UInt8);
        let dtype = ArrowDataType::FixedSizeList(
            Box::new(ArrowField::new("test", ArrowDataType::UInt8, true)),
            3,
        );
        let arr = FixedSizeListArray::new(dtype, empty_s.to_arrow(0), None);
        let s = Series::try_from(("test", arr.to_boxed())).unwrap();
        let json = serde_json::to_string(&s).unwrap();
        let out = serde_json::from_reader::<_, Series>(json.as_bytes()).unwrap();
        assert!(s.series_equal_missing(&out));
    }

    #[test]
    #[cfg(feature = "dtype-decimal")]
    fn test_serde_decimal_series_owned_json() {
        let s = Series::new("decimal", &["4.2", "4.4", "4.7"])
            .cast(&DataType::Decimal(Some(38), Some(1)))
            .unwrap();
        let json = serde_json::to_string(&s).unwrap();
        let out = serde_json::from_reader::<_, Series>(json.as_bytes()).unwrap();
        assert_eq!(out, s);
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

        let s = Series::from_any_values_and_dtype("item", &[row_1, row_2, row_3], &dtype, false)
            .unwrap();
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
