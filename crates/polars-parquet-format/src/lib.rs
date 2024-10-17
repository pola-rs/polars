#![allow(non_camel_case_types)]
#![forbid(unsafe_code)]
#![cfg_attr(docsrs, feature(doc_cfg))]

mod parquet_format;
pub use crate::parquet_format::*;

pub mod thrift;

#[cfg(test)]
mod tests {
    use super::*;

    fn meta() -> FileMetaData {
        FileMetaData {
            version: 0,
            schema: vec![SchemaElement {
                type_: Some(Type::INT32),
                type_length: None,
                repetition_type: Some(FieldRepetitionType::REQUIRED),
                name: "aaa".to_string(),
                num_children: None,
                converted_type: Some(ConvertedType::DATE),
                scale: None,
                precision: None,
                field_id: None,
                logical_type: Some(LogicalType::DATE(Default::default())),
            }],
            num_rows: 0,
            row_groups: vec![RowGroup {
                columns: vec![ColumnChunk {
                    file_path: None,
                    file_offset: 10,
                    meta_data: Some(ColumnMetaData {
                        type_: Type::INT32,
                        encodings: vec![Encoding::PLAIN],
                        path_in_schema: vec![],
                        codec: CompressionCodec::UNCOMPRESSED,
                        num_values: 0,
                        total_uncompressed_size: 0,
                        total_compressed_size: 0,
                        key_value_metadata: Some(vec![KeyValue {
                            key: "".to_string(),
                            value: Some("".to_string()),
                        }]),
                        data_page_offset: 0,
                        index_page_offset: None,
                        dictionary_page_offset: None,
                        statistics: Some(Statistics {
                            max: None,
                            min: None,
                            null_count: Some(0),
                            distinct_count: Some(0),
                            max_value: Some(vec![]),
                            min_value: Some(vec![]),
                        }),
                        encoding_stats: Some(vec![PageEncodingStats {
                            page_type: PageType::DATA_PAGE,
                            encoding: Encoding::PLAIN,
                            count: 1,
                        }]),
                        bloom_filter_offset: None,
                    }),
                    offset_index_offset: None,
                    offset_index_length: None,
                    column_index_offset: None,
                    column_index_length: None,
                    crypto_metadata: None,
                    encrypted_column_metadata: None,
                }],
                total_byte_size: 10,
                num_rows: 10,
                sorting_columns: Some(vec![SortingColumn {
                    column_idx: 1,
                    descending: true,
                    nulls_first: true,
                }]),
                file_offset: Some(10),
                total_compressed_size: Some(10),
                ordinal: None,
            }],
            key_value_metadata: None,
            created_by: None,
            column_orders: Some(vec![ColumnOrder::TYPEORDER(Default::default())]),
            encryption_algorithm: None,
            footer_signing_key_metadata: None,
        }
    }

    #[test]
    fn basic() {
        let mut writer = vec![];
        let mut protocol = thrift::protocol::TCompactOutputProtocol::new(&mut writer);
        let metadata = meta();
        metadata.write_to_out_protocol(&mut protocol).unwrap();

        let mut prot = thrift::protocol::TCompactInputProtocol::new(writer.as_slice(), usize::MAX);
        let result = FileMetaData::read_from_in_protocol(&mut prot).unwrap();
        assert_eq!(result, metadata)
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn async_() {
        let mut writer = vec![];
        let mut protocol = thrift::protocol::TCompactOutputStreamProtocol::new(&mut writer);
        let metadata = meta();
        metadata
            .write_to_out_stream_protocol(&mut protocol)
            .await
            .unwrap();

        let mut prot =
            thrift::protocol::TCompactInputStreamProtocol::new(writer.as_slice(), usize::MAX);
        let result = FileMetaData::stream_from_in_protocol(&mut prot)
            .await
            .unwrap();
        assert_eq!(result, metadata)
    }

    fn page() -> PageHeader {
        PageHeader {
            type_: PageType::DATA_PAGE,
            uncompressed_page_size: 0,
            compressed_page_size: 0,
            crc: None,
            data_page_header: Some(DataPageHeader {
                num_values: 0,
                encoding: Encoding::PLAIN,
                definition_level_encoding: Encoding::RLE,
                repetition_level_encoding: Encoding::RLE,
                statistics: Some(Statistics {
                    max: None,
                    min: None,
                    null_count: Some(0),
                    distinct_count: Some(0),
                    max_value: Some(vec![]),
                    min_value: Some(vec![]),
                }),
            }),
            index_page_header: Some(IndexPageHeader {}),
            dictionary_page_header: Some(DictionaryPageHeader {
                num_values: 0,
                encoding: Encoding::PLAIN,
                is_sorted: Some(false),
            }),
            data_page_header_v2: Some(DataPageHeaderV2 {
                num_values: 0,
                num_nulls: 0,
                num_rows: 0,
                encoding: Encoding::PLAIN,
                definition_levels_byte_length: 0,
                repetition_levels_byte_length: 0,
                is_compressed: Some(true),
                statistics: Some(Statistics {
                    max: None,
                    min: None,
                    null_count: Some(0),
                    distinct_count: Some(0),
                    max_value: Some(vec![]),
                    min_value: Some(vec![]),
                }),
            }),
        }
    }

    #[test]
    fn basic_page() {
        let mut writer = vec![];
        let mut protocol = thrift::protocol::TCompactOutputProtocol::new(&mut writer);
        let metadata = page();
        metadata.write_to_out_protocol(&mut protocol).unwrap();

        let mut prot = thrift::protocol::TCompactInputProtocol::new(writer.as_slice(), usize::MAX);
        let result = PageHeader::read_from_in_protocol(&mut prot).unwrap();
        assert_eq!(result, metadata)
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn async_page() {
        let mut writer = vec![];
        let mut protocol = thrift::protocol::TCompactOutputStreamProtocol::new(&mut writer);
        let metadata = page();
        metadata
            .write_to_out_stream_protocol(&mut protocol)
            .await
            .unwrap();

        let mut prot =
            thrift::protocol::TCompactInputStreamProtocol::new(writer.as_slice(), usize::MAX);
        let result = PageHeader::stream_from_in_protocol(&mut prot)
            .await
            .unwrap();
        assert_eq!(result, metadata)
    }
}
