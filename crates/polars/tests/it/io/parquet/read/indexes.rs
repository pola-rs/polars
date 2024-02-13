use polars_parquet::parquet::error::Error;
use polars_parquet::parquet::indexes::{
    BooleanIndex, BoundaryOrder, ByteIndex, Index, NativeIndex, PageIndex, PageLocation,
};
use polars_parquet::parquet::read::{read_columns_indexes, read_metadata, read_pages_locations};
use polars_parquet::parquet::schema::types::{
    FieldInfo, PhysicalType, PrimitiveConvertedType, PrimitiveLogicalType, PrimitiveType,
};
use polars_parquet::parquet::schema::Repetition;

/*
import pyspark.sql  # 3.2.1
spark = pyspark.sql.SparkSession.builder.getOrCreate()
spark.conf.set("parquet.bloom.filter.enabled", True)
spark.conf.set("parquet.bloom.filter.expected.ndv", 10)
spark.conf.set("parquet.bloom.filter.max.bytes", 32)

data = [(i, f"{i}", False) for i in range(10)]
df = spark.createDataFrame(data, ["id", "string", "bool"]).repartition(1)

df.write.parquet("bla.parquet", mode = "overwrite")
*/
const FILE: &[u8] = &[
    80, 65, 82, 49, 21, 0, 21, 172, 1, 21, 138, 1, 21, 169, 161, 209, 137, 5, 28, 21, 20, 21, 0,
    21, 6, 21, 8, 0, 0, 86, 24, 2, 0, 0, 0, 20, 1, 0, 13, 1, 17, 9, 1, 22, 1, 1, 0, 3, 1, 5, 12, 0,
    0, 0, 4, 1, 5, 12, 0, 0, 0, 5, 1, 5, 12, 0, 0, 0, 6, 1, 5, 12, 0, 0, 0, 7, 1, 5, 72, 0, 0, 0,
    8, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 21, 0, 21, 112, 21, 104, 21, 138, 239, 232,
    170, 15, 28, 21, 20, 21, 0, 21, 6, 21, 8, 0, 0, 56, 40, 2, 0, 0, 0, 20, 1, 1, 0, 0, 0, 48, 1,
    5, 0, 49, 1, 5, 0, 50, 1, 5, 0, 51, 1, 5, 0, 52, 1, 5, 0, 53, 1, 5, 60, 54, 1, 0, 0, 0, 55, 1,
    0, 0, 0, 56, 1, 0, 0, 0, 57, 21, 0, 21, 16, 21, 20, 21, 202, 209, 169, 227, 4, 28, 21, 20, 21,
    0, 21, 6, 21, 8, 0, 0, 8, 28, 2, 0, 0, 0, 20, 1, 0, 0, 25, 17, 2, 25, 24, 8, 0, 0, 0, 0, 0, 0,
    0, 0, 25, 24, 8, 9, 0, 0, 0, 0, 0, 0, 0, 21, 2, 25, 22, 0, 0, 25, 17, 2, 25, 24, 1, 48, 25, 24,
    1, 57, 21, 2, 25, 22, 0, 0, 25, 17, 2, 25, 24, 1, 0, 25, 24, 1, 0, 21, 2, 25, 22, 0, 0, 25, 28,
    22, 8, 21, 188, 1, 22, 0, 0, 0, 25, 28, 22, 196, 1, 21, 150, 1, 22, 0, 0, 0, 25, 28, 22, 218,
    2, 21, 66, 22, 0, 0, 0, 21, 64, 28, 28, 0, 0, 28, 28, 0, 0, 28, 28, 0, 0, 0, 24, 130, 24, 8,
    134, 8, 68, 6, 2, 101, 128, 10, 64, 2, 38, 78, 114, 1, 64, 38, 1, 192, 194, 152, 64, 70, 0, 36,
    56, 121, 64, 0, 21, 64, 28, 28, 0, 0, 28, 28, 0, 0, 28, 28, 0, 0, 0, 8, 17, 10, 29, 5, 88, 194,
    0, 35, 208, 25, 16, 70, 68, 48, 38, 17, 16, 140, 68, 98, 56, 0, 131, 4, 193, 40, 129, 161, 160,
    1, 96, 21, 64, 28, 28, 0, 0, 28, 28, 0, 0, 28, 28, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21, 2, 25, 76, 72, 12, 115, 112,
    97, 114, 107, 95, 115, 99, 104, 101, 109, 97, 21, 6, 0, 21, 4, 37, 2, 24, 2, 105, 100, 0, 21,
    12, 37, 2, 24, 6, 115, 116, 114, 105, 110, 103, 37, 0, 76, 28, 0, 0, 0, 21, 0, 37, 2, 24, 4,
    98, 111, 111, 108, 0, 22, 20, 25, 28, 25, 60, 38, 8, 28, 21, 4, 25, 53, 0, 6, 8, 25, 24, 2,
    105, 100, 21, 2, 22, 20, 22, 222, 1, 22, 188, 1, 38, 8, 60, 24, 8, 9, 0, 0, 0, 0, 0, 0, 0, 24,
    8, 0, 0, 0, 0, 0, 0, 0, 0, 22, 0, 40, 8, 9, 0, 0, 0, 0, 0, 0, 0, 24, 8, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 25, 28, 21, 0, 21, 0, 21, 2, 0, 22, 226, 4, 0, 22, 158, 4, 21, 22, 22, 156, 3, 21, 62, 0,
    38, 196, 1, 28, 21, 12, 25, 53, 0, 6, 8, 25, 24, 6, 115, 116, 114, 105, 110, 103, 21, 2, 22,
    20, 22, 158, 1, 22, 150, 1, 38, 196, 1, 60, 54, 0, 40, 1, 57, 24, 1, 48, 0, 25, 28, 21, 0, 21,
    0, 21, 2, 0, 22, 192, 5, 0, 22, 180, 4, 21, 24, 22, 218, 3, 21, 34, 0, 38, 218, 2, 28, 21, 0,
    25, 53, 0, 6, 8, 25, 24, 4, 98, 111, 111, 108, 21, 2, 22, 20, 22, 62, 22, 66, 38, 218, 2, 60,
    24, 1, 0, 24, 1, 0, 22, 0, 40, 1, 0, 24, 1, 0, 0, 25, 28, 21, 0, 21, 0, 21, 2, 0, 22, 158, 6,
    0, 22, 204, 4, 21, 22, 22, 252, 3, 21, 34, 0, 22, 186, 3, 22, 20, 38, 8, 22, 148, 3, 20, 0, 0,
    25, 44, 24, 24, 111, 114, 103, 46, 97, 112, 97, 99, 104, 101, 46, 115, 112, 97, 114, 107, 46,
    118, 101, 114, 115, 105, 111, 110, 24, 5, 51, 46, 50, 46, 49, 0, 24, 41, 111, 114, 103, 46, 97,
    112, 97, 99, 104, 101, 46, 115, 112, 97, 114, 107, 46, 115, 113, 108, 46, 112, 97, 114, 113,
    117, 101, 116, 46, 114, 111, 119, 46, 109, 101, 116, 97, 100, 97, 116, 97, 24, 213, 1, 123, 34,
    116, 121, 112, 101, 34, 58, 34, 115, 116, 114, 117, 99, 116, 34, 44, 34, 102, 105, 101, 108,
    100, 115, 34, 58, 91, 123, 34, 110, 97, 109, 101, 34, 58, 34, 105, 100, 34, 44, 34, 116, 121,
    112, 101, 34, 58, 34, 108, 111, 110, 103, 34, 44, 34, 110, 117, 108, 108, 97, 98, 108, 101, 34,
    58, 116, 114, 117, 101, 44, 34, 109, 101, 116, 97, 100, 97, 116, 97, 34, 58, 123, 125, 125, 44,
    123, 34, 110, 97, 109, 101, 34, 58, 34, 115, 116, 114, 105, 110, 103, 34, 44, 34, 116, 121,
    112, 101, 34, 58, 34, 115, 116, 114, 105, 110, 103, 34, 44, 34, 110, 117, 108, 108, 97, 98,
    108, 101, 34, 58, 116, 114, 117, 101, 44, 34, 109, 101, 116, 97, 100, 97, 116, 97, 34, 58, 123,
    125, 125, 44, 123, 34, 110, 97, 109, 101, 34, 58, 34, 98, 111, 111, 108, 34, 44, 34, 116, 121,
    112, 101, 34, 58, 34, 98, 111, 111, 108, 101, 97, 110, 34, 44, 34, 110, 117, 108, 108, 97, 98,
    108, 101, 34, 58, 116, 114, 117, 101, 44, 34, 109, 101, 116, 97, 100, 97, 116, 97, 34, 58, 123,
    125, 125, 93, 125, 0, 24, 74, 112, 97, 114, 113, 117, 101, 116, 45, 109, 114, 32, 118, 101,
    114, 115, 105, 111, 110, 32, 49, 46, 49, 50, 46, 50, 32, 40, 98, 117, 105, 108, 100, 32, 55,
    55, 101, 51, 48, 99, 56, 48, 57, 51, 51, 56, 54, 101, 99, 53, 50, 99, 51, 99, 102, 97, 54, 99,
    51, 52, 98, 55, 101, 102, 51, 51, 50, 49, 51, 50, 50, 99, 57, 52, 41, 25, 60, 28, 0, 0, 28, 0,
    0, 28, 0, 0, 0, 182, 2, 0, 0, 80, 65, 82, 49,
];

#[test]
fn test() -> Result<(), Error> {
    let mut reader = std::io::Cursor::new(FILE);

    let expected_index = vec![
        Box::new(NativeIndex::<i64> {
            primitive_type: PrimitiveType::from_physical("id".to_string(), PhysicalType::Int64),
            indexes: vec![PageIndex {
                min: Some(0),
                max: Some(9),
                null_count: Some(0),
            }],
            boundary_order: BoundaryOrder::Ascending,
        }) as Box<dyn Index>,
        Box::new(ByteIndex {
            primitive_type: PrimitiveType {
                field_info: FieldInfo {
                    name: "string".to_string(),
                    repetition: Repetition::Optional,
                    id: None,
                },
                logical_type: Some(PrimitiveLogicalType::String),
                converted_type: Some(PrimitiveConvertedType::Utf8),
                physical_type: PhysicalType::ByteArray,
            },
            indexes: vec![PageIndex {
                min: Some(b"0".to_vec()),
                max: Some(b"9".to_vec()),
                null_count: Some(0),
            }],
            boundary_order: BoundaryOrder::Ascending,
        }),
        Box::new(BooleanIndex {
            indexes: vec![PageIndex {
                min: Some(false),
                max: Some(false),
                null_count: Some(0),
            }],
            boundary_order: BoundaryOrder::Ascending,
        }),
    ];
    let expected_page_locations = vec![
        vec![PageLocation {
            offset: 4,
            compressed_page_size: 94,
            first_row_index: 0,
        }],
        vec![PageLocation {
            offset: 98,
            compressed_page_size: 75,
            first_row_index: 0,
        }],
        vec![PageLocation {
            offset: 173,
            compressed_page_size: 33,
            first_row_index: 0,
        }],
    ];

    let metadata = read_metadata(&mut reader)?;
    let columns = &metadata.row_groups[0].columns();

    let indexes = read_columns_indexes(&mut reader, columns)?;
    assert_eq!(&indexes, &expected_index);

    let pages = read_pages_locations(&mut reader, columns)?;
    assert_eq!(pages, expected_page_locations);

    Ok(())
}
