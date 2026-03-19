mod from_thrift;

mod to_thrift;

#[cfg(test)]
mod tests {
    use crate::parquet::error::ParquetResult;
    use crate::parquet::schema::io_message::from_message;
    use crate::parquet::schema::types::ParquetType;

    fn test_round_trip(message: &str) -> ParquetResult<()> {
        let expected_schema = from_message(message)?;
        let thrift_schema = expected_schema.to_thrift();
        let thrift_schema = thrift_schema.into_iter().collect::<Vec<_>>();
        let result_schema = ParquetType::try_from_thrift(&thrift_schema)?;
        assert_eq!(result_schema, expected_schema);
        Ok(())
    }

    #[test]
    fn test_schema_type_thrift_conversion() {
        let message_type = "
    message conversions {
      REQUIRED INT64 id;
      OPTIONAL group int_array_Array (LIST) {
        REPEATED group list {
          OPTIONAL group element (LIST) {
            REPEATED group list {
              OPTIONAL INT32 element;
            }
          }
        }
      }
      OPTIONAL group int_map (MAP) {
        REPEATED group map (MAP_KEY_VALUE) {
          REQUIRED BYTE_ARRAY key (UTF8);
          OPTIONAL INT32 value;
        }
      }
      OPTIONAL group int_Map_Array (LIST) {
        REPEATED group list {
          OPTIONAL group g (MAP) {
            REPEATED group map (MAP_KEY_VALUE) {
              REQUIRED BYTE_ARRAY key (UTF8);
              OPTIONAL group value {
                OPTIONAL group H {
                  OPTIONAL group i (LIST) {
                    REPEATED group list {
                      OPTIONAL DOUBLE element;
                    }
                  }
                }
              }
            }
          }
        }
      }
      OPTIONAL group nested_struct {
        OPTIONAL INT32 A;
        OPTIONAL group b (LIST) {
          REPEATED group list {
            REQUIRED FIXED_LEN_BYTE_ARRAY (16) element;
          }
        }
      }
    }
    ";
        test_round_trip(message_type).unwrap();
    }

    #[test]
    fn test_schema_type_thrift_conversion_decimal() {
        let message_type = "
    message decimals {
      OPTIONAL INT32 field0;
      OPTIONAL INT64 field1 (DECIMAL (18, 2));
      OPTIONAL FIXED_LEN_BYTE_ARRAY (16) field2 (DECIMAL (38, 18));
      OPTIONAL BYTE_ARRAY field3 (DECIMAL (9));
    }
    ";
        test_round_trip(message_type).unwrap();
    }
}
