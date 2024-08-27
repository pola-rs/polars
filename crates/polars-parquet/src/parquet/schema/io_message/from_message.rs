// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Parquet schema parser.
//! Provides methods to parse and validate string message type into Parquet
//! [`ParquetType`](crate::parquet::schema::types::ParquetType).
//!
//! # Example
//!
//! ```rust
//! use polars_parquet::parquet::schema::io_message::from_message;
//!
//! let message_type = "
//!   message spark_schema {
//!     OPTIONAL BYTE_ARRAY a (UTF8);
//!     REQUIRED INT32 b;
//!     REQUIRED DOUBLE c;
//!     REQUIRED BOOLEAN d;
//!     OPTIONAL group e (LIST) {
//!       REPEATED group list {
//!         REQUIRED INT32 element;
//!       }
//!     }
//!   }
//! ";
//!
//! let schema = from_message(message_type).expect("Expected valid schema");
//! println!("{:?}", schema);
//! ```

use parquet_format_safe::Type;
use types::PrimitiveLogicalType;

use super::super::types::{ParquetType, TimeUnit};
use super::super::*;
use crate::parquet::error::{ParquetError, ParquetResult};
use crate::parquet::schema::types::{GroupConvertedType, PrimitiveConvertedType};

fn is_logical_type(s: &str) -> bool {
    matches!(
        s,
        "INTEGER"
            | "MAP"
            | "LIST"
            | "ENUM"
            | "DECIMAL"
            | "DATE"
            | "TIME"
            | "TIMESTAMP"
            | "STRING"
            | "JSON"
            | "BSON"
            | "UUID"
            | "UNKNOWN"
            | "INTERVAL"
    )
}

fn is_converted_type(s: &str) -> bool {
    matches!(
        s,
        "UTF8"
            | "ENUM"
            | "DECIMAL"
            | "DATE"
            | "TIME_MILLIS"
            | "TIME_MICROS"
            | "TIMESTAMP_MILLIS"
            | "TIMESTAMP_MICROS"
            | "UINT_8"
            | "UINT_16"
            | "UINT_32"
            | "UINT_64"
            | "INT_8"
            | "INT_16"
            | "INT_32"
            | "INT_64"
            | "JSON"
            | "BSON"
            | "INTERVAL"
    )
}

fn converted_group_from_str(s: &str) -> ParquetResult<GroupConvertedType> {
    Ok(match s {
        "MAP" => GroupConvertedType::Map,
        "MAP_KEY_VALUE" => GroupConvertedType::MapKeyValue,
        "LIST" => GroupConvertedType::List,
        other => {
            return Err(ParquetError::oos(format!(
                "Invalid converted type {}",
                other
            )))
        },
    })
}

fn converted_primitive_from_str(s: &str) -> Option<PrimitiveConvertedType> {
    use PrimitiveConvertedType::*;
    Some(match s {
        "UTF8" => Utf8,
        "ENUM" => Enum,
        "DECIMAL" => Decimal(0, 0),
        "DATE" => Date,
        "TIME_MILLIS" => TimeMillis,
        "TIME_MICROS" => TimeMicros,
        "TIMESTAMP_MILLIS" => TimestampMillis,
        "TIMESTAMP_MICROS" => TimestampMicros,
        "UINT_8" => Uint8,
        "UINT_16" => Uint16,
        "UINT_32" => Uint32,
        "UINT_64" => Uint64,
        "INT_8" => Int8,
        "INT_16" => Int16,
        "INT_32" => Int32,
        "INT_64" => Int64,
        "JSON" => Json,
        "BSON" => Bson,
        "INTERVAL" => Interval,
        _ => return None,
    })
}

fn repetition_from_str(s: &str) -> ParquetResult<Repetition> {
    Ok(match s {
        "REQUIRED" => Repetition::Required,
        "OPTIONAL" => Repetition::Optional,
        "REPEATED" => Repetition::Repeated,
        other => return Err(ParquetError::oos(format!("Invalid repetition {}", other))),
    })
}

fn type_from_str(s: &str) -> ParquetResult<Type> {
    match s {
        "BOOLEAN" => Ok(Type::BOOLEAN),
        "INT32" => Ok(Type::INT32),
        "INT64" => Ok(Type::INT64),
        "INT96" => Ok(Type::INT96),
        "FLOAT" => Ok(Type::FLOAT),
        "DOUBLE" => Ok(Type::DOUBLE),
        "BYTE_ARRAY" | "BINARY" => Ok(Type::BYTE_ARRAY),
        "FIXED_LEN_BYTE_ARRAY" => Ok(Type::FIXED_LEN_BYTE_ARRAY),
        other => Err(ParquetError::oos(format!("Invalid type {}", other))),
    }
}

/// Parses message type as string into a Parquet [`ParquetType`](crate::parquet::schema::types::ParquetType).
///
/// This could, for example, be used to extract individual columns.
///
/// Returns Parquet general error when parsing or validation fails.
pub fn from_message(message_type: &str) -> ParquetResult<ParquetType> {
    let mut parser = Parser {
        tokenizer: &mut Tokenizer::from_str(message_type),
    };
    parser.parse_message_type()
}

/// Tokenizer to split message type string into tokens that are separated using characters
/// defined in `is_schema_delim` method. Tokenizer also preserves delimiters as tokens.
/// Tokenizer provides Iterator interface to process tokens; it also allows to step back
/// to reprocess previous tokens.
struct Tokenizer<'a> {
    // List of all tokens for a string
    tokens: Vec<&'a str>,
    // Current index of vector
    index: usize,
}

impl<'a> Tokenizer<'a> {
    // Create tokenizer from message type string
    pub fn from_str(string: &'a str) -> Self {
        let vec = string
            .split_whitespace()
            .flat_map(Self::split_token)
            .collect();
        Tokenizer {
            tokens: vec,
            index: 0,
        }
    }

    // List of all special characters in schema
    fn is_schema_delim(c: char) -> bool {
        c == ';' || c == '{' || c == '}' || c == '(' || c == ')' || c == '=' || c == ','
    }

    /// Splits string into tokens; input string can already be token or can contain
    /// delimiters, e.g. required" -> Vec("required") and
    /// "(UTF8);" -> Vec("(", "UTF8", ")", ";")
    fn split_token(string: &str) -> Vec<&str> {
        let mut buffer: Vec<&str> = Vec::new();
        let mut tail = string;
        while let Some(index) = tail.find(Self::is_schema_delim) {
            let (h, t) = tail.split_at(index);
            if !h.is_empty() {
                buffer.push(h);
            }
            buffer.push(&t[0..1]);
            tail = &t[1..];
        }
        if !tail.is_empty() {
            buffer.push(tail);
        }
        buffer
    }

    // Move pointer to a previous element
    fn backtrack(&mut self) {
        self.index -= 1;
    }
}

impl<'a> Iterator for Tokenizer<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<&'a str> {
        if self.index < self.tokens.len() {
            self.index += 1;
            Some(self.tokens[self.index - 1])
        } else {
            None
        }
    }
}

/// Internal Schema parser.
/// Traverses message type using tokenizer and parses each group/primitive type
/// recursively.
struct Parser<'a> {
    tokenizer: &'a mut Tokenizer<'a>,
}

// Utility function to assert token on validity.
fn assert_token(token: Option<&str>, expected: &str) -> ParquetResult<()> {
    match token {
        Some(value) if value == expected => Ok(()),
        Some(other) => Err(ParquetError::oos(format!(
            "Expected '{}', found token '{}'",
            expected, other
        ))),
        None => Err(ParquetError::oos(format!(
            "Expected '{}', but no token found (None)",
            expected
        ))),
    }
}

// Utility function to parse i32 or return general error.
fn parse_i32(value: Option<&str>, not_found_msg: &str, parse_fail_msg: &str) -> ParquetResult<i32> {
    value
        .ok_or_else(|| ParquetError::oos(not_found_msg))
        .and_then(|v| {
            v.parse::<i32>()
                .map_err(|_| ParquetError::oos(parse_fail_msg))
        })
}

// Utility function to parse boolean or return general error.
#[inline]
fn parse_bool(
    value: Option<&str>,
    not_found_msg: &str,
    parse_fail_msg: &str,
) -> ParquetResult<bool> {
    value
        .ok_or_else(|| ParquetError::oos(not_found_msg))
        .and_then(|v| {
            v.to_lowercase()
                .parse::<bool>()
                .map_err(|_| ParquetError::oos(parse_fail_msg))
        })
}

// Utility function to parse TimeUnit or return general error.
fn parse_timeunit(
    value: Option<&str>,
    not_found_msg: &str,
    parse_fail_msg: &str,
) -> ParquetResult<TimeUnit> {
    value
        .ok_or_else(|| ParquetError::oos(not_found_msg))
        .and_then(|v| match v.to_uppercase().as_str() {
            "MILLIS" => Ok(TimeUnit::Milliseconds),
            "MICROS" => Ok(TimeUnit::Microseconds),
            "NANOS" => Ok(TimeUnit::Nanoseconds),
            _ => Err(ParquetError::oos(parse_fail_msg)),
        })
}

impl<'a> Parser<'a> {
    // Entry function to parse message type, uses internal tokenizer.
    fn parse_message_type(&mut self) -> ParquetResult<ParquetType> {
        // Check that message type starts with "message".
        match self.tokenizer.next() {
            Some("message") => {
                let name = self
                    .tokenizer
                    .next()
                    .ok_or_else(|| ParquetError::oos("Expected name, found None"))?;
                let fields = self.parse_child_types()?;
                Ok(ParquetType::new_root(name.to_string(), fields))
            },
            _ => Err(ParquetError::oos(
                "Message type does not start with 'message'",
            )),
        }
    }

    // Parses child types for a current group type.
    // This is only invoked on root and group types.
    fn parse_child_types(&mut self) -> ParquetResult<Vec<ParquetType>> {
        assert_token(self.tokenizer.next(), "{")?;
        let mut vec = Vec::new();
        while let Some(value) = self.tokenizer.next() {
            if value == "}" {
                break;
            } else {
                self.tokenizer.backtrack();
                vec.push(self.add_type()?);
            }
        }
        Ok(vec)
    }

    fn add_type(&mut self) -> ParquetResult<ParquetType> {
        // Parse repetition
        let repetition = self
            .tokenizer
            .next()
            .ok_or_else(|| ParquetError::oos("Expected repetition, found None"))
            .and_then(|v| repetition_from_str(&v.to_uppercase()))?;

        match self.tokenizer.next() {
            Some(group) if group.to_uppercase() == "GROUP" => self.add_group_type(repetition),
            Some(type_string) => {
                let physical_type = type_from_str(&type_string.to_uppercase())?;
                self.add_primitive_type(repetition, physical_type)
            },
            None => Err(ParquetError::oos(
                "Invalid type, could not extract next token",
            )),
        }
    }

    fn add_group_type(&mut self, repetition: Repetition) -> ParquetResult<ParquetType> {
        // Parse name of the group type
        let name = self
            .tokenizer
            .next()
            .ok_or_else(|| ParquetError::oos("Expected name, found None"))?;

        // Parse converted type if exists
        let converted_type = if let Some("(") = self.tokenizer.next() {
            let converted_type = self
                .tokenizer
                .next()
                .ok_or_else(|| ParquetError::oos("Expected converted type, found None"))
                .and_then(|v| converted_group_from_str(&v.to_uppercase()))?;
            assert_token(self.tokenizer.next(), ")")?;
            Some(converted_type)
        } else {
            self.tokenizer.backtrack();
            None
        };

        // Parse optional id
        let id = if let Some("=") = self.tokenizer.next() {
            self.tokenizer.next().and_then(|v| v.parse::<i32>().ok())
        } else {
            self.tokenizer.backtrack();
            None
        };

        let fields = self.parse_child_types()?;

        Ok(ParquetType::from_converted(
            name.to_string(),
            fields,
            repetition,
            converted_type,
            id,
        ))
    }

    fn add_primitive_type(
        &mut self,
        repetition: Repetition,
        physical_type: Type,
    ) -> ParquetResult<ParquetType> {
        // Read type length if the type is FIXED_LEN_BYTE_ARRAY.
        let length = if physical_type == Type::FIXED_LEN_BYTE_ARRAY {
            assert_token(self.tokenizer.next(), "(")?;
            let length = parse_i32(
                self.tokenizer.next(),
                "Expected length for FIXED_LEN_BYTE_ARRAY, found None",
                "Failed to parse length for FIXED_LEN_BYTE_ARRAY",
            )?;
            assert_token(self.tokenizer.next(), ")")?;
            Some(length)
        } else {
            None
        };

        // Parse name of the primitive type
        let name = self
            .tokenizer
            .next()
            .ok_or_else(|| ParquetError::oos("Expected name, found None"))?;

        // Parse logical types
        let (converted_type, logical_type) = if let Some("(") = self.tokenizer.next() {
            let (is_logical_type, converted_type, token) = self
                .tokenizer
                .next()
                .ok_or_else(|| ParquetError::oos("Expected converted or logical type, found None"))
                .and_then(|v| {
                    let string = v.to_uppercase();
                    Ok(if is_logical_type(&string) {
                        (true, None, string)
                    } else if is_converted_type(&string) {
                        (false, converted_primitive_from_str(&string), string)
                    } else {
                        return Err(ParquetError::oos(format!(
                            "Expected converted or logical type, found {}",
                            string
                        )));
                    })
                })?;

            let logical_type = if is_logical_type {
                Some(self.parse_logical_type(&token)?)
            } else {
                None
            };

            // converted type decimal
            let converted_type = match converted_type {
                Some(PrimitiveConvertedType::Decimal(_, _)) => {
                    Some(self.parse_converted_decimal()?)
                },
                other => other,
            };

            assert_token(self.tokenizer.next(), ")")?;
            (converted_type, logical_type)
        } else {
            self.tokenizer.backtrack();
            (None, None)
        };

        // Parse optional id
        let id = if let Some("=") = self.tokenizer.next() {
            self.tokenizer.next().and_then(|v| v.parse::<i32>().ok())
        } else {
            self.tokenizer.backtrack();
            None
        };
        assert_token(self.tokenizer.next(), ";")?;

        ParquetType::try_from_primitive(
            name.to_string(),
            (physical_type, length).try_into()?,
            repetition,
            converted_type,
            logical_type,
            id,
        )
    }

    fn parse_converted_decimal(&mut self) -> ParquetResult<PrimitiveConvertedType> {
        assert_token(self.tokenizer.next(), "(")?;
        // Parse precision
        let precision = parse_i32(
            self.tokenizer.next(),
            "Expected precision, found None",
            "Failed to parse precision for DECIMAL type",
        )?;

        // Parse scale
        let scale = if let Some(",") = self.tokenizer.next() {
            parse_i32(
                self.tokenizer.next(),
                "Expected scale, found None",
                "Failed to parse scale for DECIMAL type",
            )?
        } else {
            // Scale is not provided, set it to 0.
            self.tokenizer.backtrack();
            0
        };

        assert_token(self.tokenizer.next(), ")")?;
        Ok(PrimitiveConvertedType::Decimal(
            precision.try_into()?,
            scale.try_into()?,
        ))
    }

    fn parse_logical_type(&mut self, tpe: &str) -> ParquetResult<PrimitiveLogicalType> {
        Ok(match tpe {
            "ENUM" => PrimitiveLogicalType::Enum,
            "DATE" => PrimitiveLogicalType::Date,
            "DECIMAL" => {
                let (precision, scale) = if let Some("(") = self.tokenizer.next() {
                    let precision = parse_i32(
                        self.tokenizer.next(),
                        "Expected precision, found None",
                        "Failed to parse precision for DECIMAL type",
                    )?;
                    let scale = if let Some(",") = self.tokenizer.next() {
                        parse_i32(
                            self.tokenizer.next(),
                            "Expected scale, found None",
                            "Failed to parse scale for DECIMAL type",
                        )?
                    } else {
                        self.tokenizer.backtrack();
                        0
                    };
                    assert_token(self.tokenizer.next(), ")")?;
                    (precision, scale)
                } else {
                    self.tokenizer.backtrack();
                    (0, 0)
                };
                PrimitiveLogicalType::Decimal(precision.try_into()?, scale.try_into()?)
            },
            "TIME" => {
                let (unit, is_adjusted_to_utc) = if let Some("(") = self.tokenizer.next() {
                    let unit = parse_timeunit(
                        self.tokenizer.next(),
                        "Invalid timeunit found",
                        "Failed to parse timeunit for TIME type",
                    )?;
                    let is_adjusted_to_utc = if let Some(",") = self.tokenizer.next() {
                        parse_bool(
                            self.tokenizer.next(),
                            "Invalid boolean found",
                            "Failed to parse timezone info for TIME type",
                        )?
                    } else {
                        self.tokenizer.backtrack();
                        false
                    };
                    assert_token(self.tokenizer.next(), ")")?;
                    (unit, is_adjusted_to_utc)
                } else {
                    self.tokenizer.backtrack();
                    (TimeUnit::Milliseconds, false)
                };
                PrimitiveLogicalType::Time {
                    is_adjusted_to_utc,
                    unit,
                }
            },
            "TIMESTAMP" => {
                let (unit, is_adjusted_to_utc) = if let Some("(") = self.tokenizer.next() {
                    let unit = parse_timeunit(
                        self.tokenizer.next(),
                        "Invalid timeunit found",
                        "Failed to parse timeunit for TIMESTAMP type",
                    )?;
                    let is_adjusted_to_utc = if let Some(",") = self.tokenizer.next() {
                        parse_bool(
                            self.tokenizer.next(),
                            "Invalid boolean found",
                            "Failed to parse timezone info for TIMESTAMP type",
                        )?
                    } else {
                        // Invalid token for unit
                        self.tokenizer.backtrack();
                        false
                    };
                    assert_token(self.tokenizer.next(), ")")?;
                    (unit, is_adjusted_to_utc)
                } else {
                    self.tokenizer.backtrack();
                    (TimeUnit::Milliseconds, false)
                };
                PrimitiveLogicalType::Timestamp {
                    is_adjusted_to_utc,
                    unit,
                }
            },
            "INTEGER" => {
                let (bit_width, is_signed) = if let Some("(") = self.tokenizer.next() {
                    let bit_width = parse_i32(
                        self.tokenizer.next(),
                        "Invalid bit_width found",
                        "Failed to parse bit_width for INTEGER type",
                    )?;
                    let is_signed = if let Some(",") = self.tokenizer.next() {
                        parse_bool(
                            self.tokenizer.next(),
                            "Invalid boolean found",
                            "Failed to parse is_signed for INTEGER type",
                        )?
                    } else {
                        // Invalid token for unit
                        self.tokenizer.backtrack();
                        return Err(ParquetError::oos("INTEGER requires sign"));
                    };
                    assert_token(self.tokenizer.next(), ")")?;
                    (bit_width, is_signed)
                } else {
                    // Invalid token for unit
                    self.tokenizer.backtrack();
                    return Err(ParquetError::oos("INTEGER requires width and sign"));
                };
                PrimitiveLogicalType::Integer((bit_width, is_signed).into())
            },
            "STRING" => PrimitiveLogicalType::String,
            "JSON" => PrimitiveLogicalType::Json,
            "BSON" => PrimitiveLogicalType::Bson,
            "UUID" => PrimitiveLogicalType::Uuid,
            "UNKNOWN" => PrimitiveLogicalType::Unknown,
            "INTERVAL" => return Err(ParquetError::oos("Interval logical type not yet supported")),
            _ => unreachable!(),
        })
    }
}

#[cfg(test)]
mod tests {
    use types::IntegerType;

    use super::*;
    use crate::parquet::schema::types::PhysicalType;

    #[test]
    fn test_tokenize_empty_string() {
        assert_eq!(Tokenizer::from_str("").next(), None);
    }

    #[test]
    fn test_tokenize_delimiters() {
        let mut iter = Tokenizer::from_str(",;{}()=");
        assert_eq!(iter.next(), Some(","));
        assert_eq!(iter.next(), Some(";"));
        assert_eq!(iter.next(), Some("{"));
        assert_eq!(iter.next(), Some("}"));
        assert_eq!(iter.next(), Some("("));
        assert_eq!(iter.next(), Some(")"));
        assert_eq!(iter.next(), Some("="));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_tokenize_delimiters_with_whitespaces() {
        let mut iter = Tokenizer::from_str(" , ; { } ( ) = ");
        assert_eq!(iter.next(), Some(","));
        assert_eq!(iter.next(), Some(";"));
        assert_eq!(iter.next(), Some("{"));
        assert_eq!(iter.next(), Some("}"));
        assert_eq!(iter.next(), Some("("));
        assert_eq!(iter.next(), Some(")"));
        assert_eq!(iter.next(), Some("="));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_tokenize_words() {
        let mut iter = Tokenizer::from_str("abc def ghi jkl mno");
        assert_eq!(iter.next(), Some("abc"));
        assert_eq!(iter.next(), Some("def"));
        assert_eq!(iter.next(), Some("ghi"));
        assert_eq!(iter.next(), Some("jkl"));
        assert_eq!(iter.next(), Some("mno"));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_tokenize_backtrack() {
        let mut iter = Tokenizer::from_str("abc;");
        assert_eq!(iter.next(), Some("abc"));
        assert_eq!(iter.next(), Some(";"));
        iter.backtrack();
        assert_eq!(iter.next(), Some(";"));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_tokenize_message_type() {
        let schema = "
    message schema {
      required int32 a;
      optional binary c (UTF8);
      required group d {
        required int32 a;
        optional binary c (UTF8);
      }
      required group e (LIST) {
        repeated group list {
          required int32 element;
        }
      }
    }
    ";
        let iter = Tokenizer::from_str(schema);
        let mut res = Vec::new();
        for token in iter {
            res.push(token);
        }
        assert_eq!(
            res,
            vec![
                "message", "schema", "{", "required", "int32", "a", ";", "optional", "binary", "c",
                "(", "UTF8", ")", ";", "required", "group", "d", "{", "required", "int32", "a",
                ";", "optional", "binary", "c", "(", "UTF8", ")", ";", "}", "required", "group",
                "e", "(", "LIST", ")", "{", "repeated", "group", "list", "{", "required", "int32",
                "element", ";", "}", "}", "}"
            ]
        );
    }

    #[test]
    fn test_assert_token() {
        assert!(assert_token(Some("a"), "a").is_ok());
        assert!(assert_token(Some("a"), "b").is_err());
        assert!(assert_token(None, "b").is_err());
    }

    #[test]
    fn test_parse_message_type_invalid() {
        let mut iter = Tokenizer::from_str("test");
        let result = Parser {
            tokenizer: &mut iter,
        }
        .parse_message_type();
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().to_string(),
            "File out of specification: Message type does not start with 'message'"
        );
    }

    #[test]
    fn test_parse_message_type_no_name() {
        let mut iter = Tokenizer::from_str("message");
        let result = Parser {
            tokenizer: &mut iter,
        }
        .parse_message_type();
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().to_string(),
            "File out of specification: Expected name, found None"
        );
    }

    #[test]
    fn test_parse_message_type_fixed_byte_array() {
        let schema = "
    message schema {
      REQUIRED FIXED_LEN_BYTE_ARRAY col;
    }
    ";
        let mut iter = Tokenizer::from_str(schema);
        let result = Parser {
            tokenizer: &mut iter,
        }
        .parse_message_type();
        assert!(result.is_err());

        let schema = "
    message schema {
      REQUIRED FIXED_LEN_BYTE_ARRAY(16) col;
    }
    ";
        let mut iter = Tokenizer::from_str(schema);
        let result = Parser {
            tokenizer: &mut iter,
        }
        .parse_message_type();
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_message_type_decimal() {
        // It is okay for decimal to omit precision and scale with right syntax.
        // Here we test wrong syntax of decimal type

        // Invalid decimal syntax
        let schema = "
    message root {
      optional int32 f1 (DECIMAL();
    }
    ";
        let mut iter = Tokenizer::from_str(schema);
        let result = Parser {
            tokenizer: &mut iter,
        }
        .parse_message_type();
        assert!(result.is_err());

        // Invalid decimal, need precision and scale
        let schema = "
    message root {
      optional int32 f1 (DECIMAL());
    }
    ";
        let mut iter = Tokenizer::from_str(schema);
        let result = Parser {
            tokenizer: &mut iter,
        }
        .parse_message_type();
        assert!(result.is_err());

        // Invalid decimal because of `,` - has precision, needs scale
        let schema = "
    message root {
      optional int32 f1 (DECIMAL(8,));
    }
    ";
        let mut iter = Tokenizer::from_str(schema);
        let result = Parser {
            tokenizer: &mut iter,
        }
        .parse_message_type();
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_decimal_wrong() {
        // Invalid decimal because, we always require either precision or scale to be
        // specified as part of converted type
        let schema = "
    message root {
      optional int32 f3 (DECIMAL);
    }
    ";
        let mut iter = Tokenizer::from_str(schema);
        let result = Parser {
            tokenizer: &mut iter,
        }
        .parse_message_type();
        assert!(result.is_err());

        // Valid decimal (precision, scale)
        let schema = "
    message root {
      optional int32 f1 (DECIMAL(8, 3));
      optional int32 f2 (DECIMAL(8));
    }
    ";
        let mut iter = Tokenizer::from_str(schema);
        let result = Parser {
            tokenizer: &mut iter,
        }
        .parse_message_type();
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_message_type_compare_1() -> ParquetResult<()> {
        let schema = "
    message root {
      optional fixed_len_byte_array(5) f1 (DECIMAL(9, 3));
      optional fixed_len_byte_array (16) f2 (DECIMAL (38, 18));
    }
    ";
        let mut iter = Tokenizer::from_str(schema);
        let message = Parser {
            tokenizer: &mut iter,
        }
        .parse_message_type()
        .unwrap();

        let fields = vec![
            ParquetType::try_from_primitive(
                "f1".to_string(),
                PhysicalType::FixedLenByteArray(5),
                Repetition::Optional,
                None,
                Some(PrimitiveLogicalType::Decimal(9, 3)),
                None,
            )?,
            ParquetType::try_from_primitive(
                "f2".to_string(),
                PhysicalType::FixedLenByteArray(16),
                Repetition::Optional,
                None,
                Some(PrimitiveLogicalType::Decimal(38, 18)),
                None,
            )?,
        ];

        let expected = ParquetType::new_root("root".to_string(), fields);

        assert_eq!(message, expected);
        Ok(())
    }

    #[test]
    fn test_parse_message_type_compare_2() -> ParquetResult<()> {
        let schema = "
    message root {
      required group a0 {
        optional group a1 (LIST) {
          repeated binary a2 (UTF8);
        }

        optional group b1 (LIST) {
          repeated group b2 {
            optional int32 b3;
            optional double b4;
          }
        }
      }
    }
    ";
        let mut iter = Tokenizer::from_str(schema);
        let message = Parser {
            tokenizer: &mut iter,
        }
        .parse_message_type()
        .unwrap();

        let a2 = ParquetType::try_from_primitive(
            "a2".to_string(),
            PhysicalType::ByteArray,
            Repetition::Repeated,
            Some(PrimitiveConvertedType::Utf8),
            None,
            None,
        )?;
        let a1 = ParquetType::from_converted(
            "a1".to_string(),
            vec![a2],
            Repetition::Optional,
            Some(GroupConvertedType::List),
            None,
        );
        let b2 = ParquetType::from_converted(
            "b2".to_string(),
            vec![
                ParquetType::from_physical("b3".to_string(), PhysicalType::Int32),
                ParquetType::from_physical("b4".to_string(), PhysicalType::Double),
            ],
            Repetition::Repeated,
            None,
            None,
        );
        let b1 = ParquetType::from_converted(
            "b1".to_string(),
            vec![b2],
            Repetition::Optional,
            Some(GroupConvertedType::List),
            None,
        );
        let a0 = ParquetType::from_converted(
            "a0".to_string(),
            vec![a1, b1],
            Repetition::Required,
            None,
            None,
        );

        let expected = ParquetType::new_root("root".to_string(), vec![a0]);

        assert_eq!(message, expected);
        Ok(())
    }

    #[test]
    fn test_parse_message_type_compare_3() -> ParquetResult<()> {
        let schema = "
    message root {
      required int32 _1 (INT_8);
      required int32 _2 (INT_16);
      required float _3;
      required double _4;
      optional int32 _5 (DATE);
      optional binary _6 (UTF8);
    }
    ";
        let mut iter = Tokenizer::from_str(schema);
        let message = Parser {
            tokenizer: &mut iter,
        }
        .parse_message_type()
        .unwrap();

        let f1 = ParquetType::try_from_primitive(
            "_1".to_string(),
            PhysicalType::Int32,
            Repetition::Required,
            Some(PrimitiveConvertedType::Int8),
            None,
            None,
        )?;
        let f2 = ParquetType::try_from_primitive(
            "_2".to_string(),
            PhysicalType::Int32,
            Repetition::Required,
            Some(PrimitiveConvertedType::Int16),
            None,
            None,
        )?;
        let f3 = ParquetType::try_from_primitive(
            "_3".to_string(),
            PhysicalType::Float,
            Repetition::Required,
            None,
            None,
            None,
        )?;
        let f4 = ParquetType::try_from_primitive(
            "_4".to_string(),
            PhysicalType::Double,
            Repetition::Required,
            None,
            None,
            None,
        )?;
        let f5 = ParquetType::try_from_primitive(
            "_5".to_string(),
            PhysicalType::Int32,
            Repetition::Optional,
            None,
            Some(PrimitiveLogicalType::Date),
            None,
        )?;
        let f6 = ParquetType::try_from_primitive(
            "_6".to_string(),
            PhysicalType::ByteArray,
            Repetition::Optional,
            Some(PrimitiveConvertedType::Utf8),
            None,
            None,
        )?;

        let fields = vec![f1, f2, f3, f4, f5, f6];

        let expected = ParquetType::new_root("root".to_string(), fields);
        assert_eq!(message, expected);
        Ok(())
    }

    #[test]
    fn test_parse_message_type_compare_4() -> ParquetResult<()> {
        let schema = "
    message root {
      required int32 _1 (INTEGER(8,true));
      required int32 _2 (INTEGER(16,false));
      required float _3;
      required double _4;
      optional int32 _5 (DATE);
      optional int32 _6 (TIME(MILLIS,false));
      optional int64 _7 (TIME(MICROS,true));
      optional int64 _8 (TIMESTAMP(MILLIS,true));
      optional int64 _9 (TIMESTAMP(NANOS,false));
      optional binary _10 (STRING);
    }
    ";
        let mut iter = Tokenizer::from_str(schema);
        let message = Parser {
            tokenizer: &mut iter,
        }
        .parse_message_type()?;

        let f1 = ParquetType::try_from_primitive(
            "_1".to_string(),
            PhysicalType::Int32,
            Repetition::Required,
            None,
            Some(PrimitiveLogicalType::Integer(IntegerType::Int8)),
            None,
        )?;
        let f2 = ParquetType::try_from_primitive(
            "_2".to_string(),
            PhysicalType::Int32,
            Repetition::Required,
            None,
            Some(PrimitiveLogicalType::Integer(IntegerType::UInt16)),
            None,
        )?;
        let f3 = ParquetType::try_from_primitive(
            "_3".to_string(),
            PhysicalType::Float,
            Repetition::Required,
            None,
            None,
            None,
        )?;
        let f4 = ParquetType::try_from_primitive(
            "_4".to_string(),
            PhysicalType::Double,
            Repetition::Required,
            None,
            None,
            None,
        )?;
        let f5 = ParquetType::try_from_primitive(
            "_5".to_string(),
            PhysicalType::Int32,
            Repetition::Optional,
            None,
            Some(PrimitiveLogicalType::Date),
            None,
        )?;
        let f6 = ParquetType::try_from_primitive(
            "_6".to_string(),
            PhysicalType::Int32,
            Repetition::Optional,
            None,
            Some(PrimitiveLogicalType::Time {
                is_adjusted_to_utc: false,
                unit: TimeUnit::Milliseconds,
            }),
            None,
        )?;
        let f7 = ParquetType::try_from_primitive(
            "_7".to_string(),
            PhysicalType::Int64,
            Repetition::Optional,
            None,
            Some(PrimitiveLogicalType::Time {
                is_adjusted_to_utc: true,
                unit: TimeUnit::Microseconds,
            }),
            None,
        )?;
        let f8 = ParquetType::try_from_primitive(
            "_8".to_string(),
            PhysicalType::Int64,
            Repetition::Optional,
            None,
            Some(PrimitiveLogicalType::Timestamp {
                is_adjusted_to_utc: true,
                unit: TimeUnit::Milliseconds,
            }),
            None,
        )?;
        let f9 = ParquetType::try_from_primitive(
            "_9".to_string(),
            PhysicalType::Int64,
            Repetition::Optional,
            None,
            Some(PrimitiveLogicalType::Timestamp {
                is_adjusted_to_utc: false,
                unit: TimeUnit::Nanoseconds,
            }),
            None,
        )?;

        let f10 = ParquetType::try_from_primitive(
            "_10".to_string(),
            PhysicalType::ByteArray,
            Repetition::Optional,
            None,
            Some(PrimitiveLogicalType::String),
            None,
        )?;

        let fields = vec![f1, f2, f3, f4, f5, f6, f7, f8, f9, f10];

        let expected = ParquetType::new_root("root".to_string(), fields);
        assert_eq!(message, expected);
        Ok(())
    }
}
