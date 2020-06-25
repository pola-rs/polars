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

//! Parquet schema printer.
//! Provides methods to print Parquet file schema and list file metadata.
//!
//! # Example
//!
//! ```rust
//! use parquet::{
//!     file::reader::{FileReader, SerializedFileReader},
//!     schema::printer::{print_file_metadata, print_parquet_metadata, print_schema},
//! };
//! use std::{fs::File, path::Path};
//!
//! // Open a file
//! let path = Path::new("test.parquet");
//! if let Ok(file) = File::open(&path) {
//!     let reader = SerializedFileReader::new(file).unwrap();
//!     let parquet_metadata = reader.metadata();
//!
//!     print_parquet_metadata(&mut std::io::stdout(), &parquet_metadata);
//!     print_file_metadata(&mut std::io::stdout(), &parquet_metadata.file_metadata());
//!
//!     print_schema(
//!         &mut std::io::stdout(),
//!         &parquet_metadata.file_metadata().schema(),
//!     );
//! }
//! ```

use std::{fmt, io};

use crate::basic::{LogicalType, Type as PhysicalType};
use crate::file::metadata::{
    ColumnChunkMetaData, FileMetaData, ParquetMetaData, RowGroupMetaData,
};
use crate::schema::types::Type;

/// Prints Parquet metadata [`ParquetMetaData`](crate::file::metadata::ParquetMetaData)
/// information.
#[allow(unused_must_use)]
pub fn print_parquet_metadata(out: &mut io::Write, metadata: &ParquetMetaData) {
    print_file_metadata(out, &metadata.file_metadata());
    writeln!(out, "");
    writeln!(out, "");
    writeln!(out, "num of row groups: {}", metadata.num_row_groups());
    writeln!(out, "row groups:");
    writeln!(out, "");
    for (i, rg) in metadata.row_groups().iter().enumerate() {
        writeln!(out, "row group {}:", i);
        print_dashes(out, 80);
        print_row_group_metadata(out, rg);
    }
}

/// Prints file metadata [`FileMetaData`](crate::file::metadata::FileMetaData)
/// information.
#[allow(unused_must_use)]
pub fn print_file_metadata(out: &mut io::Write, file_metadata: &FileMetaData) {
    writeln!(out, "version: {}", file_metadata.version());
    writeln!(out, "num of rows: {}", file_metadata.num_rows());
    if let Some(created_by) = file_metadata.created_by().as_ref() {
        writeln!(out, "created by: {}", created_by);
    }
    if let Some(metadata) = file_metadata.key_value_metadata() {
        writeln!(out, "metadata:");
        for kv in metadata.iter() {
            writeln!(
                out,
                "  {}: {}",
                &kv.key,
                kv.value.as_ref().unwrap_or(&"".to_owned())
            );
        }
    }
    let schema = file_metadata.schema();
    print_schema(out, schema);
}

/// Prints Parquet [`Type`](crate::schema::types::Type) information.
#[allow(unused_must_use)]
pub fn print_schema(out: &mut io::Write, tp: &Type) {
    // TODO: better if we can pass fmt::Write to Printer.
    // But how can we make it to accept both io::Write & fmt::Write?
    let mut s = String::new();
    {
        let mut printer = Printer::new(&mut s);
        printer.print(tp);
    }
    writeln!(out, "{}", s);
}

#[allow(unused_must_use)]
fn print_row_group_metadata(out: &mut io::Write, rg_metadata: &RowGroupMetaData) {
    writeln!(out, "total byte size: {}", rg_metadata.total_byte_size());
    writeln!(out, "num of rows: {}", rg_metadata.num_rows());
    writeln!(out, "");
    writeln!(out, "num of columns: {}", rg_metadata.num_columns());
    writeln!(out, "columns: ");
    for (i, cc) in rg_metadata.columns().iter().enumerate() {
        writeln!(out, "");
        writeln!(out, "column {}:", i);
        print_dashes(out, 80);
        print_column_chunk_metadata(out, cc);
    }
}

#[allow(unused_must_use)]
fn print_column_chunk_metadata(out: &mut io::Write, cc_metadata: &ColumnChunkMetaData) {
    writeln!(out, "column type: {}", cc_metadata.column_type());
    writeln!(out, "column path: {}", cc_metadata.column_path());
    let encoding_strs: Vec<_> = cc_metadata
        .encodings()
        .iter()
        .map(|e| format!("{}", e))
        .collect();
    writeln!(out, "encodings: {}", encoding_strs.join(" "));
    let file_path_str = match cc_metadata.file_path() {
        None => "N/A",
        Some(ref fp) => *fp,
    };
    writeln!(out, "file path: {}", file_path_str);
    writeln!(out, "file offset: {}", cc_metadata.file_offset());
    writeln!(out, "num of values: {}", cc_metadata.num_values());
    writeln!(
        out,
        "total compressed size (in bytes): {}",
        cc_metadata.compressed_size()
    );
    writeln!(
        out,
        "total uncompressed size (in bytes): {}",
        cc_metadata.uncompressed_size()
    );
    writeln!(out, "data page offset: {}", cc_metadata.data_page_offset());
    let index_page_offset_str = match cc_metadata.index_page_offset() {
        None => "N/A".to_owned(),
        Some(ipo) => ipo.to_string(),
    };
    writeln!(out, "index page offset: {}", index_page_offset_str);
    let dict_page_offset_str = match cc_metadata.dictionary_page_offset() {
        None => "N/A".to_owned(),
        Some(dpo) => dpo.to_string(),
    };
    writeln!(out, "dictionary page offset: {}", dict_page_offset_str);
    let statistics_str = match cc_metadata.statistics() {
        None => "N/A".to_owned(),
        Some(stats) => stats.to_string(),
    };
    writeln!(out, "statistics: {}", statistics_str);
    writeln!(out, "");
}

#[allow(unused_must_use)]
fn print_dashes(out: &mut io::Write, num: i32) {
    for _ in 0..num {
        write!(out, "-");
    }
    writeln!(out, "");
}

const INDENT_WIDTH: i32 = 2;

/// Struct for printing Parquet message type.
struct Printer<'a> {
    output: &'a mut fmt::Write,
    indent: i32,
}

#[allow(unused_must_use)]
impl<'a> Printer<'a> {
    fn new(output: &'a mut fmt::Write) -> Self {
        Printer { output, indent: 0 }
    }

    fn print_indent(&mut self) {
        for _ in 0..self.indent {
            write!(self.output, " ");
        }
    }
}

#[allow(unused_must_use)]
impl<'a> Printer<'a> {
    pub fn print(&mut self, tp: &Type) {
        self.print_indent();
        match tp {
            &Type::PrimitiveType {
                ref basic_info,
                physical_type,
                type_length,
                scale,
                precision,
            } => {
                let phys_type_str = match physical_type {
                    PhysicalType::FIXED_LEN_BYTE_ARRAY => {
                        // We need to include length for fixed byte array
                        format!("{} ({})", physical_type, type_length)
                    }
                    _ => format!("{}", physical_type),
                };
                // Also print logical type if it is available
                let logical_type_str = match basic_info.logical_type() {
                    LogicalType::NONE => format!(""),
                    decimal @ LogicalType::DECIMAL => {
                        // For decimal type we should print precision and scale if they
                        // are > 0, e.g. DECIMAL(9, 2) -
                        // DECIMAL(9) - DECIMAL
                        let precision_scale = match (precision, scale) {
                            (p, s) if p > 0 && s > 0 => format!(" ({}, {})", p, s),
                            (p, 0) if p > 0 => format!(" ({})", p),
                            _ => format!(""),
                        };
                        format!(" ({}{})", decimal, precision_scale)
                    }
                    other_logical_type => format!(" ({})", other_logical_type),
                };
                write!(
                    self.output,
                    "{} {} {}{};",
                    basic_info.repetition(),
                    phys_type_str,
                    basic_info.name(),
                    logical_type_str
                );
            }
            &Type::GroupType {
                ref basic_info,
                ref fields,
            } => {
                if basic_info.has_repetition() {
                    let r = basic_info.repetition();
                    write!(self.output, "{} group {} ", r, basic_info.name());
                    if basic_info.logical_type() != LogicalType::NONE {
                        write!(self.output, "({}) ", basic_info.logical_type());
                    }
                    writeln!(self.output, "{{");
                } else {
                    writeln!(self.output, "message {} {{", basic_info.name());
                }

                self.indent += INDENT_WIDTH;
                for c in fields {
                    self.print(&c);
                    writeln!(self.output, "");
                }
                self.indent -= INDENT_WIDTH;
                self.print_indent();
                write!(self.output, "}}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::rc::Rc;

    use crate::basic::{Repetition, Type as PhysicalType};
    use crate::schema::{parser::parse_message_type, types::Type};

    fn assert_print_parse_message(message: Type) {
        let mut s = String::new();
        {
            let mut p = Printer::new(&mut s);
            p.print(&message);
        }
        let parsed = parse_message_type(&s).unwrap();
        assert_eq!(message, parsed);
    }

    #[test]
    fn test_print_primitive_type() {
        let mut s = String::new();
        {
            let mut p = Printer::new(&mut s);
            let foo = Type::primitive_type_builder("foo", PhysicalType::INT32)
                .with_repetition(Repetition::REQUIRED)
                .with_logical_type(LogicalType::INT_32)
                .build()
                .unwrap();
            p.print(&foo);
        }
        assert_eq!(&mut s, "REQUIRED INT32 foo (INT_32);");
    }

    #[test]
    fn test_print_primitive_type_without_logical() {
        let mut s = String::new();
        {
            let mut p = Printer::new(&mut s);
            let foo = Type::primitive_type_builder("foo", PhysicalType::DOUBLE)
                .with_repetition(Repetition::REQUIRED)
                .build()
                .unwrap();
            p.print(&foo);
        }
        assert_eq!(&mut s, "REQUIRED DOUBLE foo;");
    }

    #[test]
    fn test_print_group_type() {
        let mut s = String::new();
        {
            let mut p = Printer::new(&mut s);
            let f1 = Type::primitive_type_builder("f1", PhysicalType::INT32)
                .with_repetition(Repetition::REQUIRED)
                .with_logical_type(LogicalType::INT_32)
                .with_id(0)
                .build();
            let f2 = Type::primitive_type_builder("f2", PhysicalType::BYTE_ARRAY)
                .with_logical_type(LogicalType::UTF8)
                .with_id(1)
                .build();
            let f3 =
                Type::primitive_type_builder("f3", PhysicalType::FIXED_LEN_BYTE_ARRAY)
                    .with_repetition(Repetition::REPEATED)
                    .with_logical_type(LogicalType::INTERVAL)
                    .with_length(12)
                    .with_id(2)
                    .build();
            let mut struct_fields = Vec::new();
            struct_fields.push(Rc::new(f1.unwrap()));
            struct_fields.push(Rc::new(f2.unwrap()));
            let foo = Type::group_type_builder("foo")
                .with_repetition(Repetition::OPTIONAL)
                .with_fields(&mut struct_fields)
                .with_id(1)
                .build()
                .unwrap();
            let mut fields = Vec::new();
            fields.push(Rc::new(foo));
            fields.push(Rc::new(f3.unwrap()));
            let message = Type::group_type_builder("schema")
                .with_fields(&mut fields)
                .with_id(2)
                .build()
                .unwrap();
            p.print(&message);
        }
        let expected = "message schema {
  OPTIONAL group foo {
    REQUIRED INT32 f1 (INT_32);
    OPTIONAL BYTE_ARRAY f2 (UTF8);
  }
  REPEATED FIXED_LEN_BYTE_ARRAY (12) f3 (INTERVAL);
}";
        assert_eq!(&mut s, expected);
    }

    #[test]
    fn test_print_and_parse_primitive() {
        let a2 = Type::primitive_type_builder("a2", PhysicalType::BYTE_ARRAY)
            .with_repetition(Repetition::REPEATED)
            .with_logical_type(LogicalType::UTF8)
            .build()
            .unwrap();

        let a1 = Type::group_type_builder("a1")
            .with_repetition(Repetition::OPTIONAL)
            .with_logical_type(LogicalType::LIST)
            .with_fields(&mut vec![Rc::new(a2)])
            .build()
            .unwrap();

        let b3 = Type::primitive_type_builder("b3", PhysicalType::INT32)
            .with_repetition(Repetition::OPTIONAL)
            .build()
            .unwrap();

        let b4 = Type::primitive_type_builder("b4", PhysicalType::DOUBLE)
            .with_repetition(Repetition::OPTIONAL)
            .build()
            .unwrap();

        let b2 = Type::group_type_builder("b2")
            .with_repetition(Repetition::REPEATED)
            .with_logical_type(LogicalType::NONE)
            .with_fields(&mut vec![Rc::new(b3), Rc::new(b4)])
            .build()
            .unwrap();

        let b1 = Type::group_type_builder("b1")
            .with_repetition(Repetition::OPTIONAL)
            .with_logical_type(LogicalType::LIST)
            .with_fields(&mut vec![Rc::new(b2)])
            .build()
            .unwrap();

        let a0 = Type::group_type_builder("a0")
            .with_repetition(Repetition::REQUIRED)
            .with_fields(&mut vec![Rc::new(a1), Rc::new(b1)])
            .build()
            .unwrap();

        let message = Type::group_type_builder("root")
            .with_fields(&mut vec![Rc::new(a0)])
            .build()
            .unwrap();

        assert_print_parse_message(message);
    }

    #[test]
    fn test_print_and_parse_nested() {
        let f1 = Type::primitive_type_builder("f1", PhysicalType::INT32)
            .with_repetition(Repetition::REQUIRED)
            .with_logical_type(LogicalType::INT_32)
            .build()
            .unwrap();

        let f2 = Type::primitive_type_builder("f2", PhysicalType::BYTE_ARRAY)
            .with_repetition(Repetition::OPTIONAL)
            .with_logical_type(LogicalType::UTF8)
            .build()
            .unwrap();

        let foo = Type::group_type_builder("foo")
            .with_repetition(Repetition::OPTIONAL)
            .with_fields(&mut vec![Rc::new(f1), Rc::new(f2)])
            .build()
            .unwrap();

        let f3 = Type::primitive_type_builder("f3", PhysicalType::FIXED_LEN_BYTE_ARRAY)
            .with_repetition(Repetition::REPEATED)
            .with_logical_type(LogicalType::INTERVAL)
            .with_length(12)
            .build()
            .unwrap();

        let message = Type::group_type_builder("schema")
            .with_fields(&mut vec![Rc::new(foo), Rc::new(f3)])
            .build()
            .unwrap();

        assert_print_parse_message(message);
    }

    #[test]
    fn test_print_and_parse_decimal() {
        let f1 = Type::primitive_type_builder("f1", PhysicalType::INT32)
            .with_repetition(Repetition::OPTIONAL)
            .with_logical_type(LogicalType::DECIMAL)
            .with_precision(9)
            .with_scale(2)
            .build()
            .unwrap();

        let f2 = Type::primitive_type_builder("f2", PhysicalType::INT32)
            .with_repetition(Repetition::OPTIONAL)
            .with_logical_type(LogicalType::DECIMAL)
            .with_precision(9)
            .with_scale(0)
            .build()
            .unwrap();

        let message = Type::group_type_builder("schema")
            .with_fields(&mut vec![Rc::new(f1), Rc::new(f2)])
            .build()
            .unwrap();

        assert_print_parse_message(message);
    }
}
