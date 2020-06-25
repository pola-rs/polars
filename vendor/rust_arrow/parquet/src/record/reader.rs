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

//! Contains implementation of record assembly and converting Parquet types into
//! [`Row`](crate::record::api::Row)s.

use std::{collections::HashMap, fmt, rc::Rc};

use crate::basic::{LogicalType, Repetition};
use crate::errors::{ParquetError, Result};
use crate::file::reader::{FileReader, RowGroupReader};
use crate::record::{
    api::{make_list, make_map, make_row, Field, Row},
    triplet::TripletIter,
};
use crate::schema::types::{ColumnPath, SchemaDescPtr, SchemaDescriptor, Type, TypePtr};

/// Default batch size for a reader
const DEFAULT_BATCH_SIZE: usize = 1024;

/// Tree builder for `Reader` enum.
/// Serves as a container of options for building a reader tree and a builder, and
/// accessing a records iterator [`RowIter`].
pub struct TreeBuilder {
    // Batch size (>= 1) for triplet iterators
    batch_size: usize,
}

impl TreeBuilder {
    /// Creates new tree builder with default parameters.
    pub fn new() -> Self {
        Self {
            batch_size: DEFAULT_BATCH_SIZE,
        }
    }

    /// Sets batch size for this tree builder.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Creates new root reader for provided schema and row group.
    pub fn build(
        &self,
        descr: SchemaDescPtr,
        row_group_reader: &RowGroupReader,
    ) -> Reader {
        // Prepare lookup table of column path -> original column index
        // This allows to prune columns and map schema leaf nodes to the column readers
        let mut paths: HashMap<ColumnPath, usize> = HashMap::new();
        let row_group_metadata = row_group_reader.metadata();

        for col_index in 0..row_group_reader.num_columns() {
            let col_meta = row_group_metadata.column(col_index);
            let col_path = col_meta.column_path().clone();
            paths.insert(col_path, col_index);
        }

        // Build child readers for the message type
        let mut readers = Vec::new();
        let mut path = Vec::new();

        for field in descr.root_schema().get_fields() {
            let reader = self.reader_tree(
                field.clone(),
                &mut path,
                0,
                0,
                &paths,
                row_group_reader,
            );
            readers.push(reader);
        }

        // Return group reader for message type,
        // it is always required with definition level 0
        Reader::GroupReader(None, 0, readers)
    }

    /// Creates iterator of `Row`s directly from schema descriptor and row group.
    pub fn as_iter(
        &self,
        descr: SchemaDescPtr,
        row_group_reader: &RowGroupReader,
    ) -> ReaderIter {
        let num_records = row_group_reader.metadata().num_rows() as usize;
        ReaderIter::new(self.build(descr, row_group_reader), num_records)
    }

    /// Builds tree of readers for the current schema recursively.
    fn reader_tree(
        &self,
        field: TypePtr,
        mut path: &mut Vec<String>,
        mut curr_def_level: i16,
        mut curr_rep_level: i16,
        paths: &HashMap<ColumnPath, usize>,
        row_group_reader: &RowGroupReader,
    ) -> Reader {
        assert!(field.get_basic_info().has_repetition());
        // Update current definition and repetition levels for this type
        let repetition = field.get_basic_info().repetition();
        match repetition {
            Repetition::OPTIONAL => {
                curr_def_level += 1;
            }
            Repetition::REPEATED => {
                curr_def_level += 1;
                curr_rep_level += 1;
            }
            _ => {}
        }

        path.push(String::from(field.name()));
        let reader = if field.is_primitive() {
            let col_path = ColumnPath::new(path.to_vec());
            let orig_index = *paths.get(&col_path).unwrap();
            let col_descr = row_group_reader
                .metadata()
                .column(orig_index)
                .column_descr_ptr();
            let col_reader = row_group_reader.get_column_reader(orig_index).unwrap();
            let column = TripletIter::new(col_descr, col_reader, self.batch_size);
            Reader::PrimitiveReader(field, column)
        } else {
            match field.get_basic_info().logical_type() {
                // List types
                LogicalType::LIST => {
                    assert_eq!(
                        field.get_fields().len(),
                        1,
                        "Invalid list type {:?}",
                        field
                    );

                    let repeated_field = field.get_fields()[0].clone();
                    assert_eq!(
                        repeated_field.get_basic_info().repetition(),
                        Repetition::REPEATED,
                        "Invalid list type {:?}",
                        field
                    );

                    if Reader::is_element_type(&repeated_field) {
                        // Support for backward compatible lists
                        let reader = self.reader_tree(
                            repeated_field.clone(),
                            &mut path,
                            curr_def_level,
                            curr_rep_level,
                            paths,
                            row_group_reader,
                        );

                        Reader::RepeatedReader(
                            field,
                            curr_def_level,
                            curr_rep_level,
                            Box::new(reader),
                        )
                    } else {
                        let child_field = repeated_field.get_fields()[0].clone();

                        path.push(String::from(repeated_field.name()));

                        let reader = self.reader_tree(
                            child_field,
                            &mut path,
                            curr_def_level + 1,
                            curr_rep_level + 1,
                            paths,
                            row_group_reader,
                        );

                        path.pop();

                        Reader::RepeatedReader(
                            field,
                            curr_def_level,
                            curr_rep_level,
                            Box::new(reader),
                        )
                    }
                }
                // Map types (key-value pairs)
                LogicalType::MAP | LogicalType::MAP_KEY_VALUE => {
                    assert_eq!(
                        field.get_fields().len(),
                        1,
                        "Invalid map type: {:?}",
                        field
                    );
                    assert!(
                        !field.get_fields()[0].is_primitive(),
                        "Invalid map type: {:?}",
                        field
                    );

                    let key_value_type = field.get_fields()[0].clone();
                    assert_eq!(
                        key_value_type.get_basic_info().repetition(),
                        Repetition::REPEATED,
                        "Invalid map type: {:?}",
                        field
                    );
                    assert_eq!(
                        key_value_type.get_fields().len(),
                        2,
                        "Invalid map type: {:?}",
                        field
                    );

                    path.push(String::from(key_value_type.name()));

                    let key_type = &key_value_type.get_fields()[0];
                    assert!(
                        key_type.is_primitive(),
                        "Map key type is expected to be a primitive type, but found {:?}",
                        key_type
                    );
                    let key_reader = self.reader_tree(
                        key_type.clone(),
                        &mut path,
                        curr_def_level + 1,
                        curr_rep_level + 1,
                        paths,
                        row_group_reader,
                    );

                    let value_type = &key_value_type.get_fields()[1];
                    let value_reader = self.reader_tree(
                        value_type.clone(),
                        &mut path,
                        curr_def_level + 1,
                        curr_rep_level + 1,
                        paths,
                        row_group_reader,
                    );

                    path.pop();

                    Reader::KeyValueReader(
                        field,
                        curr_def_level,
                        curr_rep_level,
                        Box::new(key_reader),
                        Box::new(value_reader),
                    )
                }
                // A repeated field that is neither contained by a `LIST`- or
                // `MAP`-annotated group nor annotated by `LIST` or `MAP`
                // should be interpreted as a required list of required
                // elements where the element type is the type of the field.
                _ if repetition == Repetition::REPEATED => {
                    let required_field = Type::group_type_builder(field.name())
                        .with_repetition(Repetition::REQUIRED)
                        .with_logical_type(field.get_basic_info().logical_type())
                        .with_fields(&mut Vec::from(field.get_fields()))
                        .build()
                        .unwrap();

                    path.pop();

                    let reader = self.reader_tree(
                        Rc::new(required_field),
                        &mut path,
                        curr_def_level,
                        curr_rep_level,
                        paths,
                        row_group_reader,
                    );

                    Reader::RepeatedReader(
                        field,
                        curr_def_level - 1,
                        curr_rep_level - 1,
                        Box::new(reader),
                    )
                }
                // Group types (structs)
                _ => {
                    let mut readers = Vec::new();
                    for child in field.get_fields() {
                        let reader = self.reader_tree(
                            child.clone(),
                            &mut path,
                            curr_def_level,
                            curr_rep_level,
                            paths,
                            row_group_reader,
                        );
                        readers.push(reader);
                    }
                    Reader::GroupReader(Some(field), curr_def_level, readers)
                }
            }
        };
        path.pop();

        Reader::option(repetition, curr_def_level, reader)
    }
}

/// Reader tree for record assembly
pub enum Reader {
    // Primitive reader with type information and triplet iterator
    PrimitiveReader(TypePtr, TripletIter),
    // Optional reader with definition level of a parent and a reader
    OptionReader(i16, Box<Reader>),
    // Group (struct) reader with type information, definition level and list of child
    // readers. When it represents message type, type information is None
    GroupReader(Option<TypePtr>, i16, Vec<Reader>),
    // Reader for repeated values, e.g. lists, contains type information, definition
    // level, repetition level and a child reader
    RepeatedReader(TypePtr, i16, i16, Box<Reader>),
    // Reader of key-value pairs, e.g. maps, contains type information, definition
    // level, repetition level, child reader for keys and child reader for values
    KeyValueReader(TypePtr, i16, i16, Box<Reader>, Box<Reader>),
}

impl Reader {
    /// Wraps reader in option reader based on repetition.
    fn option(repetition: Repetition, def_level: i16, reader: Reader) -> Self {
        if repetition == Repetition::OPTIONAL {
            Reader::OptionReader(def_level - 1, Box::new(reader))
        } else {
            reader
        }
    }

    /// Returns true if repeated type is an element type for the list.
    /// Used to determine legacy list types.
    /// This method is copied from Spark Parquet reader and is based on the reference:
    /// https://github.com/apache/parquet-format/blob/master/LogicalTypes.md
    ///   #backward-compatibility-rules
    fn is_element_type(repeated_type: &Type) -> bool {
        // For legacy 2-level list types with primitive element type, e.g.:
        //
        //    // ARRAY<INT> (nullable list, non-null elements)
        //    optional group my_list (LIST) {
        //      repeated int32 element;
        //    }
        //
        repeated_type.is_primitive() ||
    // For legacy 2-level list types whose element type is a group type with 2 or more
    // fields, e.g.:
    //
    //    // ARRAY<STRUCT<str: STRING, num: INT>> (nullable list, non-null elements)
    //    optional group my_list (LIST) {
    //      repeated group element {
    //        required binary str (UTF8);
    //        required int32 num;
    //      };
    //    }
    //
    repeated_type.is_group() && repeated_type.get_fields().len() > 1 ||
    // For legacy 2-level list types generated by parquet-avro (Parquet version < 1.6.0),
    // e.g.:
    //
    //    // ARRAY<STRUCT<str: STRING>> (nullable list, non-null elements)
    //    optional group my_list (LIST) {
    //      repeated group array {
    //        required binary str (UTF8);
    //      };
    //    }
    //
    repeated_type.name() == "array" ||
    // For Parquet data generated by parquet-thrift, e.g.:
    //
    //    // ARRAY<STRUCT<str: STRING>> (nullable list, non-null elements)
    //    optional group my_list (LIST) {
    //      repeated group my_list_tuple {
    //        required binary str (UTF8);
    //      };
    //    }
    //
    repeated_type.name().ends_with("_tuple")
    }

    /// Reads current record as `Row` from the reader tree.
    /// Automatically advances all necessary readers.
    /// This must be called on the root level reader (i.e., for Message type).
    /// Otherwise, it will panic.
    fn read(&mut self) -> Row {
        match *self {
            Reader::GroupReader(_, _, ref mut readers) => {
                let mut fields = Vec::new();
                for reader in readers {
                    fields.push((String::from(reader.field_name()), reader.read_field()));
                }
                make_row(fields)
            }
            _ => panic!("Cannot call read() on {}", self),
        }
    }

    /// Reads current record as `Field` from the reader tree.
    /// Automatically advances all necessary readers.
    fn read_field(&mut self) -> Field {
        match *self {
            Reader::PrimitiveReader(_, ref mut column) => {
                let value = column.current_value();
                column.read_next().unwrap();
                value
            }
            Reader::OptionReader(def_level, ref mut reader) => {
                if reader.current_def_level() > def_level {
                    reader.read_field()
                } else {
                    reader.advance_columns();
                    Field::Null
                }
            }
            Reader::GroupReader(_, def_level, ref mut readers) => {
                let mut fields = Vec::new();
                for reader in readers {
                    if reader.repetition() != Repetition::OPTIONAL
                        || reader.current_def_level() > def_level
                    {
                        fields.push((
                            String::from(reader.field_name()),
                            reader.read_field(),
                        ));
                    } else {
                        reader.advance_columns();
                        fields.push((String::from(reader.field_name()), Field::Null));
                    }
                }
                let row = make_row(fields);
                Field::Group(row)
            }
            Reader::RepeatedReader(_, def_level, rep_level, ref mut reader) => {
                let mut elements = Vec::new();
                loop {
                    if reader.current_def_level() > def_level {
                        elements.push(reader.read_field());
                    } else {
                        reader.advance_columns();
                        // If the current definition level is equal to the definition
                        // level of this repeated type, then the
                        // result is an empty list and the repetition level
                        // will always be <= rl.
                        break;
                    }

                    // This covers case when we are out of repetition levels and should
                    // close the group, or there are no values left to
                    // buffer.
                    if !reader.has_next() || reader.current_rep_level() <= rep_level {
                        break;
                    }
                }
                Field::ListInternal(make_list(elements))
            }
            Reader::KeyValueReader(
                _,
                def_level,
                rep_level,
                ref mut keys,
                ref mut values,
            ) => {
                let mut pairs = Vec::new();
                loop {
                    if keys.current_def_level() > def_level {
                        pairs.push((keys.read_field(), values.read_field()));
                    } else {
                        keys.advance_columns();
                        values.advance_columns();
                        // If the current definition level is equal to the definition
                        // level of this repeated type, then the
                        // result is an empty list and the repetition level
                        // will always be <= rl.
                        break;
                    }

                    // This covers case when we are out of repetition levels and should
                    // close the group, or there are no values left to
                    // buffer.
                    if !keys.has_next() || keys.current_rep_level() <= rep_level {
                        break;
                    }
                }

                Field::MapInternal(make_map(pairs))
            }
        }
    }

    /// Returns field name for the current reader.
    fn field_name(&self) -> &str {
        match *self {
            Reader::PrimitiveReader(ref field, _) => field.name(),
            Reader::OptionReader(_, ref reader) => reader.field_name(),
            Reader::GroupReader(ref opt, ..) => match opt {
                &Some(ref field) => field.name(),
                &None => panic!("Field is None for group reader"),
            },
            Reader::RepeatedReader(ref field, ..) => field.name(),
            Reader::KeyValueReader(ref field, ..) => field.name(),
        }
    }

    /// Returns repetition for the current reader.
    fn repetition(&self) -> Repetition {
        match *self {
            Reader::PrimitiveReader(ref field, _) => field.get_basic_info().repetition(),
            Reader::OptionReader(_, ref reader) => reader.repetition(),
            Reader::GroupReader(ref opt, ..) => match opt {
                &Some(ref field) => field.get_basic_info().repetition(),
                &None => panic!("Field is None for group reader"),
            },
            Reader::RepeatedReader(ref field, ..) => field.get_basic_info().repetition(),
            Reader::KeyValueReader(ref field, ..) => field.get_basic_info().repetition(),
        }
    }

    /// Returns true, if current reader has more values, false otherwise.
    /// Method does not advance internal iterator.
    fn has_next(&self) -> bool {
        match *self {
            Reader::PrimitiveReader(_, ref column) => column.has_next(),
            Reader::OptionReader(_, ref reader) => reader.has_next(),
            Reader::GroupReader(_, _, ref readers) => readers.first().unwrap().has_next(),
            Reader::RepeatedReader(_, _, _, ref reader) => reader.has_next(),
            Reader::KeyValueReader(_, _, _, ref keys, _) => keys.has_next(),
        }
    }

    /// Returns current definition level,
    /// Method does not advance internal iterator.
    fn current_def_level(&self) -> i16 {
        match *self {
            Reader::PrimitiveReader(_, ref column) => column.current_def_level(),
            Reader::OptionReader(_, ref reader) => reader.current_def_level(),
            Reader::GroupReader(_, _, ref readers) => match readers.first() {
                Some(reader) => reader.current_def_level(),
                None => panic!("Current definition level: empty group reader"),
            },
            Reader::RepeatedReader(_, _, _, ref reader) => reader.current_def_level(),
            Reader::KeyValueReader(_, _, _, ref keys, _) => keys.current_def_level(),
        }
    }

    /// Returns current repetition level.
    /// Method does not advance internal iterator.
    fn current_rep_level(&self) -> i16 {
        match *self {
            Reader::PrimitiveReader(_, ref column) => column.current_rep_level(),
            Reader::OptionReader(_, ref reader) => reader.current_rep_level(),
            Reader::GroupReader(_, _, ref readers) => match readers.first() {
                Some(reader) => reader.current_rep_level(),
                None => panic!("Current repetition level: empty group reader"),
            },
            Reader::RepeatedReader(_, _, _, ref reader) => reader.current_rep_level(),
            Reader::KeyValueReader(_, _, _, ref keys, _) => keys.current_rep_level(),
        }
    }

    /// Advances leaf columns for the current reader.
    fn advance_columns(&mut self) {
        match *self {
            Reader::PrimitiveReader(_, ref mut column) => {
                column.read_next().unwrap();
            }
            Reader::OptionReader(_, ref mut reader) => {
                reader.advance_columns();
            }
            Reader::GroupReader(_, _, ref mut readers) => {
                for reader in readers {
                    reader.advance_columns();
                }
            }
            Reader::RepeatedReader(_, _, _, ref mut reader) => {
                reader.advance_columns();
            }
            Reader::KeyValueReader(_, _, _, ref mut keys, ref mut values) => {
                keys.advance_columns();
                values.advance_columns();
            }
        }
    }
}

impl fmt::Display for Reader {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let s = match self {
            Reader::PrimitiveReader(..) => "PrimitiveReader",
            Reader::OptionReader(..) => "OptionReader",
            Reader::GroupReader(..) => "GroupReader",
            Reader::RepeatedReader(..) => "RepeatedReader",
            Reader::KeyValueReader(..) => "KeyValueReader",
        };
        write!(f, "{}", s)
    }
}

// ----------------------------------------------------------------------
// Row iterators

/// The enum Either with variants That represet a reference and a box of
/// [`FileReader`](crate::file::reader::FileReader).
enum Either<'a> {
    Left(&'a FileReader),
    Right(Box<FileReader>),
}

impl<'a> Either<'a> {
    fn reader(&self) -> &FileReader {
        match *self {
            Either::Left(r) => r,
            Either::Right(ref r) => &**r,
        }
    }
}

/// Iterator of [`Row`](crate::record::api::Row)s.
/// It is used either for a single row group to iterate over data in that row group, or
/// an entire file with auto buffering of all row groups.
pub struct RowIter<'a> {
    descr: SchemaDescPtr,
    tree_builder: TreeBuilder,
    file_reader: Option<Either<'a>>,
    current_row_group: usize,
    num_row_groups: usize,
    row_iter: Option<ReaderIter>,
}

impl<'a> RowIter<'a> {
    /// Creates a new iterator of [`Row`](crate::record::api::Row)s.
    fn new(
        file_reader: Option<Either<'a>>,
        row_iter: Option<ReaderIter>,
        descr: SchemaDescPtr,
    ) -> Self {
        let tree_builder = Self::tree_builder();
        let num_row_groups = match file_reader {
            Some(ref r) => r.reader().num_row_groups(),
            None => 0,
        };

        Self {
            descr,
            file_reader,
            tree_builder,
            num_row_groups,
            row_iter: row_iter,
            current_row_group: 0,
        }
    }

    /// Creates iterator of [`Row`](crate::record::api::Row)s for all row groups in a
    /// file.
    pub fn from_file(proj: Option<Type>, reader: &'a FileReader) -> Result<Self> {
        let either = Either::Left(reader);
        let descr = Self::get_proj_descr(
            proj,
            reader.metadata().file_metadata().schema_descr_ptr(),
        )?;

        Ok(Self::new(Some(either), None, descr))
    }

    /// Creates iterator of [`Row`](crate::record::api::Row)s for a specific row group.
    pub fn from_row_group(
        proj: Option<Type>,
        reader: &'a RowGroupReader,
    ) -> Result<Self> {
        let descr = Self::get_proj_descr(proj, reader.metadata().schema_descr_ptr())?;
        let tree_builder = Self::tree_builder();
        let row_iter = tree_builder.as_iter(descr.clone(), reader);

        // For row group we need to set `current_row_group` >= `num_row_groups`, because
        // we only have one row group and can't buffer more.
        Ok(Self::new(None, Some(row_iter), descr))
    }

    /// Creates a iterator of [`Row`](crate::record::api::Row)s from a
    /// [`FileReader`](crate::file::reader::FileReader) using the full file schema.
    pub fn from_file_into(reader: Box<FileReader>) -> Self {
        let either = Either::Right(reader);
        let descr = either
            .reader()
            .metadata()
            .file_metadata()
            .schema_descr_ptr();

        Self::new(Some(either), None, descr)
    }

    /// Tries to create a iterator of [`Row`](crate::record::api::Row)s using projections.
    /// Returns a error if a file reader is not the source of this iterator.
    ///
    /// The Projected schema can be a subset of or equal to the file schema,
    /// when it is None, full file schema is assumed.
    pub fn project(self, proj: Option<Type>) -> Result<Self> {
        match self.file_reader {
            Some(ref either) => {
                let schema = either
                    .reader()
                    .metadata()
                    .file_metadata()
                    .schema_descr_ptr();
                let descr = Self::get_proj_descr(proj, schema)?;

                Ok(Self::new(self.file_reader, None, descr))
            }
            None => Err(general_err!("File reader is required to use projections")),
        }
    }

    /// Helper method to get schema descriptor for projected schema.
    /// If projection is None, then full schema is returned.
    #[inline]
    fn get_proj_descr(
        proj: Option<Type>,
        root_descr: SchemaDescPtr,
    ) -> Result<SchemaDescPtr> {
        match proj {
            Some(projection) => {
                // check if projection is part of file schema
                let root_schema = root_descr.root_schema();
                if !root_schema.check_contains(&projection) {
                    return Err(general_err!("Root schema does not contain projection"));
                }
                Ok(Rc::new(SchemaDescriptor::new(Rc::new(projection))))
            }
            None => Ok(root_descr),
        }
    }

    /// Returns common tree builder, so the same settings are applied to both iterators
    /// from file reader and row group.
    #[inline]
    fn tree_builder() -> TreeBuilder {
        TreeBuilder::new()
    }
}

impl<'a> Iterator for RowIter<'a> {
    type Item = Row;

    fn next(&mut self) -> Option<Row> {
        let mut row = None;
        if let Some(ref mut iter) = self.row_iter {
            row = iter.next();
        }

        while row.is_none() && self.current_row_group < self.num_row_groups {
            // We do not expect any failures when accessing a row group, and file reader
            // must be set for selecting next row group.
            if let Some(ref either) = self.file_reader {
                let file_reader = either.reader();
                let row_group_reader = &*file_reader
                    .get_row_group(self.current_row_group)
                    .expect("Row group is required to advance");

                let mut iter = self
                    .tree_builder
                    .as_iter(self.descr.clone(), row_group_reader);

                row = iter.next();

                self.current_row_group += 1;
                self.row_iter = Some(iter);
            }
        }

        row
    }
}

/// Internal iterator of [`Row`](crate::record::api::Row)s for a reader.
pub struct ReaderIter {
    root_reader: Reader,
    records_left: usize,
}

impl ReaderIter {
    fn new(mut root_reader: Reader, num_records: usize) -> Self {
        // Prepare root reader by advancing all column vectors
        root_reader.advance_columns();
        Self {
            root_reader,
            records_left: num_records,
        }
    }
}

impl Iterator for ReaderIter {
    type Item = Row;

    fn next(&mut self) -> Option<Row> {
        if self.records_left > 0 {
            self.records_left -= 1;
            Some(self.root_reader.read())
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::errors::{ParquetError, Result};
    use crate::file::reader::{FileReader, SerializedFileReader};
    use crate::record::api::{Field, Row, RowAccessor, RowFormatter};
    use crate::schema::parser::parse_message_type;
    use crate::util::test_common::{get_test_file, get_test_path};
    use std::convert::TryFrom;

    // Convenient macros to assemble row, list, map, and group.

    macro_rules! row {
        () => {
            {
                let result = Vec::new();
                make_row(result)
            }
        };
        ( $( $e:expr ), + ) => {
            {
                let mut result = Vec::new();
                $(
                    result.push($e);
                )*
                    make_row(result)
            }
        }
    }

    macro_rules! list {
        () => {
            {
                let result = Vec::new();
                Field::ListInternal(make_list(result))
            }
        };
        ( $( $e:expr ), + ) => {
            {
                let mut result = Vec::new();
                $(
                    result.push($e);
                )*
                    Field::ListInternal(make_list(result))
            }
        }
    }

    macro_rules! map {
        () => {
            {
                let result = Vec::new();
                Field::MapInternal(make_map(result))
            }
        };
        ( $( $e:expr ), + ) => {
            {
                let mut result = Vec::new();
                $(
                    result.push($e);
                )*
                    Field::MapInternal(make_map(result))
            }
        }
    }

    macro_rules! group {
        ( $( $e:expr ), * ) => {
            {
                Field::Group(row!($( $e ), *))
            }
        }
    }

    #[test]
    fn test_file_reader_rows_nulls() {
        let rows = test_file_reader_rows("nulls.snappy.parquet", None).unwrap();
        let expected_rows = vec![
            row![(
                "b_struct".to_string(),
                group![("b_c_int".to_string(), Field::Null)]
            )],
            row![(
                "b_struct".to_string(),
                group![("b_c_int".to_string(), Field::Null)]
            )],
            row![(
                "b_struct".to_string(),
                group![("b_c_int".to_string(), Field::Null)]
            )],
            row![(
                "b_struct".to_string(),
                group![("b_c_int".to_string(), Field::Null)]
            )],
            row![(
                "b_struct".to_string(),
                group![("b_c_int".to_string(), Field::Null)]
            )],
            row![(
                "b_struct".to_string(),
                group![("b_c_int".to_string(), Field::Null)]
            )],
            row![(
                "b_struct".to_string(),
                group![("b_c_int".to_string(), Field::Null)]
            )],
            row![(
                "b_struct".to_string(),
                group![("b_c_int".to_string(), Field::Null)]
            )],
        ];
        assert_eq!(rows, expected_rows);
    }

    #[test]
    fn test_file_reader_rows_nonnullable() {
        let rows = test_file_reader_rows("nonnullable.impala.parquet", None).unwrap();
        let expected_rows = vec![row![
            ("ID".to_string(), Field::Long(8)),
            ("Int_Array".to_string(), list![Field::Int(-1)]),
            (
                "int_array_array".to_string(),
                list![list![Field::Int(-1), Field::Int(-2)], list![]]
            ),
            (
                "Int_Map".to_string(),
                map![(Field::Str("k1".to_string()), Field::Int(-1))]
            ),
            (
                "int_map_array".to_string(),
                list![
                    map![],
                    map![(Field::Str("k1".to_string()), Field::Int(1))],
                    map![],
                    map![]
                ]
            ),
            (
                "nested_Struct".to_string(),
                group![
                    ("a".to_string(), Field::Int(-1)),
                    ("B".to_string(), list![Field::Int(-1)]),
                    (
                        "c".to_string(),
                        group![(
                            "D".to_string(),
                            list![list![group![
                                ("e".to_string(), Field::Int(-1)),
                                ("f".to_string(), Field::Str("nonnullable".to_string()))
                            ]]]
                        )]
                    ),
                    ("G".to_string(), map![])
                ]
            )
        ]];
        assert_eq!(rows, expected_rows);
    }

    #[test]
    fn test_file_reader_rows_nullable() {
        let rows = test_file_reader_rows("nullable.impala.parquet", None).unwrap();
        let expected_rows = vec![
            row![
                ("id".to_string(), Field::Long(1)),
                (
                    "int_array".to_string(),
                    list![Field::Int(1), Field::Int(2), Field::Int(3)]
                ),
                (
                    "int_array_Array".to_string(),
                    list![
                        list![Field::Int(1), Field::Int(2)],
                        list![Field::Int(3), Field::Int(4)]
                    ]
                ),
                (
                    "int_map".to_string(),
                    map![
                        (Field::Str("k1".to_string()), Field::Int(1)),
                        (Field::Str("k2".to_string()), Field::Int(100))
                    ]
                ),
                (
                    "int_Map_Array".to_string(),
                    list![map![(Field::Str("k1".to_string()), Field::Int(1))]]
                ),
                (
                    "nested_struct".to_string(),
                    group![
                        ("A".to_string(), Field::Int(1)),
                        ("b".to_string(), list![Field::Int(1)]),
                        (
                            "C".to_string(),
                            group![(
                                "d".to_string(),
                                list![
                                    list![
                                        group![
                                            ("E".to_string(), Field::Int(10)),
                                            (
                                                "F".to_string(),
                                                Field::Str("aaa".to_string())
                                            )
                                        ],
                                        group![
                                            ("E".to_string(), Field::Int(-10)),
                                            (
                                                "F".to_string(),
                                                Field::Str("bbb".to_string())
                                            )
                                        ]
                                    ],
                                    list![group![
                                        ("E".to_string(), Field::Int(11)),
                                        ("F".to_string(), Field::Str("c".to_string()))
                                    ]]
                                ]
                            )]
                        ),
                        (
                            "g".to_string(),
                            map![(
                                Field::Str("foo".to_string()),
                                group![(
                                    "H".to_string(),
                                    group![("i".to_string(), list![Field::Double(1.1)])]
                                )]
                            )]
                        )
                    ]
                )
            ],
            row![
                ("id".to_string(), Field::Long(2)),
                (
                    "int_array".to_string(),
                    list![
                        Field::Null,
                        Field::Int(1),
                        Field::Int(2),
                        Field::Null,
                        Field::Int(3),
                        Field::Null
                    ]
                ),
                (
                    "int_array_Array".to_string(),
                    list![
                        list![Field::Null, Field::Int(1), Field::Int(2), Field::Null],
                        list![Field::Int(3), Field::Null, Field::Int(4)],
                        list![],
                        Field::Null
                    ]
                ),
                (
                    "int_map".to_string(),
                    map![
                        (Field::Str("k1".to_string()), Field::Int(2)),
                        (Field::Str("k2".to_string()), Field::Null)
                    ]
                ),
                (
                    "int_Map_Array".to_string(),
                    list![
                        map![
                            (Field::Str("k3".to_string()), Field::Null),
                            (Field::Str("k1".to_string()), Field::Int(1))
                        ],
                        Field::Null,
                        map![]
                    ]
                ),
                (
                    "nested_struct".to_string(),
                    group![
                        ("A".to_string(), Field::Null),
                        ("b".to_string(), list![Field::Null]),
                        (
                            "C".to_string(),
                            group![(
                                "d".to_string(),
                                list![
                                    list![
                                        group![
                                            ("E".to_string(), Field::Null),
                                            ("F".to_string(), Field::Null)
                                        ],
                                        group![
                                            ("E".to_string(), Field::Int(10)),
                                            (
                                                "F".to_string(),
                                                Field::Str("aaa".to_string())
                                            )
                                        ],
                                        group![
                                            ("E".to_string(), Field::Null),
                                            ("F".to_string(), Field::Null)
                                        ],
                                        group![
                                            ("E".to_string(), Field::Int(-10)),
                                            (
                                                "F".to_string(),
                                                Field::Str("bbb".to_string())
                                            )
                                        ],
                                        group![
                                            ("E".to_string(), Field::Null),
                                            ("F".to_string(), Field::Null)
                                        ]
                                    ],
                                    list![
                                        group![
                                            ("E".to_string(), Field::Int(11)),
                                            (
                                                "F".to_string(),
                                                Field::Str("c".to_string())
                                            )
                                        ],
                                        Field::Null
                                    ],
                                    list![],
                                    Field::Null
                                ]
                            )]
                        ),
                        (
                            "g".to_string(),
                            map![
                                (
                                    Field::Str("g1".to_string()),
                                    group![(
                                        "H".to_string(),
                                        group![(
                                            "i".to_string(),
                                            list![Field::Double(2.2), Field::Null]
                                        )]
                                    )]
                                ),
                                (
                                    Field::Str("g2".to_string()),
                                    group![(
                                        "H".to_string(),
                                        group![("i".to_string(), list![])]
                                    )]
                                ),
                                (Field::Str("g3".to_string()), Field::Null),
                                (
                                    Field::Str("g4".to_string()),
                                    group![(
                                        "H".to_string(),
                                        group![("i".to_string(), Field::Null)]
                                    )]
                                ),
                                (
                                    Field::Str("g5".to_string()),
                                    group![("H".to_string(), Field::Null)]
                                )
                            ]
                        )
                    ]
                )
            ],
            row![
                ("id".to_string(), Field::Long(3)),
                ("int_array".to_string(), list![]),
                ("int_array_Array".to_string(), list![Field::Null]),
                ("int_map".to_string(), map![]),
                ("int_Map_Array".to_string(), list![Field::Null, Field::Null]),
                (
                    "nested_struct".to_string(),
                    group![
                        ("A".to_string(), Field::Null),
                        ("b".to_string(), Field::Null),
                        ("C".to_string(), group![("d".to_string(), list![])]),
                        ("g".to_string(), map![])
                    ]
                )
            ],
            row![
                ("id".to_string(), Field::Long(4)),
                ("int_array".to_string(), Field::Null),
                ("int_array_Array".to_string(), list![]),
                ("int_map".to_string(), map![]),
                ("int_Map_Array".to_string(), list![]),
                (
                    "nested_struct".to_string(),
                    group![
                        ("A".to_string(), Field::Null),
                        ("b".to_string(), Field::Null),
                        ("C".to_string(), group![("d".to_string(), Field::Null)]),
                        ("g".to_string(), Field::Null)
                    ]
                )
            ],
            row![
                ("id".to_string(), Field::Long(5)),
                ("int_array".to_string(), Field::Null),
                ("int_array_Array".to_string(), Field::Null),
                ("int_map".to_string(), map![]),
                ("int_Map_Array".to_string(), Field::Null),
                (
                    "nested_struct".to_string(),
                    group![
                        ("A".to_string(), Field::Null),
                        ("b".to_string(), Field::Null),
                        ("C".to_string(), Field::Null),
                        (
                            "g".to_string(),
                            map![(
                                Field::Str("foo".to_string()),
                                group![(
                                    "H".to_string(),
                                    group![(
                                        "i".to_string(),
                                        list![Field::Double(2.2), Field::Double(3.3)]
                                    )]
                                )]
                            )]
                        )
                    ]
                )
            ],
            row![
                ("id".to_string(), Field::Long(6)),
                ("int_array".to_string(), Field::Null),
                ("int_array_Array".to_string(), Field::Null),
                ("int_map".to_string(), Field::Null),
                ("int_Map_Array".to_string(), Field::Null),
                ("nested_struct".to_string(), Field::Null)
            ],
            row![
                ("id".to_string(), Field::Long(7)),
                ("int_array".to_string(), Field::Null),
                (
                    "int_array_Array".to_string(),
                    list![Field::Null, list![Field::Int(5), Field::Int(6)]]
                ),
                (
                    "int_map".to_string(),
                    map![
                        (Field::Str("k1".to_string()), Field::Null),
                        (Field::Str("k3".to_string()), Field::Null)
                    ]
                ),
                ("int_Map_Array".to_string(), Field::Null),
                (
                    "nested_struct".to_string(),
                    group![
                        ("A".to_string(), Field::Int(7)),
                        (
                            "b".to_string(),
                            list![Field::Int(2), Field::Int(3), Field::Null]
                        ),
                        (
                            "C".to_string(),
                            group![(
                                "d".to_string(),
                                list![list![], list![Field::Null], Field::Null]
                            )]
                        ),
                        ("g".to_string(), Field::Null)
                    ]
                )
            ],
        ];
        assert_eq!(rows, expected_rows);
    }

    #[test]
    fn test_file_reader_rows_projection() {
        let schema = "
      message spark_schema {
        REQUIRED DOUBLE c;
        REQUIRED INT32 b;
      }
    ";
        let schema = parse_message_type(&schema).unwrap();
        let rows =
            test_file_reader_rows("nested_maps.snappy.parquet", Some(schema)).unwrap();
        let expected_rows = vec![
            row![
                ("c".to_string(), Field::Double(1.0)),
                ("b".to_string(), Field::Int(1))
            ],
            row![
                ("c".to_string(), Field::Double(1.0)),
                ("b".to_string(), Field::Int(1))
            ],
            row![
                ("c".to_string(), Field::Double(1.0)),
                ("b".to_string(), Field::Int(1))
            ],
            row![
                ("c".to_string(), Field::Double(1.0)),
                ("b".to_string(), Field::Int(1))
            ],
            row![
                ("c".to_string(), Field::Double(1.0)),
                ("b".to_string(), Field::Int(1))
            ],
            row![
                ("c".to_string(), Field::Double(1.0)),
                ("b".to_string(), Field::Int(1))
            ],
        ];
        assert_eq!(rows, expected_rows);
    }

    #[test]
    fn test_iter_columns_in_row() {
        let r = row![
            ("c".to_string(), Field::Double(1.0)),
            ("b".to_string(), Field::Int(1))
        ];
        let mut result = Vec::new();
        for (name, record) in r.get_column_iter() {
            result.push((name, record));
        }
        assert_eq!(
            vec![
                (&"c".to_string(), &Field::Double(1.0)),
                (&"b".to_string(), &Field::Int(1))
            ],
            result
        );
    }

    #[test]
    fn test_file_reader_rows_projection_map() {
        let schema = "
      message spark_schema {
        OPTIONAL group a (MAP) {
          REPEATED group key_value {
            REQUIRED BYTE_ARRAY key (UTF8);
            OPTIONAL group value (MAP) {
              REPEATED group key_value {
                REQUIRED INT32 key;
                REQUIRED BOOLEAN value;
              }
            }
          }
        }
      }
    ";
        let schema = parse_message_type(&schema).unwrap();
        let rows =
            test_file_reader_rows("nested_maps.snappy.parquet", Some(schema)).unwrap();
        let expected_rows = vec![
            row![(
                "a".to_string(),
                map![(
                    Field::Str("a".to_string()),
                    map![
                        (Field::Int(1), Field::Bool(true)),
                        (Field::Int(2), Field::Bool(false))
                    ]
                )]
            )],
            row![(
                "a".to_string(),
                map![(
                    Field::Str("b".to_string()),
                    map![(Field::Int(1), Field::Bool(true))]
                )]
            )],
            row![(
                "a".to_string(),
                map![(Field::Str("c".to_string()), Field::Null)]
            )],
            row![("a".to_string(), map![(Field::Str("d".to_string()), map![])])],
            row![(
                "a".to_string(),
                map![(
                    Field::Str("e".to_string()),
                    map![(Field::Int(1), Field::Bool(true))]
                )]
            )],
            row![(
                "a".to_string(),
                map![(
                    Field::Str("f".to_string()),
                    map![
                        (Field::Int(3), Field::Bool(true)),
                        (Field::Int(4), Field::Bool(false)),
                        (Field::Int(5), Field::Bool(true))
                    ]
                )]
            )],
        ];
        assert_eq!(rows, expected_rows);
    }

    #[test]
    fn test_file_reader_rows_projection_list() {
        let schema = "
      message spark_schema {
        OPTIONAL group a (LIST) {
          REPEATED group list {
            OPTIONAL group element (LIST) {
              REPEATED group list {
                OPTIONAL group element (LIST) {
                  REPEATED group list {
                    OPTIONAL BYTE_ARRAY element (UTF8);
                  }
                }
              }
            }
          }
        }
      }
    ";
        let schema = parse_message_type(&schema).unwrap();
        let rows =
            test_file_reader_rows("nested_lists.snappy.parquet", Some(schema)).unwrap();
        let expected_rows = vec![
            row![(
                "a".to_string(),
                list![
                    list![
                        list![Field::Str("a".to_string()), Field::Str("b".to_string())],
                        list![Field::Str("c".to_string())]
                    ],
                    list![Field::Null, list![Field::Str("d".to_string())]]
                ]
            )],
            row![(
                "a".to_string(),
                list![
                    list![
                        list![Field::Str("a".to_string()), Field::Str("b".to_string())],
                        list![Field::Str("c".to_string()), Field::Str("d".to_string())]
                    ],
                    list![Field::Null, list![Field::Str("e".to_string())]]
                ]
            )],
            row![(
                "a".to_string(),
                list![
                    list![
                        list![Field::Str("a".to_string()), Field::Str("b".to_string())],
                        list![Field::Str("c".to_string()), Field::Str("d".to_string())],
                        list![Field::Str("e".to_string())]
                    ],
                    list![Field::Null, list![Field::Str("f".to_string())]]
                ]
            )],
        ];
        assert_eq!(rows, expected_rows);
    }

    #[test]
    fn test_file_reader_rows_invalid_projection() {
        let schema = "
      message spark_schema {
        REQUIRED INT32 key;
        REQUIRED BOOLEAN value;
      }
    ";
        let schema = parse_message_type(&schema).unwrap();
        let res = test_file_reader_rows("nested_maps.snappy.parquet", Some(schema));
        assert!(res.is_err());
        assert_eq!(
            res.unwrap_err(),
            general_err!("Root schema does not contain projection")
        );
    }

    #[test]
    fn test_row_group_rows_invalid_projection() {
        let schema = "
      message spark_schema {
        REQUIRED INT32 key;
        REQUIRED BOOLEAN value;
      }
    ";
        let schema = parse_message_type(&schema).unwrap();
        let res = test_row_group_rows("nested_maps.snappy.parquet", Some(schema));
        assert!(res.is_err());
        assert_eq!(
            res.unwrap_err(),
            general_err!("Root schema does not contain projection")
        );
    }

    #[test]
    #[should_panic(expected = "Invalid map type")]
    fn test_file_reader_rows_invalid_map_type() {
        let schema = "
      message spark_schema {
        OPTIONAL group a (MAP) {
          REPEATED group key_value {
            REQUIRED BYTE_ARRAY key (UTF8);
            OPTIONAL group value (MAP) {
              REPEATED group key_value {
                REQUIRED INT32 key;
              }
            }
          }
        }
      }
    ";
        let schema = parse_message_type(&schema).unwrap();
        test_file_reader_rows("nested_maps.snappy.parquet", Some(schema)).unwrap();
    }

    #[test]
    fn test_file_reader_iter() -> Result<()> {
        let path = get_test_path("alltypes_plain.parquet");
        let vec = vec![path]
            .iter()
            .map(|p| SerializedFileReader::try_from(p.as_path()).unwrap())
            .flat_map(|r| RowIter::from_file_into(Box::new(r)))
            .flat_map(|r| r.get_int(0))
            .collect::<Vec<_>>();

        assert_eq!(vec, vec![4, 5, 6, 7, 2, 3, 0, 1]);

        Ok(())
    }

    #[test]
    fn test_file_reader_iter_projection() -> Result<()> {
        let path = get_test_path("alltypes_plain.parquet");
        let values = vec![path]
            .iter()
            .map(|p| SerializedFileReader::try_from(p.as_path()).unwrap())
            .flat_map(|r| {
                let schema = "message schema { OPTIONAL INT32 id; }";
                let proj = parse_message_type(&schema).ok();

                RowIter::from_file_into(Box::new(r)).project(proj).unwrap()
            })
            .map(|r| format!("id:{}", r.fmt(0)))
            .collect::<Vec<_>>()
            .join(", ");

        assert_eq!(values, "id:4, id:5, id:6, id:7, id:2, id:3, id:0, id:1");

        Ok(())
    }

    #[test]
    fn test_file_reader_iter_projection_err() {
        let schema = "
      message spark_schema {
        REQUIRED INT32 key;
        REQUIRED BOOLEAN value;
      }
    ";
        let proj = parse_message_type(&schema).ok();
        let path = get_test_path("nested_maps.snappy.parquet");
        let reader = SerializedFileReader::try_from(path.as_path()).unwrap();
        let res = RowIter::from_file_into(Box::new(reader)).project(proj);

        assert!(res.is_err());
        assert_eq!(
            res.err().unwrap(),
            general_err!("Root schema does not contain projection")
        );
    }

    #[test]
    fn test_tree_reader_handle_repeated_fields_with_no_annotation() {
        // Array field `phoneNumbers` does not contain LIST annotation.
        // We parse it as struct with `phone` repeated field as array.
        let rows = test_file_reader_rows("repeated_no_annotation.parquet", None).unwrap();
        let expected_rows = vec![
            row![
                ("id".to_string(), Field::Int(1)),
                ("phoneNumbers".to_string(), Field::Null)
            ],
            row![
                ("id".to_string(), Field::Int(2)),
                ("phoneNumbers".to_string(), Field::Null)
            ],
            row![
                ("id".to_string(), Field::Int(3)),
                (
                    "phoneNumbers".to_string(),
                    group![("phone".to_string(), list![])]
                )
            ],
            row![
                ("id".to_string(), Field::Int(4)),
                (
                    "phoneNumbers".to_string(),
                    group![(
                        "phone".to_string(),
                        list![group![
                            ("number".to_string(), Field::Long(5555555555)),
                            ("kind".to_string(), Field::Null)
                        ]]
                    )]
                )
            ],
            row![
                ("id".to_string(), Field::Int(5)),
                (
                    "phoneNumbers".to_string(),
                    group![(
                        "phone".to_string(),
                        list![group![
                            ("number".to_string(), Field::Long(1111111111)),
                            ("kind".to_string(), Field::Str("home".to_string()))
                        ]]
                    )]
                )
            ],
            row![
                ("id".to_string(), Field::Int(6)),
                (
                    "phoneNumbers".to_string(),
                    group![(
                        "phone".to_string(),
                        list![
                            group![
                                ("number".to_string(), Field::Long(1111111111)),
                                ("kind".to_string(), Field::Str("home".to_string()))
                            ],
                            group![
                                ("number".to_string(), Field::Long(2222222222)),
                                ("kind".to_string(), Field::Null)
                            ],
                            group![
                                ("number".to_string(), Field::Long(3333333333)),
                                ("kind".to_string(), Field::Str("mobile".to_string()))
                            ]
                        ]
                    )]
                )
            ],
        ];

        assert_eq!(rows, expected_rows);
    }

    fn test_file_reader_rows(file_name: &str, schema: Option<Type>) -> Result<Vec<Row>> {
        let file = get_test_file(file_name);
        let file_reader: Box<FileReader> = Box::new(SerializedFileReader::new(file)?);
        let iter = file_reader.get_row_iter(schema)?;
        Ok(iter.collect())
    }

    fn test_row_group_rows(file_name: &str, schema: Option<Type>) -> Result<Vec<Row>> {
        let file = get_test_file(file_name);
        let file_reader: Box<FileReader> = Box::new(SerializedFileReader::new(file)?);
        // Check the first row group only, because files will contain only single row
        // group
        let row_group_reader = file_reader.get_row_group(0).unwrap();
        let iter = row_group_reader.get_row_iter(schema)?;
        Ok(iter.collect())
    }
}
