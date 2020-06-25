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

//! A two-dimensional batch of column-oriented data with a defined
//! [schema](crate::datatypes::Schema).

use std::sync::Arc;

use crate::array::*;
use crate::datatypes::*;
use crate::error::{ArrowError, Result};

/// A two-dimensional batch of column-oriented data with a defined
/// [schema](crate::datatypes::Schema).
///
/// A `RecordBatch` is a two-dimensional dataset of a number of
/// contiguous arrays, each the same length.
/// A record batch has a schema which must match its arrays’
/// datatypes.
///
/// Record batches are a convenient unit of work for various
/// serialization and computation functions, possibly incremental.  
/// See also [CSV reader](crate::csv::Reader) and
/// [JSON reader](crate::json::Reader).
#[derive(Clone, Debug)]
pub struct RecordBatch {
    schema: Arc<Schema>,
    columns: Vec<Arc<Array>>,
}

impl RecordBatch {
    /// Creates a `RecordBatch` from a schema and columns.
    ///
    /// Expects the following:
    ///  * the vec of columns to not be empty
    ///  * the schema and column data types to have equal lengths
    ///    and match
    ///  * each array in columns to have the same length
    ///
    /// If the conditions are not met, an error is returned.
    ///
    /// # Example
    ///
    /// ```
    /// use std::sync::Arc;
    /// use some::array::Int32Array;
    /// use some::datatypes::{Schema, Field, DataType};
    /// use some::record_batch::RecordBatch;
    ///
    /// # fn main() -> some::error::Result<()> {
    /// let id_array = Int32Array::from(vec![1, 2, 3, 4, 5]);
    /// let schema = Schema::new(vec![
    ///     Field::new("id", DataType::Int32, false)
    /// ]);
    ///
    /// let batch = RecordBatch::try_new(
    ///     Arc::new(schema),
    ///     vec![Arc::new(id_array)]
    /// )?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn try_new(schema: Arc<Schema>, columns: Vec<ArrayRef>) -> Result<Self> {
        // check that there are some columns
        if columns.is_empty() {
            return Err(ArrowError::InvalidArgumentError(
                "at least one column must be defined to create a record batch"
                    .to_string(),
            ));
        }
        // check that number of fields in schema match column length
        if schema.fields().len() != columns.len() {
            return Err(ArrowError::InvalidArgumentError(format!(
                "number of columns({}) must match number of fields({}) in schema",
                columns.len(),
                schema.fields().len(),
            )));
        }
        // check that all columns have the same row count, and match the schema
        let len = columns[0].data().len();

        for (i, column) in columns.iter().enumerate() {
            if column.len() != len {
                return Err(ArrowError::InvalidArgumentError(
                    "all columns in a record batch must have the same length".to_string(),
                ));
            }
            if column.data_type() != schema.field(i).data_type() {
                return Err(ArrowError::InvalidArgumentError(format!(
                    "column types must match schema types, expected {:?} but found {:?} at column index {}",
                    schema.field(i).data_type(),
                    column.data_type(),
                    i)));
            }
        }
        Ok(RecordBatch { schema, columns })
    }

    /// Returns the [`Schema`](crate::datatypes::Schema) of the record batch.
    pub fn schema(&self) -> &Arc<Schema> {
        &self.schema
    }

    /// Returns the number of columns in the record batch.
    ///
    /// # Example
    ///
    /// ```
    /// use std::sync::Arc;
    /// use some::array::Int32Array;
    /// use some::datatypes::{Schema, Field, DataType};
    /// use some::record_batch::RecordBatch;
    ///
    /// # fn main() -> some::error::Result<()> {
    /// let id_array = Int32Array::from(vec![1, 2, 3, 4, 5]);
    /// let schema = Schema::new(vec![
    ///     Field::new("id", DataType::Int32, false)
    /// ]);
    ///
    /// let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(id_array)])?;
    ///
    /// assert_eq!(batch.num_columns(), 1);
    /// # Ok(())
    /// # }
    /// ```
    pub fn num_columns(&self) -> usize {
        self.columns.len()
    }

    /// Returns the number of rows in each column.
    ///
    /// # Panics
    ///
    /// Panics if the `RecordBatch` contains no columns.
    ///
    /// # Example
    ///
    /// ```
    /// use std::sync::Arc;
    /// use some::array::Int32Array;
    /// use some::datatypes::{Schema, Field, DataType};
    /// use some::record_batch::RecordBatch;
    ///
    /// # fn main() -> some::error::Result<()> {
    /// let id_array = Int32Array::from(vec![1, 2, 3, 4, 5]);
    /// let schema = Schema::new(vec![
    ///     Field::new("id", DataType::Int32, false)
    /// ]);
    ///
    /// let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(id_array)])?;
    ///
    /// assert_eq!(batch.num_rows(), 5);
    /// # Ok(())
    /// # }
    /// ```
    pub fn num_rows(&self) -> usize {
        self.columns[0].data().len()
    }

    /// Get a reference to a column's array by index.
    ///
    /// # Panics
    ///
    /// Panics if `index` is outside of `0..num_columns`.
    pub fn column(&self, index: usize) -> &ArrayRef {
        &self.columns[index]
    }

    /// Get a reference to all columns in the record batch.
    pub fn columns(&self) -> &[ArrayRef] {
        &self.columns[..]
    }
}

impl From<&StructArray> for RecordBatch {
    /// Create a record batch from struct array.
    ///
    /// This currently does not flatten and nested struct types
    fn from(struct_array: &StructArray) -> Self {
        if let DataType::Struct(fields) = struct_array.data_type() {
            let schema = Schema::new(fields.clone());
            let columns = struct_array.boxed_fields.clone();
            RecordBatch {
                schema: Arc::new(schema),
                columns,
            }
        } else {
            unreachable!("unable to get datatype as struct")
        }
    }
}

impl Into<StructArray> for RecordBatch {
    fn into(self) -> StructArray {
        self.schema
            .fields
            .iter()
            .zip(self.columns.iter())
            .map(|t| (t.0.clone(), t.1.clone()))
            .collect::<Vec<(Field, ArrayRef)>>()
            .into()
    }
}

/// Definition of record batch reader.
pub trait RecordBatchReader {
    /// Returns schemas of this record batch reader.
    /// Implementation of this trait should guarantee that all record batches returned
    /// by this reader should have same schema as returned from this method.
    fn schema(&mut self) -> SchemaRef;

    /// Returns next record batch.
    fn next_batch(&mut self) -> Result<Option<RecordBatch>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::buffer::*;

    #[test]
    fn create_record_batch() {
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Utf8, false),
        ]);

        let v = vec![1, 2, 3, 4, 5];
        let array_data = ArrayData::builder(DataType::Int32)
            .len(5)
            .add_buffer(Buffer::from(v.to_byte_slice()))
            .build();
        let a = Int32Array::from(array_data);

        let v = vec![b'a', b'b', b'c', b'd', b'e'];
        let offset_data = vec![0, 1, 2, 3, 4, 5, 6];
        let array_data = ArrayData::builder(DataType::Utf8)
            .len(5)
            .add_buffer(Buffer::from(offset_data.to_byte_slice()))
            .add_buffer(Buffer::from(v.to_byte_slice()))
            .build();
        let b = BinaryArray::from(array_data);

        let record_batch =
            RecordBatch::try_new(Arc::new(schema), vec![Arc::new(a), Arc::new(b)])
                .unwrap();

        assert_eq!(5, record_batch.num_rows());
        assert_eq!(2, record_batch.num_columns());
        assert_eq!(&DataType::Int32, record_batch.schema().field(0).data_type());
        assert_eq!(&DataType::Utf8, record_batch.schema().field(1).data_type());
        assert_eq!(5, record_batch.column(0).data().len());
        assert_eq!(5, record_batch.column(1).data().len());
    }

    #[test]
    fn create_record_batch_schema_mismatch() {
        let schema = Schema::new(vec![Field::new("a", DataType::Int32, false)]);

        let a = Int64Array::from(vec![1, 2, 3, 4, 5]);

        let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(a)]);
        assert!(!batch.is_ok());
    }

    #[test]
    fn create_record_batch_record_mismatch() {
        let schema = Schema::new(vec![Field::new("a", DataType::Int32, false)]);

        let a = Int32Array::from(vec![1, 2, 3, 4, 5]);
        let b = Int32Array::from(vec![1, 2, 3, 4, 5]);

        let batch =
            RecordBatch::try_new(Arc::new(schema), vec![Arc::new(a), Arc::new(b)]);
        assert!(!batch.is_ok());
    }

    #[test]
    fn create_record_batch_from_struct_array() {
        let boolean_data = ArrayData::builder(DataType::Boolean)
            .len(4)
            .add_buffer(Buffer::from([12_u8]))
            .build();
        let int_data = ArrayData::builder(DataType::Int32)
            .len(4)
            .add_buffer(Buffer::from([42, 28, 19, 31].to_byte_slice()))
            .build();
        let struct_array = StructArray::from(vec![
            (
                Field::new("b", DataType::Boolean, false),
                Arc::new(BooleanArray::from(vec![false, false, true, true]))
                    as Arc<Array>,
            ),
            (
                Field::new("c", DataType::Int32, false),
                Arc::new(Int32Array::from(vec![42, 28, 19, 31])),
            ),
        ]);

        let batch = RecordBatch::from(&struct_array);
        assert_eq!(2, batch.num_columns());
        assert_eq!(4, batch.num_rows());
        assert_eq!(
            struct_array.data_type(),
            &DataType::Struct(batch.schema().fields().to_vec())
        );
        assert_eq!(batch.column(0).data(), boolean_data);
        assert_eq!(batch.column(1).data(), int_data);
    }
}
