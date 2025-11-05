//! Contains [`RecordBatchT`], a container of [`Array`] where every array has the
//! same length.

use polars_error::{PolarsResult, polars_ensure};

use crate::array::{Array, ArrayRef};
use crate::datatypes::{ArrowSchema, ArrowSchemaRef};

/// A vector of trait objects of [`Array`] where every item has
/// the same length, [`RecordBatchT::len`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordBatchT<A: AsRef<dyn Array>> {
    height: usize,
    schema: ArrowSchemaRef,
    arrays: Vec<A>,
}

pub type RecordBatch = RecordBatchT<ArrayRef>;

impl<A: AsRef<dyn Array>> RecordBatchT<A> {
    /// Creates a new [`RecordBatchT`].
    ///
    /// # Panics
    ///
    /// I.f.f. the length does not match the length of any of the arrays
    pub fn new(length: usize, schema: ArrowSchemaRef, arrays: Vec<A>) -> Self {
        Self::try_new(length, schema, arrays).unwrap()
    }

    /// Creates a new [`RecordBatchT`].
    ///
    /// # Error
    ///
    /// I.f.f. the height does not match the length of any of the arrays
    pub fn try_new(height: usize, schema: ArrowSchemaRef, arrays: Vec<A>) -> PolarsResult<Self> {
        polars_ensure!(
            schema.len() == arrays.len(),
            ComputeError: "RecordBatch requires an equal number of fields and arrays",
        );
        polars_ensure!(
            arrays.iter().all(|arr| arr.as_ref().len() == height),
            ComputeError: "RecordBatch requires all its arrays to have an equal number of rows",
        );

        Ok(Self {
            height,
            schema,
            arrays,
        })
    }

    /// returns the [`Array`]s in [`RecordBatchT`]
    pub fn arrays(&self) -> &[A] {
        &self.arrays
    }

    /// returns the [`ArrowSchema`]s in [`RecordBatchT`]
    pub fn schema(&self) -> &ArrowSchema {
        &self.schema
    }

    /// returns the [`Array`]s in [`RecordBatchT`]
    pub fn columns(&self) -> &[A] {
        &self.arrays
    }

    /// returns the number of rows of every array
    pub fn len(&self) -> usize {
        self.height
    }

    /// returns the number of rows of every array
    pub fn height(&self) -> usize {
        self.height
    }

    /// returns the number of arrays
    pub fn width(&self) -> usize {
        self.arrays.len()
    }

    /// returns whether the columns have any rows
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Consumes [`RecordBatchT`] into its underlying arrays.
    /// The arrays are guaranteed to have the same length
    pub fn into_arrays(self) -> Vec<A> {
        self.arrays
    }

    /// Consumes [`RecordBatchT`] into its underlying schema and arrays.
    /// The arrays are guaranteed to have the same length
    pub fn into_schema_and_arrays(self) -> (ArrowSchemaRef, Vec<A>) {
        (self.schema, self.arrays)
    }
}

impl<A: AsRef<dyn Array>> From<RecordBatchT<A>> for Vec<A> {
    fn from(c: RecordBatchT<A>) -> Self {
        c.into_arrays()
    }
}

impl<A: AsRef<dyn Array>> std::ops::Deref for RecordBatchT<A> {
    type Target = [A];

    #[inline]
    fn deref(&self) -> &[A] {
        self.arrays()
    }
}
