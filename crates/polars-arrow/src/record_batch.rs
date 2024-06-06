//! Contains [`RecordBatchT`], a container of [`Array`] where every array has the
//! same length.

use polars_error::{polars_bail, PolarsResult};

use crate::array::{Array, ArrayRef};

/// A vector of trait objects of [`Array`] where every item has
/// the same length, [`RecordBatchT::len`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordBatchT<A: AsRef<dyn Array>> {
    arrays: Vec<A>,
}

pub type RecordBatch = RecordBatchT<ArrayRef>;

impl<A: AsRef<dyn Array>> RecordBatchT<A> {
    /// Creates a new [`RecordBatchT`].
    /// # Panic
    /// Iff the arrays do not have the same length
    pub fn new(arrays: Vec<A>) -> Self {
        Self::try_new(arrays).unwrap()
    }

    /// Creates a new [`RecordBatchT`].
    /// # Error
    /// Iff the arrays do not have the same length
    pub fn try_new(arrays: Vec<A>) -> PolarsResult<Self> {
        if !arrays.is_empty() {
            let len = arrays.first().unwrap().as_ref().len();
            if arrays
                .iter()
                .map(|array| array.as_ref())
                .any(|array| array.len() != len)
            {
                polars_bail!(ComputeError:
                    "RecordBatch requires all its arrays to have an equal number of rows".to_string(),
                );
            }
        }
        Ok(Self { arrays })
    }

    /// returns the [`Array`]s in [`RecordBatchT`]
    pub fn arrays(&self) -> &[A] {
        &self.arrays
    }

    /// returns the [`Array`]s in [`RecordBatchT`]
    pub fn columns(&self) -> &[A] {
        &self.arrays
    }

    /// returns the number of rows of every array
    pub fn len(&self) -> usize {
        self.arrays
            .first()
            .map(|x| x.as_ref().len())
            .unwrap_or_default()
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
