use std::sync::Arc;

use polars_error::{polars_bail, PolarsResult};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::{Field, Metadata};

/// An ordered sequence of [`Field`]s with associated [`Metadata`].
///
/// [`ArrowSchema`] is an abstraction used to read from, and write to, Arrow IPC format,
/// Apache Parquet, and Apache Avro. All these formats have a concept of a schema
/// with fields and metadata.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ArrowSchema {
    /// The fields composing this schema.
    pub fields: Vec<Field>,
    /// Optional metadata.
    pub metadata: Metadata,
}

pub type ArrowSchemaRef = Arc<ArrowSchema>;

impl ArrowSchema {
    /// Attaches a [`Metadata`] to [`ArrowSchema`]
    #[inline]
    pub fn with_metadata(self, metadata: Metadata) -> Self {
        Self {
            fields: self.fields,
            metadata,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    /// Returns a new [`ArrowSchema`] with a subset of all fields whose `predicate`
    /// evaluates to true.
    pub fn filter<F: Fn(usize, &Field) -> bool>(self, predicate: F) -> Self {
        let fields = self
            .fields
            .into_iter()
            .enumerate()
            .filter_map(|(index, f)| {
                if (predicate)(index, &f) {
                    Some(f)
                } else {
                    None
                }
            })
            .collect();

        ArrowSchema {
            fields,
            metadata: self.metadata,
        }
    }

    pub fn try_project(&self, indices: &[usize]) -> PolarsResult<Self> {
        let fields = indices.iter().map(|&i| {
            let Some(out) = self.fields.get(i) else {
                polars_bail!(
                    SchemaFieldNotFound: "projection index {} is out of bounds for schema of length {}",
                    i, self.fields.len()
                );
            };

            Ok(out.clone())
        }).collect::<PolarsResult<Vec<_>>>()?;

        Ok(ArrowSchema {
            fields,
            metadata: self.metadata.clone(),
        })
    }
}

impl From<Vec<Field>> for ArrowSchema {
    fn from(fields: Vec<Field>) -> Self {
        Self {
            fields,
            ..Default::default()
        }
    }
}
