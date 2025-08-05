pub mod builder;
use std::sync::Arc;

use arrow::bitmap::Bitmap;
use polars_core::prelude::{DataType, PlHashMap};
use polars_core::scalar::Scalar;
use polars_core::schema::{Schema, SchemaRef};
use polars_error::PolarsResult;
use polars_utils::pl_str::PlSmallStr;

use crate::nodes::io_sources::multi_scan::components::column_selector::ColumnSelector;

/// Encapsulates projection logic, including column mapping / renaming / casting information.
/// Intended to be used for the projected file columns.
#[derive(Debug, Clone)]
pub enum Projection {
    /// Represents a projection that does not have column renaming. Note however that the dtypes may
    /// not match that in the file.
    /// Readers may directly take and use the inner `SchemaRef` of this enum variant.
    Plain(SchemaRef),
    /// Constructed from physical ID schemas (e.g. Iceberg).
    /// Note: Do not access the inner fields of this directly - only use the function interface
    /// of the enum.
    Mapped {
        /// Projected output schema.
        /// Key: output_name.
        projected_schema: SchemaRef,
        /// Contains entries for columns which require e.g. renaming / casting.
        /// Key: Index in `projected_schema`.
        /// Value: (source_name, source_dtype, transform).
        mapping: Option<Arc<PlHashMap<usize, ProjectionTransform>>>,
        missing_columns_mask: Option<Bitmap>,
        missing_column_defaults: Option<Arc<PlHashMap<usize, Scalar>>>,
    },
}

impl Projection {
    /// Returns the full projected schema, keyed by the output name.
    pub fn projected_schema(&self) -> &SchemaRef {
        match self {
            Projection::Plain(schema) => schema,
            Projection::Mapped {
                projected_schema, ..
            } => projected_schema,
        }
    }

    /// Constructs a `Plain` projection that can be applied before this projection. This can be
    /// sent to readers that cannot handle mapped projections.
    ///
    /// If this projection is a `Mapped` variant, the returned projection will contain the set of
    /// source names before renaming.
    pub fn get_plain_pre_projection(&self) -> Projection {
        match self {
            Projection::Plain(projected_schema)
            | Projection::Mapped {
                projected_schema,
                mapping: None,
                missing_columns_mask: None,
                missing_column_defaults: None,
            } => Projection::Plain(projected_schema.clone()),

            Projection::Mapped {
                projected_schema,
                mapping,
                missing_columns_mask,
                missing_column_defaults: _,
            } => {
                let mut pre_projection = Schema::with_capacity(projected_schema.len());

                for (index, (output_name, output_dtype)) in projected_schema.iter().enumerate() {
                    // Important: If this column was determined as missing it must be excluded
                    // from the pre-projected schema. The file may contain a column with this
                    // name, but with a different physical ID. We do not to accidentally load
                    // such a column.
                    if missing_columns_mask
                        .as_ref()
                        .is_some_and(|x| x.get_bit(index))
                    {
                        continue;
                    }

                    let (source_name, source_dtype) =
                        mapping.as_ref().and_then(|x| x.get(&index)).map_or(
                            (output_name, output_dtype),
                            |ProjectionTransform {
                                 source_name,
                                 source_dtype,
                                 transform: _,
                             }| (source_name, source_dtype),
                        );

                    if let Some(existing) = pre_projection.get(source_name.as_str()) {
                        // If this fails it means that the same physical ID maps to multiple output
                        // columns that have different types.
                        assert_eq!(existing, source_dtype);
                    }

                    pre_projection.insert(source_name.clone(), source_dtype.clone());
                }

                Projection::Plain(Arc::new(pre_projection))
            },
        }
    }

    /// Returns `None` if `index >= self.len()`, or if `self` is a `Mapped` variant and the column
    /// was identified as missing.
    pub fn get_mapped_projection_ref_by_index(
        &self,
        index: usize,
    ) -> Option<MappedProjectionRef<'_>> {
        Some(match self {
            Projection::Plain(projected_schema) => {
                let (output_name, output_dtype) = projected_schema.get_at_index(index)?;
                MappedProjectionRef {
                    source_name: output_name,
                    output_name,
                    output_dtype,
                    resolved_transform: None,
                }
            },

            Projection::Mapped {
                projected_schema,
                mapping,
                missing_columns_mask,
                missing_column_defaults: _,
            } => {
                let (output_name, output_dtype) = projected_schema.get_at_index(index)?;

                if missing_columns_mask
                    .as_ref()
                    .is_some_and(|x| x.get_bit(index))
                {
                    // Column is missing from file
                    return None;
                }

                let (source_name, resolved_transform) =
                    mapping.as_ref().and_then(|x| x.get(&index)).map_or(
                        (output_name, None),
                        |ProjectionTransform {
                             source_name,
                             source_dtype,
                             transform,
                         }| {
                            (
                                source_name,
                                Some(ResolvedTransformRef {
                                    source_dtype,
                                    transform,
                                }),
                            )
                        },
                    );

                MappedProjectionRef {
                    source_name,
                    output_name,
                    output_dtype,
                    resolved_transform,
                }
            },
        })
    }

    /// Returns `None` if `output_name` is not found.
    pub fn get_mapped_projection_ref_by_output_name(
        &self,
        output_name: &str,
    ) -> Option<MappedProjectionRef<'_>> {
        let index = self.projected_schema().index_of(output_name)?;
        self.get_mapped_projection_ref_by_index(index)
    }

    pub fn iter_non_missing_columns(&self) -> impl Iterator<Item = MappedProjectionRef<'_>> {
        (0..self.projected_schema().len())
            .filter_map(|i| self.get_mapped_projection_ref_by_index(i))
    }

    /// # Returns
    /// Returns the number of missing columns if `self` is a `Mapped` variant. Otherwise returns `None`.
    pub fn num_missing_columns(&self) -> Option<usize> {
        match self {
            Projection::Plain(_) => None,
            Projection::Mapped {
                missing_columns_mask,
                ..
            } => Some(missing_columns_mask.as_ref().map_or(0, |x| x.set_bits())),
        }
    }

    /// Removes all column mapping information from this projection.
    pub fn clear_projection_transforms(&mut self) {
        match self {
            Projection::Plain(_) => {},

            Projection::Mapped {
                projected_schema: _,
                mapping,
                missing_columns_mask: _,
                missing_column_defaults: _,
            } => {
                *mapping = None;
            },
        }
    }

    pub fn has_projection_transforms(&self) -> bool {
        match self {
            Projection::Plain(_) => false,

            Projection::Mapped {
                projected_schema: _,
                mapping,
                missing_columns_mask: _,
                missing_column_defaults: _,
            } => mapping.is_some(),
        }
    }

    /// Retrieve a configured default value for when the column is missing from the file.
    ///
    /// # Panics
    /// Panics if `output_name` is not in the projected schema.
    pub fn get_default_value_by_output_name(&self, output_name: &str) -> Option<&Scalar> {
        match self {
            Projection::Plain(_) => None,

            Projection::Mapped {
                projected_schema,
                missing_column_defaults,
                ..
            } => {
                let missing_column_defaults = missing_column_defaults.as_ref()?;
                missing_column_defaults.get(&projected_schema.index_of(output_name).unwrap())
            },
        }
    }

    /// Returns an iterator of names in the projected schema that are missing from the file.
    ///
    /// # Panics
    /// Panics if `self` is a `Plain` variant and `reader_schema` is `None`.
    ///
    /// # Returns
    /// `(output_name, output_dtype, default_value)`
    pub fn iter_missing_columns(
        &self,
        reader_schema: Option<&Schema>,
    ) -> PolarsResult<impl Iterator<Item = (&PlSmallStr, &DataType, Option<&Scalar>)>> {
        let iter_len = match self {
            Projection::Plain(projected_schema) => projected_schema.len(),

            Projection::Mapped {
                projected_schema,
                mapping: _,
                missing_columns_mask,
                missing_column_defaults: _,
            } => {
                // For the `Mapped` case, the missing columns will have been resolved to a bitmap
                // by the `ProjectionBuilder`.
                missing_columns_mask.as_ref().map_or(0, |x| {
                    assert_eq!(x.len(), projected_schema.len());
                    x.len()
                })
            },
        };

        Ok((0..iter_len).filter_map(move |i| match self {
            Projection::Plain(projected_schema) => {
                let (projected_name, dtype) = projected_schema.get_at_index(i).unwrap();

                (!reader_schema.unwrap().contains(projected_name)).then_some((
                    projected_name,
                    dtype,
                    None,
                ))
            },

            Projection::Mapped {
                projected_schema,
                mapping: _,
                missing_columns_mask,
                missing_column_defaults,
            } => missing_columns_mask.as_ref().unwrap().get_bit(i).then(|| {
                let (name, dtype) = projected_schema.get_at_index(i).unwrap();
                let opt_default = missing_column_defaults.as_ref().and_then(|x| x.get(&i));

                (name, dtype, opt_default)
            }),
        }))
    }
}

#[derive(Debug)]
pub struct ProjectionTransform {
    pub source_name: PlSmallStr,
    pub source_dtype: DataType,
    pub transform: ColumnSelector,
}

/// Represents a projected column whose source column may have a different name and/or type.
#[derive(Debug)]
pub struct MappedProjectionRef<'a> {
    pub source_name: &'a PlSmallStr,
    pub output_name: &'a PlSmallStr,
    pub output_dtype: &'a DataType,
    pub resolved_transform: Option<ResolvedTransformRef<'a>>,
}

#[derive(Debug, Clone)]
pub struct ResolvedTransformRef<'a> {
    /// The input dtype that `self.transform` was resolved with.
    pub source_dtype: &'a DataType,
    transform: &'a ColumnSelector,
}

impl ResolvedTransformRef<'_> {
    /// Attaches the resolved transforms to this `input_selector`.
    pub fn attach_transforms(&self, input_selector: ColumnSelector) -> ColumnSelector {
        let mut selector = self.transform.clone();
        selector.replace_input(input_selector);
        selector
    }
}
