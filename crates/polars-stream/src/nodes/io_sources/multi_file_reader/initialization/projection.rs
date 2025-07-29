use std::sync::Arc;

use arrow::bitmap::MutableBitmap;
use polars_core::frame::DataFrame;
use polars_core::prelude::{InitHashMaps, PlHashMap};
use polars_core::schema::iceberg::{IcebergSchema, IcebergSchemaRef};
use polars_core::schema::{Schema, SchemaRef};
use polars_error::{PolarsResult, polars_err};
use polars_plan::dsl::{CastColumnsPolicy, ColumnMapping, MissingColumnsPolicy};
use polars_plan::plans::hive::HivePartitionsDf;
use polars_utils::pl_str::PlSmallStr;

use crate::nodes::io_sources::multi_file_reader::extra_ops::column_selector::{
    ColumnSelector, ColumnSelectorBuilder,
};
use crate::nodes::io_sources::multi_file_reader::reader_interface::Projection;
use crate::nodes::io_sources::multi_file_reader::reader_interface::projection::ProjectionTransform;

/// Provides projections for columns that are sourced from the file.
#[derive(Debug, Clone)]
pub enum ProjectionBuilder {
    Plain(SchemaRef),
    Iceberg {
        /// `(output_name, output_dtype)`
        projected_schema: SchemaRef,
        /// `(physical_id, iceberg_column)`
        projected_iceberg_schema: IcebergSchemaRef,
    },
}

impl ProjectionBuilder {
    /// Returns the full projected schema, keyed by the output name.
    fn projected_schema(&self) -> &SchemaRef {
        match self {
            ProjectionBuilder::Plain(schema) => schema,
            ProjectionBuilder::Iceberg {
                projected_schema, ..
            } => projected_schema,
        }
    }

    pub fn new(projected_schema: SchemaRef, column_mapping: Option<&ColumnMapping>) -> Self {
        match column_mapping {
            None => ProjectionBuilder::Plain(projected_schema),
            Some(ColumnMapping::Iceberg(iceberg_schema)) => {
                // Note: `projected_schema` is derived from `iceberg_schema` during IR resolution.
                // Many of the assertions below are based on this invariant.
                assert!(projected_schema.len() <= iceberg_schema.len());

                let projected_physical_ids_lookup: PlHashMap<PlSmallStr, u32> = iceberg_schema
                    .iter()
                    .filter(|(_, col)| projected_schema.contains(&col.name))
                    .map(|(key, col)| (col.name.clone(), *key))
                    .collect();

                assert_eq!(projected_physical_ids_lookup.len(), projected_schema.len());

                let projected_iceberg_schema: IcebergSchemaRef = Arc::new(
                    projected_schema
                        .iter()
                        .map(|(output_name, output_dtype)| {
                            let physical_id = projected_physical_ids_lookup
                                .get(output_name.as_str())
                                .unwrap();

                            let column = iceberg_schema.get(physical_id).unwrap();

                            assert_eq!(&column.type_.to_polars_dtype(), output_dtype);

                            (*physical_id, column.clone())
                        })
                        .collect(),
                );

                assert_eq!(projected_iceberg_schema.len(), projected_schema.len());

                Self::Iceberg {
                    projected_schema,
                    projected_iceberg_schema,
                }
            },
        }
    }

    /// Builds a potentially mapped `Projection` (i.e. one containing renames/casting).
    ///
    /// # Returns
    /// Returns a `Plain` variant if `self` is a `Plain` variant and the `file_schema` is `None`.
    ///
    /// # Panics
    /// * If `self` is the `Iceberg` variant and `file_iceberg_schema` is `None`.
    pub fn build_projection(
        &self,
        file_schema: Option<&Schema>,
        file_iceberg_schema: Option<&IcebergSchema>,
        cast_columns_policy: CastColumnsPolicy,
    ) -> PolarsResult<Projection> {
        let selector_builder = ColumnSelectorBuilder {
            cast_columns_policy,
            // This should not be used by `attach_transforms()`.
            missing_columns_policy: MissingColumnsPolicy::Raise,
        };

        Ok(match self {
            Self::Plain(projected_schema) => {
                let Some(file_schema) = file_schema else {
                    return Ok(Projection::Plain(projected_schema.clone()));
                };

                let mut mapping: Option<PlHashMap<usize, ProjectionTransform>> = None;
                let mut missing_columns_mask: Option<MutableBitmap> = None;

                for (index, (projected_name, projected_dtype)) in
                    projected_schema.iter().enumerate()
                {
                    let Some(incoming_dtype) = file_schema.get(projected_name) else {
                        missing_columns_mask
                            .get_or_insert_with(|| {
                                MutableBitmap::from_len_zeroed(projected_schema.len())
                            })
                            .set(index, true);

                        continue;
                    };

                    match selector_builder.attach_transforms(
                        ColumnSelector::Position(0),
                        incoming_dtype,
                        projected_dtype,
                        projected_name,
                    )? {
                        ColumnSelector::Position(0) => {},
                        selector => {
                            mapping
                                .get_or_insert_with(|| {
                                    PlHashMap::with_capacity(projected_schema.len())
                                })
                                .insert(
                                    index,
                                    ProjectionTransform {
                                        source_name: projected_name.clone(),
                                        source_dtype: incoming_dtype.clone(),
                                        transform: selector,
                                    },
                                );
                        },
                    }
                }

                Projection::Mapped {
                    projected_schema: projected_schema.clone(),
                    mapping: mapping.map(Arc::new),
                    missing_columns_mask: missing_columns_mask.map(|x| x.freeze()),
                }
            },

            Self::Iceberg {
                projected_schema,
                projected_iceberg_schema,
            } => (|| {
                let file_iceberg_schema = file_iceberg_schema.ok_or_else(|| {
                    polars_err!(
                        ComputeError:
                        "reader file_arrow_schema() returned None"
                    )
                })?;

                let mut mapping: Option<PlHashMap<usize, ProjectionTransform>> = None;
                let mut missing_columns_mask: Option<MutableBitmap> = None;

                for (index, (physical_id, output_iceberg_column)) in
                    projected_iceberg_schema.iter().enumerate()
                {
                    let Some(incoming_iceberg_column) = file_iceberg_schema.get(physical_id) else {
                        missing_columns_mask
                            .get_or_insert_with(|| {
                                MutableBitmap::from_len_zeroed(projected_iceberg_schema.len())
                            })
                            .set(index, true);

                        continue;
                    };

                    match selector_builder.attach_iceberg_transforms(
                        ColumnSelector::Position(0),
                        incoming_iceberg_column,
                        output_iceberg_column,
                    )? {
                        ColumnSelector::Position(0) => {
                            assert_eq!(incoming_iceberg_column.name, output_iceberg_column.name);
                        },
                        selector => {
                            mapping
                                .get_or_insert_with(|| {
                                    PlHashMap::with_capacity(projected_iceberg_schema.len())
                                })
                                .insert(
                                    index,
                                    ProjectionTransform {
                                        source_name: incoming_iceberg_column.name.clone(),
                                        source_dtype: incoming_iceberg_column
                                            .type_
                                            .to_polars_dtype(),
                                        transform: selector,
                                    },
                                );
                        },
                    }
                }

                PolarsResult::Ok(Projection::Mapped {
                    projected_schema: projected_schema.clone(),
                    mapping: mapping.map(Arc::new),
                    missing_columns_mask: missing_columns_mask.map(|x| x.freeze()),
                })
            })()
            .map_err(|e| {
                e.wrap_msg(|msg| format!("failed to resolve Iceberg column mapping: {msg}"))
            })?,
        })
    }

    pub fn num_projections(&self) -> usize {
        self.projected_schema().len()
    }
}

/// Returns the schema containing columns to project from the file.
///
/// Note: This is used during IR lowering.
///
/// # Returns
/// `(ProjectionBuilder, FullFileSchema)`, where `ProjectionBuilder` holds information on
/// the columns to be projected from the file.
pub fn resolve_projections(
    // Final output schema of the MultiScan. Includes e.g. row index, missing columns etc, and the
    // projection is applied.
    final_output_schema: &Schema,
    // Full schema inferred for the file
    file_schema: &SchemaRef,
    // Hive parts
    hive_parts: &mut Option<HivePartitionsDf>,

    // TODO: One day update IR conversion to avoid attaching these to the file schema :')
    row_index_name: Option<&str>,
    include_file_paths: Option<&str>,
    column_mapping: Option<&ColumnMapping>,
) -> (ProjectionBuilder, SchemaRef) {
    if let Some(hive_parts) = hive_parts.as_mut() {
        let projected_hive_parts: HivePartitionsDf = hive_parts
            .df()
            .get_columns()
            .iter()
            .filter(|c| final_output_schema.contains(c.name()))
            .cloned()
            .collect::<DataFrame>()
            .into();

        *hive_parts = projected_hive_parts;
    }

    let hive_schema = hive_parts.as_ref().map(|x| x.schema().as_ref());

    // We will assume that aside from hive, there are no duplicate column names with other column
    // adding functions (e.g. row index, include file paths).

    let projected_file_schema: Schema = file_schema
        .iter()
        .filter_map(|(name, dtype)| {
            let in_final = final_output_schema.contains(name);
            let in_hive = hive_schema.is_some_and(|x| x.contains(name));
            let is_row_index_col = row_index_name.is_some_and(|x| name == x);
            let is_file_path_col = include_file_paths.is_some_and(|x| name == x);
            (in_final && !(in_hive || is_file_path_col || is_row_index_col))
                .then(|| (name.clone(), dtype.clone()))
        })
        .collect();

    // Resolve full file schema.
    // TODO: Update IR conversion to avoid attaching row index / file path
    let mut full_file_schema = file_schema.clone();

    if row_index_name.is_some_and(|x| full_file_schema.contains(x)) {
        Arc::make_mut(&mut full_file_schema).shift_remove(row_index_name.unwrap());
    }

    if include_file_paths.is_some_and(|x| full_file_schema.contains(x)) {
        Arc::make_mut(&mut full_file_schema).shift_remove(include_file_paths.unwrap());
    }

    let projection_builder =
        ProjectionBuilder::new(Arc::new(projected_file_schema), column_mapping);

    (projection_builder, full_file_schema)
}
