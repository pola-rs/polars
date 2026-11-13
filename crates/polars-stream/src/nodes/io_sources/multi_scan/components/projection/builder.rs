use std::sync::Arc;

use polars_arrow::bitmap::MutableBitmap;
use polars_core::prelude::{InitHashMaps, PlHashMap, PlHashSet};
use polars_core::scalar::Scalar;
use polars_core::schema::iceberg::{
    IcebergColumn, IcebergColumnType, IcebergSchema, IcebergSchemaRef,
};
use polars_core::schema::{Schema, SchemaRef};
use polars_error::{PolarsResult, polars_err};
use polars_plan::dsl::default_values::IcebergDefaultFieldValues;
use polars_plan::dsl::{
    CastColumnsPolicy, ColumnMapping, FieldSelection, MissingColumnsPolicy, ScanFieldSelection,
};
use polars_utils::pl_str::PlSmallStr;

use crate::nodes::io_sources::multi_scan::components::column_selector::ColumnSelector;
use crate::nodes::io_sources::multi_scan::components::column_selector::builder::{
    ColumnSelectorBuilder, build_iceberg_default_value, collect_partitioned_structs,
};
use crate::nodes::io_sources::multi_scan::components::default_field_values::IcebergDefaultValueProviderRef;
use crate::nodes::io_sources::multi_scan::components::projection::{
    Projection, ProjectionTransform,
};

/// Provides projections for columns that are sourced from the file.
#[derive(Debug, Clone)]
pub enum ProjectionBuilder {
    Plain(SchemaRef),
    Iceberg {
        /// `(output_name, output_dtype)`
        projected_schema: SchemaRef,
        /// `(physical_id, iceberg_column)`
        projected_iceberg_schema: IcebergSchemaRef,
        /// Whether the table has a name mapping, for data files without field IDs. This cannot
        /// be derived from the mapped names in `projected_iceberg_schema`: the mapping may name
        /// none of the projected columns (e.g. only fields dropped from the schema).
        table_has_name_mapping: bool,
        /// Used for filling missing fields.
        default_values: Option<Arc<IcebergDefaultFieldValues>>,
        /// The struct fields read of each projected column, indexed like `projected_schema`;
        /// `None` if every column is read whole. The output dtypes in `projected_schema` are
        /// pruned to them, the columns in `projected_iceberg_schema` are not.
        field_selections: Option<Arc<[FieldSelection]>>,
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

    /// `field_selection`: the struct fields read of the columns not read whole (see
    /// [`ScanFieldSelection`]), keyed by output name; `projected_schema` has their pruned dtypes.
    pub fn new(
        projected_schema: SchemaRef,
        column_mapping: Option<&ColumnMapping>,
        default_values: Option<Arc<IcebergDefaultFieldValues>>,
        field_selection: Option<&ScanFieldSelection>,
    ) -> Self {
        match column_mapping {
            None => ProjectionBuilder::Plain(projected_schema),
            Some(ColumnMapping::Iceberg {
                schema: iceberg_schema,
                has_name_mapping,
            }) => {
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

                            let selection = field_selection
                                .map_or(&FieldSelection::All, |s| s.get(output_name));
                            assert_eq!(
                                &selection.prune_dtype(&column.type_.to_polars_dtype().unwrap()),
                                output_dtype
                            );

                            (*physical_id, column.clone())
                        })
                        .collect(),
                );

                assert_eq!(projected_iceberg_schema.len(), projected_schema.len());

                let field_selections = field_selection
                    .filter(|s| projected_schema.iter_names().any(|n| !s.get(n).is_all()))
                    .map(|s| {
                        projected_schema
                            .iter_names()
                            .map(|n| s.get(n).clone())
                            .collect()
                    });

                Self::Iceberg {
                    projected_schema,
                    projected_iceberg_schema,
                    table_has_name_mapping: *has_name_mapping,
                    default_values,
                    field_selections,
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
    /// * If `self` is the `Iceberg` variant and `file_iceberg_schema` is `None` or `scan_source_idx` is `usize::MAX`.
    pub fn build_projection(
        &self,
        file_schema: Option<&Schema>,
        file_iceberg_schema: Option<&IcebergSchema>,
        cast_columns_policy: CastColumnsPolicy,
        scan_source_idx: usize,
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

                if mapping.is_none() && missing_columns_mask.is_none() {
                    return Ok(Projection::Plain(projected_schema.clone()));
                }

                Projection::Mapped {
                    projected_schema: projected_schema.clone(),
                    mapping: mapping.map(Arc::new),
                    missing_columns_mask: missing_columns_mask.map(|x| x.freeze()),
                    missing_column_defaults: None,
                }
            },

            Self::Iceberg {
                projected_schema,
                projected_iceberg_schema,
                table_has_name_mapping: _,
                default_values,
                field_selections,
            } => (|| {
                let file_iceberg_schema = file_iceberg_schema.ok_or_else(|| {
                    polars_err!(
                        ComputeError:
                        "reader file_arrow_schema() returned None"
                    )
                })?;

                // usize::MAX is used as None.
                assert_ne!(scan_source_idx, usize::MAX);

                let mut mapping: Option<PlHashMap<usize, ProjectionTransform>> = None;
                let mut missing_columns_mask: Option<MutableBitmap> = None;
                let mut missing_column_defaults: Option<PlHashMap<usize, Scalar>> = None;
                let mut partitioned_structs = PlHashSet::new();
                let iceberg_default_value_provider = default_values.as_deref().map(|x| {
                    let provider = IcebergDefaultValueProviderRef::new(x, scan_source_idx);
                    if field_selections.is_none() {
                        return provider;
                    }
                    // Defaults are built for pruned columns: whether a missing struct is valid
                    // depends on partition values of fields that may be pruned away.
                    for column in projected_iceberg_schema.values() {
                        collect_partitioned_structs(provider, column, &mut partitioned_structs);
                    }
                    provider.with_partitioned_structs(&partitioned_structs)
                });

                for (index, (physical_id, output_iceberg_column)) in
                    projected_iceberg_schema.iter().enumerate()
                {
                    let Some(incoming_iceberg_column) = file_iceberg_schema.get(physical_id) else {
                        missing_columns_mask
                            .get_or_insert_with(|| {
                                MutableBitmap::from_len_zeroed(projected_iceberg_schema.len())
                            })
                            .set(index, true);

                        if let Some(iceberg_default_value_provider) = iceberg_default_value_provider
                        {
                            // Of the output dtype: only the selected struct fields.
                            let pruned;
                            let output_column = match field_selections.as_deref() {
                                Some(s) => {
                                    pruned = prune_iceberg_column(output_iceberg_column, &s[index]);
                                    &pruned
                                },
                                None => output_iceberg_column,
                            };
                            if let Some(default) = build_iceberg_default_value(
                                iceberg_default_value_provider,
                                output_column,
                            )? {
                                missing_column_defaults
                                    .get_or_insert_with(|| {
                                        PlHashMap::with_capacity(projected_iceberg_schema.len())
                                    })
                                    .insert(index, default);
                            }
                        }

                        continue;
                    };

                    if let Some(selection) = field_selections
                        .as_deref()
                        .map(|s| &s[index])
                        .filter(|s| !s.is_all())
                    {
                        // The cast policies are checked on the whole column, so that they raise
                        // as when it is read whole.
                        selector_builder.attach_iceberg_transforms(
                            ColumnSelector::Position(0),
                            incoming_iceberg_column,
                            output_iceberg_column,
                            iceberg_default_value_provider,
                        )?;

                        // Only the selected fields are read: map them to the fields of the file
                        // by physical ID, so that each is read once whatever its name.
                        let output_column = prune_iceberg_column(output_iceberg_column, selection);
                        let incoming_column =
                            prune_iceberg_column_by_id(incoming_iceberg_column, &output_column);

                        let transform = selector_builder.attach_iceberg_transforms(
                            ColumnSelector::Position(0),
                            &incoming_column,
                            &output_column,
                            iceberg_default_value_provider,
                        )?;

                        mapping
                            .get_or_insert_with(|| {
                                PlHashMap::with_capacity(projected_iceberg_schema.len())
                            })
                            .insert(
                                index,
                                ProjectionTransform {
                                    source_name: incoming_column.name.clone(),
                                    source_dtype: incoming_column.type_.to_polars_dtype()?,
                                    transform,
                                },
                            );
                        continue;
                    }

                    match selector_builder.attach_iceberg_transforms(
                        ColumnSelector::Position(0),
                        incoming_iceberg_column,
                        output_iceberg_column,
                        iceberg_default_value_provider,
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
                                            .to_polars_dtype()?,
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
                    missing_column_defaults: missing_column_defaults.map(Arc::new),
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

    pub fn projected_names(&self) -> impl Iterator<Item = &PlSmallStr> {
        self.projected_schema().iter_names()
    }
}

/// `column` with only the struct fields in `selection` (by name).
fn prune_iceberg_column(column: &IcebergColumn, selection: &FieldSelection) -> IcebergColumn {
    use IcebergColumnType as ICT;

    let type_ = match (&column.type_, selection) {
        (ICT::Struct(fields), FieldSelection::Fields(selected)) => ICT::Struct(
            fields
                .iter()
                .filter_map(|(id, field)| {
                    let selection = selected.get(&field.name)?;
                    Some((*id, prune_iceberg_column(field, selection)))
                })
                .collect(),
        ),
        (ICT::List(inner), FieldSelection::Element(element)) => {
            ICT::List(Box::new(prune_iceberg_column(inner, element)))
        },
        (ICT::FixedSizeList(inner, width), FieldSelection::Element(element)) => {
            ICT::FixedSizeList(Box::new(prune_iceberg_column(inner, element)), *width)
        },
        (type_, _) => type_.clone(),
    };

    IcebergColumn {
        type_,
        ..column.clone()
    }
}

/// `incoming` (a column of the file) with only the struct fields `target` (a pruned column of the
/// table) has, matched by physical ID.
fn prune_iceberg_column_by_id(incoming: &IcebergColumn, target: &IcebergColumn) -> IcebergColumn {
    use IcebergColumnType as ICT;

    let type_ = match (&incoming.type_, &target.type_) {
        (ICT::Struct(fields), ICT::Struct(target_fields)) => ICT::Struct(
            fields
                .iter()
                .filter_map(|(id, field)| {
                    let target = target_fields.get(id)?;
                    Some((*id, prune_iceberg_column_by_id(field, target)))
                })
                .collect(),
        ),
        (ICT::List(inner), ICT::List(target)) => {
            ICT::List(Box::new(prune_iceberg_column_by_id(inner, target)))
        },
        (ICT::FixedSizeList(inner, width), ICT::FixedSizeList(target, _)) => {
            ICT::FixedSizeList(Box::new(prune_iceberg_column_by_id(inner, target)), *width)
        },
        (type_, _) => type_.clone(),
    };

    IcebergColumn {
        type_,
        ..incoming.clone()
    }
}
