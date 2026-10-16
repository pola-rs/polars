use std::borrow::Cow;
use std::sync::Arc;

use polars_arrow::datatypes::ArrowSchema;
use polars_core::prelude::{ArrowField, Column, DataType};
use polars_core::schema::Schema;
use polars_error::PolarsResult;
use polars_plan::dsl::CastColumnsPolicy;
use polars_plan::plans::predicates::null_count_dtype;
use polars_utils::pl_str::PlSmallStr;

use crate::nodes::io_sources::multi_scan::components::column_selector::ColumnSelector;
use crate::nodes::io_sources::multi_scan::components::projection::MappedProjectionRef;
use crate::nodes::io_sources::multi_scan::components::projection::builder::ProjectionBuilder;
use crate::nodes::io_sources::multi_scan::reader_interface::Projection;

pub fn resolve_arrow_field_projections(
    file_arrow_schema: &ArrowSchema,
    file_schema: &Schema,
    projection: Projection,
    cast_columns_policy: CastColumnsPolicy,
) -> PolarsResult<Arc<[ArrowFieldProjection]>> {
    let projection: Projection = match projection {
        Projection::Plain(projected_schema) => ProjectionBuilder::new(projected_schema, None, None)
            .build_projection(Some(file_schema), None, cast_columns_policy, usize::MAX)?,
        Projection::Mapped { .. } => projection,
    };

    Ok(projection
        .iter_non_missing_columns()
        .map(
            |MappedProjectionRef {
                 source_name,
                 output_name,
                 output_dtype,
                 resolved_transform,
             }| {
                let arrow_field = file_arrow_schema.get(source_name.as_str()).unwrap().clone();

                let Some(resolved_transform) = resolved_transform else {
                    assert_eq!(source_name, output_name);

                    return ArrowFieldProjection::Plain(arrow_field);
                };

                assert_eq!(
                    resolved_transform.source_dtype,
                    file_schema.get(source_name.as_str()).unwrap()
                );

                ArrowFieldProjection::Mapped {
                    arrow_field,
                    output_name: output_name.clone(),
                    output_dtype: output_dtype.clone(),
                    transform: resolved_transform.attach_transforms(ColumnSelector::Position(0)),
                }
            },
        )
        .collect::<Arc<[ArrowFieldProjection]>>())
}

/// Represents a potentially mapped (i.e. cast and/or renamed) arrow field projection.
#[derive(Debug)]
pub enum ArrowFieldProjection {
    Plain(ArrowField),
    Mapped {
        arrow_field: ArrowField,
        output_name: PlSmallStr,
        output_dtype: DataType,
        transform: ColumnSelector,
    },
}

impl ArrowFieldProjection {
    #[inline]
    pub fn arrow_field(&self) -> &ArrowField {
        match self {
            Self::Plain(field) => field,
            Self::Mapped { arrow_field, .. } => arrow_field,
        }
    }

    pub fn output_name(&self) -> &PlSmallStr {
        match self {
            Self::Plain(field) => &field.name,
            Self::Mapped { output_name, .. } => output_name,
        }
    }

    pub fn output_dtype(&self) -> Cow<'_, DataType> {
        match self {
            Self::Plain(field) => Cow::Owned(DataType::from_arrow_field(field)),
            Self::Mapped { output_dtype, .. } => Cow::Borrowed(output_dtype),
        }
    }

    pub fn apply_transform(&self, column: Column) -> PolarsResult<Column> {
        match self {
            Self::Plain(_) => Ok(column),
            Self::Mapped {
                transform,
                output_dtype,
                ..
            } => apply_selector(transform, column, output_dtype),
        }
    }

    /// Like [`Self::apply_transform`], but for the `null_count` statistic.
    pub fn apply_null_count_transform(&self, column: Column) -> PolarsResult<Column> {
        match self {
            Self::Plain(_) => Ok(column),
            Self::Mapped {
                transform,
                output_dtype,
                ..
            } => {
                let nc_dtype = null_count_dtype(output_dtype);

                match transform.null_count_transform() {
                    Some(selector) => apply_selector(&selector, column, &nc_dtype),
                    None => Ok(Column::full_null(
                        PlSmallStr::EMPTY,
                        column.len(),
                        &nc_dtype,
                    )),
                }
            },
        }
    }
}

fn apply_selector(
    selector: &ColumnSelector,
    column: Column,
    expected_dtype: &DataType,
) -> PolarsResult<Column> {
    let output_height = column.len();
    let out = selector.select_from_columns(&[column], output_height)?;

    debug_assert_eq!(out.dtype(), expected_dtype);

    Ok(out)
}
