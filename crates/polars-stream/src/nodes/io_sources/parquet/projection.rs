use std::sync::Arc;

use polars_arrow::datatypes::{ArrowDataType, ArrowSchema};
use polars_core::config;
use polars_core::prelude::{ArrowField, Column, DataType, PlHashMap, PlIndexMap};
use polars_core::schema::Schema;
use polars_error::{PolarsResult, polars_ensure};
use polars_io::prelude::FileMetadata;
use polars_parquet::read::{RowGroupMetadata, n_columns};
use polars_plan::dsl::{
    CastColumnsPolicy, ExtraColumnsPolicy, FieldSelection, MissingColumnsPolicy, ScanFieldSelection,
};
use polars_utils::pl_str::PlSmallStr;

use crate::nodes::io_sources::multi_scan::components::column_selector::ColumnSelector;
use crate::nodes::io_sources::multi_scan::components::column_selector::builder::ColumnSelectorBuilder;
use crate::nodes::io_sources::multi_scan::components::projection::MappedProjectionRef;
use crate::nodes::io_sources::multi_scan::components::projection::builder::ProjectionBuilder;
use crate::nodes::io_sources::multi_scan::reader_interface::Projection;

/// A struct column (or list of them) of which only some leaves are read (see
/// [`ScanFieldSelection`]).
#[derive(Debug)]
pub struct PrunedField {
    /// How to decode the column: one per distinct choice of the leaves read only for a struct
    /// validity, which is made per row group. Never empty.
    pub decodes: Vec<PrunedDecode>,
    /// The index into `decodes` of each row group; `None` if there is only one.
    pub row_group_decodes: Option<Arc<[u32]>>,
    /// The dtype the reader outputs: the selected fields that the file has.
    pub dtype: DataType,
    /// The full dtype of the column in the file and in the scan schema: the cast policies (e.g.
    /// on missing or extra struct fields) are checked on these, as for a column read whole.
    pub unpruned_dtypes: Option<(DataType, DataType)>,
}

/// How to decode a pruned column in a row group.
#[derive(Debug)]
pub struct PrunedDecode {
    /// The field to decode: the selected fields that the file has, plus a leaf read only for the
    /// struct validity where none of them is.
    pub arrow_field: ArrowField,
    /// The leaves to read, as ascending indices into the leaves of the column.
    pub leaves: Arc<[usize]>,
    /// Narrows a column decoded with `arrow_field` to the output dtype, if they differ.
    pub narrow: Option<ColumnSelector>,
}

impl PrunedField {
    /// How to decode row group `row_group_idx`.
    pub fn decode(&self, row_group_idx: usize) -> &PrunedDecode {
        match &self.row_group_decodes {
            Some(idxs) => &self.decodes[idxs[row_group_idx] as usize],
            None => &self.decodes[0],
        }
    }

    /// Whether some row group is decoded with leaves not in the output.
    pub fn narrows(&self) -> bool {
        self.decodes.iter().any(|d| d.narrow.is_some())
    }
}

/// The struct columns of the file of which only some fields are read. `metadata` gives the
/// compressed sizes to pick the cheapest leaf for a struct validity in each row group.
pub fn resolve_pruned_fields(
    file_arrow_schema: &ArrowSchema,
    field_selection: &ScanFieldSelection,
    scan_schema: Option<&Schema>,
    metadata: &FileMetadata,
) -> PolarsResult<PlHashMap<PlSmallStr, Arc<PrunedField>>> {
    let mut out = PlHashMap::default();

    for (name, selection) in field_selection.0.iter() {
        let Some(arrow_field) = file_arrow_schema.get(name.as_str()) else {
            continue;
        };
        if selection.is_all() {
            continue;
        }

        let leaf_sizes = |rg: &RowGroupMetadata| -> Vec<i64> {
            rg.columns_under_root_iter(name)
                .map(|iter| iter.map(|c| c.compressed_size().unwrap_or(0)).collect())
                .unwrap_or_default()
        };

        // The dtype the plan expects of the column in this file.
        let full_dtype = DataType::from_arrow_field(arrow_field)?;
        let expected_dtype = selection.prune_dtype(&full_dtype);

        let first_sizes = metadata
            .row_groups
            .first()
            .map(leaf_sizes)
            .unwrap_or_default();
        let mut pruner = Pruner::new(&first_sizes);
        let decoded = pruner.prune(arrow_field, selection, 0);
        let dtype = DataType::from_arrow_field(&selection.prune_arrow_field(arrow_field))?;

        let narrow_to = |decoded: &ArrowField, dtype: &DataType| -> PolarsResult<_> {
            let decoded_dtype = DataType::from_arrow_field(decoded)?;
            (&decoded_dtype != dtype)
                .then(|| {
                    ColumnSelectorBuilder {
                        cast_columns_policy: CastColumnsPolicy {
                            extra_struct_fields: ExtraColumnsPolicy::Ignore,
                            ..Default::default()
                        },
                        missing_columns_policy: MissingColumnsPolicy::Raise,
                    }
                    .attach_transforms(
                        ColumnSelector::Position(0),
                        &decoded_dtype,
                        dtype,
                        name,
                    )
                })
                .transpose()
        };

        let unpruned_dtypes = scan_schema
            .and_then(|s| s.get(name))
            .map(|scan_dtype| (full_dtype.clone(), scan_dtype.clone()));

        // The reader and the planner disagree on the pruned dtype (e.g. an arrow type the leaf
        // walk does not prune the way `prune_dtype` does): read the column whole and narrow it,
        // rather than return other fields than the plan expects.
        if dtype != expected_dtype {
            if config::verbose() {
                eprintln!(
                    "[ParquetFileReader]: reading pruned column '{name}' whole: \
                    pruned dtype {dtype:?} != expected {expected_dtype:?}"
                );
            }
            let narrow = narrow_to(arrow_field, &expected_dtype)?;
            out.insert(
                name.clone(),
                Arc::new(PrunedField {
                    decodes: vec![PrunedDecode {
                        arrow_field: arrow_field.clone(),
                        leaves: (0..n_columns(arrow_field.dtype())).collect(),
                        narrow,
                    }],
                    row_group_decodes: None,
                    dtype: expected_dtype,
                    unpruned_dtypes,
                }),
            );
            continue;
        }

        let mut decodes = vec![PrunedDecode {
            narrow: narrow_to(&decoded, &dtype)?,
            arrow_field: decoded,
            leaves: std::mem::take(&mut pruner.leaves).into(),
        }];

        // The leaves read for a struct validity are the cheapest in each row group.
        let row_group_decodes = if pruner.chose_validity_leaf && metadata.row_groups.len() > 1 {
            let mut idxs = Vec::with_capacity(metadata.row_groups.len());
            idxs.push(0u32);
            for rg in &metadata.row_groups[1..] {
                let sizes = leaf_sizes(rg);
                let mut pruner = Pruner::new(&sizes);
                let decoded = pruner.prune(arrow_field, selection, 0);
                let idx = match decodes
                    .iter()
                    .position(|d| d.leaves.as_ref() == pruner.leaves.as_slice())
                {
                    Some(idx) => idx,
                    None => {
                        decodes.push(PrunedDecode {
                            narrow: narrow_to(&decoded, &dtype)?,
                            arrow_field: decoded,
                            leaves: pruner.leaves.into(),
                        });
                        decodes.len() - 1
                    },
                };
                idxs.push(idx as u32);
            }
            (decodes.len() > 1).then(|| idxs.into())
        } else {
            None
        };

        out.insert(
            name.clone(),
            Arc::new(PrunedField {
                decodes,
                row_group_decodes,
                dtype,
                unpruned_dtypes,
            }),
        );
    }

    Ok(out)
}

/// `field_selection` narrowed to the struct fields of the dtypes in `scan_schema`, for a cast to
/// them that ignores extra struct fields: a file column with fields the scan schema leaves out
/// reads only the others.
pub fn narrow_to_scan_schema(
    field_selection: Option<&ScanFieldSelection>,
    scan_schema: &Schema,
    file_arrow_schema: &ArrowSchema,
) -> Option<ScanFieldSelection> {
    let mut out = field_selection.map_or_else(PlIndexMap::default, |s| s.0.clone());

    for (name, scan_dtype) in scan_schema.iter() {
        if !scan_dtype.is_nested() {
            continue;
        }
        let Some(arrow_field) = file_arrow_schema.get(name.as_str()) else {
            continue;
        };
        let file_dtype = DataType::from_arrow_field(arrow_field).ok()?;
        let target =
            FieldSelection::of_target_dtype(&file_dtype, scan_dtype).normalize(&file_dtype);
        if target.is_all() {
            continue;
        }
        let selection = match out.get(name) {
            Some(selection) => selection.intersect(&target).normalize(&file_dtype),
            None => target,
        };
        out.insert(name.clone(), selection);
    }

    (!out.is_empty()).then_some(ScanFieldSelection(out))
}

/// Walks a column and its selection, collecting the leaves to read.
struct Pruner<'a> {
    /// The compressed size of each leaf of the column in the row group.
    leaf_sizes: &'a [i64],
    /// The leaves to read, ascending.
    leaves: Vec<usize>,
    /// Whether a leaf was read only for a struct validity: its choice depends on `leaf_sizes`.
    chose_validity_leaf: bool,
}

impl<'a> Pruner<'a> {
    fn new(leaf_sizes: &'a [i64]) -> Self {
        Self {
            leaf_sizes,
            leaves: Vec::new(),
            chose_validity_leaf: false,
        }
    }

    /// Prunes `field` (whose leaves start at `offset`) to `selection`. Returns the field to
    /// decode; the field output is `selection.prune_arrow_field(field)`.
    fn prune(
        &mut self,
        field: &ArrowField,
        selection: &FieldSelection,
        offset: usize,
    ) -> ArrowField {
        match (field.dtype(), selection) {
            (ArrowDataType::Struct(children), FieldSelection::Fields(selected)) => {
                let mut decoded_children = Vec::new();
                let mut child_offset = offset;

                for child in children {
                    if let Some(child_selection) = selected.get(&child.name) {
                        decoded_children.push(self.prune(child, child_selection, child_offset));
                    }
                    child_offset += n_columns(child.dtype());
                }

                // The struct validity comes from the definition levels of a leaf: read the
                // cheapest one if no selected field is in the file. Under a list, this leaf
                // also gives the list offsets and validity.
                if decoded_children.is_empty()
                    && let Some(choice) =
                        cheapest_leaf_of_children(children, offset, self.leaf_sizes)
                {
                    let child = &children[choice.path[0]];
                    self.chose_validity_leaf = true;
                    decoded_children.push(validity_leaf_field(
                        child,
                        &choice.path[1..],
                        choice.offset,
                        &mut self.leaves,
                    ));
                }

                field.with_dtype(ArrowDataType::Struct(decoded_children))
            },
            // A list has no leaves of its own: its offsets and validity come with those of the
            // element.
            (ArrowDataType::List(inner), FieldSelection::Element(element)) => {
                let decoded = self.prune(inner, element, offset);
                field.with_dtype(ArrowDataType::List(Box::new(decoded)))
            },
            (ArrowDataType::LargeList(inner), FieldSelection::Element(element)) => {
                let decoded = self.prune(inner, element, offset);
                field.with_dtype(ArrowDataType::LargeList(Box::new(decoded)))
            },
            (ArrowDataType::FixedSizeList(inner, width), FieldSelection::Element(element)) => {
                let decoded = self.prune(inner, element, offset);
                field.with_dtype(ArrowDataType::FixedSizeList(Box::new(decoded), *width))
            },
            _ => {
                self.leaves
                    .extend(offset..offset + n_columns(field.dtype()));
                field.clone()
            },
        }
    }
}

/// The leaf chosen to read a struct validity from.
#[derive(Debug, Clone, PartialEq, Eq)]
struct LeafChoice {
    /// `(under a list, compressed size)`: the cost of reading the leaf.
    cost: (bool, i64),
    /// The child index taken at each struct level down to the leaf.
    path: Vec<usize>,
    /// The offset of the leaves of the field at the end of `path`.
    offset: usize,
}

/// The cheapest leaf of the children of a struct whose leaves start at `offset`: no list on the
/// way first, then the smallest compressed size (the first of equals). A field that is not a
/// struct is a leaf here, read whole.
///
/// One pass over the fields, so linear in their number (a choice made per level, comparing the
/// cost of each child's own choice, would be exponential in the depth when costs are recomputed).
fn cheapest_leaf_of_children(
    children: &[ArrowField],
    offset: usize,
    leaf_sizes: &[i64],
) -> Option<LeafChoice> {
    // Paths are built leaf first.
    cheapest_leaf_rev(children, offset, leaf_sizes)
        .1
        .map(|mut choice| {
            choice.path.reverse();
            choice
        })
}

/// The number of leaves of the `children` of a struct and their cheapest leaf, with its path in
/// reverse order.
fn cheapest_leaf_rev(
    children: &[ArrowField],
    offset: usize,
    leaf_sizes: &[i64],
) -> (usize, Option<LeafChoice>) {
    let mut best: Option<LeafChoice> = None;
    let mut child_offset = offset;

    for (i, child) in children.iter().enumerate() {
        let (n, choice) = match child.dtype() {
            ArrowDataType::Struct(grandchildren) => {
                cheapest_leaf_rev(grandchildren, child_offset, leaf_sizes)
            },
            dtype => {
                let n = n_columns(dtype);
                let size = (child_offset..child_offset + n)
                    .map(|i| leaf_sizes.get(i).copied().unwrap_or(0))
                    .sum();
                let choice = (n > 0).then(|| LeafChoice {
                    cost: (dtype.is_nested(), size),
                    path: Vec::new(),
                    offset: child_offset,
                });
                (n, choice)
            },
        };
        if let Some(mut choice) = choice
            && best.as_ref().is_none_or(|b| choice.cost < b.cost)
        {
            choice.path.push(i);
            best = Some(choice);
        }
        child_offset += n;
    }

    (child_offset - offset, best)
}

/// `field` with only the leaf at `path` (child indices of nested structs), whose leaves start at
/// `offset`, pushing them to `leaves`. Read for the validity of the struct above it.
fn validity_leaf_field(
    field: &ArrowField,
    path: &[usize],
    offset: usize,
    leaves: &mut Vec<usize>,
) -> ArrowField {
    match (field.dtype(), path.split_first()) {
        (ArrowDataType::Struct(children), Some((&i, rest))) => {
            let child = validity_leaf_field(&children[i], rest, offset, leaves);
            let mut field = field.clone();
            field.dtype = ArrowDataType::Struct(vec![child]);
            field
        },
        (dtype, _) => {
            leaves.extend(offset..offset + n_columns(dtype));
            field.clone()
        },
    }
}

pub fn resolve_arrow_field_projections(
    file_arrow_schema: &ArrowSchema,
    file_schema: &Schema,
    projection: Projection,
    cast_columns_policy: CastColumnsPolicy,
    pruned_fields: &PlHashMap<PlSmallStr, Arc<PrunedField>>,
    metadata: &FileMetadata,
) -> PolarsResult<Arc<[ArrowFieldProjection]>> {
    // A mapped projection (by Iceberg field IDs) may read a column with a pruned dtype: read only
    // its leaves. Columns read through several outputs are read with the union of their
    // selections, so that no leaf is read twice for one column.
    let mapped_pruned_fields;
    let pruned_fields = if let Projection::Mapped { .. } = &projection {
        let mut selections: PlIndexMap<PlSmallStr, FieldSelection> = PlIndexMap::default();
        for MappedProjectionRef {
            source_name,
            output_dtype,
            resolved_transform,
            ..
        } in projection.iter_non_missing_columns()
        {
            if pruned_fields.contains_key(source_name) {
                continue;
            }
            let source_dtype = resolved_transform.map_or(output_dtype, |t| t.source_dtype);
            let Some(file_dtype) = file_schema.get(source_name) else {
                continue;
            };
            let selection =
                FieldSelection::of_narrowed_dtype(file_dtype, source_dtype).normalize(file_dtype);
            match selections.get_mut(source_name) {
                Some(merged) => merged.union(selection),
                None => {
                    selections.insert(source_name.clone(), selection);
                },
            }
        }
        selections.retain(|_, s| !s.is_all());

        if selections.is_empty() {
            pruned_fields
        } else {
            let mut out = pruned_fields.clone();
            out.extend(resolve_pruned_fields(
                file_arrow_schema,
                &ScanFieldSelection(selections),
                None,
                metadata,
            )?);
            mapped_pruned_fields = out;
            &mapped_pruned_fields
        }
    } else {
        pruned_fields
    };

    // Check the cast policies on the pruned columns as if they were read whole, so that e.g. a
    // struct field missing from the file raises as it does without the pruning.
    for MappedProjectionRef { source_name, .. } in projection.iter_non_missing_columns() {
        if let Some((file_dtype, scan_dtype)) = pruned_fields
            .get(source_name)
            .and_then(|p| p.unpruned_dtypes.as_ref())
        {
            ColumnSelectorBuilder {
                cast_columns_policy: cast_columns_policy.clone(),
                missing_columns_policy: MissingColumnsPolicy::Raise,
            }
            .attach_transforms(
                ColumnSelector::Position(0),
                file_dtype,
                scan_dtype,
                source_name,
            )?;
        }
    }

    let projection: Projection = match projection {
        Projection::Plain(projected_schema) => ProjectionBuilder::new(
            projected_schema,
            None,
            None,
            None,
        )
        .build_projection(Some(file_schema), None, cast_columns_policy, usize::MAX)?,
        Projection::Mapped { .. } => projection,
    };

    projection
        .iter_non_missing_columns()
        .map(
            |MappedProjectionRef {
                 source_name,
                 output_name,
                 output_dtype,
                 resolved_transform,
             }| {
                let pruned = pruned_fields.get(source_name).cloned();
                let arrow_field = match &pruned {
                    Some(pruned) => pruned.decodes[0].arrow_field.clone(),
                    None => file_arrow_schema.get(source_name.as_str()).unwrap().clone(),
                };

                let Some(resolved_transform) = resolved_transform else {
                    assert_eq!(source_name, output_name);

                    let dtype = match &pruned {
                        Some(pruned) => pruned.dtype.clone(),
                        None => file_schema.get(source_name.as_str()).unwrap().clone(),
                    };
                    return Ok(ArrowFieldProjection::Plain(arrow_field, dtype, pruned));
                };

                // The transform reads the dtype it was resolved with: the file's, or a pruned one
                // that the decoded column is first narrowed to.
                let decoded_dtype = match &pruned {
                    Some(pruned) => &pruned.dtype,
                    None => file_schema.get(source_name.as_str()).unwrap(),
                };
                let input = if decoded_dtype == resolved_transform.source_dtype {
                    ColumnSelector::Position(0)
                } else {
                    assert!(pruned.is_some());
                    ColumnSelectorBuilder {
                        cast_columns_policy: CastColumnsPolicy {
                            extra_struct_fields: ExtraColumnsPolicy::Ignore,
                            ..Default::default()
                        },
                        missing_columns_policy: MissingColumnsPolicy::Raise,
                    }
                    .attach_transforms(
                        ColumnSelector::Position(0),
                        decoded_dtype,
                        resolved_transform.source_dtype,
                        source_name,
                    )?
                };

                Ok(ArrowFieldProjection::Mapped {
                    arrow_field,
                    output_name: output_name.clone(),
                    output_dtype: output_dtype.clone(),
                    transform: resolved_transform.attach_transforms(input),
                    pruned,
                })
            },
        )
        .collect::<PolarsResult<Arc<[ArrowFieldProjection]>>>()
}

/// Represents a potentially mapped (i.e. cast and/or renamed) arrow field projection.
///
/// A pruned struct column (see [`PrunedField`]) decodes [`Self::decode_field`] from the selected
/// leaves of the row group; [`Self::apply_transform`] then narrows it to the pruned dtype first.
/// `arrow_field` is the field decoded in the first row group.
#[derive(Debug)]
pub enum ArrowFieldProjection {
    /// The field, the dtype it is read as, and its pruning.
    Plain(ArrowField, DataType, Option<Arc<PrunedField>>),
    Mapped {
        arrow_field: ArrowField,
        output_name: PlSmallStr,
        output_dtype: DataType,
        transform: ColumnSelector,
        pruned: Option<Arc<PrunedField>>,
    },
}

impl ArrowFieldProjection {
    #[inline]
    pub fn arrow_field(&self) -> &ArrowField {
        match self {
            Self::Plain(field, ..) => field,
            Self::Mapped { arrow_field, .. } => arrow_field,
        }
    }

    pub fn pruned(&self) -> Option<&PrunedField> {
        match self {
            Self::Plain(_, _, pruned) | Self::Mapped { pruned, .. } => pruned.as_deref(),
        }
    }

    /// The leaves to read in row group `row_group_idx`, as indices into the leaves of the
    /// column. `None`: all of them.
    pub fn leaves(&self, row_group_idx: usize) -> Option<&[usize]> {
        self.pruned()
            .map(|p| p.decode(row_group_idx).leaves.as_ref())
    }

    /// The field to decode in row group `row_group_idx`.
    pub fn decode_field(&self, row_group_idx: usize) -> &ArrowField {
        match self.pruned() {
            Some(pruned) => &pruned.decode(row_group_idx).arrow_field,
            None => self.arrow_field(),
        }
    }

    /// Whether only some leaves of the column are read.
    pub fn is_pruned(&self) -> bool {
        self.pruned().is_some()
    }

    pub fn output_name(&self) -> &PlSmallStr {
        match self {
            Self::Plain(field, ..) => &field.name,
            Self::Mapped { output_name, .. } => output_name,
        }
    }

    pub fn output_dtype(&self) -> &DataType {
        match self {
            Self::Plain(_, dtype, _) => dtype,
            Self::Mapped { output_dtype, .. } => output_dtype,
        }
    }

    /// Transforms a column decoded with [`Self::decode_field`] of row group `row_group_idx`
    /// (`arrow_field` for statistics, row group 0).
    pub fn apply_transform(&self, column: Column, row_group_idx: usize) -> PolarsResult<Column> {
        let column = match self.pruned() {
            Some(pruned) => {
                let column = match &pruned.decode(row_group_idx).narrow {
                    Some(narrow) => {
                        let height = column.len();
                        narrow.select_from_columns(&[column], height)?
                    },
                    None => column,
                };
                // The decoded fields must be the ones the file schema reported to the scan.
                polars_ensure!(
                    column.dtype() == &pruned.dtype,
                    ComputeError: "struct-field pruning: column '{}' decoded as {:?}, expected {:?}",
                    column.name(), column.dtype(), pruned.dtype
                );
                column
            },
            None => column,
        };

        match self {
            Self::Plain(..) => Ok(column),
            Self::Mapped {
                transform,
                output_dtype,
                ..
            } => {
                let output_height = column.len();
                let out = transform.select_from_columns(&[column], output_height)?;

                debug_assert_eq!(out.dtype(), output_dtype);

                Ok(out)
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn field(name: &str, dtype: ArrowDataType) -> ArrowField {
        ArrowField::new(name.into(), dtype, true)
    }

    fn struct_of(children: Vec<ArrowField>) -> ArrowDataType {
        ArrowDataType::Struct(children)
    }

    /// No list on the way first, then the smallest compressed size.
    #[test]
    fn validity_leaf_choice_rule() {
        let list = ArrowDataType::LargeList(Box::new(field("item", ArrowDataType::Int8)));
        let children = vec![
            field("l", list),
            field("big", ArrowDataType::Int64),
            field(
                "s",
                struct_of(vec![
                    field("x", ArrowDataType::Int64),
                    field("y", ArrowDataType::Int8),
                ]),
            ),
        ];
        // Leaves: l.item, big, s.x, s.y.
        // Sizes past `leaf_sizes` count as 0: the first leaf not under a list wins.
        let choice = cheapest_leaf_of_children(&children, 10, &[1, 50, 40, 30]).unwrap();
        assert_eq!(choice.cost, (false, 0));
        assert_eq!(choice.path, vec![1]);
        assert_eq!(choice.offset, 11);

        let choice = cheapest_leaf_of_children(&children, 0, &[1, 50, 40, 30]).unwrap();
        assert_eq!(choice.cost, (false, 30));
        assert_eq!(choice.path, vec![2, 1]);
        assert_eq!(choice.offset, 3);

        let s = field("s", struct_of(children));
        let sizes = [1, 50, 40, 30];
        let mut pruner = Pruner::new(&sizes);
        let selection = FieldSelection::validity_only();
        let decoded = pruner.prune(&s, &selection, 0);
        assert_eq!(pruner.leaves, vec![3]);
        assert!(pruner.chose_validity_leaf);
        assert_eq!(selection.prune_arrow_field(&s).dtype(), &struct_of(vec![]));
        assert_eq!(
            decoded.dtype(),
            &struct_of(vec![field(
                "s",
                struct_of(vec![field("y", ArrowDataType::Int8)])
            )])
        );
    }

    /// Lists have no leaves of their own: an element selection reads the element's leaves, and
    /// one leaf for the offsets and validity if no field of the element is read.
    #[test]
    fn prune_list_of_struct() {
        let element = struct_of(vec![
            field("a", ArrowDataType::Int64),
            field("b", ArrowDataType::Int8),
        ]);
        let l = field(
            "l",
            ArrowDataType::LargeList(Box::new(field("item", element))),
        );
        let list_of =
            |children| ArrowDataType::LargeList(Box::new(field("item", struct_of(children))));

        let selection =
            FieldSelection::element(FieldSelection::field("b".into(), FieldSelection::All));
        let mut pruner = Pruner::new(&[5, 9]);
        let decoded = pruner.prune(&l, &selection, 0);
        let output = selection.prune_arrow_field(&l);
        assert_eq!(pruner.leaves, vec![1]);
        assert!(!pruner.chose_validity_leaf);
        assert_eq!(decoded, output);
        assert_eq!(
            output.dtype(),
            &list_of(vec![field("b", ArrowDataType::Int8)])
        );

        let selection = FieldSelection::element(FieldSelection::validity_only());
        let mut pruner = Pruner::new(&[5, 9]);
        let decoded = pruner.prune(&l, &selection, 0);
        let output = selection.prune_arrow_field(&l);
        assert_eq!(pruner.leaves, vec![0]);
        assert!(pruner.chose_validity_leaf);
        assert_eq!(output.dtype(), &list_of(vec![]));
        assert_eq!(
            decoded.dtype(),
            &list_of(vec![field("a", ArrowDataType::Int64)])
        );
    }

    /// The choice is linear in the depth of nesting (it was exponential).
    #[test]
    fn validity_leaf_choice_deep_nesting() {
        let depth = 200;
        let mut dtype = ArrowDataType::Int64;
        for _ in 0..depth {
            dtype = struct_of(vec![field("p", dtype), field("q", ArrowDataType::Int64)]);
        }
        let s = field("s", dtype);
        // Leaves: the innermost `p` first, then the `q`s from the inside out. Make the
        // innermost `p` the cheapest.
        let mut leaf_sizes = vec![10; depth + 1];
        leaf_sizes[0] = 1;

        let mut pruner = Pruner::new(&leaf_sizes);
        let decoded = pruner.prune(&s, &FieldSelection::validity_only(), 0);
        assert_eq!(pruner.leaves, vec![0]);
        assert_eq!(n_columns(decoded.dtype()), 1);
    }
}
