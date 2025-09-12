use std::borrow::Cow;
use std::ops::Deref;
use std::sync::Mutex;

use arrow::datatypes::ArrowSchemaRef;
use either::Either;
use polars_core::prelude::*;
use polars_utils::idx_vec::UnitVec;
use polars_utils::{format_pl_smallstr, unitvec};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::prelude::*;

impl DslPlan {
    // Warning! This should not be used on the DSL internally.
    // All schema resolving should be done during conversion to [`IR`].

    /// Compute the schema. This requires conversion to [`IR`] and type-resolving.
    pub fn compute_schema(&self) -> PolarsResult<SchemaRef> {
        let mut lp_arena = Default::default();
        let mut expr_arena = Default::default();
        let node = to_alp(
            self.clone(),
            &mut expr_arena,
            &mut lp_arena,
            &mut OptFlags::schema_only(),
        )?;

        Ok(lp_arena.get(node).schema(&lp_arena).into_owned())
    }
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub struct FileInfo {
    /// Schema of the physical file.
    ///
    /// Notes:
    /// - Does not include logical columns like `include_file_path` and row index.
    /// - Always includes all hive columns.
    pub schema: SchemaRef,
    /// Stores the schema used for the reader, as the main schema can contain
    /// extra hive columns.
    pub reader_schema: Option<Either<ArrowSchemaRef, SchemaRef>>,
    /// - known size
    /// - estimated size (set to usize::max if unknown).
    pub row_estimation: (Option<usize>, usize),
}

// Manual default because `row_estimation.1` needs to be `usize::MAX`.
impl Default for FileInfo {
    fn default() -> Self {
        FileInfo {
            schema: Default::default(),
            reader_schema: None,
            row_estimation: (None, usize::MAX),
        }
    }
}

impl FileInfo {
    /// Constructs a new [`FileInfo`].
    pub fn new(
        schema: SchemaRef,
        reader_schema: Option<Either<ArrowSchemaRef, SchemaRef>>,
        row_estimation: (Option<usize>, usize),
    ) -> Self {
        Self {
            schema,
            reader_schema,
            row_estimation,
        }
    }

    /// Merge the [`Schema`] of a [`HivePartitions`] with the schema of this [`FileInfo`].
    pub fn update_schema_with_hive_schema(&mut self, hive_schema: SchemaRef) {
        let schema = Arc::make_mut(&mut self.schema);

        for field in hive_schema.iter_fields() {
            if let Some(existing) = schema.get_mut(&field.name) {
                *existing = field.dtype().clone();
            } else {
                schema
                    .insert_at_index(schema.len(), field.name, field.dtype.clone())
                    .unwrap();
            }
        }
    }
}

pub(crate) fn det_join_schema(
    schema_left: &SchemaRef,
    schema_right: &SchemaRef,
    left_on: &[ExprIR],
    right_on: &[ExprIR],
    options: &JoinOptionsIR,
    expr_arena: &Arena<AExpr>,
) -> PolarsResult<SchemaRef> {
    match &options.args.how {
        // semi and anti joins are just filtering operations
        // the schema will never change.
        #[cfg(feature = "semi_anti_join")]
        JoinType::Semi | JoinType::Anti => Ok(schema_left.clone()),
        // Right-join with coalesce enabled will coalesce LHS columns into RHS columns (i.e. LHS columns
        // are removed). This is the opposite of what a left join does so it has its own codepath.
        //
        // E.g. df(cols=[A, B]).right_join(df(cols=[A, B]), on=A, coalesce=True)
        //
        // will result in
        //
        // df(cols=[B, A, B_right])
        JoinType::Right if options.args.should_coalesce() => {
            // Get join names.
            let mut join_on_left: PlHashSet<_> = PlHashSet::with_capacity(left_on.len());
            for e in left_on {
                let field = e.field(schema_left, expr_arena)?;
                join_on_left.insert(field.name);
            }

            let mut join_on_right: PlHashSet<_> = PlHashSet::with_capacity(right_on.len());
            for e in right_on {
                let field = e.field(schema_right, expr_arena)?;
                join_on_right.insert(field.name);
            }

            // For the error message
            let mut suffixed = None;

            let new_schema = Schema::with_capacity(schema_left.len() + schema_right.len())
                // Columns from left, excluding those used as join keys
                .hstack(schema_left.iter().filter_map(|(name, dtype)| {
                    if join_on_left.contains(name) {
                        return None;
                    }

                    Some((name.clone(), dtype.clone()))
                }))?
                // Columns from right
                .hstack(schema_right.iter().map(|(name, dtype)| {
                    suffixed = None;

                    let in_left_schema = schema_left.contains(name.as_str());
                    let is_coalesced = join_on_left.contains(name.as_str());

                    if in_left_schema && !is_coalesced {
                        suffixed = Some(format_pl_smallstr!("{}{}", name, options.args.suffix()));
                        (suffixed.clone().unwrap(), dtype.clone())
                    } else {
                        (name.clone(), dtype.clone())
                    }
                }))
                .map_err(|e| {
                    if let Some(column) = suffixed {
                        join_suffix_duplicate_help_msg(&column)
                    } else {
                        e
                    }
                })?;

            Ok(Arc::new(new_schema))
        },
        how => {
            let mut new_schema = Schema::with_capacity(schema_left.len() + schema_right.len())
                .hstack(schema_left.iter_fields())?;

            let is_coalesced = options.args.should_coalesce();

            let mut join_on_right: PlIndexSet<_> = PlIndexSet::with_capacity(right_on.len());
            for e in right_on {
                let field = e.field(schema_right, expr_arena)?;
                join_on_right.insert(field.name);
            }

            let mut right_by: PlHashSet<&PlSmallStr> = PlHashSet::default();
            #[cfg(feature = "asof_join")]
            if let JoinType::AsOf(asof_options) = &options.args.how {
                if let Some(v) = &asof_options.right_by {
                    right_by.extend(v.iter());
                }
            }

            for (name, dtype) in schema_right.iter() {
                // Asof join by columns are coalesced
                if right_by.contains(name) {
                    // Do not add suffix. The column of the left table will be used
                    continue;
                }

                if is_coalesced
                    && let Some(idx) = join_on_right.get_index_of(name)
                    && {
                        let mut need_to_include_column = false;

                        // Handles coalescing of asof-joins.
                        // Asof joins are not equi-joins
                        // so the columns that are joined on, may have different
                        // values so if the right has a different name, it is added to the schema
                        #[cfg(feature = "asof_join")]
                        if matches!(how, JoinType::AsOf(_)) {
                            let field_left = left_on[idx].field(schema_left, expr_arena)?;
                            need_to_include_column = field_left.name != name;
                        }

                        !need_to_include_column
                    }
                {
                    // Column will be coalesced into an already added LHS column.
                    continue;
                }

                // For the error message.
                let mut suffixed = None;
                let (name, dtype) = if schema_left.contains(name) {
                    suffixed = Some(format_pl_smallstr!("{}{}", name, options.args.suffix()));
                    (suffixed.clone().unwrap(), dtype.clone())
                } else {
                    (name.clone(), dtype.clone())
                };

                new_schema.try_insert(name, dtype).map_err(|e| {
                    if let Some(column) = suffixed {
                        join_suffix_duplicate_help_msg(&column)
                    } else {
                        e
                    }
                })?;
            }

            Ok(Arc::new(new_schema))
        },
    }
}

fn join_suffix_duplicate_help_msg(column_name: &str) -> PolarsError {
    polars_err!(
        Duplicate:
        "\
column with name '{}' already exists

You may want to try:
- renaming the column prior to joining
- using the `suffix` parameter to specify a suffix different to the default one ('_right')",
        column_name
    )
}

// We don't use an `Arc<Mutex>` because caches should live in different query plans.
// For that reason we have a specialized deep clone.
#[derive(Default)]
pub struct CachedSchema(Mutex<Option<SchemaRef>>);

impl AsRef<Mutex<Option<SchemaRef>>> for CachedSchema {
    fn as_ref(&self) -> &Mutex<Option<SchemaRef>> {
        &self.0
    }
}

impl Deref for CachedSchema {
    type Target = Mutex<Option<SchemaRef>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Clone for CachedSchema {
    fn clone(&self) -> Self {
        let inner = self.0.lock().unwrap();
        Self(Mutex::new(inner.clone()))
    }
}

impl CachedSchema {
    pub fn get(&self) -> Option<SchemaRef> {
        self.0.lock().unwrap().clone()
    }
}

pub fn get_input(lp_arena: &Arena<IR>, lp_node: Node) -> UnitVec<Node> {
    let plan = lp_arena.get(lp_node);
    let mut inputs: UnitVec<Node> = unitvec!();

    // Used to get the schema of the input.
    if is_scan(plan) {
        inputs.push(lp_node);
    } else {
        plan.copy_inputs(&mut inputs);
    };
    inputs
}

/// Retrieves the schema of the first LP input, or that of the `lp_node` if there
/// are no inputs.
///
/// # Panics
/// Panics if this `lp_node` does not have inputs and is not a `Scan` or `PythonScan`.
pub fn get_input_schema(lp_arena: &Arena<IR>, lp_node: Node) -> Cow<'_, SchemaRef> {
    let inputs = get_input(lp_arena, lp_node);
    if inputs.is_empty() {
        // Files don't have an input, so we must take their schema.
        Cow::Borrowed(lp_arena.get(lp_node).scan_schema())
    } else {
        let input = inputs[0];
        lp_arena.get(input).schema(lp_arena)
    }
}
