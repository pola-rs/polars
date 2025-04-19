use std::ops::Deref;
use std::sync::Mutex;

use arrow::datatypes::ArrowSchemaRef;
use either::Either;
use polars_core::prelude::*;
use polars_utils::format_pl_smallstr;
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
            schema: schema.clone(),
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

#[cfg(feature = "streaming")]
fn estimate_sizes(
    known_size: Option<usize>,
    estimated_size: usize,
    filter_count: usize,
) -> (Option<usize>, usize) {
    match (known_size, filter_count) {
        (Some(known_size), 0) => (Some(known_size), estimated_size),
        (None, 0) => (None, estimated_size),
        (_, _) => (
            None,
            (estimated_size as f32 * 0.9f32.powf(filter_count as f32)) as usize,
        ),
    }
}

#[cfg(feature = "streaming")]
pub fn set_estimated_row_counts(
    root: Node,
    lp_arena: &mut Arena<IR>,
    expr_arena: &Arena<AExpr>,
    mut _filter_count: usize,
    scratch: &mut Vec<Node>,
) -> (Option<usize>, usize, usize) {
    use IR::*;

    fn apply_slice(out: &mut (Option<usize>, usize, usize), slice: Option<(i64, usize)>) {
        if let Some((_, len)) = slice {
            out.0 = out.0.map(|known_size| std::cmp::min(len, known_size));
            out.1 = std::cmp::min(len, out.1);
        }
    }

    match lp_arena.get(root) {
        Filter { predicate, input } => {
            _filter_count += expr_arena
                .iter(predicate.node())
                .filter(|(_, ae)| matches!(ae, AExpr::BinaryExpr { .. }))
                .count()
                + 1;
            set_estimated_row_counts(*input, lp_arena, expr_arena, _filter_count, scratch)
        },
        Slice { input, len, .. } => {
            let len = *len as usize;
            let mut out =
                set_estimated_row_counts(*input, lp_arena, expr_arena, _filter_count, scratch);
            apply_slice(&mut out, Some((0, len)));
            out
        },
        Union { .. } => {
            if let Union {
                inputs,
                mut options,
            } = lp_arena.take(root)
            {
                let mut sum_output = (None, 0usize);
                for input in &inputs {
                    let mut out =
                        set_estimated_row_counts(*input, lp_arena, expr_arena, 0, scratch);
                    if let Some((_offset, len)) = options.slice {
                        apply_slice(&mut out, Some((0, len)))
                    }
                    // todo! deal with known as well
                    let out = estimate_sizes(out.0, out.1, out.2);
                    sum_output.1 = sum_output.1.saturating_add(out.1);
                }
                options.rows = sum_output;
                lp_arena.replace(root, Union { inputs, options });
                (sum_output.0, sum_output.1, 0)
            } else {
                unreachable!()
            }
        },
        Join { .. } => {
            if let Join {
                input_left,
                input_right,
                mut options,
                schema,
                left_on,
                right_on,
            } = lp_arena.take(root)
            {
                let mut_options = Arc::make_mut(&mut options);
                let (known_size, estimated_size, filter_count_left) =
                    set_estimated_row_counts(input_left, lp_arena, expr_arena, 0, scratch);
                mut_options.rows_left =
                    estimate_sizes(known_size, estimated_size, filter_count_left);
                let (known_size, estimated_size, filter_count_right) =
                    set_estimated_row_counts(input_right, lp_arena, expr_arena, 0, scratch);
                mut_options.rows_right =
                    estimate_sizes(known_size, estimated_size, filter_count_right);

                let mut out = match options.args.how {
                    JoinType::Left => {
                        let (known_size, estimated_size) = options.rows_left;
                        (known_size, estimated_size, filter_count_left)
                    },
                    JoinType::Cross | JoinType::Full => {
                        let (known_size_left, estimated_size_left) = options.rows_left;
                        let (known_size_right, estimated_size_right) = options.rows_right;
                        match (known_size_left, known_size_right) {
                            (Some(l), Some(r)) => {
                                (Some(l * r), estimated_size_left, estimated_size_right)
                            },
                            _ => (None, estimated_size_left * estimated_size_right, 0),
                        }
                    },
                    _ => {
                        let (known_size_left, estimated_size_left) = options.rows_left;
                        let (known_size_right, estimated_size_right) = options.rows_right;
                        if estimated_size_left > estimated_size_right {
                            (known_size_left, estimated_size_left, 0)
                        } else {
                            (known_size_right, estimated_size_right, 0)
                        }
                    },
                };
                apply_slice(&mut out, options.args.slice);
                lp_arena.replace(
                    root,
                    Join {
                        input_left,
                        input_right,
                        options,
                        schema,
                        left_on,
                        right_on,
                    },
                );
                out
            } else {
                unreachable!()
            }
        },
        DataFrameScan { df, .. } => {
            let len = df.height();
            (Some(len), len, _filter_count)
        },
        Scan { file_info, .. } => {
            let (known_size, estimated_size) = file_info.row_estimation;
            (known_size, estimated_size, _filter_count)
        },
        #[cfg(feature = "python")]
        PythonScan { .. } => {
            // TODO! get row estimation.
            (None, usize::MAX, _filter_count)
        },
        lp => {
            lp.copy_inputs(scratch);
            let mut sum_output = (None, 0, 0);
            while let Some(input) = scratch.pop() {
                let out =
                    set_estimated_row_counts(input, lp_arena, expr_arena, _filter_count, scratch);
                sum_output.1 += out.1;
                sum_output.2 += out.2;
                sum_output.0 = match sum_output.0 {
                    None => out.0,
                    p => p,
                };
            }
            sum_output
        },
    }
}

pub(crate) fn det_join_schema(
    schema_left: &SchemaRef,
    schema_right: &SchemaRef,
    left_on: &[ExprIR],
    right_on: &[ExprIR],
    options: &JoinOptions,
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
                let field = e.field(schema_left, Context::Default, expr_arena)?;
                join_on_left.insert(field.name);
            }

            let mut join_on_right: PlHashSet<_> = PlHashSet::with_capacity(right_on.len());
            for e in right_on {
                let field = e.field(schema_right, Context::Default, expr_arena)?;
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
        _how => {
            let mut new_schema = Schema::with_capacity(schema_left.len() + schema_right.len())
                .hstack(schema_left.iter_fields())?;

            let is_coalesced = options.args.should_coalesce();

            let mut _asof_pre_added_rhs_keys: PlHashSet<PlSmallStr> = PlHashSet::new();

            // Handles coalescing of asof-joins.
            // Asof joins are not equi-joins
            // so the columns that are joined on, may have different
            // values so if the right has a different name, it is added to the schema
            #[cfg(feature = "asof_join")]
            if matches!(_how, JoinType::AsOf(_)) {
                for (left_on, right_on) in left_on.iter().zip(right_on) {
                    let field_left = left_on.field(schema_left, Context::Default, expr_arena)?;
                    let field_right = right_on.field(schema_right, Context::Default, expr_arena)?;

                    if is_coalesced && field_left.name != field_right.name {
                        _asof_pre_added_rhs_keys.insert(field_right.name.clone());

                        if schema_left.contains(&field_right.name) {
                            new_schema.with_column(
                                _join_suffix_name(&field_right.name, options.args.suffix()),
                                field_right.dtype,
                            );
                        } else {
                            new_schema.with_column(field_right.name, field_right.dtype);
                        }
                    }
                }
            }

            let mut join_on_right: PlHashSet<_> = PlHashSet::with_capacity(right_on.len());
            for e in right_on {
                let field = e.field(schema_right, Context::Default, expr_arena)?;
                join_on_right.insert(field.name);
            }

            for (name, dtype) in schema_right.iter() {
                #[cfg(feature = "asof_join")]
                {
                    if let JoinType::AsOf(asof_options) = &options.args.how {
                        // Asof adds keys earlier
                        if _asof_pre_added_rhs_keys.contains(name) {
                            continue;
                        }

                        // Asof join by columns are coalesced
                        if asof_options
                            .right_by
                            .as_deref()
                            .is_some_and(|x| x.contains(name))
                        {
                            // Do not add suffix. The column of the left table will be used
                            continue;
                        }
                    }
                }

                if join_on_right.contains(name.as_str()) && is_coalesced {
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
