use polars_core::prelude::*;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::prelude::*;

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FileInfo {
    pub schema: SchemaRef,
    // - known size
    // - estimated size
    pub row_estimation: (Option<usize>, usize),
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
    lp_arena: &mut Arena<ALogicalPlan>,
    expr_arena: &Arena<AExpr>,
    mut _filter_count: usize,
) -> (Option<usize>, usize, usize) {
    use ALogicalPlan::*;

    fn apply_slice(out: &mut (Option<usize>, usize, usize), slice: Option<(i64, usize)>) {
        if let Some((_, len)) = slice {
            out.0 = out.0.map(|known_size| std::cmp::min(len, known_size));
            out.1 = std::cmp::min(len, out.1);
        }
    }

    match lp_arena.get(root) {
        Selection { predicate, input } => {
            _filter_count += expr_arena
                .iter(*predicate)
                .filter(|(_, ae)| matches!(ae, AExpr::BinaryExpr { .. }))
                .count()
                + 1;
            set_estimated_row_counts(*input, lp_arena, expr_arena, _filter_count)
        }
        Slice { input, len, .. } => {
            let len = *len as usize;
            let mut out = set_estimated_row_counts(*input, lp_arena, expr_arena, _filter_count);
            apply_slice(&mut out, Some((0, len)));
            out
        }
        Union { .. } => {
            if let Union {
                inputs,
                mut options,
            } = lp_arena.take(root)
            {
                let mut sum_output = (None, 0);
                for input in &inputs {
                    let mut out = set_estimated_row_counts(*input, lp_arena, expr_arena, 0);
                    if options.slice {
                        apply_slice(&mut out, Some((0, options.slice_len as usize)))
                    }
                    // todo! deal with known as well
                    let out = estimate_sizes(out.0, out.1, out.2);
                    sum_output.1 += out.1;
                }
                options.rows = sum_output;
                lp_arena.replace(root, Union { inputs, options });
                (sum_output.0, sum_output.1, 0)
            } else {
                unreachable!()
            }
        }
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
                let (known_size, estimated_size, filter_count_left) =
                    set_estimated_row_counts(input_left, lp_arena, expr_arena, 0);
                options.rows_left = estimate_sizes(known_size, estimated_size, filter_count_left);
                let (known_size, estimated_size, filter_count_right) =
                    set_estimated_row_counts(input_right, lp_arena, expr_arena, 0);
                options.rows_right = estimate_sizes(known_size, estimated_size, filter_count_right);

                let mut out = match options.how {
                    JoinType::Left => {
                        let (known_size, estimated_size) = options.rows_left;
                        (known_size, estimated_size, filter_count_left)
                    }
                    JoinType::Cross | JoinType::Outer => {
                        let (known_size_left, estimated_size_left) = options.rows_left;
                        let (known_size_right, estimated_size_right) = options.rows_right;
                        match (known_size_left, known_size_right) {
                            (Some(l), Some(r)) => {
                                (Some(l * r), estimated_size_left, estimated_size_right)
                            }
                            _ => (None, estimated_size_left * estimated_size_right, 0),
                        }
                    }
                    _ => {
                        let (known_size_left, estimated_size_left) = options.rows_left;
                        let (known_size_right, estimated_size_right) = options.rows_right;
                        if estimated_size_left > estimated_size_right {
                            (known_size_left, estimated_size_left, 0)
                        } else {
                            (known_size_right, estimated_size_right, 0)
                        }
                    }
                };
                apply_slice(&mut out, options.slice);
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
        }
        DataFrameScan { df, .. } => {
            let len = df.height();
            (Some(len), len, _filter_count)
        }
        #[cfg(feature = "csv-file")]
        CsvScan { file_info, .. } => {
            let (known_size, estimated_size) = file_info.row_estimation;
            (known_size, estimated_size, _filter_count)
        }
        #[cfg(feature = "ipc")]
        IpcScan { file_info, .. } => {
            let (known_size, estimated_size) = file_info.row_estimation;
            (known_size, estimated_size, _filter_count)
        }
        #[cfg(feature = "parquet")]
        ParquetScan { file_info, .. } => {
            let (known_size, estimated_size) = file_info.row_estimation;
            (known_size, estimated_size, _filter_count)
        }
        #[cfg(feature = "python")]
        PythonScan { .. } => {
            // TODO! get row estimation.
            (None, usize::MAX, _filter_count)
        }
        AnonymousScan { options, .. } => {
            let size = options.n_rows;
            (size, size.unwrap_or(usize::MAX), _filter_count)
        }
        lp => {
            let input = lp.get_input().unwrap();
            set_estimated_row_counts(input, lp_arena, expr_arena, _filter_count)
        }
    }
}

pub(crate) fn det_join_schema(
    schema_left: &SchemaRef,
    schema_right: &SchemaRef,
    left_on: &[Expr],
    right_on: &[Expr],
    options: &JoinOptions,
) -> PolarsResult<SchemaRef> {
    match options.how {
        // semi and anti joins are just filtering operations
        // the schema will never change.
        #[cfg(feature = "semi_anti_join")]
        JoinType::Semi | JoinType::Anti => Ok(schema_left.clone()),
        _ => {
            // column names of left table
            let mut names: PlHashSet<&str> =
                PlHashSet::with_capacity(schema_left.len() + schema_right.len());
            let mut new_schema = Schema::with_capacity(schema_left.len() + schema_right.len());

            for (name, dtype) in schema_left.iter() {
                names.insert(name.as_str());
                new_schema.with_column(name.to_string(), dtype.clone());
            }

            // make sure that expression are assigned to the schema
            // an expression can have an alias, and change a dtype.
            // we only do this for the left hand side as the right hand side
            // is dropped.
            let mut arena = Arena::with_capacity(8);
            for e in left_on {
                let field = e.to_field_amortized(schema_left, Context::Default, &mut arena)?;
                new_schema.with_column(field.name, field.dtype);
                arena.clear();
            }
            // except in asof joins. Asof joins are not equi-joins
            // so the columns that are joined on, may have different
            // values so if the right has a different name, it is added to the schema
            #[cfg(feature = "asof_join")]
            if let JoinType::AsOf(_) = &options.how {
                for (left_on, right_on) in left_on.iter().zip(right_on) {
                    let field_left =
                        left_on.to_field_amortized(schema_left, Context::Default, &mut arena)?;
                    let field_right =
                        right_on.to_field_amortized(schema_right, Context::Default, &mut arena)?;
                    if field_left.name != field_right.name {
                        if schema_left.contains(&field_right.name) {
                            use polars_core::frame::hash_join::_join_suffix_name;
                            new_schema.with_column(
                                _join_suffix_name(&field_right.name, options.suffix.as_ref()),
                                field_right.dtype,
                            );
                        } else {
                            new_schema.with_column(field_right.name, field_right.dtype);
                        }
                    }
                }
            }

            let mut right_names: PlHashSet<_> = PlHashSet::with_capacity(right_on.len());
            for e in right_on {
                let field = e.to_field_amortized(schema_right, Context::Default, &mut arena)?;
                right_names.insert(field.name);
            }

            for (name, dtype) in schema_right.iter() {
                if !right_names.contains(name.as_str()) {
                    if names.contains(name.as_str()) {
                        #[cfg(feature = "asof_join")]
                        if let JoinType::AsOf(asof_options) = &options.how {
                            if let (Some(left_by), Some(right_by)) =
                                (&asof_options.left_by, &asof_options.right_by)
                            {
                                {
                                    // Do not add suffix. The column of the left table will be used
                                    if left_by.contains(name) && right_by.contains(name) {
                                        continue;
                                    }
                                }
                            }
                        }

                        let new_name = format!("{}{}", name, options.suffix.as_ref());
                        new_schema.with_column(new_name, dtype.clone());
                    } else {
                        new_schema.with_column(name.to_string(), dtype.clone());
                    }
                }
            }

            Ok(Arc::new(new_schema))
        }
    }
}
