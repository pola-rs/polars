use std::sync::Arc;

use polars_core::error::{PolarsResult, polars_bail, polars_ensure};
use polars_core::prelude::{DataType, Field, PlHashMap, PlHashSet};
use polars_core::schema::Schema;
use polars_utils::arena::{Arena, Node};

use super::{AExpr, IR, OptimizationRule};
use crate::dsl::{FileSinkType, FileType, PartitionSinkTypeIR, PartitionVariantIR, SinkTypeIR};
use crate::plans::Context;
use crate::plans::conversion::get_schema;

pub struct TypeCheckRule;

impl OptimizationRule for TypeCheckRule {
    fn optimize_plan(
        &mut self,
        ir_arena: &mut Arena<IR>,
        expr_arena: &mut Arena<AExpr>,
        node: Node,
    ) -> PolarsResult<Option<IR>> {
        let ir = ir_arena.get(node);
        match ir {
            IR::Scan {
                predicate: Some(predicate),
                ..
            } => {
                let input_schema = get_schema(ir_arena, node);
                let dtype = predicate.dtype(input_schema.as_ref(), Context::Default, expr_arena)?;

                polars_ensure!(
                    matches!(dtype, DataType::Boolean | DataType::Unknown(_)),
                    InvalidOperation: "filter predicate must be of type `Boolean`, got `{dtype:?}`"
                );

                Ok(None)
            },
            IR::Filter { predicate, .. } => {
                let input_schema = get_schema(ir_arena, node);
                let dtype = predicate.dtype(input_schema.as_ref(), Context::Default, expr_arena)?;

                polars_ensure!(
                    matches!(dtype, DataType::Boolean | DataType::Unknown(_)),
                    InvalidOperation: "filter predicate must be of type `Boolean`, got `{dtype:?}`"
                );

                Ok(None)
            },
            #[cfg(feature = "parquet")]
            IR::Sink { input: _, payload } => {
                use polars_io::prelude::{
                    ChildFieldOverwrites, ParquetFieldOverwrites, ParquetWriteOptions,
                };

                fn type_check_parquet_field_overwrites(
                    field_overwrites: &[ParquetFieldOverwrites],
                    schema: &Schema,
                ) -> PolarsResult<()> {
                    enum Item<'a> {
                        /// List / Array
                        ListLike(&'a DataType, &'a ParquetFieldOverwrites),
                        Struct(&'a [Field], &'a [ParquetFieldOverwrites]),
                    }

                    let mut stack = Vec::new();

                    fn push_children<'a>(
                        stack: &mut Vec<Item<'a>>,
                        children: &'a ChildFieldOverwrites,
                        dtype: &'a DataType,
                    ) -> PolarsResult<()> {
                        match children {
                            ChildFieldOverwrites::None => {},
                            ChildFieldOverwrites::ListLike(child_overwrites) => {
                                let Some(child_dtype) = dtype.inner_dtype() else {
                                    polars_bail!(InvalidOperation: "cannot give a parquet field overwrite with a single child to a non-list / non-array column");
                                };
                                stack.push(Item::ListLike(child_dtype, child_overwrites.as_ref()));
                            },
                            ChildFieldOverwrites::Struct(child_overwrites) => {
                                let DataType::Struct(fields) = dtype else {
                                    polars_bail!(InvalidOperation: "cannot give parquet field overwrite with multiple children to a non-struct column");
                                };
                                stack.push(Item::Struct(
                                    fields.as_slice(),
                                    child_overwrites.as_slice(),
                                ));
                            },
                        }
                        Ok(())
                    }

                    let mut fields_lut = PlHashMap::default();
                    let mut seen = PlHashSet::default();

                    for o in field_overwrites {
                        let Some(name) = &o.name else {
                            polars_bail!(InvalidOperation: "cannot do a top-level parquet field overwrite without name");
                        };

                        let dtype = schema.try_get(name.as_str())?;

                        if !seen.insert(name.as_str()) {
                            polars_bail!(InvalidOperation: "duplicate parquet field overwrite for struct field `{name}`");
                        }

                        push_children(&mut stack, &o.children, dtype)?;
                    }

                    while let Some(item) = stack.pop() {
                        match item {
                            Item::ListLike(dt, o) => {
                                if o.name.is_some() {
                                    polars_bail!(InvalidOperation: "parquet field overwrite list child cannot have name");
                                };
                                push_children(&mut stack, &o.children, dt)?;
                            },
                            Item::Struct(fields, os) => {
                                // @NOTE: Avoid quadratic behavior through HashMap.
                                fields_lut.clear();
                                seen.clear();

                                fields_lut.extend(fields.iter().map(|f| (f.name().as_str(), f)));
                                for o in os {
                                    let Some(name) = &o.name else {
                                        polars_bail!(InvalidOperation: "cannot do a struct child parquet field overwrite without name");
                                    };

                                    let Some(field) = fields_lut.get(name.as_str()) else {
                                        polars_bail!(InvalidOperation: "cannot find parquet field overwrite struct field `{name}`");
                                    };

                                    if !seen.insert(name.as_str()) {
                                        polars_bail!(InvalidOperation: "duplicate parquet field overwrite for struct field `{name}`");
                                    }

                                    push_children(&mut stack, &o.children, field.dtype())?;
                                }
                            },
                        }
                    }

                    Ok(())
                }

                match payload {
                    SinkTypeIR::File(FileSinkType {
                        file_type: FileType::Parquet(write_options @ ParquetWriteOptions { .. }),
                        ..
                    }) if !write_options.field_overwrites.is_empty() => {
                        let input_schema = get_schema(ir_arena, node);
                        type_check_parquet_field_overwrites(
                            &write_options.field_overwrites,
                            &input_schema,
                        )?;
                    },
                    SinkTypeIR::Partition(PartitionSinkTypeIR {
                        file_type: FileType::Parquet(write_options @ ParquetWriteOptions { .. }),
                        variant,
                        ..
                    }) if !write_options.field_overwrites.is_empty() => {
                        let mut input_schema = get_schema(ir_arena, node);

                        if let PartitionVariantIR::ByKey {
                            key_exprs,
                            include_key,
                        }
                        | PartitionVariantIR::Parted {
                            key_exprs,
                            include_key,
                        } = variant
                        {
                            let mut input_schema_mut = input_schema.as_ref().as_ref().clone();
                            for e in key_exprs {
                                let field =
                                    e.field(&input_schema_mut, Context::Default, expr_arena)?;
                                if *include_key {
                                    input_schema_mut.insert(field.name, field.dtype);
                                } else {
                                    input_schema_mut.remove(&field.name);
                                }
                            }
                            input_schema = std::borrow::Cow::Owned(Arc::new(input_schema_mut));
                        }

                        type_check_parquet_field_overwrites(
                            &write_options.field_overwrites,
                            &input_schema,
                        )?;
                    },
                    _ => {},
                }
                Ok(None)
            },
            _ => Ok(None),
        }
    }
}
