//! Struct-field pruning for projection pushdown: which fields of each column are needed, the
//! expression rewrites that drop unneeded fields, and the schema refresh once dtypes narrowed.
//!
//! A node delivers at least the selected fields of a column: a consumer never relies on a field
//! being absent, as fields are read by name. Where several values of one struct column must have
//! equal dtypes (`concat`, `merge_sorted`), the inputs are pruned to the fields they have in
//! common after the schemas are refreshed.
//!
//! See `agent-docs/nested-projection-pushdown/01-field-pruning.md`.

use std::sync::Arc;

use polars_core::error::PolarsResult;
#[cfg(feature = "dtype-struct")]
use polars_core::prelude::StructFields;
use polars_core::prelude::{DataType, PlIndexMap, PlIndexSet};
use polars_core::schema::Schema;
#[allow(clippy::disallowed_types)]
use polars_utils::aliases::{PlHashMap, PlHashSet};
use polars_utils::arena::{Arena, Node};
use polars_utils::pl_str::PlSmallStr;
#[cfg(feature = "dtype-struct")]
use polars_utils::scratch_vec::ScratchVec;
use recursive::recursive;

#[cfg(feature = "dtype-struct")]
use crate::dsl::StructEvalVariant;
use crate::dsl::{EvalVariant, FieldSelection};
#[cfg(feature = "dtype-array")]
use crate::plans::IRArrayFunction;
use crate::plans::aexpr::ToFieldContext;
#[cfg(feature = "dtype-struct")]
use crate::plans::projection_height::{
    ExprHeightOptions, ExprProjectionHeight, aexpr_projection_height_rec_with,
};
use crate::plans::{
    AExpr, ExprIR, IR, IRAggExpr, IRFunctionExpr, IRListFunction, OutputName, det_join_schema,
};
#[cfg(feature = "dtype-struct")]
use crate::plans::{IRBooleanFunction, IRStructFunction};
use crate::prelude::ProjectionOptions;

/// The projected columns, each with the parts of its value that are needed. Ordered: projection
/// pushdown uses the order as the order of the projected columns.
pub type ColumnSelection = PlIndexMap<PlSmallStr, FieldSelection>;

pub trait ColumnSelectionExt {
    /// Adds column `name` with `selection`, merged with what is already selected of it. All
    /// inserts go through here: a plain `insert` would let one consumer's selection overwrite
    /// another's.
    fn insert_merged(&mut self, name: PlSmallStr, selection: FieldSelection);

    /// Adds column `name`, needed whole.
    fn insert_whole(&mut self, name: PlSmallStr) {
        self.insert_merged(name, FieldSelection::All)
    }

    fn extend_whole<I: IntoIterator<Item = PlSmallStr>>(&mut self, names: I) {
        for name in names {
            self.insert_whole(name)
        }
    }

    fn extend_merged<I: IntoIterator<Item = (PlSmallStr, FieldSelection)>>(&mut self, iter: I) {
        for (name, selection) in iter {
            self.insert_merged(name, selection)
        }
    }

    /// Whether some column is not needed whole.
    fn has_field_selections(&self) -> bool;

    /// Makes every selection canonical for its dtype in `schema` (see
    /// [`FieldSelection::normalize`]).
    fn normalize(&mut self, schema: &Schema);

    /// Selects every column whole.
    fn select_all_whole(&mut self);
}

impl ColumnSelectionExt for ColumnSelection {
    fn insert_merged(&mut self, name: PlSmallStr, selection: FieldSelection) {
        crate::dsl::insert_merged(self, name, selection)
    }

    fn has_field_selections(&self) -> bool {
        self.values().any(|s| !s.is_all())
    }

    fn normalize(&mut self, schema: &Schema) {
        for (name, selection) in self.iter_mut() {
            if selection.is_all() {
                continue;
            }

            *selection = match schema.get(name) {
                Some(dtype) => std::mem::take(selection).normalize(dtype),
                None => FieldSelection::All,
            };
        }
    }

    fn select_all_whole(&mut self) {
        for selection in self.values_mut() {
            *selection = FieldSelection::All;
        }
    }
}

/// Collects what an expression needs of its input columns, given what is needed of its value,
/// and rewrites it to drop the struct fields nobody reads.
pub(super) struct SelectionWalker<'a> {
    pub expr_arena: &'a mut Arena<AExpr>,
    /// If unset, every column is needed whole and nothing is rewritten.
    prune: bool,
    /// The schema the expression is evaluated on. Set where expressions may be rewritten; then
    /// [`Self::walk`] may return a new node.
    pub rewrite_schema: Option<&'a Schema>,
    /// Whether an expression was rewritten, changing the dtype of its value.
    pub rewrote: bool,
    /// Whether the height of the node being walked matters only up to broadcasting: it is (a
    /// struct field of) the expression of a `with_columns`.
    height_free: bool,
    /// While walking the bodies of `list.eval` / `arr.eval` (innermost last): what is needed of
    /// `pl.element()` in each, `None` while it is not read.
    element_selections: Vec<Option<FieldSelection>>,
}

impl<'a> SelectionWalker<'a> {
    pub fn new(
        expr_arena: &'a mut Arena<AExpr>,
        prune: bool,
        rewrite_schema: Option<&'a Schema>,
    ) -> Self {
        Self {
            expr_arena,
            prune,
            rewrite_schema: rewrite_schema.filter(|_| prune),
            rewrote: false,
            height_free: false,
            element_selections: Vec::new(),
        }
    }

    /// [`Self::walk`] on the expression of `e`, which takes the rewritten expression.
    /// `broadcast`: whether a value of height 1 is broadcast to the height of the frame, as in
    /// `with_columns`.
    pub fn walk_expr_ir(
        &mut self,
        e: &mut ExprIR,
        selection: &FieldSelection,
        out: &mut ColumnSelection,
        broadcast: bool,
    ) {
        self.height_free = broadcast;
        let new_node = self.walk(e.node(), selection, out);
        if new_node != e.node() {
            // The name may have come from a dropped input.
            let name = e.output_name().clone();
            e.set_node(new_node);
            e.set_alias(name);
        }
    }

    /// Adds what `node` needs of its input columns to `out`, for its value to be needed with
    /// `selection`. Returns the node of the rewritten expression, `node` if it is unchanged.
    pub fn walk(
        &mut self,
        node: Node,
        selection: &FieldSelection,
        out: &mut ColumnSelection,
    ) -> Node {
        self.walk_rec(node, selection, out, None)
    }

    /// Adds the input columns of `node` to `out`, each needed whole or for the struct fields that
    /// `node` reads. Does not rewrite.
    pub fn collect(
        expr_arena: &mut Arena<AExpr>,
        prune: bool,
        node: Node,
        out: &mut ColumnSelection,
    ) {
        let new_node =
            SelectionWalker::new(expr_arena, prune, None).walk(node, &FieldSelection::All, out);
        debug_assert_eq!(new_node, node);
    }

    /// `struct_fields`: the fields read of the struct of the innermost `StructEval`, through
    /// `pl.field()`.
    #[recursive]
    fn walk_rec(
        &mut self,
        node: Node,
        selection: &FieldSelection,
        out: &mut ColumnSelection,
        mut struct_fields: Option<&mut FieldSelection>,
    ) -> Node {
        let height_free = std::mem::take(&mut self.height_free);
        let ae = self.expr_arena.get(node);

        match ae {
            AExpr::Column(name) => {
                let selection = if self.prune {
                    selection.clone()
                } else {
                    FieldSelection::All
                };
                out.insert_merged(name.clone(), selection);
                return node;
            },
            #[cfg(feature = "dtype-struct")]
            AExpr::StructField(name) => {
                if let Some(struct_fields) = struct_fields {
                    struct_fields.insert_child(name.clone(), selection.clone());
                }
                return node;
            },
            _ => {},
        }

        if matches!(ae, AExpr::Element) {
            if let Some(element_selection) = self.element_selections.last_mut() {
                let selection = if self.prune {
                    selection.clone()
                } else {
                    FieldSelection::All
                };
                match element_selection {
                    Some(s) => s.union(selection),
                    None => *element_selection = Some(selection),
                }
            }
            return node;
        }

        if matches!(ae, AExpr::Literal(_) | AExpr::Len) {
            return node;
        }

        let ae = ae.clone();

        match ae {
            #[cfg(feature = "dtype-struct")]
            AExpr::Function {
                input,
                function: IRFunctionExpr::StructExpr(function),
                options,
            } => {
                let function_node = |input: Vec<ExprIR>, function| AExpr::Function {
                    input,
                    function: IRFunctionExpr::StructExpr(function),
                    options,
                };

                match function {
                    IRStructFunction::FieldByName(name) => {
                        let input_selection =
                            FieldSelection::field(name.clone(), selection.clone());
                        self.height_free = height_free;
                        let new_input =
                            self.walk_rec(input[0].node(), &input_selection, out, struct_fields);
                        self.with_new_inputs(
                            node,
                            function_node(input, IRStructFunction::FieldByName(name)),
                            &[new_input],
                        )
                    },

                    IRStructFunction::PrefixFields(ref affix)
                    | IRStructFunction::SuffixFields(ref affix) => {
                        let is_prefix = matches!(function, IRStructFunction::PrefixFields(_));
                        let Some(input_selection) = map_field_names(selection, |name| {
                            if is_prefix {
                                name.strip_prefix(affix.as_str())
                            } else {
                                name.strip_suffix(affix.as_str())
                            }
                        }) else {
                            return self.walk_default(node, out, struct_fields);
                        };
                        self.height_free = height_free;
                        let new_input =
                            self.walk_rec(input[0].node(), &input_selection, out, struct_fields);
                        self.with_new_inputs(node, function_node(input, function), &[new_input])
                    },

                    // The value is needed whole: exactly the fields left after the drop.
                    IRStructFunction::DropFields(..) | IRStructFunction::RenameFields(_)
                        if selection.is_all() =>
                    {
                        self.walk_default(node, out, struct_fields)
                    },

                    // The fields kept are those not dropped, read by name.
                    IRStructFunction::DropFields(_, strict) => {
                        if let Some(schema) = self.rewrite_schema {
                            // Rewritten to a selection of the kept fields that are read: the
                            // input alone would bring the dropped fields back where its fields
                            // are not pruned, e.g. into an `unnest`.
                            let Ok(DataType::Struct(kept_fields)) = self
                                .expr_arena
                                .get(node)
                                .to_dtype(&ToFieldContext::new(self.expr_arena, schema))
                            else {
                                return self.walk_default(node, out, struct_fields);
                            };
                            let evaluation = kept_fields
                                .iter()
                                .filter(|field| selection.child(field.name()).is_some())
                                .map(|field| {
                                    ExprIR::new(
                                        self.expr_arena
                                            .add(AExpr::StructField(field.name().clone())),
                                        OutputName::Alias(field.name().clone()),
                                    )
                                })
                                .collect();

                            self.height_free = height_free;
                            let new_input =
                                self.walk_rec(input[0].node(), selection, out, struct_fields);
                            self.rewrote = true;
                            self.expr_arena.add(AExpr::StructEval {
                                expr: new_input,
                                evaluation,
                                variant: StructEvalVariant::Select,
                            })
                        } else if !strict {
                            self.height_free = height_free;
                            let new_input =
                                self.walk_rec(input[0].node(), selection, out, struct_fields);
                            self.with_new_inputs(node, function_node(input, function), &[new_input])
                        } else {
                            // A strict drop errors on fields that the narrowed input lacks.
                            self.walk_default(node, out, struct_fields)
                        }
                    },

                    // Positional: rewritten to a selection of the kept fields by name.
                    IRStructFunction::RenameFields(new_names) => {
                        let Some(schema) = self.rewrite_schema else {
                            return self.walk_default(node, out, struct_fields);
                        };
                        let Ok(DataType::Struct(old_fields)) = self
                            .expr_arena
                            .get(input[0].node())
                            .to_dtype(&ToFieldContext::new(self.expr_arena, schema))
                        else {
                            return self.walk_default(node, out, struct_fields);
                        };
                        // Duplicate names raise, as without the rewrite.
                        if old_fields.len() != new_names.len()
                            || new_names.iter().collect::<PlIndexSet<_>>().len() != new_names.len()
                        {
                            return self.walk_default(node, out, struct_fields);
                        }

                        let mut input_selection = FieldSelection::validity_only();
                        let mut evaluation = Vec::new();

                        for (old, new) in old_fields.iter().zip(new_names.iter()) {
                            if let Some(field_selection) = selection.child(new) {
                                input_selection
                                    .insert_child(old.name().clone(), field_selection.clone());
                                evaluation.push(ExprIR::new(
                                    self.expr_arena.add(AExpr::StructField(old.name().clone())),
                                    OutputName::Alias(new.clone()),
                                ));
                            }
                        }

                        self.height_free = height_free;
                        let new_input =
                            self.walk_rec(input[0].node(), &input_selection, out, struct_fields);
                        self.rewrote = true;
                        self.expr_arena.add(AExpr::StructEval {
                            expr: new_input,
                            evaluation,
                            variant: StructEvalVariant::Select,
                        })
                    },

                    _ => self.walk_default(node, out, struct_fields),
                }
            },

            #[cfg(feature = "dtype-struct")]
            AExpr::Function {
                input,
                function: IRFunctionExpr::AsStruct,
                options,
            } if !selection.is_all() => self.walk_as_struct(
                node,
                input,
                options,
                selection,
                height_free,
                out,
                struct_fields,
            ),

            #[cfg(feature = "dtype-struct")]
            AExpr::Function {
                input,
                function:
                    function @ (IRFunctionExpr::Boolean(
                        IRBooleanFunction::IsNull
                        | IRBooleanFunction::IsNotNull
                        | IRBooleanFunction::HasNulls,
                    )
                    | IRFunctionExpr::NullCount),
                options,
            } => {
                let new_input = self.walk_rec(
                    input[0].node(),
                    &FieldSelection::validity_only(),
                    out,
                    struct_fields,
                );
                self.with_new_inputs(
                    node,
                    AExpr::Function {
                        input,
                        function,
                        options,
                    },
                    &[new_input],
                )
            },

            #[cfg(feature = "dtype-struct")]
            AExpr::StructEval {
                expr,
                evaluation,
                variant,
            } => self.walk_struct_eval(
                node,
                expr,
                evaluation,
                variant,
                selection,
                out,
                struct_fields,
            ),

            // Only the validity, or the row count.
            #[cfg(feature = "dtype-struct")]
            AExpr::Agg(IRAggExpr::Count {
                input,
                include_nulls,
            }) => {
                let new_input =
                    self.walk_rec(input, &FieldSelection::validity_only(), out, struct_fields);
                self.with_new_inputs(
                    node,
                    AExpr::Agg(IRAggExpr::Count {
                        input,
                        include_nulls,
                    }),
                    &[new_input],
                )
            },

            // Row selections: the value keeps its fields.
            AExpr::Agg(
                ref agg @ (IRAggExpr::First(input)
                | IRAggExpr::Last(input)
                | IRAggExpr::FirstNonNull(input)
                | IRAggExpr::LastNonNull(input)
                | IRAggExpr::Item { input, .. }),
            ) => {
                let new_input = self.walk_rec(input, selection, out, struct_fields);
                self.with_new_inputs(node, AExpr::Agg(agg.clone()), &[new_input])
            },
            // The value is a list of the input's values.
            AExpr::Agg(ref agg @ IRAggExpr::Implode { input, .. }) => {
                let input_selection = match selection {
                    FieldSelection::Element(element) => element.as_ref().clone(),
                    FieldSelection::Fields(fields) if fields.is_empty() => {
                        FieldSelection::validity_only()
                    },
                    _ => FieldSelection::All,
                };
                let new_input = self.walk_rec(input, &input_selection, out, struct_fields);
                self.with_new_inputs(node, AExpr::Agg(agg.clone()), &[new_input])
            },
            // Every branch is (part of) the value. They are rewritten to the selected fields, as
            // their dtypes must be equal and each may come back with other fields.
            AExpr::Ternary {
                truthy,
                falsy,
                predicate,
            } if !selection.is_all() => {
                match self.walk_branches(
                    &[truthy, falsy],
                    selection,
                    out,
                    struct_fields.as_deref_mut(),
                ) {
                    Some(branches) => {
                        let new_predicate =
                            self.walk_rec(predicate, &FieldSelection::All, out, struct_fields);
                        self.expr_arena.add(AExpr::Ternary {
                            truthy: branches[0],
                            falsy: branches[1],
                            predicate: new_predicate,
                        })
                    },
                    None => self.walk_default(node, out, struct_fields),
                }
            },
            AExpr::Function {
                ref input,
                function: IRFunctionExpr::FillNull | IRFunctionExpr::Coalesce,
                ..
            } if !selection.is_all() => {
                let branches: Vec<Node> = input.iter().map(|e| e.node()).collect();
                match self.walk_branches(&branches, selection, out, struct_fields.as_deref_mut()) {
                    Some(branches) => {
                        let AExpr::Function {
                            mut input,
                            function,
                            options,
                        } = ae
                        else {
                            unreachable!()
                        };
                        for (e, node) in input.iter_mut().zip(branches) {
                            e.set_node(node);
                        }
                        self.expr_arena.add(AExpr::Function {
                            input,
                            function,
                            options,
                        })
                    },
                    None => self.walk_default(node, out, struct_fields),
                }
            },
            AExpr::Filter { input, by } => {
                let new_input = self.walk_rec(input, selection, out, struct_fields.as_deref_mut());
                let new_by = self.walk_rec(by, &FieldSelection::All, out, struct_fields);
                self.with_new_inputs(node, ae, &[new_input, new_by])
            },
            AExpr::Slice {
                input,
                offset,
                length,
            } => {
                let new_input = self.walk_rec(input, selection, out, struct_fields.as_deref_mut());
                let new_offset = self.walk_rec(
                    offset,
                    &FieldSelection::All,
                    out,
                    struct_fields.as_deref_mut(),
                );
                let new_length = self.walk_rec(length, &FieldSelection::All, out, struct_fields);
                self.with_new_inputs(node, ae, &[new_input, new_offset, new_length])
            },
            AExpr::Gather { expr, idx, .. } => {
                let new_expr = self.walk_rec(expr, selection, out, struct_fields.as_deref_mut());
                let new_idx = self.walk_rec(idx, &FieldSelection::All, out, struct_fields);
                self.with_new_inputs(node, ae, &[new_expr, new_idx])
            },
            AExpr::SortBy { expr, ref by, .. } => {
                let mut new_children = Vec::with_capacity(by.len() + 1);
                new_children.push(self.walk_rec(
                    expr,
                    selection,
                    out,
                    struct_fields.as_deref_mut(),
                ));
                for by in by.clone() {
                    new_children.push(self.walk_rec(
                        by,
                        &FieldSelection::All,
                        out,
                        struct_fields.as_deref_mut(),
                    ));
                }
                self.with_new_inputs(node, ae, &new_children)
            },

            // Lists: the value is (part of) an element. Only an element read at a constant index
            // or through a plain chain of struct fields is pruned (SPARK-42163, SPARK-38529).
            AExpr::Explode { expr, .. } => {
                let new_expr = self.walk_rec(
                    expr,
                    &FieldSelection::element(selection.clone()),
                    out,
                    struct_fields,
                );
                self.with_new_inputs(node, ae, &[new_expr])
            },
            AExpr::Function {
                ref input,
                ref function,
                ..
            } if is_element_get(function) && self.is_constant_index(input[1].node()) => {
                let (list, idx) = (input[0].node(), input[1].node());
                let new_list = self.walk_rec(
                    list,
                    &FieldSelection::element(selection.clone()),
                    out,
                    struct_fields.as_deref_mut(),
                );
                let new_idx = self.walk_rec(idx, &FieldSelection::All, out, struct_fields);
                self.with_new_inputs(node, ae, &[new_list, new_idx])
            },
            // Rows, or elements of each list, selected: the value keeps the fields of the input.
            AExpr::Function {
                ref input,
                ref function,
                ..
            } if is_row_selection(function) => {
                let mut new_inputs = Vec::with_capacity(input.len());
                for (i, e) in input.iter().enumerate() {
                    let input_selection = if i == 0 {
                        selection
                    } else {
                        &FieldSelection::All
                    };
                    new_inputs.push(self.walk_rec(
                        e.node(),
                        input_selection,
                        out,
                        struct_fields.as_deref_mut(),
                    ));
                }
                self.with_new_inputs(node, ae, &new_inputs)
            },
            // Only the offsets and the validity.
            AExpr::Function {
                ref input,
                function: IRFunctionExpr::ListExpr(IRListFunction::Length),
                ..
            } => {
                let new_input = self.walk_rec(
                    input[0].node(),
                    &FieldSelection::validity_only(),
                    out,
                    struct_fields,
                );
                self.with_new_inputs(node, ae, &[new_input])
            },
            // The body is walked (not rewritten) for what it needs of `pl.element()`. In
            // `list.eval` / `arr.eval` the value's elements are the body's values; the body of
            // `list.agg` / `arr.agg` (or a non-list `cumulative_eval`) is needed whole.
            AExpr::Eval {
                expr,
                evaluation,
                variant,
            } => {
                let body_selection = match variant {
                    EvalVariant::List | EvalVariant::Array { .. } => match selection {
                        FieldSelection::Fields(fields) if fields.is_empty() => {
                            FieldSelection::validity_only()
                        },
                        selection => selection.element_selection().clone(),
                    },
                    EvalVariant::ListAgg
                    | EvalVariant::ArrayAgg
                    | EvalVariant::Cumulative { .. } => FieldSelection::All,
                };
                let element = self.walk_eval_body(
                    evaluation,
                    &body_selection,
                    out,
                    struct_fields.as_deref_mut(),
                );
                let input_selection = match variant {
                    EvalVariant::Cumulative { .. } => FieldSelection::All,
                    _ => FieldSelection::element(element),
                };
                let new_expr = self.walk_rec(expr, &input_selection, out, struct_fields);
                if new_expr == expr {
                    return node;
                }
                // `evaluation` is not an input, so it is not replaced.
                self.expr_arena.add(AExpr::Eval {
                    expr: new_expr,
                    evaluation,
                    variant,
                })
            },

            _ => self.walk_default(node, out, struct_fields),
        }
    }

    /// Whether `node` is a scalar literal.
    fn is_constant_index(&self, node: Node) -> bool {
        matches!(self.expr_arena.get(node), AExpr::Literal(lv) if lv.is_scalar())
    }

    /// Walks the body of an `eval` for its value to be needed with `selection`, without
    /// rewriting it. Returns what it needs of `pl.element()`: for its height at least.
    fn walk_eval_body(
        &mut self,
        evaluation: Node,
        selection: &FieldSelection,
        out: &mut ColumnSelection,
        struct_fields: Option<&mut FieldSelection>,
    ) -> FieldSelection {
        let rewrite_schema = self.rewrite_schema.take();
        self.element_selections.push(None);
        // `pl.field()` in the body reads the struct of an enclosing `struct.eval`.
        let new_evaluation = self.walk_rec(evaluation, selection, out, struct_fields);
        debug_assert_eq!(new_evaluation, evaluation);
        let element = self.element_selections.pop().unwrap();
        self.rewrite_schema = rewrite_schema;
        element.unwrap_or_else(FieldSelection::validity_only)
    }

    /// Walks `branches`, each (part of) the value, and prunes each to the fields `selection`
    /// selects of their common dtype. `None` if they can't be: not rewriting, or dtypes that
    /// differ or can't be resolved.
    fn walk_branches(
        &mut self,
        branches: &[Node],
        selection: &FieldSelection,
        out: &mut ColumnSelection,
        mut struct_fields: Option<&mut FieldSelection>,
    ) -> Option<Vec<Node>> {
        let schema = self.rewrite_schema?;
        let ctx = ToFieldContext::new(self.expr_arena, schema);
        let dtypes = branches
            .iter()
            .map(|b| self.expr_arena.get(*b).to_dtype(&ctx))
            .collect::<PolarsResult<Vec<DataType>>>()
            .ok()?;
        let dtype = dtypes.first()?;
        if dtypes.iter().any(|d| d != dtype) {
            return None;
        }
        let selection = selection.clone().normalize(dtype);
        if selection.is_all() {
            return None;
        }
        let pruned_dtype = selection.prune_dtype(dtype);
        // Not to a struct without fields (validity only), which a `struct.eval` would build.
        #[cfg(feature = "dtype-struct")]
        if pruned_dtype.contains_dtype_recursive(&DataType::Struct(StructFields::default())) {
            return None;
        }

        let new_branches = branches
            .iter()
            .map(|branch| {
                let new_branch =
                    self.walk_rec(*branch, &selection, out, struct_fields.as_deref_mut());
                prune_to_dtype(new_branch, dtype, &pruned_dtype, self.expr_arena)
            })
            .collect();
        self.rewrote = true;
        Some(new_branches)
    }

    /// Every child is needed whole.
    fn walk_default(
        &mut self,
        node: Node,
        out: &mut ColumnSelection,
        mut struct_fields: Option<&mut FieldSelection>,
    ) -> Node {
        let ae = self.expr_arena.get(node);

        #[cfg(feature = "dtype-struct")]
        if let AExpr::StructEval {
            expr,
            evaluation,
            variant,
        } = ae
        {
            let (expr, evaluation, variant) = (*expr, evaluation.clone(), *variant);
            return self.walk_struct_eval(
                node,
                expr,
                evaluation,
                variant,
                &FieldSelection::All,
                out,
                struct_fields,
            );
        }

        let mut children = Vec::new();
        ae.children_rev(&mut children);
        if children.is_empty() {
            return node;
        }
        children.reverse();

        let mut changed = false;
        for child in children.iter_mut() {
            let new_child = self.walk_rec(
                *child,
                &FieldSelection::All,
                out,
                struct_fields.as_deref_mut(),
            );
            changed |= new_child != *child;
            *child = new_child;
        }

        if !changed {
            return node;
        }

        let ae = self
            .expr_arena
            .get(node)
            .clone()
            .replace_children(&children);
        self.expr_arena.add(ae)
    }

    /// `ae` (the expression at `node`) with its inputs replaced, as a new node if they changed.
    fn with_new_inputs(&mut self, node: Node, ae: AExpr, new_children: &[Node]) -> Node {
        let mut children = Vec::new();
        ae.children_rev(&mut children);
        children.reverse();

        if children.as_slice() == new_children {
            return node;
        }

        self.expr_arena.add(ae.replace_children(new_children))
    }

    #[cfg(feature = "dtype-struct")]
    #[allow(clippy::too_many_arguments)]
    fn walk_as_struct(
        &mut self,
        node: Node,
        input: Vec<ExprIR>,
        options: crate::prelude::FunctionOptions,
        selection: &FieldSelection,
        height_free: bool,
        out: &mut ColumnSelection,
        mut struct_fields: Option<&mut FieldSelection>,
    ) -> Node {
        let mut keep: Vec<bool> = input
            .iter()
            .map(|e| selection.child(e.output_name()).is_some())
            .collect();

        let can_drop = if self.rewrite_schema.is_none() || keep.iter().all(|k| *k) {
            false
        } else if height_free {
            // Only the height up to broadcasting matters: keep the selected fields, or the
            // cheapest one (a literal if there is one) as a struct needs a field. An input that
            // may change the height (a `filter`, a `Range` literal) is kept too: `pl.struct`
            // raises a `ShapeError` if its height disagrees with the others'.
            let mut nodes_scratch = ScratchVec::default();
            let mut heights_scratch = ScratchVec::default();
            for (e, keep) in input.iter().zip(keep.iter_mut()) {
                if !*keep {
                    let height = aexpr_projection_height_rec_with(
                        e.node(),
                        self.expr_arena,
                        &mut nodes_scratch,
                        &mut heights_scratch,
                        ExprHeightOptions {
                            structural_unknowns_only: true,
                            // A range zipped with a column of one row sets the height too.
                            range_is_unknown: true,
                            ..Default::default()
                        },
                    );
                    *keep = matches!(height, ExprProjectionHeight::Unknown);
                }
            }
            if !keep.iter().any(|k| *k) {
                let i = input
                    .iter()
                    .position(|e| matches!(self.expr_arena.get(e.node()), AExpr::Literal(_)))
                    .unwrap_or(0);
                keep[i] = true;
            }
            true
        } else {
            // Dropping fields must not change the height of the struct: keep an input of the
            // full height. E.g. a struct of a column and a sum keeps its height without the
            // column only if the sum is not all that remains.
            // A range zipped with a column has the range's height, not necessarily the frame's.
            let mut nodes_scratch = ScratchVec::default();
            let mut heights_scratch = ScratchVec::default();
            let heights: Vec<ExprProjectionHeight> = input
                .iter()
                .map(|e| {
                    aexpr_projection_height_rec_with(
                        e.node(),
                        self.expr_arena,
                        &mut nodes_scratch,
                        &mut heights_scratch,
                        ExprHeightOptions {
                            range_is_unknown: true,
                            ..Default::default()
                        },
                    )
                })
                .collect();
            let full_height = ExprProjectionHeight::zipped_projection_height(heights.clone());
            let kept_height = ExprProjectionHeight::zipped_projection_height(
                heights
                    .iter()
                    .zip(&keep)
                    .filter(|(_, k)| **k)
                    .map(|(h, _)| *h),
            );

            // Only inputs of a known height may be dropped, and only if one of the full height
            // stays. `Unknown` (or a `Range` literal) inputs may disagree with the others, in
            // which case `pl.struct` broadcasts a unit-length one or raises a `ShapeError`, so
            // dropping or keeping them alone could change the result.
            use ExprProjectionHeight as H;
            let known = |h: &H| matches!(h, H::Column | H::Scalar);
            if !heights.iter().all(known) {
                false
            } else if kept_height == full_height && keep.iter().any(|k| *k) {
                true
            } else {
                match heights.iter().position(|h| *h == full_height) {
                    Some(i) => {
                        keep[i] = true;
                        true
                    },
                    None => false,
                }
            }
        };

        if can_drop {
            let mut new_input = Vec::with_capacity(input.len());

            for (e, keep) in input.into_iter().zip(keep) {
                if !keep {
                    continue;
                }
                // An input kept for its height alone is needed for its validity at most.
                let child_selection = selection
                    .child(e.output_name())
                    .cloned()
                    .unwrap_or_else(FieldSelection::validity_only);
                let new_node = self.walk_rec(
                    e.node(),
                    &child_selection,
                    out,
                    struct_fields.as_deref_mut(),
                );
                let mut e = e;
                e.set_node(new_node);
                new_input.push(e);
            }

            self.rewrote = true;
            return self.expr_arena.add(AExpr::Function {
                input: new_input,
                function: IRFunctionExpr::AsStruct,
                options,
            });
        }

        // Not rewritten: every field is still computed, so the unselected ones are read whole.
        let new_children: Vec<Node> = input
            .iter()
            .map(|e| {
                let child_selection = selection
                    .child(e.output_name())
                    .unwrap_or(&FieldSelection::All)
                    .clone();
                self.walk_rec(
                    e.node(),
                    &child_selection,
                    out,
                    struct_fields.as_deref_mut(),
                )
            })
            .collect();

        self.with_new_inputs(
            node,
            AExpr::Function {
                input,
                function: IRFunctionExpr::AsStruct,
                options,
            },
            &new_children,
        )
    }

    #[cfg(feature = "dtype-struct")]
    #[allow(clippy::too_many_arguments)]
    fn walk_struct_eval(
        &mut self,
        node: Node,
        expr: Node,
        evaluation: Vec<ExprIR>,
        variant: StructEvalVariant,
        selection: &FieldSelection,
        out: &mut ColumnSelection,
        struct_fields: Option<&mut FieldSelection>,
    ) -> Node {
        // The fields of the input struct read through `pl.field()`.
        let mut read_fields = FieldSelection::validity_only();
        // Columns read by the new fields, added after those of the input to keep the column order.
        let mut evaluation_out = ColumnSelection::default();
        let mut new_evaluation = Vec::with_capacity(evaluation.len());
        let mut changed = false;

        // The height of the input struct, for whether a new field may be dropped.
        let mut nodes_scratch = ScratchVec::default();
        let mut heights_scratch = ScratchVec::default();
        let options = ExprHeightOptions {
            range_is_unknown: true,
            ..Default::default()
        };
        let input_height = aexpr_projection_height_rec_with(
            expr,
            self.expr_arena,
            &mut nodes_scratch,
            &mut heights_scratch,
            options,
        );

        for e in evaluation.iter() {
            let field_selection = match selection.child(e.output_name()) {
                Some(s) => s.clone(),
                // A new field nobody reads, if it cannot broadcast the struct to another height
                // (e.g. a column added to a scalar struct).
                None if self.rewrite_schema.is_some() && {
                    use ExprProjectionHeight as H;
                    let height = aexpr_projection_height_rec_with(
                        e.node(),
                        self.expr_arena,
                        &mut nodes_scratch,
                        &mut heights_scratch,
                        ExprHeightOptions {
                            struct_field: input_height,
                            ..options
                        },
                    );
                    height == H::Scalar || (height == H::Column && input_height == H::Column)
                } =>
                {
                    changed = true;
                    continue;
                },
                None => FieldSelection::All,
            };

            let new_node = self.walk_rec(
                e.node(),
                &field_selection,
                &mut evaluation_out,
                Some(&mut read_fields),
            );
            changed |= new_node != e.node();
            let mut e = e.clone();
            e.set_node(new_node);
            new_evaluation.push(e);
        }

        let input_selection = match (variant, selection) {
            (StructEvalVariant::WithFields, FieldSelection::All) => FieldSelection::All,
            (StructEvalVariant::WithFields, FieldSelection::Fields(fields)) => {
                // The fields passed through from the input, and those the new fields read.
                let mut input_selection = FieldSelection::Fields(Box::new(
                    fields
                        .iter()
                        .filter(|(name, _)| !evaluation.iter().any(|e| e.output_name() == *name))
                        .map(|(name, s)| (name.clone(), s.clone()))
                        .collect(),
                ));
                input_selection.union(read_fields);
                input_selection
            },
            (StructEvalVariant::Select, _) => read_fields,
            (StructEvalVariant::WithFields, FieldSelection::Element(_)) => FieldSelection::All,
        };

        let new_expr = self.walk_rec(expr, &input_selection, out, struct_fields);
        changed |= new_expr != expr;
        out.extend_merged(evaluation_out);

        if !changed {
            return node;
        }

        if new_evaluation.len() != evaluation.len() {
            self.rewrote = true;
        }

        // No new field is read: the input is the result (an empty `with_fields` is rejected by
        // the streaming engine).
        if matches!(variant, StructEvalVariant::WithFields) && new_evaluation.is_empty() {
            return new_expr;
        }

        self.expr_arena.add(AExpr::StructEval {
            expr: new_expr,
            evaluation: new_evaluation,
            variant,
        })
    }
}

/// Whether `function` selects rows of its first input (or elements of each of its lists), so
/// that the value's rows (or elements) are some of the input's.
fn is_row_selection(function: &IRFunctionExpr) -> bool {
    match function {
        IRFunctionExpr::Reverse
        | IRFunctionExpr::Shift
        | IRFunctionExpr::DropNulls
        | IRFunctionExpr::MinBy { .. }
        | IRFunctionExpr::MaxBy { .. } => true,
        IRFunctionExpr::ListExpr(f) => match f {
            IRListFunction::Slice | IRListFunction::Shift => true,
            #[cfg(feature = "list_gather")]
            IRListFunction::Gather(_) | IRListFunction::GatherEvery => true,
            #[cfg(feature = "list_drop_nulls")]
            IRListFunction::DropNulls => true,
            #[cfg(feature = "list_sample")]
            IRListFunction::Sample { .. } => true,
            _ => false,
        },
        _ => false,
    }
}

/// Whether `function` is `list.get` or `arr.get` (and so `first` / `last`).
fn is_element_get(function: &IRFunctionExpr) -> bool {
    match function {
        IRFunctionExpr::ListExpr(IRListFunction::Get(_)) => true,
        #[cfg(feature = "dtype-array")]
        IRFunctionExpr::ArrayExpr(IRArrayFunction::Get(_)) => true,
        _ => false,
    }
}

/// `selection` with each field name mapped by `f`, `None` if `f` fails on some name.
#[cfg(feature = "dtype-struct")]
fn map_field_names<'a>(
    selection: &'a FieldSelection,
    f: impl Fn(&'a str) -> Option<&'a str>,
) -> Option<FieldSelection> {
    let FieldSelection::Fields(fields) = selection else {
        return Some(FieldSelection::All);
    };

    let mut out = FieldSelection::validity_only();
    for (name, s) in fields.iter() {
        out.insert_child(PlSmallStr::from_str(f(name.as_str())?), s.clone());
    }
    Some(out)
}

/// What is needed of struct `column` (of `dtype`) that `unnest` turns into columns, given what is
/// needed of the output columns `out_names`.
pub(super) fn unnested_struct_selection(
    column: &str,
    dtype: &DataType,
    separator: Option<&PlSmallStr>,
    out_names: &ColumnSelection,
) -> FieldSelection {
    #[cfg(feature = "dtype-struct")]
    if let DataType::Struct(fields) = dtype {
        let mut selection = FieldSelection::validity_only();

        for field in fields {
            let out_name = match separator {
                Some(separator) => {
                    polars_utils::format_pl_smallstr!("{column}{separator}{}", field.name())
                },
                None => field.name().clone(),
            };
            if let Some(field_selection) = out_names.get(&out_name) {
                selection.insert_child(field.name().clone(), field_selection.clone());
            }
        }

        // Unnesting no field would lose the height.
        if selection.fields().is_some_and(|f| !f.is_empty()) {
            return selection;
        }
    }

    let _ = (column, dtype, separator, out_names);
    FieldSelection::All
}

/// Recomputes the schemas of the nodes above narrowed dtypes. `changed`: the nodes whose own
/// output dtypes narrowed (pruned scans, rewritten expressions).
#[allow(clippy::disallowed_types)]
pub(super) fn refresh_schemas(
    root: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
    changed: &PlHashSet<Node>,
) {
    let mut memo = PlHashMap::default();
    refresh_rec(root, ir_arena, expr_arena, changed, &mut memo);
}

/// Returns whether the schema of `node` changed.
#[recursive]
#[allow(clippy::disallowed_types)]
fn refresh_rec(
    node: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
    changed: &PlHashSet<Node>,
    memo: &mut PlHashMap<Node, bool>,
) -> bool {
    if let Some(out) = memo.get(&node) {
        return *out;
    }

    let inputs: Vec<Node> = ir_arena.get(node).inputs().collect();
    let mut inputs_changed = false;
    for input in inputs.iter() {
        inputs_changed |= refresh_rec(*input, ir_arena, expr_arena, changed, memo);
    }

    let self_changed = changed.contains(&node);
    let out = if !inputs_changed && !self_changed {
        false
    } else {
        // Other nodes derive their schema from their inputs: compare to the stored one.
        let stored_schema = matches!(
            ir_arena.get(node),
            IR::Select { .. }
                | IR::HStack { .. }
                | IR::SimpleProjection { .. }
                | IR::Join { .. }
                | IR::HConcat { .. }
                | IR::GroupBy { .. }
                | IR::Window { .. }
        )
        .then(|| ir_arena.get(node).schema(ir_arena).into_owned());

        recompute_schema(node, &inputs, ir_arena, expr_arena);

        self_changed
            || stored_schema
                .is_none_or(|old| old != ir_arena.get(node).schema(ir_arena).into_owned())
    };

    memo.insert(node, out);
    out
}

fn recompute_schema(
    node: Node,
    inputs: &[Node],
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
) {
    let input_schemas: Vec<Arc<Schema>> = inputs
        .iter()
        .map(|i| ir_arena.get(*i).schema(ir_arena).into_owned())
        .collect();

    match ir_arena.get_mut(node) {
        IR::Select { expr, schema, .. } => {
            let input_schema = &input_schemas[0];
            let mut new_schema = Schema::with_capacity(expr.len());
            for e in expr.iter_mut() {
                let dtype = expr_dtype(e, input_schema, expr_arena)
                    .unwrap_or_else(|| schema.get(e.output_name()).unwrap().clone());
                new_schema.insert(e.output_name().clone(), dtype);
            }
            *schema = Arc::new(new_schema);
        },
        IR::HStack { exprs, schema, .. } => {
            let input_schema = &input_schemas[0];
            let mut expr_dtypes =
                PlIndexMap::with_capacity_and_hasher(exprs.len(), Default::default());
            for e in exprs.iter_mut() {
                if !expr_dtypes.contains_key(e.output_name()) {
                    let dtype = expr_dtype(e, input_schema, expr_arena);
                    expr_dtypes.insert(e.output_name().clone(), dtype);
                }
            }
            let mut new_schema = Schema::with_capacity(schema.len());
            for (name, old_dtype) in schema.iter() {
                let dtype = match expr_dtypes.swap_remove(name) {
                    Some(dtype) => dtype,
                    None => input_schema.get(name).cloned(),
                };
                new_schema.insert(name.clone(), dtype.unwrap_or_else(|| old_dtype.clone()));
            }
            *schema = Arc::new(new_schema);
        },
        IR::SimpleProjection { columns, .. } => {
            let input_schema = &input_schemas[0];
            *columns = Arc::new(
                columns
                    .iter()
                    .map(|(name, dtype)| {
                        (
                            name.clone(),
                            input_schema.get(name).unwrap_or(dtype).clone(),
                        )
                    })
                    .collect(),
            );
        },
        IR::Join {
            schema, options, ..
        } => {
            if let Ok(new_schema) = det_join_schema(&input_schemas[0], &input_schemas[1], options) {
                *schema = Arc::new(
                    schema
                        .iter()
                        .map(|(name, dtype)| {
                            (name.clone(), new_schema.get(name).unwrap_or(dtype).clone())
                        })
                        .collect(),
                );
            }
        },
        IR::HConcat { schema, .. } => {
            *schema = Arc::new(
                schema
                    .iter()
                    .map(|(name, dtype)| {
                        let dtype = input_schemas
                            .iter()
                            .find_map(|s| s.get(name))
                            .unwrap_or(dtype);
                        (name.clone(), dtype.clone())
                    })
                    .collect(),
            );
        },
        IR::GroupBy { aggs, schema, .. } => {
            let input_schema = &input_schemas[0];
            let mut new_schema = schema.as_ref().clone();
            for e in aggs.iter_mut() {
                if let Some(dtype) = expr_dtype(e, input_schema, expr_arena) {
                    *new_schema.get_mut(e.output_name()).unwrap() = dtype;
                }
            }
            *schema = Arc::new(new_schema);
        },
        IR::MapFunction { function, .. } => function.clear_cached_schema(),
        IR::Union { .. } => {
            align_input_dtypes(node, inputs, &input_schemas, ir_arena, expr_arena);
        },
        #[cfg(feature = "merge_sorted")]
        IR::MergeSorted { .. } => {
            align_input_dtypes(node, inputs, &input_schemas, ir_arena, expr_arena);
        },
        // Schemas derived from the input (`Filter`, `Sort`, `Cache`, ..), or nodes whose inputs
        // are read whole (`Window` and a `GroupBy` with `apply` ask `All` of every column).
        _ => {},
    }
}

/// The dtype of `e` on `input_schema`, resetting its cached dtype.
fn expr_dtype(
    e: &mut ExprIR,
    input_schema: &Schema,
    expr_arena: &Arena<AExpr>,
) -> Option<DataType> {
    let dtype = expr_arena
        .get(e.node())
        .to_dtype(&ToFieldContext::new(expr_arena, input_schema))
        .and_then(|dtype| dtype.materialize_unknown(true));
    match dtype {
        Ok(dtype) => {
            e.set_dtype(dtype.clone());
            Some(dtype)
        },
        Err(err) => {
            // The expression resolved before the rewrite, so this is a bug in it; the caller
            // keeps the old dtype, which is wrong if the fields of the expression were pruned.
            debug_assert!(
                false,
                "struct-field pruning: dtype of rewritten expression: {err}"
            );
            None
        },
    }
}

/// Prunes the inputs of a node that concatenates them vertically to the struct fields they all
/// have, as their selections may have been merged with different other consumers.
fn align_input_dtypes(
    node: Node,
    inputs: &[Node],
    input_schemas: &[Arc<Schema>],
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
) {
    if input_schemas.windows(2).all(|w| w[0] == w[1]) {
        return;
    }

    let mut common = input_schemas[0].as_ref().clone();
    for schema in &input_schemas[1..] {
        for (name, dtype) in common.iter_mut() {
            if let Some(other) = schema.get(name) {
                // Top-level columns are read, so they do not differ.
                if let Some(common) = intersect_dtype(dtype, other) {
                    *dtype = common;
                }
            }
        }
    }
    let common = Arc::new(common);

    for (i, (input, input_schema)) in inputs.iter().zip(input_schemas).enumerate() {
        if input_schema.as_ref() == common.as_ref() {
            continue;
        }

        let new_input = pruning_select(*input, input_schema, common.clone(), expr_arena);
        let new_input = ir_arena.add(new_input);

        *ir_arena.get_mut(node).inputs_mut().nth(i).unwrap() = new_input;
    }
}

/// A `Select` of `input`, with schema `input_schema`, that prunes each column to its dtype in
/// `schema`.
pub(super) fn pruning_select(
    input: Node,
    input_schema: &Schema,
    schema: Arc<Schema>,
    expr_arena: &mut Arena<AExpr>,
) -> IR {
    let expr = schema
        .iter()
        .map(|(name, dtype)| {
            let column = expr_arena.add(AExpr::Column(name.clone()));
            let pruned = prune_to_dtype(column, input_schema.get(name).unwrap(), dtype, expr_arena);
            ExprIR::new(pruned, OutputName::Alias(name.clone())).with_dtype(dtype.clone())
        })
        .collect();

    IR::Select {
        input,
        expr,
        schema,
        options: ProjectionOptions {
            run_parallel: false,
            duplicate_check: false,
            should_broadcast: false,
            maintain_dataframe_height: false,
        },
    }
}

/// The struct fields `a` and `b` both have, recursively, or `None` if they differ otherwise.
///
/// The inputs of a vertical concatenation had equal dtypes before the pruning, so a field whose
/// dtype differs is one that nobody reads: an input it reverted on dropped an unread `with_fields`
/// that overwrote it. Such a field is left out.
fn intersect_dtype(a: &DataType, b: &DataType) -> Option<DataType> {
    Some(match (a, b) {
        #[cfg(feature = "dtype-struct")]
        (DataType::Struct(a_fields), DataType::Struct(b_fields)) => {
            let b_fields = fields_by_name(b_fields);
            DataType::Struct(a_fields.filter_map_fields(|a| {
                let b = b_fields.get(a.name().as_str())?;
                intersect_dtype(a.dtype(), b.dtype())
            }))
        },
        (DataType::List(a), DataType::List(b)) => DataType::List(Box::new(intersect_dtype(a, b)?)),
        #[cfg(feature = "dtype-array")]
        (DataType::Array(a, width), DataType::Array(b, _)) => {
            DataType::Array(Box::new(intersect_dtype(a, b)?), *width)
        },
        _ if a == b => a.clone(),
        _ => return None,
    })
}

/// The fields of a struct by name, for lookups that are not linear in the width.
#[cfg(feature = "dtype-struct")]
fn fields_by_name(
    fields: &[polars_core::prelude::Field],
) -> PlIndexMap<&str, &polars_core::prelude::Field> {
    fields.iter().map(|f| (f.name().as_str(), f)).collect()
}

/// An expression reading the struct value (or list of them) of `node` (of dtype `from`) pruned
/// to `to`, keeping the outer validity.
fn prune_to_dtype(
    node: Node,
    from: &DataType,
    to: &DataType,
    expr_arena: &mut Arena<AExpr>,
) -> Node {
    if from == to {
        return node;
    }

    match (from, to) {
        #[cfg(feature = "dtype-struct")]
        (DataType::Struct(from_fields), DataType::Struct(to_fields)) => {
            let from_fields = fields_by_name(from_fields);
            let evaluation = to_fields
                .iter()
                .map(|to_field| {
                    let from_field = from_fields[to_field.name().as_str()];
                    let field = expr_arena.add(AExpr::StructField(to_field.name().clone()));
                    let field =
                        prune_to_dtype(field, from_field.dtype(), to_field.dtype(), expr_arena);
                    ExprIR::new(field, OutputName::Alias(to_field.name().clone()))
                })
                .collect();

            expr_arena.add(AExpr::StructEval {
                expr: node,
                evaluation,
                variant: StructEvalVariant::Select,
            })
        },
        (DataType::List(from), DataType::List(to)) => {
            let element = expr_arena.add(AExpr::Element);
            let evaluation = prune_to_dtype(element, from, to, expr_arena);
            expr_arena.add(AExpr::Eval {
                expr: node,
                evaluation,
                variant: EvalVariant::List,
            })
        },
        #[cfg(feature = "dtype-array")]
        (DataType::Array(from, _), DataType::Array(to, _)) => {
            let element = expr_arena.add(AExpr::Element);
            let evaluation = prune_to_dtype(element, from, to, expr_arena);
            expr_arena.add(AExpr::Eval {
                expr: node,
                evaluation,
                variant: EvalVariant::Array { as_list: false },
            })
        },
        _ => node,
    }
}
