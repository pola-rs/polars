use std::sync::Arc;

use arrow::bitmap::{Bitmap, MutableBitmap};
use polars_core::schema::SchemaRef;
use polars_utils::aliases::PlHashMap;
use polars_utils::arena::{Arena, Node};

use super::aexpr::{AAggExpr, AExpr};
use super::alp::IR;
use super::expr_ir::{ExprIR, OutputName};
use super::ProjectionOptions;
use crate::logical_plan::projection_expr::ProjectionExprs;

pub struct ClusterWithColumns<'a> {
    lp_arena: &'a mut Arena<IR>,
    expr_arena: &'a mut Arena<AExpr>,
}

/// Utility structure to contain all the information of a [`IR::HStack`] i.e. a `WITH_COLUMNS` node
struct WithColumnsMut<'a> {
    input: &'a mut Node,
    exprs: &'a mut ProjectionExprs,
    schema: &'a mut SchemaRef,
    options: &'a mut ProjectionOptions,
}

/// Status of the columns in an IR expression, which contains which [`ColumnName`]'s are written to
/// and which are read from.
struct ExprColumnStatus {
    /// Columns that are "live" in a expression i.e. columns that fetched or read from
    live: MutableBitmap,
    /// Columns that are "generated" by a expression i.e. columns that assigned or written to
    generate: MutableBitmap,
}

/// Status of the columns in an IR expression, which contains which [`ColumnName`]'s are written to
/// and which are read from.
struct ExprColumnStatusFrozen {
    /// Columns that are "live" in a expression i.e. columns that fetched or read from
    live: Bitmap,
    /// Columns that are "generated" by a expression i.e. columns that assigned or written to
    generate: Bitmap,
}

// // @NOTE: We use a u128 here, because the space would otherwise be used by the fat-pointer of the
// // boxed slice anyway.
// pub enum BitSet {
//     Short(u128),
//     Long(Box<[u128]>),
// }
//
// impl fmt::Debug for BitSet {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         f.debug_list().entries(self.iter()).finish()
//     }
// }
//
// pub struct BitSetIter<'a> {
//     inner: &'a BitSet,
//     offset: usize,
// }
//
// impl<'a> Iterator for BitSetIter<'a> {
//     type Item = usize;
//
//     fn next(&mut self) -> Option<Self::Item> {
//         match self.inner {
//             BitSet::Short(n) => {
//                 if self.offset >= BitSet::SHORT_SIZE {
//                     return None;
//                 }
//
//                 let shifted = n >> self.offset;
//
//                 if *n == 0 {
//                     self.offset = BitSet::SHORT_SIZE;
//                     return None;
//                 }
//
//                 let trailing_zeros = shifted.trailing_zeros();
//                 self.offset += trailing_zeros as usize;
//                 let idx = self.offset;
//                 self.offset += 1;
//
//                 Some(idx)
//             },
//             BitSet::Long(ns) => loop {
//                 if self.offset >= BitSet::ELEMENT_SIZE * ns.len() {
//                     return None;
//                 }
//
//                 let element_idx = self.offset / BitSet::ELEMENT_SIZE;
//                 let element_offset = self.offset % BitSet::ELEMENT_SIZE;
//
//                 let n = ns[element_idx];
//                 let shifted = n >> element_offset;
//
//                 if shifted == 0 {
//                     self.offset = (element_idx + 1) * BitSet::ELEMENT_SIZE;
//                     continue;
//                 }
//
//                 let trailing_zeros = shifted.trailing_zeros();
//                 self.offset += trailing_zeros as usize;
//                 let idx = self.offset;
//                 self.offset += 1;
//
//                 break Some(idx);
//             },
//         }
//     }
// }
//
// impl BitSet {
//     const SHORT_SIZE: usize = 128;
//     const ELEMENT_SIZE: usize = 128;
//
//     pub fn new(length: usize) -> Self {
//         if length <= Self::SHORT_SIZE {
//             Self::Short(0)
//         } else {
//             let num_elements =
//                 (length / Self::ELEMENT_SIZE) + usize::from(length % Self::ELEMENT_SIZE != 0);
//             Self::Long(vec![0; num_elements].into_boxed_slice())
//         }
//     }
//
//     pub fn iter(&self) -> BitSetIter {
//         BitSetIter {
//             inner: self,
//             offset: 0,
//         }
//     }
//
//     pub fn insert(&mut self, at: usize) {
//         match self {
//             BitSet::Short(ref mut n) => *n |= 1 << at,
//             BitSet::Long(ref mut ns) => {
//                 ns[at / Self::ELEMENT_SIZE] |= 1 << (at % Self::ELEMENT_SIZE)
//             },
//         }
//     }
//
//     pub fn is_empty(&self) -> bool {
//         match self {
//             BitSet::Short(n) => *n == 0,
//             BitSet::Long(ns) => ns.iter().all(|n| *n == 0),
//         }
//     }
//
//     pub fn has_intersection(&self, other: &Self) -> bool {
//         match (self, other) {
//             (Self::Short(lhs), Self::Short(rhs)) => lhs & rhs != 0,
//             (Self::Long(lhs), Self::Long(rhs)) if lhs.len() == rhs.len() => {
//                 lhs.iter().zip(rhs.iter()).any(|(lhs, rhs)| lhs & rhs != 0)
//             },
//             _ => panic!("has_intersection between bitsets of differing lengths"),
//         }
//     }
// }

impl<'a> WithColumnsMut<'a> {
    fn from_ir(ir: &'a mut IR) -> Option<Self> {
        let IR::HStack {
            input,
            exprs,
            schema,
            options,
        } = ir
        else {
            return None;
        };

        Some(Self {
            input,
            exprs,
            schema,
            options,
        })
    }
}

impl ExprColumnStatus {
    fn new() -> Self {
        Self {
            live: MutableBitmap::with_capacity(8),
            generate: MutableBitmap::with_capacity(8),
        }
    }

    fn name_to_idx<'a>(
        &mut self,
        column_map: &mut PlHashMap<&'a str, usize>,
        name: &'a str,
    ) -> usize {
        let size = column_map.len();
        let idx = *column_map.entry(name).or_insert_with(|| size);

        // @NOTE: We need to account for new names here, but also defined in other instances using
        // the same `column_map`.
        let size = column_map.len();
        if self.live.len() < size {
            self.live.extend_constant(size - self.live.len(), false);
            self.generate
                .extend_constant(size - self.generate.len(), false);
        }

        idx
    }

    fn freeze_with_size(mut self, size: usize) -> ExprColumnStatusFrozen {
        if self.live.len() < size {
            self.live.extend_constant(size - self.live.len(), false);
            self.generate
                .extend_constant(size - self.generate.len(), false);
        }

        ExprColumnStatusFrozen {
            live: self.live.freeze(),
            generate: self.generate.freeze(),
        }
    }

    fn append_expr_node<'a>(
        &mut self,
        column_map: &mut PlHashMap<&'a str, usize>,
        node: Node,
        expr_arena: &'a Arena<AExpr>,
    ) {
        let expr = expr_arena.get(node);

        match expr {
            AExpr::Explode(expr) => self.append_expr_node(column_map, *expr, expr_arena),
            AExpr::Alias(expr, alias) => {
                let idx = self.name_to_idx(column_map, alias);
                self.generate.set(idx, true);
                self.append_expr_node(column_map, *expr, expr_arena)
            },
            AExpr::Column(col) => {
                let idx = self.name_to_idx(column_map, col);
                self.live.set(idx, true);
            },
            AExpr::Literal(_) => {},
            AExpr::BinaryExpr { left, op: _, right } => {
                self.append_expr_node(column_map, *left, expr_arena);
                self.append_expr_node(column_map, *right, expr_arena);
            },
            AExpr::Cast {
                expr,
                data_type: _,
                strict: _,
            } => self.append_expr_node(column_map, *expr, expr_arena),
            AExpr::Sort { expr, options: _ } => {
                self.append_expr_node(column_map, *expr, expr_arena)
            },
            AExpr::Gather {
                expr,
                idx,
                returns_scalar: _,
            } => {
                self.append_expr_node(column_map, *expr, expr_arena);
                self.append_expr_node(column_map, *idx, expr_arena);
            },
            AExpr::SortBy {
                expr,
                by,
                sort_options: _,
            } => {
                self.append_expr_node(column_map, *expr, expr_arena);
                for i in by.iter() {
                    self.append_expr_node(column_map, *i, expr_arena);
                }
            },
            AExpr::Filter { input, by } => {
                self.append_expr_node(column_map, *input, expr_arena);
                self.append_expr_node(column_map, *by, expr_arena);
            },
            AExpr::Agg(aagg_expr) => match aagg_expr {
                AAggExpr::Min {
                    input,
                    propagate_nans: _,
                }
                | AAggExpr::Max {
                    input,
                    propagate_nans: _,
                }
                | AAggExpr::Median(input)
                | AAggExpr::NUnique(input)
                | AAggExpr::First(input)
                | AAggExpr::Last(input)
                | AAggExpr::Mean(input)
                | AAggExpr::Implode(input)
                | AAggExpr::Sum(input)
                | AAggExpr::Count(input, _)
                | AAggExpr::Std(input, _)
                | AAggExpr::Var(input, _)
                | AAggExpr::AggGroups(input) => {
                    self.append_expr_node(column_map, *input, expr_arena)
                },
                AAggExpr::Quantile {
                    expr,
                    quantile,
                    interpol: _,
                } => {
                    self.append_expr_node(column_map, *expr, expr_arena);
                    self.append_expr_node(column_map, *quantile, expr_arena);
                },
            },
            AExpr::Ternary {
                predicate,
                truthy,
                falsy,
            } => {
                self.append_expr_node(column_map, *predicate, expr_arena);
                self.append_expr_node(column_map, *truthy, expr_arena);
                self.append_expr_node(column_map, *falsy, expr_arena);
            },
            AExpr::AnonymousFunction {
                input,
                function: _,
                output_type: _,
                options: _,
            } => {
                for i in input.iter() {
                    self.append_expr_ir(column_map, i, expr_arena);
                }
            },
            AExpr::Function {
                input,
                function: _,
                options: _,
            } => {
                for i in input.iter() {
                    self.append_expr_ir(column_map, i, expr_arena);
                }

                // @TODO: Do we also have to do something with the `function`?
                //
                // From my first impression, the answer is no, but probably someone else should
                // look at this for a second.
            },
            AExpr::Window {
                function,
                partition_by,
                options: _,
            } => {
                self.append_expr_node(column_map, *function, expr_arena);
                for i in partition_by.iter() {
                    self.append_expr_node(column_map, *i, expr_arena);
                }
            },
            AExpr::Wildcard => {
                // @TODO: I am not really sure if we need to account for this in some way. Does
                // this mean all of the columns?
                //
                // I don't think so.
            },
            AExpr::Slice {
                input,
                offset,
                length,
            } => {
                self.append_expr_node(column_map, *input, expr_arena);
                self.append_expr_node(column_map, *offset, expr_arena);
                self.append_expr_node(column_map, *length, expr_arena);
            },
            AExpr::Len => {},
            AExpr::Nth(_) => {},
        }
    }

    fn append_expr_ir<'a>(
        &mut self,
        column_map: &mut PlHashMap<&'a str, usize>,
        expr_ir: &'a ExprIR,
        expr_arena: &'a Arena<AExpr>,
    ) {
        match expr_ir.output_name_inner() {
            OutputName::None => {},
            OutputName::LiteralLhs(s) | OutputName::ColumnLhs(s) => {
                let idx = self.name_to_idx(column_map, s);
                self.live.set(idx, true);
                self.generate.set(idx, true);
            },
            OutputName::Alias(s) => {
                let idx = self.name_to_idx(column_map, s);
                self.generate.set(idx, true);
            },
        }

        self.append_expr_node(column_map, expr_ir.node(), expr_arena);
    }

    fn from_expr_ir<'a>(
        column_map: &mut PlHashMap<&'a str, usize>,
        expr: &'a ExprIR,
        expr_arena: &'a Arena<AExpr>,
    ) -> Self {
        let mut column_status = Self::new();
        column_status.append_expr_ir(column_map, expr, expr_arena);
        column_status
    }
}

impl<'a> ClusterWithColumns<'a> {
    pub fn new(lp_arena: &'a mut Arena<IR>, expr_arena: &'a mut Arena<AExpr>) -> Self {
        Self {
            lp_arena,
            expr_arena,
        }
    }

    pub fn optimize(&mut self, current: Node) {
        while let IR::HStack { input, .. } = self.lp_arena.get(current) {
            let input = *input;

            let [current_ir, input_ir] = self.lp_arena.get_many_mut([current, input]);

            let Some(current_wc) = WithColumnsMut::from_ir(current_ir) else {
                unreachable!();
            };
            let Some(input_wc) = WithColumnsMut::from_ir(input_ir) else {
                break;
            };

            // @NOTE
            // We can pushdown any column that utilizes no live columns that are generated in the
            // input.

            let mut column_map = PlHashMap::default();

            let input_colstat: Vec<ExprColumnStatus> = input_wc
                .exprs
                .as_exprs()
                .iter()
                .map(|expr| ExprColumnStatus::from_expr_ir(&mut column_map, expr, self.expr_arena))
                .collect();
            let current_colstat: Vec<ExprColumnStatus> = current_wc
                .exprs
                .as_exprs()
                .iter()
                .map(|expr| ExprColumnStatus::from_expr_ir(&mut column_map, expr, self.expr_arena))
                .collect();

            let child_colstat: Vec<ExprColumnStatusFrozen> = input_colstat
                .into_iter()
                .map(|s| s.freeze_with_size(column_map.len()))
                .collect();
            let current_colstat: Vec<ExprColumnStatusFrozen> = current_colstat
                .into_iter()
                .map(|s| s.freeze_with_size(column_map.len()))
                .collect();

            // @NOTE: This satisfies the borrow checker
            let num_names = column_map.len();
            drop(column_map);

            let mut pushable = MutableBitmap::with_capacity(input_wc.exprs.len());

            let mut input_generate = Bitmap::new_zeroed(num_names);
            for colstat in &child_colstat {
                use std::ops::BitOr;
                input_generate = input_generate.bitor(&colstat.generate);
            }

            // Check for every expression in the current WITH_COLUMNS node whether
            for colstat in &current_colstat {
                use std::ops::BitAnd;

                let has_intersection = input_generate.bitand(&colstat.live).set_bits() > 0;
                let is_pushable = !has_intersection;

                pushable.push(is_pushable);
            }

            let pushable = pushable.freeze();

            // There is nothing to push down. Move on.
            if pushable.set_bits() == 0 {
                break;
            }

            // If all columns are pushable, we can merge the input into the current. This should be
            // a relatively common case.
            if pushable.set_bits() == pushable.len() {
                // @NOTE: To keep the schema correct, we reverse the order here. As a
                // `WITH_COLUMNS` higher up produces later columns. This also allows us not to
                // have to deal with schemas.
                input_wc
                    .exprs
                    .exprs_mut()
                    .extend(std::mem::take(current_wc.exprs.exprs_mut()));
                std::mem::swap(current_wc.exprs.exprs_mut(), input_wc.exprs.exprs_mut());

                *current_wc.input = *input_wc.input;
                // @TODO: Is this allowed?
                *current_wc.options = *input_wc.options;

                self.lp_arena.take(input);

                // Since we merged the current and input nodes and the input node might have
                // optimizations with their input, we loop again on this node.
                continue;
            }

            let mut current_schema = current_wc.schema.as_ref().clone();
            let mut input_schema = input_wc.schema.as_ref().clone();

            // @NOTE: We don't have to insert a SimpleProjection or redo the `current_schema` if
            // `pushable` contains only 0..N for some N.
            let mut current_exprs = std::mem::take(current_wc.exprs.exprs_mut())
                .into_iter()
                .zip(pushable)
                .enumerate()
                .filter_map(|(i, (expr, do_pushdown))| {
                    if do_pushdown {
                        // @NOTE: It would be nice to have a `remove_at_index` here.
                        let (column, _) = current_wc.schema.get_at_index(i).unwrap();

                        input_wc.exprs.exprs_mut().push(expr);
                        let datatype = current_schema.remove(column).unwrap();
                        input_schema.with_column(column.clone(), datatype);

                        None
                    } else {
                        Some(expr)
                    }
                })
                .collect();

            std::mem::swap(&mut current_exprs, current_wc.exprs.exprs_mut());

            current_schema.merge(input_schema.clone());

            let proj_schema = current_wc.schema.clone();

            *current_wc.schema = Arc::new(current_schema);
            *input_wc.schema = Arc::new(input_schema);

            // @NOTE: Here we add a simple projection to make sure that the output still
            // has the right schema.
            // @TODO: I don't think we always have to add a simple projection, maybe we can
            // filter out some of the cases where we don't have to add it.
            let moved_current = self.lp_arena.add(IR::Invalid);
            let projection = IR::SimpleProjection {
                input: moved_current,
                columns: proj_schema,
            };
            let current = self.lp_arena.replace(current, projection);
            self.lp_arena.replace(moved_current, current);

            // We know that this node is done optimizing
            break;
        }

        let ir = self.lp_arena.get(current);
        for child in ir.get_inputs().iter() {
            self.optimize(*child);
        }
    }
}
