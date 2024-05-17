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

struct WithColumnsMut<'a> {
    input: &'a mut Node,
    exprs: &'a mut ProjectionExprs,
    schema: &'a mut SchemaRef,
    options: &'a mut ProjectionOptions,
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

struct ColumnStatus {
    live: MutableBitmap,
    assign: MutableBitmap,
}

struct ColumnStatusFrozen {
    live: Bitmap,
    assign: Bitmap,
}

impl ColumnStatus {
    fn new() -> Self {
        Self {
            live: MutableBitmap::new(),
            assign: MutableBitmap::new(),
        }
    }

    fn name_to_idx<'a>(
        &mut self,
        name_map: &mut PlHashMap<&'a str, usize>,
        name: &'a str,
    ) -> usize {
        let size = name_map.len();
        let idx = *name_map.entry(name).or_insert_with(|| size);

        let size = name_map.len();
        if self.live.len() < size {
            self.live.extend_constant(size - self.live.len(), false);
            self.assign.extend_constant(size - self.assign.len(), false);
        }

        idx
    }

    fn freeze_with_size(mut self, size: usize) -> ColumnStatusFrozen {
        if self.live.len() < size {
            self.live.extend_constant(size - self.live.len(), false);
            self.assign.extend_constant(size - self.assign.len(), false);
        }

        ColumnStatusFrozen {
            live: self.live.freeze(),
            assign: self.assign.freeze(),
        }
    }

    fn append_expr_node<'a>(
        &mut self,
        name_map: &mut PlHashMap<&'a str, usize>,
        node: Node,
        expr_arena: &'a Arena<AExpr>,
    ) {
        let expr = expr_arena.get(node);

        match expr {
            AExpr::Explode(expr) => self.append_expr_node(name_map, *expr, expr_arena),
            AExpr::Alias(expr, alias) => {
                let idx = self.name_to_idx(name_map, alias);
                self.assign.set(idx, true);
                self.append_expr_node(name_map, *expr, expr_arena)
            },
            AExpr::Column(col) => {
                let idx = self.name_to_idx(name_map, col);
                self.live.set(idx, true);
            },
            AExpr::Literal(_) => {},
            AExpr::BinaryExpr { left, op: _, right } => {
                self.append_expr_node(name_map, *left, expr_arena);
                self.append_expr_node(name_map, *right, expr_arena);
            },
            AExpr::Cast {
                expr,
                data_type: _,
                strict: _,
            } => self.append_expr_node(name_map, *expr, expr_arena),
            AExpr::Sort { expr, options: _ } => self.append_expr_node(name_map, *expr, expr_arena),
            AExpr::Gather {
                expr,
                idx,
                returns_scalar: _,
            } => {
                self.append_expr_node(name_map, *expr, expr_arena);
                self.append_expr_node(name_map, *idx, expr_arena);
            },
            AExpr::SortBy {
                expr,
                by,
                sort_options: _,
            } => {
                self.append_expr_node(name_map, *expr, expr_arena);
                for i in by.iter() {
                    self.append_expr_node(name_map, *i, expr_arena);
                }
            },
            AExpr::Filter { input, by } => {
                self.append_expr_node(name_map, *input, expr_arena);
                self.append_expr_node(name_map, *by, expr_arena);
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
                | AAggExpr::AggGroups(input) => self.append_expr_node(name_map, *input, expr_arena),
                AAggExpr::Quantile {
                    expr,
                    quantile,
                    interpol: _,
                } => {
                    self.append_expr_node(name_map, *expr, expr_arena);
                    self.append_expr_node(name_map, *quantile, expr_arena);
                },
            },
            AExpr::Ternary {
                predicate,
                truthy,
                falsy,
            } => {
                self.append_expr_node(name_map, *predicate, expr_arena);
                self.append_expr_node(name_map, *truthy, expr_arena);
                self.append_expr_node(name_map, *falsy, expr_arena);
            },
            AExpr::AnonymousFunction {
                input,
                function: _,
                output_type: _,
                options: _,
            } => {
                for i in input.iter() {
                    self.append_expr_ir(name_map, i, expr_arena);
                }
            },
            AExpr::Function {
                input,
                function: _,
                options: _,
            } => {
                for i in input.iter() {
                    self.append_expr_ir(name_map, i, expr_arena);
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
                self.append_expr_node(name_map, *function, expr_arena);
                for i in partition_by.iter() {
                    self.append_expr_node(name_map, *i, expr_arena);
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
                self.append_expr_node(name_map, *input, expr_arena);
                self.append_expr_node(name_map, *offset, expr_arena);
                self.append_expr_node(name_map, *length, expr_arena);
            },
            AExpr::Len => {},
            AExpr::Nth(_) => {},
        }
    }

    fn append_expr_ir<'a>(
        &mut self,
        name_map: &mut PlHashMap<&'a str, usize>,
        expr_ir: &'a ExprIR,
        expr_arena: &'a Arena<AExpr>,
    ) {
        match expr_ir.output_name_inner() {
            OutputName::None => {},
            OutputName::LiteralLhs(s) | OutputName::ColumnLhs(s) => {
                let idx = self.name_to_idx(name_map, s);
                self.live.set(idx, true);
                self.assign.set(idx, true);
            },
            OutputName::Alias(s) => {
                let idx = self.name_to_idx(name_map, s);
                self.assign.set(idx, true);
            },
        }

        self.append_expr_node(name_map, expr_ir.node(), expr_arena);
    }

    fn from_expr_ir<'a>(
        name_map: &mut PlHashMap<&'a str, usize>,
        expr: &'a ExprIR,
        expr_arena: &'a Arena<AExpr>,
    ) -> Self {
        // @TODO: This should not be hardcoded
        let mut column_status = Self::new();
        column_status.append_expr_ir(name_map, expr, expr_arena);
        column_status
    }
}

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

enum CWCOptimization {
    PushdownAll,
    PushdownSubset(Bitmap),
}

impl<'a> ClusterWithColumns<'a> {
    pub fn new(lp_arena: &'a mut Arena<IR>, expr_arena: &'a mut Arena<AExpr>) -> Self {
        Self {
            lp_arena,
            expr_arena,
        }
    }

    pub fn optimize(&mut self, root: Node) {
        self._optimize(root);
    }

    fn _optimize(&mut self, root: Node) {
        loop {
            let ir = self.lp_arena.get(root);
            let IR::HStack { input, .. } = ir else {
                break;
            };
            let input = *input;

            let [current, child] = self.lp_arena.get_many_mut([root, input]);

            let Some(with_columns) = WithColumnsMut::from_ir(current) else {
                unreachable!();
            };
            let Some(child_with_columns) = WithColumnsMut::from_ir(child) else {
                break;
            };

            // @NOTE
            // There are dependencies i.f.f. `child` assigns to a column that is used by `parent`

            let mut name_map = PlHashMap::default();

            let child_statuses: Vec<ColumnStatus> = child_with_columns
                .exprs
                .as_exprs()
                .iter()
                .map(|expr| ColumnStatus::from_expr_ir(&mut name_map, expr, self.expr_arena))
                .collect();
            let current_statuses: Vec<ColumnStatus> = with_columns
                .exprs
                .as_exprs()
                .iter()
                .map(|expr| ColumnStatus::from_expr_ir(&mut name_map, expr, self.expr_arena))
                .collect();

            let child_statuses: Vec<ColumnStatusFrozen> = child_statuses
                .into_iter()
                .map(|s| s.freeze_with_size(name_map.len()))
                .collect();
            let current_statuses: Vec<ColumnStatusFrozen> = current_statuses
                .into_iter()
                .map(|s| s.freeze_with_size(name_map.len()))
                .collect();

            // @NOTE: This satisfies the borrow checker
            let num_names = name_map.len();
            drop(name_map);

            let mut pushable = MutableBitmap::with_capacity(child_with_columns.exprs.len());

            let mut child_assign = Bitmap::new_zeroed(num_names);
            for child_status in &child_statuses {
                use std::ops::BitOr;
                child_assign = child_assign.bitor(&child_status.assign);
            }

            for current_status in &current_statuses {
                use std::ops::BitAnd;

                let has_intersection = child_assign.bitand(&current_status.live).set_bits() > 0;
                let is_pushable = !has_intersection;

                pushable.push(is_pushable);
            }

            let pushable = pushable.freeze();

            let optimization = match pushable.set_bits() {
                0 => break,
                x if x == pushable.len() => CWCOptimization::PushdownAll,
                _ => CWCOptimization::PushdownSubset(pushable),
            };

            match optimization {
                CWCOptimization::PushdownAll => {
                    // @NOTE: To keep the schema correct, we reverse the order here. As a
                    // `WITH_COLUMNS` higher up produces later columns.
                    child_with_columns
                        .exprs
                        .exprs_mut()
                        .extend(std::mem::take(with_columns.exprs.exprs_mut()));
                    std::mem::swap(
                        with_columns.exprs.exprs_mut(),
                        child_with_columns.exprs.exprs_mut(),
                    );

                    *with_columns.input = *child_with_columns.input;
                    *with_columns.options = *child_with_columns.options;

                    self.lp_arena.take(input);
                },
                CWCOptimization::PushdownSubset(exprs) => {
                    let mut current_exprs = std::mem::take(with_columns.exprs.exprs_mut())
                        .into_iter()
                        .zip(exprs)
                        .filter_map(|(expr, do_pushdown)| {
                            if do_pushdown {
                                child_with_columns.exprs.exprs_mut().push(expr);
                                None
                            } else {
                                Some(expr)
                            }
                        })
                        .collect();

                    std::mem::swap(&mut current_exprs, with_columns.exprs.exprs_mut());

                    // @NOTE: Here we add a simple projection to make sure that the output still
                    // has the right schema.
                    // @TODO: I don't think we always have to add a simple projection, maybe we can
                    // filter out some of the cases where we don't have to add it.
                    // @TODO: Do we need to adjust the schema of current?
                    let schema = with_columns.schema.clone();
                    let new = self.lp_arena.add(IR::Invalid);
                    let projection = IR::SimpleProjection {
                        input: new,
                        columns: schema,
                    };
                    let current = self.lp_arena.replace(root, projection);
                    self.lp_arena.replace(new, current);
                },
            }
        }

        let ir = self.lp_arena.get(root);
        for child in ir.get_inputs().iter() {
            self._optimize(*child);
        }
    }
}
