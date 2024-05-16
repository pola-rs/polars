use arrow::bitmap::{Bitmap, MutableBitmap};
use polars_core::schema::SchemaRef;
use polars_utils::aliases::PlHashMap;
use polars_utils::arena::{Arena, Node};

use super::aexpr::{AAggExpr, AExpr};
use super::alp::IR;
use super::expr_ir::{ExprIR, OutputName};
use super::ProjectionOptions;
use crate::logical_plan::projection_expr::ProjectionExprs;
use crate::prelude::AnonymousScan;

pub struct ClusterWithColumns<'a> {
    lp_arena: &'a mut Arena<IR>,
    expr_arena: &'a mut Arena<AExpr>,
}

struct WithColumnsRef<'a> {
    input: Node,
    exprs: &'a ProjectionExprs,
    schema: &'a SchemaRef,
    options: &'a ProjectionOptions,
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
        let next = self.live.len();

        self.live.push(false);
        self.assign.push(false);

        *name_map.entry(name).or_insert_with(|| next)
    }

    fn freeze(self) -> ColumnStatusFrozen {
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
            AExpr::Alias(expr, _) => self.append_expr_node(name_map, *expr, expr_arena),
            AExpr::Column(col) => {
                let idx = self.name_to_idx(name_map, &col);
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

                todo!()
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
            OutputName::LiteralLhs(s) | OutputName::ColumnLhs(s) | OutputName::Alias(s) => {
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

impl<'a> WithColumnsRef<'a> {
    fn from_ir(ir: &'a IR) -> Option<Self> {
        let IR::HStack {
            input,
            exprs,
            schema,
            options,
        } = ir
        else {
            return None;
        };

        let input = *input;

        Some(Self {
            input,
            exprs,
            schema,
            options,
        })
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
    PullUpAll,
    PullUpSubset(Bitmap),
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

    fn has_optimization(&self, root: Node) -> Option<CWCOptimization> {
        let ir = self.lp_arena.get(root);
        let with_columns = WithColumnsRef::from_ir(ir)?;

        let child = ir.get_input()?;
        let child = self.lp_arena.get(child);
        let child_with_columns = WithColumnsRef::from_ir(child)?;

        // @NOTE
        // There are dependencies i.f.f.
        // - `child` assigns to a column that is used by `parent`

        let mut name_map = PlHashMap::default();

        let child_statuses: Vec<ColumnStatusFrozen> = child_with_columns
            .exprs
            .as_exprs()
            .iter()
            .map(|expr| ColumnStatus::from_expr_ir(&mut name_map, expr, self.expr_arena).freeze())
            .collect();
        let current_statuses: Vec<ColumnStatusFrozen> = with_columns
            .exprs
            .as_exprs()
            .iter()
            .map(|expr| ColumnStatus::from_expr_ir(&mut name_map, expr, self.expr_arena).freeze())
            .collect();

        let mut pullable = MutableBitmap::new();

        pullable.reserve(child_with_columns.exprs.len());
        for _ in 0..child_with_columns.exprs.len() {
            pullable.push(false);
        }

        let mut all_pullable = true;
        for (i, current_status) in current_statuses.iter().enumerate() {
            let mut can_be_merged = true;

            for child_status in child_statuses.iter() {
                use std::ops::BitAnd;
                if (current_status.assign.bitand(&child_status.live)).is_empty() {
                    can_be_merged = false;
                }
            }

            all_pullable |= can_be_merged;
            if can_be_merged {
                pullable.set(i, true);
            }
        }

        if all_pullable {
            Some(CWCOptimization::PullUpAll)
        } else if pullable.is_empty() {
            None
        } else {
            Some(CWCOptimization::PullUpSubset(pullable.freeze()))
        }
    }

    fn apply_optimization(&mut self, root: Node, optimization: CWCOptimization) {
        let ir = self.lp_arena.get(root);
        let child_node = ir.get_input().unwrap();

        match optimization {
            CWCOptimization::PullUpAll => {
                let child = self.lp_arena.take(child_node);
                let IR::HStack {
                    input,
                    exprs,
                    schema,
                    options,
                } = child
                else {
                    panic!("Specified node is not a HStack. Instead found: {child:?}");
                };

                let ir = self.lp_arena.get_mut(root);
                let with_columns = WithColumnsMut::from_ir(ir).unwrap();
                with_columns.exprs.exprs_mut().extend(exprs);
                *with_columns.input = input;
                *with_columns.schema = schema;
                *with_columns.options = options;
            },
            CWCOptimization::PullUpSubset(exprs) => {
                let child = self.lp_arena.get_mut(child_node);
                let child_with_columns = WithColumnsMut::from_ir(child).unwrap();
                let child_exprs = std::mem::take(child_with_columns.exprs.exprs_mut());

                let ir = self.lp_arena.get_mut(root);
                let with_columns = WithColumnsMut::from_ir(ir).unwrap();

                let mut child_exprs = child_exprs
                    .into_iter()
                    .zip(exprs)
                    .filter_map(|(expr, do_pullup)| {
                        if do_pullup {
                            with_columns.exprs.exprs_mut().push(expr);
                            None
                        } else {
                            Some(expr)
                        }
                    })
                    .collect();

                let child = self.lp_arena.get_mut(child_node);
                let child_with_columns = WithColumnsMut::from_ir(child).unwrap();
                std::mem::swap(&mut child_exprs, child_with_columns.exprs.exprs_mut());
            },
        }
    }

    fn _optimize(&mut self, root: Node) {
        let ir = self.lp_arena.get(root);
        if let Some(child) = ir.get_input() {
            self._optimize(child);

            if let Some(optimization) = self.has_optimization(root) {
                self.apply_optimization(root, optimization);
            }
        }
    }
}
