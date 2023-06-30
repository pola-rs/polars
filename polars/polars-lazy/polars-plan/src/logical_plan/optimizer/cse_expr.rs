use std::rc::Rc;

use super::*;
use crate::logical_plan::visitor::{TreeNode, VisitRecursion};
use crate::prelude::visitor::{AexprNode, TreeNodeRewriter, TreeNodeVisitor};

type Identifier = String;
type SubExprMap = PlHashMap<Identifier, (Node, usize, DataType)>;
type IdentifierArray = Vec<(usize, Identifier)>;

#[derive(Debug)]
enum VisitRecord {
    /// entered a new expression
    Entered(usize),
    /// every visited sub-expression pushes their identifier to the stack
    SubExprId(Identifier),
}

struct ExprIdentifierVisitor<'a> {
    expr_set: &'a mut SubExprMap,
    identifier_array: &'a mut Vec<(usize, Identifier)>,
    schema: &'a Schema,
    node_count: usize,
    series_number: usize,
    visit_stack: Vec<VisitRecord>,
}

impl ExprIdentifierVisitor<'_> {
    fn new<'a>(
        expr_set: &'a mut SubExprMap,
        identifier_array: &'a mut IdentifierArray,
        schema: &'a Schema,
    ) -> ExprIdentifierVisitor<'a> {
        ExprIdentifierVisitor {
            expr_set,
            identifier_array,
            schema,
            node_count: 0,
            series_number: 0,
            visit_stack: vec![],
        }
    }

    /// pop all visit-records until an `Entered` is found. We accumulate a `SubExprId`s
    /// to `id`. Finally we return the expression `idx` and `Identifier`.
    /// This works due to the stack.
    /// If we traverse another expression in the mean time, it will get popped of the stack first
    /// so the returned identifier belongs to a single sub-expression
    fn pop_until_entered(&mut self) -> (usize, Identifier) {
        let mut id = String::new();

        while let Some(item) = self.visit_stack.pop() {
            match item {
                VisitRecord::Entered(idx) => return (idx, id),
                // TODO! Don't like this. Node to expr and to format already shows the whole lineage
                // so this seems overly redundant and O^2
                VisitRecord::SubExprId(s) => {
                    id.push_str(s.as_str());
                }
            }
        }
        unreachable!()
    }

    fn accept_node(&self, ae: &AExpr) -> bool {
        !matches!(ae, AExpr::Column(_) | AExpr::Count | AExpr::Literal(_))
    }
}

impl TreeNodeVisitor for ExprIdentifierVisitor<'_> {
    type Node = AexprNode;

    fn pre_visit(&mut self, _node: &Self::Node) -> PolarsResult<VisitRecursion> {
        self.visit_stack.push(VisitRecord::Entered(self.node_count));
        self.node_count += 1;
        self.identifier_array
            .push((Default::default(), Default::default()));
        Ok(VisitRecursion::Continue)
    }
    fn post_visit(&mut self, node: &Self::Node) -> PolarsResult<VisitRecursion> {
        self.series_number += 1;

        let (idx, mut sub_expr_id) = self.pop_until_entered();
        if !self.accept_node(node.to_aexpr()) {
            self.identifier_array[idx].0 = self.series_number;
            self.visit_stack
                .push(VisitRecord::SubExprId(format!("{}", node.to_expr())));
            return Ok(VisitRecursion::Continue);
        }

        let id = format!("{}{}", node.to_expr(), sub_expr_id);
        self.identifier_array[idx] = (self.series_number, id.clone());
        self.visit_stack.push(VisitRecord::SubExprId(id.clone()));

        let dtype = node.with_arena(|arena| {
            node.to_aexpr()
                .get_type(&self.schema, Context::Default, arena)
        })?;

        self.expr_set
            .entry(id)
            .or_insert_with(|| (node.node(), 0, dtype))
            .1 += 1;
        Ok(VisitRecursion::Continue)
    }
}


struct CommonSubExprRewriter<'a> {
    sub_expr_map: &'a SubExprMap,
    identifier_array: &'a IdentifierArray,
    /// keep track of the replaced identifiers
    replaced_identifiers: &'a PlHashSet<Identifier>,

    max_series_number: usize,
    /// current nodes index in `identifier_array`
    current_idx: usize
}

// impl TreeNodeRewriter for CommonSubExprRewriter<'_> {
//     type Node = AexprNode;
//
//     fn pre_visit(&mut self, node: &Self::Node) -> PolarsResult<VisitRecursion> {
//         todo!()
//     }
//
//     fn mut
// }

struct CommonSubExprElimination {
    processed: PlHashSet<Node>,
    sub_expr_map: SubExprMap,
}

fn process_expression(
    node: Node,
    expr_arena: &mut Arena<AExpr>,
    sub_expr_map: &mut SubExprMap,
    schema: &Schema,
) -> PolarsResult<IdentifierArray> {
    let mut id_array = vec![];
    let mut visitor = ExprIdentifierVisitor::new(sub_expr_map, &mut id_array, schema);
    AexprNode::with_context(node, expr_arena, |ae_node| ae_node.visit(&mut visitor))?;

    Ok(id_array)
}

impl CommonSubExprElimination {
    pub fn new() -> Self {
        Self {
            processed: Default::default(),
            sub_expr_map: Default::default(),
        }
    }
}

impl OptimizationRule for CommonSubExprElimination {
    fn optimize_plan(
        &mut self,
        lp_arena: &mut Arena<ALogicalPlan>,
        expr_arena: &mut Arena<AExpr>,
        node: Node,
    ) -> Option<ALogicalPlan> {
        if !self.processed.insert(node) {
            return None;
        }

        self.sub_expr_map.clear();
        match lp_arena.get(node) {
            ALogicalPlan::Projection {
                expr,
                input,
                schema,
            } => {
                expr.into_iter().map(|node| {
                    let array =
                        process_expression(*node, expr_arena, &mut self.sub_expr_map, schema)
                            .unwrap();
                });

                None
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_foo() {
        let e = (col("f00").sum() * col("bar")).sum() + col("f00").sum();
        let mut arena = Arena::new();
        let node = to_aexpr(e, &mut arena);

        let mut expr_set = Default::default();
        let mut id_array = vec![];
        let mut schema = Schema::new();
        schema.with_column("f00".into(), DataType::Int32);
        schema.with_column("bar".into(), DataType::Int32);
        let mut visitor = ExprIdentifierVisitor::new(&mut expr_set, &mut id_array, &schema);

        AexprNode::with_context(node, &mut arena, |ae_node| ae_node.visit(&mut visitor)).unwrap();

        for (id, pl) in visitor.expr_set {
            let e = node_to_expr(pl.0, &arena);
            dbg!(e, pl.1);
        }
    }

    #[test]
    fn test_foo2() {}
}
