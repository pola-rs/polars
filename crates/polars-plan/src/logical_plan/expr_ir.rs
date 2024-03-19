use super::*;

#[derive(Clone, Debug)]
pub struct ExprIR {
    output_dtype: Option<DataType>,
    /// Name that this expression refers to via `col(<name>)`
    /// This is `None` for literals.
    left_most_input_name: Option<Arc<str>>,
    /// Output name of this expression.
    output_name: Option<Arc<str>>,
    /// Reduced expression.
    /// This expression is pruned from `alias` and already expanded.
    node: Node
}

impl ExprIR {
    pub(crate) fn new(
        node: Node,
        left_most_input_name: Option<Arc<str>>,
        output_name: Arc<str>) -> Self {
        ExprIR {
            output_dtype: None,
            left_most_input_name,
            output_name: Some(output_name),
            node
        }
    }

    pub(crate) fn new_minimal(node: Node) -> Self {
        Self {
            node,
            output_dtype: None,
            left_most_input_name: None,
            output_name: None
        }

    }

    #[inline]
    pub(crate) fn node(&self) -> Node {
        self.node
    }

    pub(crate) fn set_node(&mut self, node: Node) {
        self.node = node;
    }

    pub(crate) fn output_name_arc(&self) -> &Arc<str> {
        self.output_name.as_ref().unwrap()
    }

    pub(crate) fn output_name(&self) -> &str {
        self.output_name_arc().as_ref()
    }

    pub(crate) fn output_dtype(&self) -> &DataType {
        &self.output_dtype.unwrap()
    }

    pub(crate) fn to_field(&self) -> Field {
        Field::new(
            self.output_name(),
            self.output_dtype().clone()
        )
    }

    pub(crate) fn to_expr(&self, expr_arena: &Arena<AExpr>) -> Expr {
        let out = node_to_expr(self.node, expr_arena);

        match (&self.left_most_input_name, &self.output_name) {
            (Some(i), Some(o)) => {
                if i == o {
                    out
                } else {
                    out.alias(o)
                }
            },
            (Some(_), None) => out,
            (None, None) => out,
            (None, Some(o)) => out.alias(o),
        }
    }
}
