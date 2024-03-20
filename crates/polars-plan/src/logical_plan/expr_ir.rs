use super::*;

type Name = Arc<str>;

#[derive(Default)]
pub(crate) enum OutputName {
    #[default]
    None,
    LiteralLhs(Name),
    ColumnLhs(Name),
    Alias(Name)
}

impl OutputName {
    fn unwrap(&self) -> &Name {
        match self {
            OutputName::Alias(name) => name,
            OutputName::ColumnLhs(name) => name,
            OutputName::LiteralLhs(name) => name,
            OutputName::None => panic!("no output name set")
        }
    }

    pub(crate) fn is_none(&self) -> bool {
        matches!(self, OutputName::None)
    }
    pub(crate) fn is_alias(&self) -> bool {
        matches!(self, OutputName::Alias(_))
    }
}

#[derive(Clone, Debug)]
pub struct ExprIR {
    output_dtype: Option<DataType>,
    /// Name that this expression refers to via `col(<name>)`
    /// This is `None` for literals.
    left_most_input_name: Option<Arc<str>>,
    /// Output name of this expression.
    output_name: OutputName,
    /// Reduced expression.
    /// This expression is pruned from `alias` and already expanded.
    node: Node
}

impl ExprIR {
    pub(crate) fn new(
        node: Node,
        left_most_input_name: Option<Arc<str>>,
        output_name: OutputName) -> Self {
        ExprIR {
            output_dtype: None,
            left_most_input_name,
            output_name,
            node
        }
    }

    pub(crate) fn new_minimal(node: Node) -> Self {
        Self {
            node,
            output_dtype: None,
            left_most_input_name: None,
            output_name: OutputName::None
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
        self.output_name.unwrap()
    }

    pub(crate) fn output_name(&self) -> &str {
        self.output_name_arc().as_ref()
    }

    pub(crate) fn left_most_input_name_arc(&self) -> &Name {
        self.left_most_input_name.as_ref().unwrap()
    }

    pub(crate) fn left_most_input_name(&self) -> &str {
        self.left_most_input_name_arc().as_ref()
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

        match &self.output_name {
            OutputName::Alias(name) => out.alias(name.as_ref()),
            _ => out
        }
    }

    pub(crate) fn get_alias(&self) -> Option<&Name> {
        match &self.output_name {
            OutputName::Alias(name) => Some(name),
            _ => None
        }
    }
}


/// A Node that is restricted to `AExpr::Column`
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub(crate) struct ColumnNode(pub(crate) Node);

impl From<ColumnNode> for Node {
    fn from(value: ColumnNode) -> Self {
        value.0
    }
}