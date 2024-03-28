use std::hash::Hash;
#[cfg(feature = "cse")]
use std::hash::Hasher;

use super::*;
use crate::constants::LITERAL_NAME;

#[derive(Default, Debug, Clone, Hash, PartialEq, Eq)]
pub enum OutputName {
    #[default]
    None,
    LiteralLhs(ColumnName),
    ColumnLhs(ColumnName),
    Alias(ColumnName),
}

impl OutputName {
    fn unwrap(&self) -> &ColumnName {
        match self {
            OutputName::Alias(name) => name,
            OutputName::ColumnLhs(name) => name,
            OutputName::LiteralLhs(name) => name,
            OutputName::None => panic!("no output name set"),
        }
    }

    pub(crate) fn is_none(&self) -> bool {
        matches!(self, OutputName::None)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExprIR {
    /// Output name of this expression.
    output_name: OutputName,
    /// Reduced expression.
    /// This expression is pruned from `alias` and already expanded.
    node: Node,
}

impl ExprIR {
    pub fn new(node: Node, output_name: OutputName) -> Self {
        debug_assert!(!output_name.is_none());
        ExprIR { output_name, node }
    }

    pub fn from_node(node: Node, arena: &Arena<AExpr>) -> Self {
        let mut out = Self {
            node,
            output_name: OutputName::None,
        };
        out.node = node;
        for (_, ae) in arena.iter(node) {
            match ae {
                AExpr::Column(name) => {
                    out.output_name = OutputName::ColumnLhs(name.clone());
                    break;
                },
                AExpr::Literal(lv) => {
                    if let LiteralValue::Series(s) = lv {
                        out.output_name = OutputName::LiteralLhs(s.name().into());
                    } else {
                        out.output_name = OutputName::LiteralLhs(LITERAL_NAME.into());
                    }
                    break;
                },
                AExpr::Alias(node, name) => {
                    out.output_name = OutputName::ColumnLhs(name.clone());
                    out.node = *node;
                    break;
                },
                _ => {},
            }
        }
        debug_assert!(!out.output_name.is_none());
        out
    }

    #[inline]
    pub fn node(&self) -> Node {
        self.node
    }

    pub(crate) fn set_node(&mut self, node: Node) {
        self.node = node;
    }

    #[cfg(feature = "cse")]
    pub(crate) fn set_alias(&mut self, name: ColumnName) {
        self.output_name = OutputName::Alias(name)
    }

    pub(crate) fn output_name_arc(&self) -> &Arc<str> {
        self.output_name.unwrap()
    }

    pub(crate) fn output_name(&self) -> &str {
        self.output_name_arc().as_ref()
    }

    pub(crate) fn to_expr(&self, expr_arena: &Arena<AExpr>) -> Expr {
        let out = node_to_expr(self.node, expr_arena);

        match &self.output_name {
            OutputName::Alias(name) => out.alias(name.as_ref()),
            _ => out,
        }
    }

    pub fn get_alias(&self) -> Option<&ColumnName> {
        match &self.output_name {
            OutputName::Alias(name) => Some(name),
            _ => None,
        }
    }

    // Utility for debugging.
    #[cfg(debug_assertions)]
    #[allow(dead_code)]
    pub(crate) fn print(&self, expr_arena: &Arena<AExpr>) {
        eprintln!("{:?}", self.to_expr(expr_arena))
    }

    pub(crate) fn has_alias(&self) -> bool {
        matches!(self.output_name, OutputName::Alias(_))
    }

    #[cfg(feature = "cse")]
    pub(crate) fn traverse_and_hash<H: Hasher>(&self, expr_arena: &Arena<AExpr>, state: &mut H) {
        traverse_and_hash_aexpr(self.node, expr_arena, state);
        if let Some(alias) = self.get_alias() {
            alias.hash(state)
        }
    }
}

impl AsRef<ExprIR> for ExprIR {
    fn as_ref(&self) -> &ExprIR {
        self
    }
}

/// A Node that is restricted to `AExpr::Column`
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct ColumnNode(pub(crate) Node);

impl From<ColumnNode> for Node {
    fn from(value: ColumnNode) -> Self {
        value.0
    }
}
impl From<&ExprIR> for Node {
    fn from(value: &ExprIR) -> Self {
        value.node()
    }
}

pub(crate) fn name_to_expr_ir(name: &str, expr_arena: &mut Arena<AExpr>) -> ExprIR {
    let name = ColumnName::from(name);
    let node = expr_arena.add(AExpr::Column(name.clone()));
    ExprIR::new(node, OutputName::ColumnLhs(name))
}

pub(crate) fn names_to_expr_irs<I: IntoIterator<Item = S>, S: AsRef<str>>(
    names: I,
    expr_arena: &mut Arena<AExpr>,
) -> Vec<ExprIR> {
    names
        .into_iter()
        .map(|name| {
            let name = name.as_ref();
            name_to_expr_ir(name, expr_arena)
        })
        .collect()
}
