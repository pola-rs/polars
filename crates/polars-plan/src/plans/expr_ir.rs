use std::borrow::Borrow;
use std::hash::Hash;
#[cfg(feature = "cse")]
use std::hash::Hasher;

#[cfg(feature = "ir_serde")]
use serde::{Deserialize, Serialize};

use super::*;
use crate::constants::{get_len_name, LITERAL_NAME};

#[derive(Default, Debug, Clone, Hash, PartialEq, Eq)]
#[cfg_attr(feature = "ir_serde", derive(Serialize, Deserialize))]
pub enum OutputName {
    /// No not yet set.
    #[default]
    None,
    /// The most left-hand-side literal will be the output name.
    LiteralLhs(ColumnName),
    /// The most left-hand-side column will be the output name.
    ColumnLhs(ColumnName),
    /// Rename the output as `ColumnName`.
    Alias(ColumnName),
    #[cfg(feature = "dtype-struct")]
    /// A struct field.
    Field(ColumnName),
}

impl OutputName {
    pub fn unwrap(&self) -> &ColumnName {
        match self {
            OutputName::Alias(name) => name,
            OutputName::ColumnLhs(name) => name,
            OutputName::LiteralLhs(name) => name,
            #[cfg(feature = "dtype-struct")]
            OutputName::Field(name) => name,
            OutputName::None => panic!("no output name set"),
        }
    }

    pub(crate) fn is_none(&self) -> bool {
        matches!(self, OutputName::None)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "ir_serde", derive(Serialize, Deserialize))]
pub struct ExprIR {
    /// Output name of this expression.
    output_name: OutputName,
    /// Reduced expression.
    /// This expression is pruned from `alias` and already expanded.
    node: Node,
}

impl Borrow<Node> for ExprIR {
    fn borrow(&self) -> &Node {
        &self.node
    }
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
                AExpr::Function {
                    input, function, ..
                } => {
                    match function {
                        #[cfg(feature = "dtype-struct")]
                        FunctionExpr::StructExpr(StructFunction::FieldByName(name)) => {
                            out.output_name = OutputName::Field(name.clone());
                        },
                        _ => {
                            if input.is_empty() {
                                out.output_name = OutputName::LiteralLhs(ColumnName::from(
                                    format!("{}", function),
                                ));
                            } else {
                                out.output_name = input[0].output_name.clone();
                            }
                        },
                    }
                    break;
                },
                AExpr::AnonymousFunction { input, options, .. } => {
                    if input.is_empty() {
                        out.output_name = OutputName::LiteralLhs(ColumnName::from(options.fmt_str));
                    } else {
                        out.output_name = input[0].output_name.clone();
                    }
                    break;
                },
                AExpr::Len => out.output_name = OutputName::LiteralLhs(get_len_name()),
                AExpr::Alias(_, _) => {
                    // Should be removed during conversion.
                    #[cfg(debug_assertions)]
                    {
                        unreachable!()
                    }
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

    /// Create a `ExprIR` structure that implements display
    pub fn display<'a>(&'a self, expr_arena: &'a Arena<AExpr>) -> ExprIRDisplay<'a> {
        ExprIRDisplay {
            node: self.node(),
            output_name: self.output_name_inner(),
            expr_arena,
        }
    }

    pub(crate) fn set_node(&mut self, node: Node) {
        self.node = node;
    }

    #[cfg(feature = "cse")]
    pub(crate) fn set_alias(&mut self, name: ColumnName) {
        self.output_name = OutputName::Alias(name)
    }

    pub fn output_name_inner(&self) -> &OutputName {
        &self.output_name
    }

    pub(crate) fn output_name_arc(&self) -> &Arc<str> {
        self.output_name.unwrap()
    }

    pub fn output_name(&self) -> &str {
        self.output_name_arc().as_ref()
    }

    pub fn to_expr(&self, expr_arena: &Arena<AExpr>) -> Expr {
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

    /// Gets any name except one deriving from `Column`.
    pub(crate) fn get_non_projected_name(&self) -> Option<&ColumnName> {
        match &self.output_name {
            OutputName::Alias(name) => Some(name),
            #[cfg(feature = "dtype-struct")]
            OutputName::Field(name) => Some(name),
            OutputName::LiteralLhs(name) => Some(name),
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
