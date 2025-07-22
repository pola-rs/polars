use std::borrow::Borrow;
use std::hash::Hash;
#[cfg(feature = "cse")]
use std::hash::Hasher;
use std::sync::OnceLock;

use polars_utils::format_pl_smallstr;
#[cfg(feature = "ir_serde")]
use serde::{Deserialize, Serialize};

use super::*;
use crate::constants::{get_len_name, get_literal_name};

#[derive(Clone, Debug)]
#[cfg_attr(feature = "ir_serde", derive(Serialize, Deserialize))]
pub struct ExprIR {
    /// Output name of this expression.
    output_name: PlSmallStr,
    /// Output dtype of this expression
    /// Reduced expression.
    /// This expression is pruned from `alias` and already expanded.
    node: Node,
    #[cfg_attr(feature = "ir_serde", serde(skip))]
    output_dtype: OnceLock<DataType>,
}

impl Eq for ExprIR {}

impl PartialEq for ExprIR {
    fn eq(&self, other: &Self) -> bool {
        self.node == other.node && self.output_name == other.output_name
    }
}

impl Borrow<Node> for ExprIR {
    fn borrow(&self) -> &Node {
        &self.node
    }
}

impl ExprIR {
    pub fn new(node: Node, output_name: PlSmallStr) -> Self {
        ExprIR {
            output_name,
            node,
            output_dtype: OnceLock::new(),
        }
    }

    pub fn with_dtype(self, dtype: DataType) -> Self {
        let _ = self.output_dtype.set(dtype);
        self
    }

    pub(crate) fn set_dtype(&mut self, dtype: DataType) {
        self.output_dtype = OnceLock::from(dtype);
    }

    pub fn from_node(node: Node, arena: &Arena<AExpr>) -> Self {
        let mut out = Self {
            node,
            output_name: PlSmallStr::EMPTY,
            output_dtype: OnceLock::new(),
        };
        for (_, ae) in arena.iter(node) {
            match ae {
                AExpr::Column(name) => {
                    out.output_name = name.clone();
                    break;
                },
                AExpr::Literal(lv) => {
                    if let LiteralValue::Series(s) = lv {
                        out.output_name = s.name().clone();
                    } else {
                        out.output_name = get_literal_name().clone();
                    }
                    break;
                },
                AExpr::Function {
                    input, function, ..
                } => {
                    match function {
                        #[cfg(feature = "dtype-struct")]
                        IRFunctionExpr::StructExpr(IRStructFunction::FieldByName(name)) => {
                            out.output_name = name.clone();
                        },
                        _ => {
                            if input.is_empty() {
                                out.output_name = format_pl_smallstr!("{}", function);
                            } else {
                                out.output_name = input[0].output_name.clone();
                            }
                        },
                    }
                    break;
                },
                AExpr::AnonymousFunction { input, fmt_str, .. } => {
                    if input.is_empty() {
                        out.output_name = fmt_str.as_ref().clone();
                    } else {
                        out.output_name = input[0].output_name.clone();
                    }
                    break;
                },
                AExpr::Len => {
                    out.output_name = get_len_name();
                    break;
                },
                _ => {},
            }
        }
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
            output_name: Some(self.output_name().as_str()),
            expr_arena,
        }
    }

    pub(crate) fn set_node(&mut self, node: Node) {
        self.node = node;
        self.output_dtype = OnceLock::new();
    }

    pub(crate) fn set_output_name(&mut self, name: PlSmallStr) {
        self.output_name = name
    }

    pub fn with_output_name(&self, output_name: PlSmallStr) -> Self {
        Self {
            output_name,
            node: self.node,
            output_dtype: self.output_dtype.clone(),
        }
    }

    pub fn output_name(&self) -> &PlSmallStr {
        &self.output_name
    }

    pub fn to_expr(&self, expr_arena: &Arena<AExpr>) -> Expr {
        let out = node_to_expr(self.node, expr_arena);
        out.alias(self.output_name().clone())
    }

    // Utility for debugging.
    #[cfg(debug_assertions)]
    #[allow(dead_code)]
    pub(crate) fn print(&self, expr_arena: &Arena<AExpr>) {
        eprintln!("{:?}", self.to_expr(expr_arena))
    }

    #[cfg(feature = "cse")]
    pub(crate) fn traverse_and_hash<H: Hasher>(&self, expr_arena: &Arena<AExpr>, state: &mut H) {
        traverse_and_hash_aexpr(self.node, expr_arena, state);
        self.output_name.hash(state);
    }

    pub fn is_scalar(&self, expr_arena: &Arena<AExpr>) -> bool {
        is_scalar_ae(self.node, expr_arena)
    }

    pub fn dtype(
        &self,
        schema: &Schema,
        ctxt: Context,
        expr_arena: &Arena<AExpr>,
    ) -> PolarsResult<&DataType> {
        match self.output_dtype.get() {
            Some(dtype) => Ok(dtype),
            None => {
                let dtype = expr_arena
                    .get(self.node)
                    .to_dtype(schema, ctxt, expr_arena)?;
                let _ = self.output_dtype.set(dtype);
                Ok(self.output_dtype.get().unwrap())
            },
        }
    }

    pub fn field(
        &self,
        schema: &Schema,
        ctxt: Context,
        expr_arena: &Arena<AExpr>,
    ) -> PolarsResult<Field> {
        let dtype = self.dtype(schema, ctxt, expr_arena)?;
        let name = self.output_name();
        Ok(Field::new(name.clone(), dtype.clone()))
    }

    pub fn into_inner(self) -> (Node, PlSmallStr) {
        (self.node, self.output_name)
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

pub(crate) fn name_to_expr_ir(name: PlSmallStr, expr_arena: &mut Arena<AExpr>) -> ExprIR {
    let node = expr_arena.add(AExpr::Column(name.clone()));
    ExprIR::new(node, name)
}

pub(crate) fn names_to_expr_irs<I, S>(names: I, expr_arena: &mut Arena<AExpr>) -> Vec<ExprIR>
where
    I: IntoIterator<Item = S>,
    S: Into<PlSmallStr>,
{
    names
        .into_iter()
        .map(|name| {
            let name = name.into();
            name_to_expr_ir(name, expr_arena)
        })
        .collect()
}
