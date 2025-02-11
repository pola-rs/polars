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

#[derive(Default, Debug, Clone, Hash, PartialEq, Eq)]
#[cfg_attr(feature = "ir_serde", derive(Serialize, Deserialize))]
pub enum OutputName {
    /// No not yet set.
    #[default]
    None,
    /// The most left-hand-side literal will be the output name.
    LiteralLhs(PlSmallStr),
    /// The most left-hand-side column will be the output name.
    ColumnLhs(PlSmallStr),
    /// Rename the output as `PlSmallStr`.
    Alias(PlSmallStr),
    #[cfg(feature = "dtype-struct")]
    /// A struct field.
    Field(PlSmallStr),
}

impl OutputName {
    pub fn get(&self) -> Option<&PlSmallStr> {
        match self {
            OutputName::Alias(name) => Some(name),
            OutputName::ColumnLhs(name) => Some(name),
            OutputName::LiteralLhs(name) => Some(name),
            #[cfg(feature = "dtype-struct")]
            OutputName::Field(name) => Some(name),
            OutputName::None => None,
        }
    }

    pub fn unwrap(&self) -> &PlSmallStr {
        self.get().expect("no output name set")
    }

    pub(crate) fn is_none(&self) -> bool {
        matches!(self, OutputName::None)
    }
}

#[derive(Debug)]
#[cfg_attr(feature = "ir_serde", derive(Serialize, Deserialize))]
pub struct ExprIR {
    /// Output name of this expression.
    output_name: OutputName,
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

impl Clone for ExprIR {
    fn clone(&self) -> Self {
        let output_dtype = OnceLock::new();
        if let Some(dt) = self.output_dtype.get() {
            output_dtype.set(dt.clone()).unwrap()
        }

        ExprIR {
            output_name: self.output_name.clone(),
            node: self.node,
            output_dtype,
        }
    }
}

impl Borrow<Node> for ExprIR {
    fn borrow(&self) -> &Node {
        &self.node
    }
}

impl ExprIR {
    pub fn new(node: Node, output_name: OutputName) -> Self {
        debug_assert!(!output_name.is_none());
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
            output_name: OutputName::None,
            output_dtype: OnceLock::new(),
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
                        out.output_name = OutputName::LiteralLhs(s.name().clone());
                    } else {
                        out.output_name = OutputName::LiteralLhs(get_literal_name().clone());
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
                                out.output_name =
                                    OutputName::LiteralLhs(format_pl_smallstr!("{}", function));
                            } else {
                                out.output_name = input[0].output_name.clone();
                            }
                        },
                    }
                    break;
                },
                AExpr::AnonymousFunction { input, options, .. } => {
                    if input.is_empty() {
                        out.output_name =
                            OutputName::LiteralLhs(PlSmallStr::from_static(options.fmt_str));
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
        self.output_dtype = OnceLock::new();
    }

    pub(crate) fn set_alias(&mut self, name: PlSmallStr) {
        self.output_name = OutputName::Alias(name)
    }

    pub fn output_name_inner(&self) -> &OutputName {
        &self.output_name
    }

    pub fn output_name(&self) -> &PlSmallStr {
        self.output_name.unwrap()
    }

    pub fn to_expr(&self, expr_arena: &Arena<AExpr>) -> Expr {
        let out = node_to_expr(self.node, expr_arena);

        match &self.output_name {
            OutputName::Alias(name) => out.alias(name.clone()),
            _ => out,
        }
    }

    pub fn get_alias(&self) -> Option<&PlSmallStr> {
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
    ExprIR::new(node, OutputName::ColumnLhs(name))
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
