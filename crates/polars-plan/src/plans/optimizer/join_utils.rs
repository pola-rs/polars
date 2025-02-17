use polars_core::error::{polars_bail, PolarsResult};
use polars_core::schema::*;
use polars_utils::arena::{Arena, Node};
use polars_utils::pl_str::PlSmallStr;

use super::{aexpr_to_leaf_names_iter, AExpr};
use crate::plans::visitor::{AexprNode, RewriteRecursion, RewritingVisitor, TreeWalker};
use crate::plans::{ExprIR, OutputName};

/// Join origin of an expression
#[derive(Debug, Clone, PartialEq, Copy)]
#[repr(u8)]
pub(crate) enum ExprOrigin {
    // Note: BitOr is implemented on this struct that relies on this exact u8
    // repr layout (i.e. treated as a bitfield).
    //
    /// Utilizes no columns
    None = 0b00,
    /// Utilizes columns from the left side of the join
    Left = 0b10,
    /// Utilizes columns from the right side of the join
    Right = 0b01,
    /// Utilizes columns from both sides of the join
    Both = 0b11,
}

impl ExprOrigin {
    /// Errors with ColumnNotFound if a column cannot be found on either side.
    pub(crate) fn get_expr_origin(
        root: Node,
        expr_arena: &Arena<AExpr>,
        left_schema: &Schema,
        right_schema: &Schema,
        suffix: &str,
    ) -> PolarsResult<ExprOrigin> {
        aexpr_to_leaf_names_iter(root, expr_arena).try_fold(
            ExprOrigin::None,
            |acc_origin, column_name| {
                Ok(acc_origin
                    | Self::get_column_origin(&column_name, left_schema, right_schema, suffix)?)
            },
        )
    }

    /// Errors with ColumnNotFound if a column cannot be found on either side.
    pub(crate) fn get_column_origin(
        column_name: &str,
        left_schema: &Schema,
        right_schema: &Schema,
        suffix: &str,
    ) -> PolarsResult<ExprOrigin> {
        Ok(if left_schema.contains(column_name) {
            ExprOrigin::Left
        } else if right_schema.contains(column_name)
            || column_name
                .strip_suffix(suffix)
                .is_some_and(|x| right_schema.contains(x))
        {
            ExprOrigin::Right
        } else {
            polars_bail!(ColumnNotFound: "{}", column_name)
        })
    }
}

impl std::ops::BitOr for ExprOrigin {
    type Output = ExprOrigin;

    fn bitor(self, rhs: Self) -> Self::Output {
        unsafe { std::mem::transmute::<u8, ExprOrigin>(self as u8 | rhs as u8) }
    }
}

impl std::ops::BitOrAssign for ExprOrigin {
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

pub(super) fn remove_suffix<'a>(
    expr: &mut ExprIR,
    expr_arena: &mut Arena<AExpr>,
    schema_rhs: &'a Schema,
    suffix: &'a str,
) {
    let schema = schema_rhs;
    // Using AexprNode::rewrite() ensures we do not mutate any nodes in-place. The nodes may be
    // used in other locations and mutating them will cause really confusing bugs, such as
    // https://github.com/pola-rs/polars/issues/20831.
    let node = AexprNode::new(expr.node())
        .rewrite(&mut RemoveSuffix { schema, suffix }, expr_arena)
        .unwrap()
        .node();

    expr.set_node(node);

    if let OutputName::ColumnLhs(colname) = expr.output_name_inner() {
        if colname.ends_with(suffix) && !schema.contains(colname.as_str()) {
            let name = PlSmallStr::from(&colname[..colname.len() - suffix.len()]);
            expr.set_columnlhs(name);
        }
    }

    struct RemoveSuffix<'a> {
        schema: &'a Schema,
        suffix: &'a str,
    }

    impl RewritingVisitor for RemoveSuffix<'_> {
        type Node = AexprNode;
        type Arena = Arena<AExpr>;

        fn pre_visit(
            &mut self,
            node: &Self::Node,
            arena: &mut Self::Arena,
        ) -> polars_core::prelude::PolarsResult<crate::prelude::visitor::RewriteRecursion> {
            let AExpr::Column(colname) = arena.get(node.node()) else {
                return Ok(RewriteRecursion::NoMutateAndContinue);
            };

            if !colname.ends_with(self.suffix) || self.schema.contains(colname.as_str()) {
                return Ok(RewriteRecursion::NoMutateAndContinue);
            }

            Ok(RewriteRecursion::MutateAndContinue)
        }

        fn mutate(
            &mut self,
            node: Self::Node,
            arena: &mut Self::Arena,
        ) -> polars_core::prelude::PolarsResult<Self::Node> {
            let AExpr::Column(colname) = arena.get(node.node()) else {
                unreachable!();
            };

            // Safety: Checked in pre_visit()
            Ok(AexprNode::new(arena.add(AExpr::Column(PlSmallStr::from(
                &colname[..colname.len() - self.suffix.len()],
            )))))
        }
    }
}

pub(super) fn split_suffix<'a>(name: &'a str, suffix: &str) -> &'a str {
    let (original, _) = name.split_at(name.len() - suffix.len());
    original
}
