use polars_core::schema::*;
#[cfg(feature = "iejoin")]
use polars_utils::arena::{Arena, Node};

use super::{aexpr_to_leaf_names_iter, AExpr};

/// Join origin of an expression
#[derive(Debug, Clone, PartialEq, Copy)]
#[repr(u8)]
pub(crate) enum ExprOrigin {
    // Note: There is a merge() function implemented on this enum that relies
    // on this exact u8 repr layout.
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
    pub(crate) fn get_expr_origin(
        root: Node,
        expr_arena: &Arena<AExpr>,
        left_schema: &SchemaRef,
        right_schema: &SchemaRef,
        suffix: &str,
    ) -> ExprOrigin {
        let mut expr_origin = ExprOrigin::None;

        for name in aexpr_to_leaf_names_iter(root, expr_arena) {
            let in_left = left_schema.contains(name.as_str());
            let in_right = right_schema.contains(name.as_str());
            let has_suffix = name.as_str().ends_with(suffix);
            let in_right = in_right
                | (has_suffix
                    && right_schema.contains(&name.as_str()[..name.len() - suffix.len()]));

            let name_origin = match (in_left, in_right, has_suffix) {
                (true, false, _) | (true, true, false) => ExprOrigin::Left,
                (false, true, _) | (true, true, true) => ExprOrigin::Right,
                (false, false, _) => {
                    unreachable!("Invalid filter column should have been filtered before")
                },
            };

            use ExprOrigin as O;
            expr_origin = match (expr_origin, name_origin) {
                (O::None, other) | (other, O::None) => other,
                (O::Left, O::Left) => O::Left,
                (O::Right, O::Right) => O::Right,
                _ => O::Both,
            };
        }

        expr_origin
    }

    /// Logical OR with another [`ExprOrigin`]
    fn merge(&mut self, other: Self) {
        *self = unsafe { std::mem::transmute::<u8, ExprOrigin>(*self as u8 | other as u8) }
    }
}

impl std::ops::BitOrAssign for ExprOrigin {
    fn bitor_assign(&mut self, rhs: Self) {
        self.merge(rhs)
    }
}

pub(super) fn split_suffix<'a>(name: &'a str, suffix: &str) -> &'a str {
    let (original, _) = name.split_at(name.len() - suffix.len());
    original
}
