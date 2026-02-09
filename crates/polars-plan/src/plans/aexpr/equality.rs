use polars_core::prelude::SortOptions;
use polars_utils::arena::{Arena, Node};

use super::{AExpr, IRAggExpr};

impl AExpr {
    pub fn is_expr_equal_to(&self, other: &Self, arena: &Arena<AExpr>) -> bool {
        let mut l_stack = Vec::new();
        let mut r_stack = Vec::new();
        self.is_expr_equal_to_amortized(other, arena, &mut l_stack, &mut r_stack)
    }

    pub fn is_expr_equal_to_amortized(
        &self,
        other: &Self,
        arena: &Arena<AExpr>,
        l_stack: &mut Vec<Node>,
        r_stack: &mut Vec<Node>,
    ) -> bool {
        l_stack.clear();
        r_stack.clear();

        // Top-Level node.
        if !self.is_expr_equal_top_level(other) {
            return false;
        }
        self.children_rev(l_stack);
        other.children_rev(r_stack);

        // Traverse node in N R L order
        loop {
            assert_eq!(l_stack.len(), r_stack.len());

            let (Some(l_node), Some(r_node)) = (l_stack.pop(), r_stack.pop()) else {
                break;
            };

            let l_expr = arena.get(l_node);
            let r_expr = arena.get(r_node);

            if !l_expr.is_expr_equal_top_level(r_expr) {
                return false;
            }
            l_expr.children_rev(l_stack);
            r_expr.children_rev(r_stack);
        }

        true
    }

    pub fn is_expr_equal_top_level(&self, other: &Self) -> bool {
        if std::mem::discriminant(self) != std::mem::discriminant(other) {
            // Fast path: different kind of expression.
            return false;
        }

        use AExpr as E;

        // @NOTE: Intentionally written as a match statement over only `self` as it forces the
        // match to be exhaustive.
        #[rustfmt::skip]
        let is_equal = match self {
            E::Explode { expr: _, options: l_options } => matches!(other, E::Explode { expr: _, options: r_options } if l_options == r_options),
            E::Column(l_name) => matches!(other, E::Column(r_name) if l_name == r_name),
            #[cfg(feature = "dtype-struct")]
            E::StructField (l_name) => matches!(other, E::StructField(r_name) if l_name == r_name),
            E::Literal(l_lit) => matches!(other, E::Literal(r_lit) if l_lit == r_lit),
            E::BinaryExpr { left: _, op: l_op, right: _ } => matches!(other, E::BinaryExpr { left: _, op: r_op, right: _ } if l_op == r_op),
            E::Cast { expr: _, dtype: l_dtype, options: l_options } => matches!(other, E::Cast { expr: _, dtype: r_dtype, options: r_options } if l_dtype == r_dtype && l_options == r_options),
            E::Sort { expr: _, options: l_options } => matches!(other, E::Sort { expr: _, options: r_options } if l_options == r_options),
            E::Gather { expr: _, idx: l_idx, returns_scalar: l_returns_scalar, null_on_oob: l_null_on_oob } => matches!(other, E::Gather { expr: _, idx: r_idx, returns_scalar: r_returns_scalar, null_on_oob: r_null_on_oob } if l_idx == r_idx && l_returns_scalar == r_returns_scalar && l_null_on_oob == r_null_on_oob),
            E::SortBy { expr: _, by: l_by, sort_options: l_sort_options } => matches!(other, E::SortBy { expr: _, by: r_by, sort_options: r_sort_options } if l_by.len() == r_by.len() && l_sort_options == r_sort_options),
            E::Agg(l_agg) => matches!(other, E::Agg(r_agg) if l_agg.is_agg_equal_top_level(r_agg)),
            E::AnonymousAgg { input: input_l, fmt_str: fmt_str_l, function: function_l } => matches!(other, E::AnonymousAgg { input: input_r, fmt_str: fmt_str_r, function: function_r} if input_l == input_r && function_l == function_r && fmt_str_l == fmt_str_r),
            E::AnonymousFunction { input: l_input, function: l_function, options: l_options, fmt_str: l_fmt_str } => matches!(other, E::AnonymousFunction { input: r_input, function: r_function, options: r_options, fmt_str: r_fmt_str } if l_input.len() == r_input.len() && l_function == r_function && l_options == r_options && l_fmt_str == r_fmt_str),
            E::Eval { expr: _, evaluation: _, variant: l_variant } => matches!(other, E::Eval { expr: _, evaluation: _, variant: r_variant } if l_variant == r_variant),
            E::Function { input: l_input, function: l_function, options: l_options } => matches!(other, E::Function { input: r_input, function: r_function, options: r_options } if l_input.len() == r_input.len() && l_function == r_function && l_options == r_options),
            #[cfg(feature = "dynamic_group_by")]
            E::Rolling { function: _, index_column: _, period: l_period, offset: l_offset, closed_window: l_closed_window } => matches!(other, E::Rolling { function: _, index_column: _, period: r_period, offset: r_offset, closed_window: r_closed_window } if l_period == r_period && l_offset == r_offset && l_closed_window == r_closed_window),
            E::Over { function: _, partition_by: l_partition_by, order_by: l_order_by, mapping: l_mapping } => matches!(other, E::Over { function: _, partition_by: r_partition_by, order_by: r_order_by, mapping: r_mapping } if l_partition_by.len() == r_partition_by.len() && l_order_by.as_ref().map(|(_, v): &(Node, SortOptions)| v) == r_order_by.as_ref().map(|(_, v): &(Node, SortOptions)| v) && l_mapping == r_mapping),

            // Discriminant check done above.
            E::Element |
            E::Filter { input: _, by: _ } |
            E::Ternary { predicate: _, truthy: _, falsy: _ } |
            E::Slice { input: _, offset: _, length: _ } |
            E::Len => true,
            #[cfg(feature = "dtype-struct")]
            E::StructEval { expr: _, evaluation: _} => true
        };

        is_equal
    }
}

impl IRAggExpr {
    pub fn is_agg_equal_top_level(&self, other: &Self) -> bool {
        if std::mem::discriminant(self) != std::mem::discriminant(other) {
            // Fast path: different kind of expression.
            return false;
        }

        use IRAggExpr as A;

        // @NOTE: Intentionally written as a match statement over only `self` as it forces the
        // match to be exhaustive.
        #[rustfmt::skip]
        let is_equal = match self {
            A::Min { input: _, propagate_nans: l_propagate_nans } => matches!(other, A::Min { input: _, propagate_nans: r_propagate_nans } if l_propagate_nans == r_propagate_nans),
            A::Max { input: _, propagate_nans: l_propagate_nans } => matches!(other, A::Max { input: _, propagate_nans: r_propagate_nans } if l_propagate_nans == r_propagate_nans),
            A::Quantile { expr: _, quantile: _, method: l_method } => matches!(other, A::Quantile { expr: _, quantile: _, method: r_method } if l_method == r_method),
            A::Count { input: _, include_nulls: l_include_nulls } => matches!(other, A::Count { input: _, include_nulls: r_include_nulls } if l_include_nulls == r_include_nulls),
            A::Item { input: _, allow_empty: l_allow_empty } => matches!(other, A::Item { input: _, allow_empty: r_allow_empty } if l_allow_empty == r_allow_empty),
            A::Std(_, l_ddof) => matches!(other, A::Std(_, r_ddof) if l_ddof == r_ddof),
            A::Var(_, l_ddof) => matches!(other, A::Var(_, r_ddof) if l_ddof == r_ddof),

            // Discriminant check done above.
            A::Median(_) |
            A::NUnique(_) |
            A::First(_) |
            A::FirstNonNull(_) |
            A::Last(_) |
            A::LastNonNull(_) |
            A::Mean(_) |
            A::Implode(_) |
            A::Sum(_) |
            A::AggGroups(_)  => true,
        };

        is_equal
    }
}
