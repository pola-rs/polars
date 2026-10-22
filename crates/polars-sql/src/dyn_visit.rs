//! Entry points for running sqlparser visitors through dynamic dispatch.
//!
//! sqlparser generates the full AST walk for every visitor type. Calling a visitor through
//! these functions generates it once per `Break` type instead. Use them instead of
//! `Visit::visit`, `VisitMut::visit` and sqlparser's `visit_expressions(_mut)`.

use std::ops::ControlFlow;

use sqlparser::ast::{
    Expr, ObjectName, Query, Select, Statement, TableFactor, ValueWithSpan, Visit, VisitMut,
    Visitor, VisitorMut,
};

struct DynVisitor<'a, B>(&'a mut dyn Visitor<Break = B>);

struct DynVisitorMut<'a, B>(&'a mut dyn VisitorMut<Break = B>);

macro_rules! forward {
    ($($method:ident($ty:ty)),* $(,)?) => {
        $(
            fn $method(&mut self, node: $ty) -> ControlFlow<B> {
                self.0.$method(node)
            }
        )*
    };
}

impl<B> Visitor for DynVisitor<'_, B> {
    type Break = B;

    forward!(
        pre_visit_query(&Query),
        post_visit_query(&Query),
        pre_visit_select(&Select),
        post_visit_select(&Select),
        pre_visit_relation(&ObjectName),
        post_visit_relation(&ObjectName),
        pre_visit_table_factor(&TableFactor),
        post_visit_table_factor(&TableFactor),
        pre_visit_expr(&Expr),
        post_visit_expr(&Expr),
        pre_visit_statement(&Statement),
        post_visit_statement(&Statement),
        pre_visit_value(&ValueWithSpan),
        post_visit_value(&ValueWithSpan),
    );
}

impl<B> VisitorMut for DynVisitorMut<'_, B> {
    type Break = B;

    forward!(
        pre_visit_query(&mut Query),
        post_visit_query(&mut Query),
        pre_visit_select(&mut Select),
        post_visit_select(&mut Select),
        pre_visit_relation(&mut ObjectName),
        post_visit_relation(&mut ObjectName),
        pre_visit_table_factor(&mut TableFactor),
        post_visit_table_factor(&mut TableFactor),
        pre_visit_expr(&mut Expr),
        post_visit_expr(&mut Expr),
        pre_visit_statement(&mut Statement),
        post_visit_statement(&mut Statement),
        pre_visit_value(&mut ValueWithSpan),
        post_visit_value(&mut ValueWithSpan),
    );
}

pub(crate) fn visit<B>(
    node: &(impl Visit + ?Sized),
    visitor: &mut dyn Visitor<Break = B>,
) -> ControlFlow<B> {
    node.visit(&mut DynVisitor(visitor))
}

pub(crate) fn visit_mut<B>(
    node: &mut (impl VisitMut + ?Sized),
    visitor: &mut dyn VisitorMut<Break = B>,
) -> ControlFlow<B> {
    node.visit(&mut DynVisitorMut(visitor))
}

/// Calls `f` on every expression in `node`, parents before children.
pub(crate) fn visit_expressions<B>(
    node: &(impl Visit + ?Sized),
    mut f: impl FnMut(&Expr) -> ControlFlow<B>,
) -> ControlFlow<B> {
    struct ExprFn<'a, B>(&'a mut dyn FnMut(&Expr) -> ControlFlow<B>);
    impl<B> Visitor for ExprFn<'_, B> {
        type Break = B;
        fn pre_visit_expr(&mut self, expr: &Expr) -> ControlFlow<B> {
            (self.0)(expr)
        }
    }
    visit(node, &mut ExprFn(&mut f))
}

/// Calls `f` on every expression in `node`, children before parents.
pub(crate) fn visit_expressions_mut<B>(
    node: &mut (impl VisitMut + ?Sized),
    mut f: impl FnMut(&mut Expr) -> ControlFlow<B>,
) -> ControlFlow<B> {
    struct ExprFn<'a, B>(&'a mut dyn FnMut(&mut Expr) -> ControlFlow<B>);
    impl<B> VisitorMut for ExprFn<'_, B> {
        type Break = B;
        fn post_visit_expr(&mut self, expr: &mut Expr) -> ControlFlow<B> {
            (self.0)(expr)
        }
    }
    visit_mut(node, &mut ExprFn(&mut f))
}
