use std::ops::Deref;

use super::*;

#[derive(Debug, Clone)]
pub struct ProjectionExprs {
    expr: Vec<ExprIR>,
    /// offset from the back
    /// `expr[expr.len() - common_sub_offset..]`
    /// are the common sub expressions
    common_sub_offset: usize,
}

impl Deref for ProjectionExprs {
    type Target = Vec<ExprIR>;

    fn deref(&self) -> &Self::Target {
        &self.expr
    }
}

impl From<Vec<ExprIR>> for ProjectionExprs {
    fn from(value: Vec<ExprIR>) -> Self {
        Self::new(value)
    }
}

impl ProjectionExprs {
    pub(crate) fn new(expr: Vec<ExprIR>) -> Self {
        Self::new_with_cse(expr, 0)
    }

    pub fn default_exprs(&self) -> &[ExprIR] {
        &self.expr[..self.expr.len() - self.common_sub_offset]
    }

    pub fn cse_exprs(&self) -> &[ExprIR] {
        &self.expr[self.expr.len() - self.common_sub_offset..]
    }

    pub(crate) fn new_with_cse(expr: Vec<ExprIR>, common_sub_offset: usize) -> Self {
        Self {
            expr,
            common_sub_offset,
        }
    }

    pub fn has_sub_exprs(&self) -> bool {
        self.common_sub_offset != 0
    }

    fn dbg_assert_no_sub_exprs(&self) {
        debug_assert!(!self.has_sub_exprs(), "should not have sub-expressions yet");
    }

    pub(crate) fn exprs(self) -> Vec<ExprIR> {
        self.dbg_assert_no_sub_exprs();
        self.expr
    }

    pub(crate) fn all_exprs(self) -> Vec<ExprIR> {
        self.expr
    }
}

impl IntoIterator for ProjectionExprs {
    type Item = ExprIR;
    type IntoIter = <Vec<ExprIR> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        assert!(!self.has_sub_exprs(), "should not have sub-expressions yet");
        self.expr.into_iter()
    }
}

impl<'a> IntoIterator for &'a ProjectionExprs {
    type Item = &'a ExprIR;
    type IntoIter = <&'a Vec<ExprIR> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        assert!(!self.has_sub_exprs(), "should not have sub-expressions yet");
        self.expr.iter()
    }
}
