use std::ops::Deref;

use super::*;

#[derive(Debug, Clone)]
pub struct ProjectionExprs {
    expr: Vec<Node>,
    /// offset from the back
    /// `expr[expr.len() - common_sub_offset..]`
    /// are the common sub expressions
    common_sub_offset: usize,
}

impl Deref for ProjectionExprs {
    type Target = Vec<Node>;

    fn deref(&self) -> &Self::Target {
        &self.expr
    }
}

impl From<Vec<Node>> for ProjectionExprs {
    fn from(value: Vec<Node>) -> Self {
        Self::new(value)
    }
}

impl FromIterator<Node> for ProjectionExprs {
    fn from_iter<T: IntoIterator<Item = Node>>(iter: T) -> Self {
        let expr = iter.into_iter().collect();
        Self::new(expr)
    }
}

impl ProjectionExprs {
    pub(crate) fn new(expr: Vec<Node>) -> Self {
        Self::new_with_cse(expr, 0)
    }

    pub fn default_exprs(&self) -> &[Node] {
        &self.expr[..self.expr.len() - self.common_sub_offset]
    }

    pub fn cse_exprs(&self) -> &[Node] {
        &self.expr[self.expr.len() - self.common_sub_offset..]
    }

    pub(crate) fn new_with_cse(expr: Vec<Node>, common_sub_offset: usize) -> Self {
        Self {
            expr,
            common_sub_offset,
        }
    }

    pub(crate) fn has_sub_exprs(&self) -> bool {
        self.common_sub_offset != 0
    }

    fn dbg_assert_no_sub_exprs(&self) {
        debug_assert!(!self.has_sub_exprs(), "should not have sub-expressions yet");
    }

    pub(crate) fn exprs(self) -> Vec<Node> {
        self.dbg_assert_no_sub_exprs();
        self.expr
    }
}

impl IntoIterator for ProjectionExprs {
    type Item = Node;
    type IntoIter = <Vec<Node> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        assert!(!self.has_sub_exprs(), "should not have sub-expressions yet");
        self.expr.into_iter()
    }
}

impl<'a> IntoIterator for &'a ProjectionExprs {
    type Item = &'a Node;
    type IntoIter = <&'a Vec<Node> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        assert!(!self.has_sub_exprs(), "should not have sub-expressions yet");
        self.expr.iter()
    }
}
