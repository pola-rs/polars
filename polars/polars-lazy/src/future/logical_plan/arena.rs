use crate::logical_plan::optimizer::{AExpr, ALogicalPlan};
use polars_core::prelude::*;
use polars_core::utils::{Arena, Node};
use std::borrow::{Borrow, BorrowMut};
use std::cell::{Ref, RefCell, RefMut};
use std::rc::Rc;

pub type ExprArenaRef = Rc<RefCell<Arena<AExpr>>>;

thread_local! {pub(crate) static EXPR_ARENA: ExprArenaRef = Rc::new(RefCell::new(Arena::default()))}

pub type LPRef = Rc<RefCell<Arena<ALogicalPlan>>>;

thread_local! {pub(crate) static LP_ARENA: LPRef = Rc::new(RefCell::new(Arena::default()))}

pub(crate) fn assign_aexpr(aexpr: AExpr) -> Node {
    EXPR_ARENA.with(|arena| (**arena).borrow_mut().add(aexpr))
}

pub(crate) fn map_aexpr<F>(node: Node, f: F)
where
    F: Fn(AExpr) -> AExpr,
{
    EXPR_ARENA.with(|arena| (**arena).borrow_mut().replace_with(node, f))
}

pub(crate) fn assign_alp(alp: ALogicalPlan) -> Node {
    LP_ARENA.with(|arena| (**arena).borrow_mut().add(alp))
}

pub(crate) fn map_alp<F>(node: Node, f: F) -> Result<()>
where
    F: FnMut(ALogicalPlan) -> Result<ALogicalPlan>,
{
    LP_ARENA.with(|arena| (**arena).borrow_mut().try_replace_with(node, f))
}

pub(crate) fn expr_arena_get() -> ExprArenaRef {
    EXPR_ARENA.with(|arena| arena.clone())
}

pub(crate) fn lp_arena_get() -> LPRef {
    LP_ARENA.with(|arena| arena.clone())
}
