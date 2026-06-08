use std::sync::Arc;

use polars_error::PolarsResult;
use polars_ops::frame::MaintainOrderJoin;

use crate::plans::{IR, OptimizationRule};

pub struct ForceMaintainOrderJoin;

impl OptimizationRule for ForceMaintainOrderJoin {
    fn optimize_plan(
        &mut self,
        ir_arena: &mut polars_utils::arena::Arena<IR>,
        _expr_arena: &mut polars_utils::arena::Arena<crate::prelude::AExpr>,
        node: polars_utils::arena::Node,
    ) -> PolarsResult<Option<IR>> {
        let IR::Join { options, .. } = ir_arena.get_mut(node) else {
            return Ok(None);
        };

        let maintain_order = match options.args.maintain_order {
            MaintainOrderJoin::None => MaintainOrderJoin::LeftRight,
            MaintainOrderJoin::Left => MaintainOrderJoin::LeftRight,
            MaintainOrderJoin::Right => MaintainOrderJoin::RightLeft,
            MaintainOrderJoin::LeftRight | MaintainOrderJoin::RightLeft => return Ok(None),
        };

        Arc::make_mut(options).args.maintain_order = maintain_order;

        Ok(Some(ir_arena.take(node)))
    }
}
