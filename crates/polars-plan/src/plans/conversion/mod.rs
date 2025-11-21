mod convert_utils;
mod dsl_to_ir;
mod ir_to_dsl;
mod stack_opt;

use std::sync::{Arc, Mutex};

pub use dsl_to_ir::*;
pub use ir_to_dsl::*;
use polars_core::prelude::*;
use polars_utils::idx_vec::UnitVec;
use polars_utils::unitvec;
use polars_utils::vec::ConvertVec;
use recursive::recursive;
pub(crate) mod type_check;
pub(crate) mod type_coercion;

pub use dsl_to_ir::{is_regex_projection, prepare_projection};
pub(crate) use stack_opt::ConversionOptimizer;

use crate::constants::get_len_name;
use crate::prelude::*;

fn expr_irs_to_exprs(expr_irs: Vec<ExprIR>, expr_arena: &Arena<AExpr>) -> Vec<Expr> {
    expr_irs.convert_owned(|e| e.to_expr(expr_arena))
}
