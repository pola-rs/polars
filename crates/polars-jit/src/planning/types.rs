use polars_plan::dsl::Expr;
use polars_utils::pl_str::PlSmallStr;

/// Everything that can be learned before Polars supplies concrete columns.
pub(crate) struct PlannedFunction {
    pub function: Expr,
    pub row_names: Box<[PlSmallStr]>,
    pub row_inputs: Box<[Expr]>,
}
