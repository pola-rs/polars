//! Translation of explicit casts.

use polars_error::{PolarsResult, polars_bail, polars_ensure};
use polars_plan::dsl::{DataTypeExpr, Expr};

use super::Translator;
use crate::jit_program::{JitProgram, ScalarType};
use crate::translate::helpers::at;

impl Translator<'_> {
    /// A cast that cannot produce null; casts that can wait for null support.
    pub(super) fn cast_expression(
        &self,
        input: &Expr,
        dtype: &DataTypeExpr,
        location: &str,
    ) -> PolarsResult<JitProgram> {
        let DataTypeExpr::Literal(dtype) = dtype else {
            polars_bail!(InvalidOperation:
                "foldv only supports casts to concrete scalar dtypes at {location}")
        };
        let target = ScalarType::from_dtype(dtype)?;
        let input = self.expression(input, None, &at(location, "input"))?;
        polars_ensure!(cast_stays_non_null(input.dtype(), target), InvalidOperation:
            "foldv cast from {} to {} may produce null and is not supported yet",
            input.dtype().dtype(), target.dtype());
        self.cast(input, target)
    }
}

/// Whether every value of `source` has a value in `target`, so the cast can never produce null.
fn cast_stays_non_null(source: ScalarType, target: ScalarType) -> bool {
    if source == target || target.is_float() {
        return true;
    }
    if source.is_float() || target == ScalarType::Boolean {
        return false;
    }
    let source_bits = source.bit_width();
    let target_bits = target.bit_width();
    (source.is_signed() == target.is_signed() && target_bits >= source_bits)
        || (!source.is_signed() && target.is_signed() && target_bits > source_bits)
}
