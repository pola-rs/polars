use super::*;

/// Get the datatypes of function arguments.
///
/// If all arguments give the same datatype or a datatype cannot be found, `Ok(None)` is returned.
pub(super) fn get_function_dtypes(
    input: &[ExprIR],
    expr_arena: &Arena<AExpr>,
    input_schema: &Schema,
    function: &FunctionExpr,
) -> PolarsResult<Option<Vec<DataType>>> {
    let mut dtypes = Vec::with_capacity(input.len());
    let mut first = true;
    for e in input {
        let Some((_, dtype)) = get_aexpr_and_type(expr_arena, e.node(), input_schema) else {
            return Ok(None);
        };

        if first {
            check_namespace(function, &dtype)?;
            first = false;
        }
        // Ignore Unknown in the inputs.
        // We will raise if we cannot find the supertype later.
        match dtype {
            DataType::Unknown(UnknownKind::Any) => {
                return Ok(None);
            },
            _ => dtypes.push(dtype),
        }
    }

    if dtypes.iter().all_equal() {
        return Ok(None);
    }
    Ok(Some(dtypes))
}

// `str` namespace belongs to `String`
// `cat` namespace belongs to `Categorical` etc.
fn check_namespace(function: &FunctionExpr, first_dtype: &DataType) -> PolarsResult<()> {
    match function {
        #[cfg(feature = "strings")]
        FunctionExpr::StringExpr(_) => {
            polars_ensure!(first_dtype == &DataType::String, InvalidOperation: "expected String type, got: {}", first_dtype)
        },
        FunctionExpr::BinaryExpr(_) => {
            polars_ensure!(first_dtype == &DataType::Binary, InvalidOperation: "expected Binary type, got: {}", first_dtype)
        },
        #[cfg(feature = "temporal")]
        FunctionExpr::TemporalExpr(_) => {
            polars_ensure!(first_dtype.is_temporal(), InvalidOperation: "expected Date(time)/Duration type, got: {}", first_dtype)
        },
        FunctionExpr::ListExpr(_) => {
            polars_ensure!(matches!(first_dtype, DataType::List(_)), InvalidOperation: "expected List type, got: {}", first_dtype)
        },
        #[cfg(feature = "dtype-array")]
        FunctionExpr::ArrayExpr(_) => {
            polars_ensure!(matches!(first_dtype, DataType::Array(_, _)), InvalidOperation: "expected Array type, got: {}", first_dtype)
        },
        #[cfg(feature = "dtype-struct")]
        FunctionExpr::StructExpr(_) => {
            polars_ensure!(matches!(first_dtype, DataType::Struct(_)), InvalidOperation: "expected Struct type, got: {}", first_dtype)
        },
        #[cfg(feature = "dtype-categorical")]
        FunctionExpr::Categorical(_) => {
            polars_ensure!(matches!(first_dtype, DataType::Categorical(_, _)), InvalidOperation: "expected Categorical type, got: {}", first_dtype)
        },
        _ => {},
    }

    Ok(())
}
