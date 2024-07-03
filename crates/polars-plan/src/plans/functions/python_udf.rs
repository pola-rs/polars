use super::*;

pub(super) fn call_python_udf(
    function: &PythonFunction,
    df: DataFrame,
    validate_output: bool,
    opt_schema: Option<&Schema>,
) -> PolarsResult<DataFrame> {
    let expected_schema = if let Some(schema) = opt_schema {
        Some(Cow::Borrowed(schema))
    }
    // only materialize if we validate the output
    else if validate_output {
        Some(Cow::Owned(df.schema()))
    }
    // do not materialize the schema, we will ignore it.
    else {
        None
    };
    let out = DataFrameUdf::call_udf(function, df)?;

    if validate_output {
        let output_schema = out.schema();
        let expected = expected_schema.unwrap();
        if expected.as_ref() != &output_schema {
            return Err(PolarsError::ComputeError(
                format!(
                    "The output schema of 'LazyFrame.map' is incorrect. Expected: {expected:?}\n\
                        Got: {output_schema:?}"
                )
                .into(),
            ));
        }
    }
    Ok(out)
}
