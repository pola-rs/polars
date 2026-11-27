use std::sync::Arc;

use delta_kernel::arrow::array::{RecordBatch, new_null_array};
use delta_kernel::engine::arrow_data::ArrowEngineData;
use delta_kernel::expressions::{ColumnName, Expression, ExpressionRef, Scalar, null_lit};
use delta_kernel::schema::{DataType, SchemaRef, StructType};
use delta_kernel::{
    EngineData, EvaluationHandler, ExpressionEvaluator, KernelResult, PredicateEvaluator,
    PredicateRef,
};

use super::with_checkpoint_stats_text;

pub(super) struct HostEvaluation(pub(super) Arc<dyn EvaluationHandler>);

const TYPED_PARTITIONS: &str = "partitionValues_parsed";

impl EvaluationHandler for HostEvaluation {
    fn new_expression_evaluator(
        &self,
        input_schema: SchemaRef,
        expression: ExpressionRef,
        output_type: DataType,
    ) -> KernelResult<Arc<dyn ExpressionEvaluator>> {
        if let DataType::Struct(output) = &output_type
            && input_schema.field("add").is_some()
            && output.field("fileConstantValues").is_some()
            && output.field("path").is_some()
        {
            if let Some(expression) = scan_projection(&expression, output) {
                return self.0.new_expression_evaluator(
                    with_checkpoint_stats_text(input_schema)?,
                    expression,
                    output_type,
                );
            }
            if output.field(TYPED_PARTITIONS).is_some() {
                let inner =
                    self.0
                        .new_expression_evaluator(input_schema, expression, output_type)?;
                return Ok(Arc::new(UnknownPartitionValues(inner)));
            }
        }
        self.0
            .new_expression_evaluator(input_schema, expression, output_type)
    }

    fn new_predicate_evaluator(
        &self,
        input_schema: SchemaRef,
        predicate: PredicateRef,
    ) -> KernelResult<Arc<dyn PredicateEvaluator>> {
        self.0.new_predicate_evaluator(input_schema, predicate)
    }

    fn create_many(
        &self,
        schema: SchemaRef,
        rows: Vec<Vec<Scalar>>,
    ) -> KernelResult<Box<dyn EngineData>> {
        self.0.create_many(schema, rows)
    }
}

fn column_is(expression: &Expression, path: &[&str]) -> bool {
    matches!(expression, Expression::Column(column) if column.path() == path)
}

fn scan_projection(expression: &ExpressionRef, output: &StructType) -> Option<ExpressionRef> {
    let Expression::Struct(fields, validity) = expression.as_ref() else {
        return None;
    };
    let index = |name| output.fields().position(|field| field.name() == name);
    if fields.len() != output.num_fields()
        || !column_is(fields.get(index("path")?)?, &["add", "path"])
    {
        return None;
    }
    let mut rewritten = None;
    // Scan replay has no Kernel predicate. Preserve raw JSON statistics for the Polars
    // parser, including Boolean bounds and decimal precision.
    if let (Some(raw), Some(typed)) = (index("stats"), index("stats_parsed")) {
        if let Expression::ParseJson(parse) = fields.get(typed)?.as_ref()
            && column_is(&parse.json_expr, &["add", "stats"])
            && column_is(fields.get(raw)?, &["add", "stats"])
        {
            let mut fields = fields.clone();
            fields[typed] = Arc::new(null_lit(output.field("stats_parsed")?.data_type().clone()));
            rewritten = Some(fields);
        } else if column_is(fields.get(typed)?, &["add", "stats_parsed"])
            && matches!(
                fields.get(raw)?.as_ref(),
                Expression::Literal(Scalar::Null(DataType::STRING))
            )
        {
            // The checkpoint reader also projects the original text for unpopulated
            // typed rows. Export it directly and leave the native struct intact.
            let mut fields = fields.clone();
            fields[raw] = Arc::new(Expression::Column(ColumnName::new(["add", "stats"])));
            rewritten = Some(fields);
        }
    }
    // Only a direct projection of the stored checkpoint field establishes the writer's
    // timestamp partition instant. Kernel's map conversion would assume a zone for raw text.
    if let Some(partitions) = index(TYPED_PARTITIONS)
        && !column_is(fields.get(partitions)?, &["add", TYPED_PARTITIONS])
    {
        rewritten.get_or_insert_with(|| fields.clone())[partitions] = Arc::new(null_lit(
            output.field(TYPED_PARTITIONS)?.data_type().clone(),
        ));
    }
    Some(match rewritten {
        Some(fields) => Arc::new(Expression::Struct(fields, validity.clone())),
        None => expression.clone(),
    })
}

// Keep the same safety rule if a future Kernel produces scan metadata with an expression
// other than the expected struct of columns and conversions.
struct UnknownPartitionValues(Arc<dyn ExpressionEvaluator>);

impl ExpressionEvaluator for UnknownPartitionValues {
    fn evaluate(&self, batch: &dyn EngineData) -> KernelResult<Box<dyn EngineData>> {
        let output = ArrowEngineData::try_from_engine_data(self.0.evaluate(batch)?)?;
        let record = output.record_batch();
        let schema = record.schema();
        let index = schema.index_of(TYPED_PARTITIONS)?;
        let mut columns = record.columns().to_vec();
        columns[index] = new_null_array(columns[index].data_type(), record.num_rows());
        let record = RecordBatch::try_new(schema, columns)?;
        Ok(Box::new(ArrowEngineData::new(record)))
    }
}
