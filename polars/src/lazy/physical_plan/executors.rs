use super::*;
use arrow::datatypes::SchemaRef;

#[derive(Debug)]
pub struct CsvExec {
    path: String,
    schema: Option<SchemaRef>,
    has_header: bool,
    delimiter: Option<u8>,
}

impl CsvExec {
    pub fn new(
        path: String,
        schema: Option<SchemaRef>,
        has_header: bool,
        delimiter: Option<u8>,
    ) -> Self {
        CsvExec {
            path,
            schema,
            has_header,
            delimiter,
        }
    }
}

impl ExecutionPlan for CsvExec {
    fn execute(&self) -> Result<DataStructure> {
        let file = std::fs::File::open(&self.path).unwrap();

        let df = CsvReader::new(file)
            .has_header(self.has_header)
            .with_batch_size(10000)
            .finish()?;
        Ok(DataStructure::DataFrame(df))
    }
}

#[derive(Debug)]
pub struct FilterExec {
    predicate: Rc<dyn PhysicalExpr>,
    input: Rc<dyn ExecutionPlan>,
}

impl FilterExec {
    pub fn new(predicate: Rc<dyn PhysicalExpr>, input: Rc<dyn ExecutionPlan>) -> Self {
        Self { predicate, input }
    }
}

impl ExecutionPlan for FilterExec {
    fn execute(&self) -> Result<DataStructure> {
        let ds = self.input.execute()?;
        let s = self.predicate.evaluate(&ds)?;
        let mask = s.bool()?;

        match ds {
            DataStructure::DataFrame(df) => Ok(df.filter(mask)?.into()),
            DataStructure::Series(s) => Ok(s.filter(mask)?.into()),
        }
    }
}
