use super::*;
use arrow::datatypes::SchemaRef;
use std::cell::RefCell;
use std::mem;

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
    fn execute(&self) -> Result<DataFrame> {
        let file = std::fs::File::open(&self.path).unwrap();

        let df = CsvReader::new(file)
            .has_header(self.has_header)
            .with_batch_size(10000)
            .finish()?;
        Ok(df)
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
    fn execute(&self) -> Result<DataFrame> {
        let df = self.input.execute()?;
        let s = self.predicate.evaluate(&df)?;
        let mask = s.bool()?;

        Ok(df.filter(mask)?)
    }
}

#[derive(Debug)]
pub struct DataFrameExec {
    df: Rc<RefCell<DataFrame>>,
}

impl DataFrameExec {
    pub(crate) fn new(df: Rc<RefCell<DataFrame>>) -> Self {
        DataFrameExec { df }
    }
}

impl ExecutionPlan for DataFrameExec {
    fn execute(&self) -> Result<DataFrame> {
        let mut ref_df = self.df.borrow_mut();
        let df = &mut *ref_df;
        let out = mem::take(df);
        Ok(out)
    }
}

#[derive(Debug)]
pub struct ProjectionExec {
    input: Rc<dyn ExecutionPlan>,
    columns: Vec<Rc<dyn PhysicalExpr>>,
}

impl ProjectionExec {
    pub(crate) fn new(input: Rc<dyn ExecutionPlan>, columns: Vec<Rc<dyn PhysicalExpr>>) -> Self {
        Self { input, columns }
    }
}

impl ExecutionPlan for ProjectionExec {
    fn execute(&self) -> Result<DataFrame> {
        // projection is only on a DataFrame so we unpack df
        let df = self.input.execute()?;
        let selected_columns = self
            .columns
            .iter()
            .map(|expr| expr.evaluate(&df))
            .collect::<Result<Vec<Series>>>()?;
        Ok(DataFrame::new_no_checks(selected_columns))
    }
}
