use super::*;
use std::cell::RefCell;
use std::mem;

#[derive(Debug)]
pub struct CsvExec {
    path: String,
    schema: Schema,
    has_header: bool,
    delimiter: Option<u8>,
}

impl CsvExec {
    pub fn new(path: String, schema: Schema, has_header: bool, delimiter: Option<u8>) -> Self {
        CsvExec {
            path,
            schema,
            has_header,
            delimiter,
        }
    }
}

impl Executor for CsvExec {
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
    input: Rc<dyn Executor>,
}

impl FilterExec {
    pub fn new(predicate: Rc<dyn PhysicalExpr>, input: Rc<dyn Executor>) -> Self {
        Self { predicate, input }
    }
}

impl Executor for FilterExec {
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

impl Executor for DataFrameExec {
    fn execute(&self) -> Result<DataFrame> {
        let mut ref_df = self.df.borrow_mut();
        let df = &mut *ref_df;
        let out = mem::take(df);
        Ok(out)
    }
}

/// Take an input Executor and a multiple expressions
#[derive(Debug)]
pub struct PipeExec {
    /// i.e. sort, projection
    operation: &'static str,
    input: Rc<dyn Executor>,
    expr: Vec<Rc<dyn PhysicalExpr>>,
}

impl PipeExec {
    pub(crate) fn new(
        operation: &'static str,
        input: Rc<dyn Executor>,
        expr: Vec<Rc<dyn PhysicalExpr>>,
    ) -> Self {
        Self {
            operation,
            input,
            expr,
        }
    }
}

impl Executor for PipeExec {
    fn execute(&self) -> Result<DataFrame> {
        let df = self.input.execute()?;

        let selected_columns = self
            .expr
            .iter()
            .map(|expr| expr.evaluate(&df))
            .collect::<Result<Vec<Series>>>()?;
        Ok(DataFrame::new_no_checks(selected_columns))
    }
}
