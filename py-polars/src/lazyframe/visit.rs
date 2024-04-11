use std::sync::RwLock;
use pyo3::prelude::*;

use polars_plan::logical_plan::IR;
use polars_plan::prelude::{AExpr, FunctionNode, PythonOptions};
use polars_plan::prelude::expr_ir::ExprIR;
use polars_plan::prelude::python_udf::PythonFunction;
use polars_utils::arena::{Arena, Node};
use super::*;

#[derive(Clone)]
#[pyclass]
struct PyExprIR {
    #[pyo3(get)]
    node: usize,
    #[pyo3(get)]
    output_name: String
}

#[pyclass]
struct NodeTraverser {
    root: Node,
    lp_arena: Arc<RwLock<Arena<IR>>>,
    expr_arena: Arc<RwLock<Arena<AExpr>>>,
    scratch: Vec<Node>,
    expr_scratch: Vec<ExprIR>
}

impl NodeTraverser {
    fn fill_inputs(&mut self) {
        let lp_arena = self.lp_arena.read().unwrap();
        let this_node = lp_arena.get(self.root);
        self.scratch.clear();
        this_node.copy_exprs(&mut self.expr_scratch);
    }

    fn fill_expressions(&mut self) {
        let lp_arena = self.lp_arena.read().unwrap();
        let this_node = lp_arena.get(self.root);
        self.expr_scratch.clear();
        this_node.copy_exprs(&mut self.expr_scratch);
    }

    fn scratch_to_list(&mut self) -> PyObject {
        Python::with_gil(|py| {
            PyList::new(py, self.scratch.drain(..).map(|node| node.0)).to_object(py)
        })
    }

    fn expr_to_list(&mut self) -> PyObject {
        Python::with_gil(|py| {
            PyList::new(py, self.expr_scratch.drain(..).map(|e| {
                PyExprIR {
                    node: e.node().0,
                    output_name: e.output_name().into()
                }.into_py(py)
            })).to_object(py)
        })
    }
}

#[pymethods]
impl NodeTraverser {
    /// Get expression nodes
    fn get_exprs(&mut self) -> PyObject {
        self.fill_expressions();
        self.expr_to_list()
    }

    /// Get input nodes
    fn get_inputs(&mut self) -> PyObject {
        self.fill_inputs();
        self.scratch_to_list()
    }

    /// Set the current node in the plan.
    fn set_node(&mut self, node: usize) {
        self.root = Node(node);
    }

    /// Set a python UDF that will replace the subtree location with this function src.
    fn set_udf(&mut self, function: PyObject, schema: Wrap<Schema>) {
        let ir = IR::PythonScan {
            options: PythonOptions{
                scan_fn: Some(function.into()),
                schema: Arc::new(schema.0),
                output_schema: None,
                with_columns: None,
                pyarrow: false,
                predicate: None,
                n_rows: None,
            },
            predicate: None
        };
        let mut lp_arena = self.lp_arena.write().unwrap();
        lp_arena.replace(self.root, ir);
    }

    fn view_current_node(&self) -> PyObject {
        // Insert Python objects/struct that map to our plan here
        todo!()
    }

    fn view_expression(&self, node: usize) -> PyObject {
        let expr_arena = self.expr_arena.read().unwrap();
        let _expr = expr_arena.get(Node(node));
        // Insert Python objects/struct that map to our expr here
        todo!()
    }
}

#[pymethods]
#[allow(clippy::should_implement_trait)]
impl PyLazyFrame {
    fn visit(&self) -> PyResult<NodeTraverser> {
        let mut lp_arena = Arena::with_capacity(16);
        let mut expr_arena = Arena::with_capacity(16);
        let root = self
            .ldf
            .clone()
            .optimize(&mut lp_arena, &mut expr_arena)
            .map_err(PyPolarsErr::from)?;
        Ok(NodeTraverser {
            root,
            lp_arena: Arc::new(RwLock::new(lp_arena)),
            expr_arena: Arc::new(RwLock::new(expr_arena)),
            scratch: vec![],
            expr_scratch: vec![]
        })
    }
}
