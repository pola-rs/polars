use std::sync::RwLock;

use polars_plan::logical_plan::{to_aexpr, Context, IR};
use polars_plan::prelude::expr_ir::ExprIR;
use polars_plan::prelude::{AExpr, PythonOptions};
use polars_utils::arena::{Arena, Node};
use pyo3::prelude::*;
use visitor::{expr_nodes, nodes};

use super::*;
use crate::raise_err;

#[derive(Clone)]
#[pyclass]
pub(crate) struct PyExprIR {
    #[pyo3(get)]
    node: usize,
    #[pyo3(get)]
    output_name: String,
}

impl From<ExprIR> for PyExprIR {
    fn from(value: ExprIR) -> Self {
        Self {
            node: value.node().0,
            output_name: value.output_name().into(),
        }
    }
}

impl From<&ExprIR> for PyExprIR {
    fn from(value: &ExprIR) -> Self {
        Self {
            node: value.node().0,
            output_name: value.output_name().into(),
        }
    }
}

#[pyclass]
struct NodeTraverser {
    root: Node,
    lp_arena: Arc<RwLock<Arena<IR>>>,
    expr_arena: Arc<RwLock<Arena<AExpr>>>,
    scratch: Vec<Node>,
    expr_scratch: Vec<ExprIR>,
    expr_mapping: Option<Vec<Node>>,
}

impl NodeTraverser {
    fn fill_inputs(&mut self) {
        let lp_arena = self.lp_arena.read().unwrap();
        let this_node = lp_arena.get(self.root);
        self.scratch.clear();
        this_node.copy_inputs(&mut self.scratch);
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
            PyList::new(
                py,
                self.expr_scratch
                    .drain(..)
                    .map(|e| PyExprIR::from(e).into_py(py)),
            )
            .to_object(py)
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

    /// Get Schema of current node as python dict<str, pl.DataType>
    fn get_schema(&self, py: Python<'_>) -> PyObject {
        let lp_arena = self.lp_arena.read().unwrap();
        let schema = lp_arena.get(self.root).schema(&lp_arena).into_owned();
        Wrap(schema.as_ref()).into_py(py)
    }

    /// Get expression dtype.
    fn get_dtype(&self, expr_node: usize, py: Python<'_>) -> PyResult<PyObject> {
        let expr_node = Node(expr_node);
        let lp_arena = self.lp_arena.read().unwrap();
        let schema = lp_arena.get(self.root).schema(&lp_arena).into_owned();
        let expr_arena = self.expr_arena.read().unwrap();
        let field = expr_arena
            .get(expr_node)
            .to_field(&schema, Context::Default, &expr_arena)
            .map_err(PyPolarsErr::from)?;
        Ok(Wrap(field.dtype).to_object(py))
    }

    /// Set the current node in the plan.
    fn set_node(&mut self, node: usize) {
        self.root = Node(node);
    }

    /// Set a python UDF that will replace the subtree location with this function src.
    fn set_udf(&mut self, function: PyObject, schema: Wrap<Schema>) {
        let ir = IR::PythonScan {
            options: PythonOptions {
                scan_fn: Some(function.into()),
                schema: Arc::new(schema.0),
                output_schema: None,
                with_columns: None,
                pyarrow: false,
                predicate: None,
                n_rows: None,
            },
            predicate: None,
        };
        let mut lp_arena = self.lp_arena.write().unwrap();
        lp_arena.replace(self.root, ir);
    }

    fn view_current_node(&self, py: Python<'_>) -> PyResult<PyObject> {
        let lp_arena = self.lp_arena.read().unwrap();
        let lp_node = lp_arena.get(self.root);
        nodes::into_py(py, lp_node)
    }

    fn view_expression(&self, py: Python<'_>, node: usize) -> PyResult<PyObject> {
        let expr_arena = self.expr_arena.read().unwrap();
        let n = match &self.expr_mapping {
            Some(mapping) => *mapping.get(node).unwrap(),
            None => Node(node),
        };
        let expr = expr_arena.get(n);
        expr_nodes::into_py(py, expr)
    }

    /// Add some expressions to the arena and return their new node ids as well
    /// as the total number of nodes in the arena.
    fn add_expressions(&mut self, expressions: Vec<PyExpr>) -> PyResult<(Vec<usize>, usize)> {
        let mut expr_arena: std::sync::RwLockWriteGuard<'_, Arena<AExpr>> =
            self.expr_arena.write().unwrap();
        Ok((
            expressions
                .iter()
                .map(|e| to_aexpr(e.inner.clone(), &mut expr_arena).0)
                .collect(),
            expr_arena.len(),
        ))
    }

    /// Set up a mapping of expression nodes used in `view_expression_node``.
    /// With a mapping set, `view_expression_node(i)` produces the node for
    /// `mapping[i]`.
    fn set_expr_mapping(&mut self, mapping: Vec<usize>) -> PyResult<()> {
        if mapping.len() != self.expr_arena.read().unwrap().len() {
            raise_err!("Invalid mapping length", ComputeError);
        }
        self.expr_mapping = Some(mapping.into_iter().map(Node).collect());
        Ok(())
    }

    /// Unset the expression mapping (reinstates the identity map)
    fn unset_expr_mapping(&mut self) {
        self.expr_mapping = None;
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
            expr_scratch: vec![],
            expr_mapping: None,
        })
    }
}
