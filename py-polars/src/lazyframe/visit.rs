use std::sync::RwLock;

use polars_plan::logical_plan::ALogicalPlan;
use polars_plan::prelude::AExpr;
use polars_utils::arena::{Arena, Node};

use super::*;

#[pyclass]
struct NodeTraverser {
    root: Node,
    lp_arena: Arc<RwLock<Arena<ALogicalPlan>>>,
    expr_arena: Arc<RwLock<Arena<AExpr>>>,
    scratch: Vec<Node>,
}

impl NodeTraverser {
    fn fill_inputs(&mut self) {
        let lp_arena = self.lp_arena.read().unwrap();
        let this_node = lp_arena.get(self.root);
        self.scratch.clear();
        this_node.copy_exprs(&mut self.scratch);
    }

    fn fill_expressions(&mut self) {
        let lp_arena = self.lp_arena.read().unwrap();
        let this_node = lp_arena.get(self.root);
        self.scratch.clear();
        this_node.copy_inputs(&mut self.scratch);
    }

    fn scratch_to_list(&mut self) -> PyObject {
        Python::with_gil(|py| {
            PyList::new(py, self.scratch.drain(..).map(|node| node.0)).to_object(py)
        })
    }
}

#[pymethods]
impl NodeTraverser {
    fn get_exprs(&mut self) -> PyObject {
        self.fill_expressions();
        self.scratch_to_list()
    }

    fn get_inputs(&mut self) -> PyObject {
        self.fill_inputs();
        self.scratch_to_list()
    }

    fn set_node(&mut self, node: usize) {
        self.root = Node(node);
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
        })
    }
}
