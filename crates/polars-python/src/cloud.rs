use std::io::Cursor;

use polars_core::error::{polars_err, to_compute_err, PolarsResult};
use polars_expr::state::ExecutionState;
use polars_mem_engine::create_physical_plan;
use polars_plan::plans::{AExpr, IRPlan, IR};
use polars_plan::prelude::{Arena, Node};
use pyo3::intern;
use pyo3::prelude::{PyAnyMethods, PyModule, Python, *};
use pyo3::types::{IntoPyDict, PyBytes};

use crate::error::PyPolarsErr;
use crate::lazyframe::visit::NodeTraverser;
use crate::{PyDataFrame, PyLazyFrame};

#[pyfunction]
pub fn prepare_cloud_plan(lf: PyLazyFrame, py: Python) -> PyResult<PyObject> {
    let plan = lf.ldf.logical_plan;
    let bytes = polars::prelude::prepare_cloud_plan(plan).map_err(PyPolarsErr::from)?;

    Ok(PyBytes::new_bound(py, &bytes).to_object(py))
}

/// Take a serialized `IRPlan` and execute it on the GPU engine.
///
/// This is done as a Python function because the `NodeTraverser` class created for this purpose
/// must exactly match the one expected by the `cudf_polars` package.
#[pyfunction]
pub fn _execute_ir_plan_with_gpu(ir_plan_ser: Vec<u8>, py: Python) -> PyResult<PyDataFrame> {
    // Deserialize into IRPlan.
    let reader = Cursor::new(ir_plan_ser);
    let mut ir_plan = ciborium::from_reader::<IRPlan, _>(reader)
        .map_err(to_compute_err)
        .map_err(PyPolarsErr::from)?;

    // Edit for use with GPU engine.
    gpu_post_opt(
        py,
        ir_plan.lp_top,
        &mut ir_plan.lp_arena,
        &mut ir_plan.expr_arena,
    )
    .map_err(PyPolarsErr::from)?;

    // Convert to physical plan.
    let mut physical_plan =
        create_physical_plan(ir_plan.lp_top, &mut ir_plan.lp_arena, &ir_plan.expr_arena)
            .map_err(PyPolarsErr::from)?;

    // Execute the plan.
    let mut state = ExecutionState::new();
    let df = physical_plan
        .execute(&mut state)
        .map_err(PyPolarsErr::from)?;

    Ok(df.into())
}

/// Prepare the IR for execution by the Polars GPU engine.
fn gpu_post_opt(
    py: Python,
    root: Node,
    lp_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
) -> PolarsResult<()> {
    // Get cuDF Python function.
    let cudf = PyModule::import_bound(py, intern!(py, "cudf_polars")).unwrap();
    let lambda = cudf.getattr(intern!(py, "execute_with_cudf")).unwrap();

    // Define cuDF config.
    let polars = PyModule::import_bound(py, intern!(py, "polars")).unwrap();
    let engine = polars.getattr(intern!(py, "GPUEngine")).unwrap();
    let kwargs = [("raise_on_fail", true)].into_py_dict_bound(py);
    let engine = engine.call((), Some(&kwargs)).unwrap();

    // Define node traverser.
    let nt = NodeTraverser::new(root, std::mem::take(lp_arena), std::mem::take(expr_arena));

    // Get a copy of the arenas.
    let arenas = nt.get_arenas();

    // Pass the node visitor which allows the Python callback to replace parts of the query plan.
    // Remove "cuda" or specify better once we have multiple post-opt callbacks.
    let kwargs = [("config", engine)].into_py_dict_bound(py);
    lambda
        .call((nt,), Some(&kwargs))
        .map_err(|e| polars_err!(ComputeError: "'cuda' conversion failed: {}", e))?;

    // Unpack the arena's.
    // At this point the `nt` is useless.
    std::mem::swap(lp_arena, &mut *arenas.0.lock().unwrap());
    std::mem::swap(expr_arena, &mut *arenas.1.lock().unwrap());

    Ok(())
}
