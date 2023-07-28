use std::fmt::Write;

use polars_core::prelude::*;
use polars_plan::dot::*;
use polars_plan::prelude::*;

use crate::prelude::*;

impl LazyFrame {
    /// Get a dot language representation of the LogicalPlan.
    pub fn to_dot(&self, optimized: bool) -> PolarsResult<String> {
        let mut s = String::with_capacity(512);

        let mut logical_plan = self.clone().get_plan_builder().build();
        if optimized {
            // initialize arena's
            let mut expr_arena = Arena::with_capacity(64);
            let mut lp_arena = Arena::with_capacity(32);

            let lp_top = self.clone().optimize_with_scratch(
                &mut lp_arena,
                &mut expr_arena,
                &mut vec![],
                true,
            )?;
            logical_plan = node_to_lp(lp_top, &expr_arena, &mut lp_arena);
        }

        let prev_node = DotNode {
            branch: 0,
            id: 0,
            fmt: "",
        };

        // maps graphviz id to label
        // we use this to create this graph
        // first we create nodes including ids to make sure they are unique
        // A [id] -- B [id]
        // B [id] -- C [id]
        //
        // then later we hide the [id] by adding this to the graph
        // A [id] [label="A"]
        // B [id] [label="B"]
        // C [id] [label="C"]

        let mut id_map = PlHashMap::with_capacity(8);
        logical_plan
            .dot(&mut s, (0, 0), prev_node, &mut id_map)
            .expect("io error");
        s.push('\n');

        for (id, label) in id_map {
            // the label is wrapped in double quotes
            // the id already is wrapped in double quotes
            writeln!(s, "{id}[label=\"{label}\"]").unwrap();
        }
        s.push_str("\n}");
        Ok(s)
    }
}
