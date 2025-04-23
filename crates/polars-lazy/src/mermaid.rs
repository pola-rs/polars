use polars_core::prelude::*;
use regex::Regex;

use crate::prelude::*;

impl LazyFrame {
    pub fn to_mermaid(&self, optimized: bool) -> PolarsResult<String> {
        // Mermaid strings are very similar to dot strings, so
        // we can reuse the dot implementation.
        let dot = self.to_dot(optimized)?;

        let edge_regex = Regex::new(r"(?P<node1>\w+) -- (?P<node2>\w+)").unwrap();
        let node_regex = Regex::new(r#"(?P<node>\w+)(\s+)?\[label="(?P<label>.*)"]"#).unwrap();

        let nodes = node_regex.captures_iter(&dot);
        let edges = edge_regex.captures_iter(&dot);

        let node_lines = nodes
            .map(|node| {
                format!(
                    "\t{}[\"{}\"]",
                    node.name("node").unwrap().as_str(),
                    node.name("label")
                        .unwrap()
                        .as_str()
                        .replace(r"\n", "\n") // replace escaped newlines
                        .replace(r#"\""#, "#quot;") // replace escaped quotes
                )
            })
            .collect::<Vec<_>>()
            .join("\n");

        let edge_lines = edges
            .map(|edge| {
                format!(
                    "\t{} --- {}",
                    edge.name("node1").unwrap().as_str(),
                    edge.name("node2").unwrap().as_str()
                )
            })
            .collect::<Vec<_>>()
            .join("\n");

        let mermaid = format!("graph TD\n{node_lines}\n{edge_lines}");

        Ok(mermaid)
    }
}
