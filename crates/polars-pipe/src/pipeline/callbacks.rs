use polars_utils::aliases::{PlHashMap, PlHashSet};
use polars_utils::arena::Node;
use crate::executors::operators::PlaceHolder;
use crate::operators::Sink;
use crate::pipeline::PhysOperator;

pub struct CallBackReplacer {
    placeholders: PlHashMap<Node, PlaceHolder>,
}

impl CallBackReplacer {
    pub fn new() -> Self {
        Self {
            placeholders: Default::default(),
        }
    }

    pub fn insert_placeholder(&mut self, node: Node, placeholder: PlaceHolder) {
        let _ = self.placeholders.insert(node, placeholder);
    }

    pub fn get_placeholder(&mut self, node: &Node) -> PlaceHolder{
        self.placeholders.get(node).unwrap().clone()
    }
}

