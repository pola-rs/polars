use std::path::PathBuf;
use std::sync::Arc;
use crate::prelude::{DslPlan, ScanSources};

impl DslPlan {

    /// Get sources paths in iteration order.
    pub fn get_sources_paths(&self) -> Vec<Arc<[PathBuf]>> {
        self.into_iter().flat_map(|subplan| match subplan {
            DslPlan::Scan { sources, .. } => {
                let src = sources.lock().unwrap();
                match &*src.sources {
                    ScanSources::Paths(paths) => Some(paths.clone()),
                    _ => None
                }
            }
            _ => None
        }).collect()
    }

    /// Set sources paths in iteration order.
    pub fn set_sources_paths(&self, new_paths: Vec<Arc<[PathBuf]>>) -> Vec<Arc<[PathBuf]>> {
        self.into_iter().flat_map(|subplan| match subplan {
            DslPlan::Scan { sources, .. } => {
                let src = sources.lock().unwrap();
                match &*src.sources {
                    ScanSources::Paths(paths) => Some(paths.clone()),
                    _ => None
                }
            }
            _ => None
        }).collect()
    }
}
