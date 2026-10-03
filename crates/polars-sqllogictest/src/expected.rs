use std::collections::BTreeSet;
use std::path::Path;

pub struct ExpectedFailures {
    entries: BTreeSet<String>,
}

impl ExpectedFailures {
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let entries = match std::fs::read_to_string(path) {
            Ok(contents) => contents
                .lines()
                .map(str::trim)
                .filter(|line| !line.is_empty() && !line.starts_with('#'))
                .map(str::to_string)
                .collect(),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => BTreeSet::new(),
            Err(err) => return Err(err),
        };
        Ok(ExpectedFailures { entries })
    }

    pub fn contains(&self, key: &str) -> bool {
        self.entries.contains(key)
    }

    pub fn iter(&self) -> impl Iterator<Item = &String> {
        self.entries.iter()
    }
}
