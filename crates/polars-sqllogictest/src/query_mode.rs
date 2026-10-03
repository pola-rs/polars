use std::path::Path;

use sqllogictest::{DB, DBOutput};

use crate::catch_panic;
use crate::engine::PolarsEngine;

fn json_str(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Runs the statements in `path`, separated by lines holding only `;;`, in one engine.
///
/// Prints one JSON line per statement: `{"ok":true,"rows":[["1","a"],...]}` with values
/// formatted as in `.slt` results, or `{"ok":false,"err":"..."}`.
pub fn run(path: &Path) {
    let text = std::fs::read_to_string(path).unwrap();
    let mut engine = PolarsEngine::new();
    for sql in text
        .split("\n;;\n")
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        let line = match catch_panic(|| engine.run(sql)) {
            Ok(Ok(DBOutput::Rows { rows, .. })) => {
                let rows: Vec<String> = rows
                    .iter()
                    .map(|row| {
                        let values: Vec<String> = row.iter().map(|v| json_str(v)).collect();
                        format!("[{}]", values.join(","))
                    })
                    .collect();
                format!("{{\"ok\":true,\"rows\":[{}]}}", rows.join(","))
            },
            Ok(Ok(_)) => "{\"ok\":true,\"rows\":[]}".to_string(),
            Ok(Err(err)) => format!("{{\"ok\":false,\"err\":{}}}", json_str(&err.to_string())),
            Err(msg) => format!("{{\"ok\":false,\"err\":{}}}", json_str(&msg)),
        };
        println!("{line}");
    }
}
