#[cfg(target_os = "linux")]
fn detect_system_memory() -> Option<usize> {
    use std::fs::File;
    use std::io::{BufRead, BufReader};
    
    let file = File::open("/proc/meminfo").ok()?;
    let reader = BufReader::new(file);
    
    for line in reader.lines() {
        let line = line.ok()?;
        if line.starts_with("MemTotal:") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                return parts[1].parse::<usize>().ok().map(|kb| kb * 1024);
            }
        }
    }
    None
}

#[cfg(target_os = "macos")]
fn detect_system_memory() -> Option<usize> {
    use std::process::Command;
    
    let output = Command::new("sysctl")
        .arg("-n")
        .arg("hw.memsize")
        .output()
        .ok()?;
        
    let bytes = String::from_utf8(output.stdout).ok()?;
    bytes.trim().parse().ok()
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn detect_system_memory() -> Option<usize> {
    None
}

pub fn get_system_memory() -> usize {
    detect_system_memory().unwrap_or(8 * 1024 * 1024 * 1024) // 8GB fallback
}