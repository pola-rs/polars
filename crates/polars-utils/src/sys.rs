pub use polars_config::total_memory;

/// Check whether a process with the given PID is currently alive.
///
/// Used by `polars_ooc::cleaner::cleanup_stale_dirs` to remove spill
/// directories left behind by dead processes on startup.
pub fn is_process_alive(pid: u32) -> bool {
    use sysinfo::{Pid, ProcessRefreshKind, System, UpdateKind};
    let pid = Pid::from_u32(pid);
    let mut sys = System::new();
    sys.refresh_processes_specifics(
        sysinfo::ProcessesToUpdate::Some(&[pid]),
        true,
        ProcessRefreshKind::nothing().with_cmd(UpdateKind::Never),
    );
    sys.process(pid).is_some()
}
