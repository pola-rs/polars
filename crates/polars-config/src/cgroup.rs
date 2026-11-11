/// The memory limit of the cgroup of this process, including the limits of
/// its parent cgroups.
///
/// The root cgroup files only hold this limit if the process has its own cgroup
/// namespace. We don't use sysinfo's process cgroup limits, as refreshing a
/// process there raises the soft open file limit of this process.
#[cfg(target_os = "linux")]
pub(crate) fn process_cgroup_memory_limit() -> Option<u64> {
    let cgroups = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    let mountinfo = std::fs::read_to_string("/proc/self/mountinfo").ok()?;
    cgroup_memory_limit(&cgroups, &mountinfo)
}

/// The memory limit for the cgroups in `cgroups` (the format of
/// /proc/self/cgroup), with the hierarchies mounted as in `mountinfo`.
#[cfg(target_os = "linux")]
fn cgroup_memory_limit(cgroups: &str, mountinfo: &str) -> Option<u64> {
    use std::path::{Path, PathBuf};

    let read_u64 = |path: &Path| {
        std::fs::read_to_string(path)
            .ok()?
            .trim()
            .parse::<u64>()
            .ok()
    };

    // The cgroup mounts as (is cgroup v2, mount root, mount point). A container
    // may mount only its own part of the hierarchy, so cgroup paths are
    // resolved against the mount root.
    let mounts: Vec<(bool, PathBuf, PathBuf)> = mountinfo
        .lines()
        .filter_map(|line| {
            let (mount, fs) = line.split_once(" - ")?;
            let mut mount = mount.split(' ');
            let mount_root = mount.nth(3)?;
            let mount_point = mount.next()?;
            let mut fs = fs.split(' ');
            let fs_type = fs.next()?;
            let super_options = fs.nth(1)?;
            let is_v2 = match fs_type {
                "cgroup2" => true,
                "cgroup" if super_options.split(',').any(|o| o == "memory") => false,
                _ => return None,
            };
            Some((
                is_v2,
                unescape_mount_path(mount_root),
                unescape_mount_path(mount_point),
            ))
        })
        .collect();

    let cgroup_limit = |is_v2: bool, path: &str| {
        mounts
            .iter()
            .filter(|(mount_is_v2, _, _)| *mount_is_v2 == is_v2)
            .filter_map(|(_, mount_root, mount_point)| {
                let dir = mount_point.join(Path::new(path).strip_prefix(mount_root).ok()?);
                if is_v2 {
                    // The limit of every parent applies. "max" fails to parse
                    // and means no limit.
                    dir.ancestors()
                        .take_while(|dir| dir.starts_with(mount_point))
                        .filter_map(|dir| read_u64(&dir.join("memory.max")))
                        .min()
                } else {
                    // The kernel reports the limit that applies, taking
                    // memory.use_hierarchy into account.
                    std::fs::read_to_string(dir.join("memory.stat"))
                        .ok()?
                        .lines()
                        .find_map(|l| l.strip_prefix("hierarchical_memory_limit "))?
                        .trim()
                        .parse::<u64>()
                        .ok()
                }
            })
            .min()
    };

    cgroups
        .lines()
        .filter_map(|line| {
            let mut fields = line.splitn(3, ':');
            let hierarchy_id = fields.next()?;
            let controllers = fields.next()?;
            let path = fields.next()?;

            if hierarchy_id == "0" && controllers.is_empty() {
                cgroup_limit(true, path)
            } else if controllers.split(',').any(|c| c == "memory") {
                cgroup_limit(false, path)
            } else {
                None
            }
        })
        .min()
}

/// Decodes the octal escapes (such as `\040` for a space) of a path in
/// /proc/self/mountinfo.
#[cfg(target_os = "linux")]
fn unescape_mount_path(path: &str) -> std::path::PathBuf {
    use std::os::unix::ffi::OsStringExt;

    let bytes = path.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'\\'
            && let Some(code) = bytes
                .get(i + 1..i + 4)
                .and_then(|d| u8::from_str_radix(std::str::from_utf8(d).ok()?, 8).ok())
        {
            out.push(code);
            i += 4;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    std::ffi::OsString::from_vec(out).into()
}

#[cfg(not(target_os = "linux"))]
pub(crate) fn process_cgroup_memory_limit() -> Option<u64> {
    None
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};

    use super::cgroup_memory_limit;

    fn fake_mount(name: &str, files: &[(&str, &str)]) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("polars-cgroup-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        for (path, content) in files {
            let path = dir.join(path);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, content).unwrap();
        }
        dir
    }

    fn mountinfo_line(root: &str, mount_point: &Path, fs: &str) -> String {
        format!(
            "36 27 0:30 {root} {} rw,relatime shared:9 - {fs}",
            mount_point.display()
        )
    }

    #[test]
    fn test_cgroup_v2_parent_limit() {
        let mount = fake_mount(
            "v2",
            &[
                ("memory.max", "max\n"),
                ("a/memory.max", "2147483648\n"),
                ("a/b/memory.max", "5368709120\n"),
            ],
        );
        let mountinfo = mountinfo_line("/", &mount, "cgroup2 cgroup2 rw,nsdelegate");
        assert_eq!(cgroup_memory_limit("0::/a/b\n", &mountinfo), Some(2 << 30));
        assert_eq!(cgroup_memory_limit("0::/\n", &mountinfo), None);
        fs::remove_dir_all(mount).unwrap();
    }

    #[test]
    fn test_cgroup_v2_subtree_mount() {
        // Only /docker/abc is mounted, with the process in its child "job".
        let mount = fake_mount(
            "v2-subtree",
            &[
                ("memory.max", "5368709120\n"),
                ("job/memory.max", "2147483648\n"),
            ],
        );
        let mountinfo = mountinfo_line("/docker/abc", &mount, "cgroup2 cgroup2 rw");
        assert_eq!(
            cgroup_memory_limit("0::/docker/abc/job\n", &mountinfo),
            Some(2 << 30)
        );
        fs::remove_dir_all(mount).unwrap();
    }

    #[test]
    fn test_cgroup_v2_skips_unrelated_mount() {
        let unrelated = fake_mount("v2-unrelated", &[("memory.max", "1073741824\n")]);
        let mount = fake_mount("v2-full", &[("a/job/memory.max", "2147483648\n")]);
        let mountinfo = [
            mountinfo_line("/b", &unrelated, "cgroup2 cgroup2 rw"),
            mountinfo_line("/", &mount, "cgroup2 cgroup2 rw"),
        ]
        .join("\n");
        assert_eq!(
            cgroup_memory_limit("0::/a/job\n", &mountinfo),
            Some(2 << 30)
        );
        fs::remove_dir_all(unrelated).unwrap();
        fs::remove_dir_all(mount).unwrap();
    }

    #[test]
    fn test_cgroup_escaped_mount_paths() {
        let mount = fake_mount("v2 escaped", &[("job/memory.max", "2147483648\n")]);
        let escaped_mount = mount.display().to_string().replace(' ', "\\040");
        let mountinfo = mountinfo_line(
            "/batch\\040jobs",
            Path::new(&escaped_mount),
            "cgroup2 cgroup2 rw",
        );
        assert_eq!(
            cgroup_memory_limit("0::/batch jobs/job\n", &mountinfo),
            Some(2 << 30)
        );
        fs::remove_dir_all(mount).unwrap();
    }

    #[test]
    fn test_cgroup_v1_subtree_mount() {
        let mount = fake_mount(
            "v1-subtree",
            &[(
                "job/memory.stat",
                "cache 0\nhierarchical_memory_limit 2147483648\n",
            )],
        );
        let mountinfo = [
            mountinfo_line(
                "/docker/abc",
                Path::new("/nonexistent/cpu"),
                "cgroup cgroup rw,cpu",
            ),
            mountinfo_line("/docker/abc", &mount, "cgroup cgroup rw,memory"),
        ]
        .join("\n");
        let cgroups = "4:cpu:/docker/abc/job\n7:memory:/docker/abc/job\n";
        assert_eq!(cgroup_memory_limit(cgroups, &mountinfo), Some(2 << 30));
        fs::remove_dir_all(mount).unwrap();
    }
}
