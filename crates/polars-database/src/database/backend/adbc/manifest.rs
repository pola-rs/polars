//! ADBC driver manifests: discovery and parsing.
//!
//! Spec: <https://arrow.apache.org/adbc/current/format/driver_manifests.html>. A manifest is a
//! `<name>.toml` file naming the driver's shared library per platform:
//!
//! ```toml
//! manifest_version = 1
//! name = "PostgreSQL"
//! version = "1.12.0"
//! [ADBC]
//! version = "v1.1.0"
//! [Driver]
//! entrypoint = "AdbcDriverInit"     # optional
//! [Driver.shared]                   # or: shared = "/path/lib.so"
//! linux_amd64 = "/path/libadbc_driver_postgresql.so"
//! ```
//!
//! Only the small TOML subset manifests use is parsed: tables, (dotted, quoted) keys, strings,
//! integers and inline tables.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use polars_error::{PolarsResult, polars_bail, polars_err};

/// A driver library located on this machine.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FoundDriver {
    /// The shared library to load.
    pub library: PathBuf,
    /// The init symbol named by the manifest, if any.
    pub entrypoint: Option<String>,
    /// The driver version from the manifest, if any.
    pub version: Option<String>,
    /// The manifest the library was found through, if any.
    pub manifest: Option<PathBuf>,
}

/// The platform tuple manifests key libraries by, e.g. `linux_amd64` or `macos_arm64`.
pub fn current_platform() -> String {
    let os = if cfg!(target_os = "windows") {
        "windows"
    } else if cfg!(target_os = "macos") {
        "macos"
    } else if cfg!(target_os = "freebsd") {
        "freebsd"
    } else if cfg!(target_os = "openbsd") {
        "openbsd"
    } else if cfg!(target_os = "linux") {
        "linux"
    } else {
        "unknown"
    };
    let arch = if cfg!(target_arch = "x86_64") {
        "amd64"
    } else if cfg!(target_arch = "aarch64") {
        if cfg!(target_endian = "little") {
            "arm64"
        } else {
            "arm64be"
        }
    } else if cfg!(target_arch = "x86") {
        "x86"
    } else if cfg!(target_arch = "arm") {
        "arm"
    } else if cfg!(target_arch = "riscv64") {
        "riscv64"
    } else if cfg!(target_arch = "powerpc64") {
        if cfg!(target_endian = "little") {
            "powerpc64le"
        } else {
            "powerpc64"
        }
    } else if cfg!(target_arch = "s390x") {
        "s390x"
    } else if cfg!(target_arch = "wasm32") {
        "wasm32"
    } else {
        "unknown"
    };
    let target = if cfg!(all(target_os = "windows", target_env = "gnu")) {
        "_mingw"
    } else if cfg!(target_env = "musl") {
        "_musl"
    } else {
        ""
    };
    format!("{os}_{arch}{target}")
}

/// The file extension of shared libraries on this platform, without the dot.
pub fn library_extension() -> &'static str {
    if cfg!(target_os = "windows") {
        "dll"
    } else if cfg!(target_os = "macos") {
        "dylib"
    } else {
        "so"
    }
}

fn is_library_ext(ext: &str) -> bool {
    matches!(ext, "so" | "dylib" | "dll")
}

static EXTRA_SEARCH_DIRS: std::sync::RwLock<Vec<PathBuf>> = std::sync::RwLock::new(Vec::new());

/// Adds a directory searched after the environment variables, e.g. the running Python
/// environment's `etc/adbc/drivers` when no `VIRTUAL_ENV` is set.
pub fn add_manifest_search_dir(dir: PathBuf) {
    let mut dirs = EXTRA_SEARCH_DIRS.write().unwrap();
    if !dirs.contains(&dir) {
        dirs.push(dir);
    }
}

/// Directories searched for `<name>.toml` manifests, in order. Directories that don't exist
/// are included, so error messages can list where Polars looked.
pub fn manifest_search_dirs() -> Vec<PathBuf> {
    let mut dirs = vec![];
    let sep = if cfg!(windows) { ';' } else { ':' };
    if let Some(paths) = std::env::var_os("ADBC_DRIVER_PATH") {
        let paths = paths.to_string_lossy().into_owned();
        dirs.extend(
            paths
                .split(sep)
                .filter(|p| !p.is_empty())
                .map(PathBuf::from),
        );
    }
    for var in ["VIRTUAL_ENV", "CONDA_PREFIX"] {
        if let Some(prefix) = std::env::var_os(var).filter(|p| !p.is_empty()) {
            dirs.push(
                PathBuf::from(prefix)
                    .join("etc")
                    .join("adbc")
                    .join("drivers"),
            );
        }
    }
    dirs.extend(EXTRA_SEARCH_DIRS.read().unwrap().iter().cloned());
    if let Some(user) = user_dir() {
        dirs.push(user);
    }
    if let Some(system) = system_dir() {
        dirs.push(system);
    }
    let mut seen = polars_utils::aliases::PlHashSet::default();
    dirs.retain(|d| seen.insert(d.clone()));
    dirs
}

fn home_dir() -> Option<PathBuf> {
    let var = if cfg!(windows) { "USERPROFILE" } else { "HOME" };
    std::env::var_os(var)
        .filter(|h| !h.is_empty())
        .map(PathBuf::from)
}

fn user_dir() -> Option<PathBuf> {
    if cfg!(target_os = "macos") {
        Some(home_dir()?.join("Library/Application Support/ADBC/Drivers"))
    } else if cfg!(windows) {
        let local = std::env::var_os("LOCALAPPDATA").filter(|p| !p.is_empty())?;
        Some(PathBuf::from(local).join("ADBC").join("Drivers"))
    } else {
        let config = std::env::var_os("XDG_CONFIG_HOME")
            .filter(|p| !p.is_empty())
            .map(PathBuf::from)
            .or_else(|| Some(home_dir()?.join(".config")))?;
        Some(config.join("adbc").join("drivers"))
    }
}

fn system_dir() -> Option<PathBuf> {
    if cfg!(target_os = "macos") {
        Some(PathBuf::from("/Library/Application Support/ADBC/Drivers"))
    } else if cfg!(windows) {
        // Windows uses the registry for system-level drivers, which is not searched.
        None
    } else {
        Some(PathBuf::from("/etc/adbc/drivers"))
    }
}

/// Locates a driver. `name` is a bare driver name (searched as `<name>.toml` in
/// [`manifest_search_dirs`]), a path to a shared library, or a path to a `.toml` manifest.
/// A manifest only counts if it names a library for this platform that exists.
pub fn find_driver(name: &str) -> Option<FoundDriver> {
    find_driver_impl(name, &manifest_search_dirs())
}

fn find_driver_impl(name: &str, dirs: &[PathBuf]) -> Option<FoundDriver> {
    if name.is_empty() {
        return None;
    }
    let path = Path::new(name);
    let ext = path.extension().and_then(|e| e.to_str());
    match ext {
        Some("toml") => return load_manifest(path).ok().flatten(),
        Some(e) if is_library_ext(e) => {
            return path.is_file().then(|| FoundDriver {
                library: path.to_path_buf(),
                entrypoint: None,
                version: None,
                manifest: None,
            });
        },
        _ => {},
    }
    // `libadbc_driver_x.so.1.2.3`
    if path.is_file() && name.contains(&format!(".{}.", library_extension())) {
        return Some(FoundDriver {
            library: path.to_path_buf(),
            entrypoint: None,
            version: None,
            manifest: None,
        });
    }
    if path.components().count() > 1 || path.is_absolute() {
        // A path without extension: `<path>.toml`, then `<path>.<lib ext>`.
        let manifest = path.with_extension("toml");
        if let Some(found) = load_manifest(&manifest).ok().flatten() {
            return Some(found);
        }
        let lib = path.with_extension(library_extension());
        return lib.is_file().then_some(FoundDriver {
            library: lib,
            entrypoint: None,
            version: None,
            manifest: None,
        });
    }
    dirs.iter().find_map(|dir| {
        load_manifest(&dir.join(format!("{name}.toml")))
            .ok()
            .flatten()
    })
}

/// Reads a manifest. `Ok(None)` if the file doesn't exist or has no existing library for this
/// platform; an error if it can't be parsed.
pub fn load_manifest(path: &Path) -> PolarsResult<Option<FoundDriver>> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Ok(None);
    };
    let manifest = Manifest::parse(&text).map_err(
        |e| polars_err!(ComputeError: "invalid ADBC driver manifest '{}': {e}", path.display()),
    )?;
    if let Some(v) = manifest.manifest_version
        && v != 1
    {
        polars_bail!(
            ComputeError: "ADBC driver manifest '{}' has unsupported manifest_version {v}",
            path.display()
        );
    }
    let Some(lib) = manifest.library_for(&current_platform()) else {
        return Ok(None);
    };
    let mut library = PathBuf::from(lib);
    if library.is_relative()
        && let Some(dir) = path.parent()
    {
        library = dir.join(library);
    }
    if !library.is_file() {
        return Ok(None);
    }
    Ok(Some(FoundDriver {
        library,
        entrypoint: manifest.entrypoint,
        version: manifest.version,
        manifest: Some(path.to_path_buf()),
    }))
}

/// The fields of a manifest Polars uses.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Manifest {
    pub name: Option<String>,
    pub version: Option<String>,
    pub manifest_version: Option<i64>,
    pub adbc_version: Option<String>,
    pub entrypoint: Option<String>,
    /// `[Driver] shared = "…"`.
    pub shared: Option<String>,
    /// `[Driver.shared]`, keyed by platform tuple.
    pub shared_by_platform: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Value {
    Str(String),
    Int(i64),
    Other,
}

impl Manifest {
    pub fn parse(text: &str) -> Result<Self, String> {
        let entries = parse_toml_subset(text)?;
        let mut m = Manifest::default();
        for (key, value) in entries {
            let s = match &value {
                Value::Str(s) => Some(s.clone()),
                _ => None,
            };
            match key.as_slice() {
                [k] if k == "name" => m.name = s,
                [k] if k == "version" => m.version = s,
                [k] if k == "manifest_version" => match value {
                    Value::Int(i) => m.manifest_version = Some(i),
                    _ => return Err("manifest_version must be an integer".into()),
                },
                [t, k] if t == "ADBC" && k == "version" => m.adbc_version = s,
                [t, k] if t == "Driver" && k == "entrypoint" => match s {
                    Some(s) => m.entrypoint = Some(s).filter(|s| !s.is_empty()),
                    None => return Err("Driver.entrypoint must be a string".into()),
                },
                [t, k] if t == "Driver" && k == "shared" => m.shared = s,
                [t, k, platform] if t == "Driver" && k == "shared" => {
                    if let Some(s) = s {
                        m.shared_by_platform.insert(platform.clone(), s);
                    }
                },
                _ => {},
            }
        }
        Ok(m)
    }

    /// The library path for a platform tuple.
    pub fn library_for(&self, platform: &str) -> Option<&str> {
        self.shared_by_platform
            .get(platform)
            .or(self.shared.as_ref())
            .map(String::as_str)
            .filter(|s| !s.is_empty())
    }
}

/// Parses the TOML subset into `(dotted key path, value)` pairs.
fn parse_toml_subset(text: &str) -> Result<Vec<(Vec<String>, Value)>, String> {
    let mut out = vec![];
    let mut table: Vec<String> = vec![];
    let mut lines = text.lines().enumerate();
    while let Some((lineno, first)) = lines.next() {
        let err = |msg: &str| format!("line {}: {msg}", lineno + 1);
        // A value that is an array may continue on the next lines.
        let mut raw = first.to_string();
        if starts_array_value(first) {
            while matches!(skip_value(&raw), Err(e) if e == "unterminated array") {
                let Some((_, next)) = lines.next() else { break };
                raw.push('\n');
                raw.push_str(next);
            }
        }
        let raw = raw.as_str();
        let mut p = Parser::new(raw);
        p.skip_ws();
        if p.at_end_of_line() {
            continue;
        }
        if p.eat('[') {
            let array_table = p.eat('[');
            let path = p.parse_key().map_err(|e| err(&e))?;
            p.skip_ws();
            if !p.eat(']') || (array_table && !p.eat(']')) {
                return Err(err("expected ']'"));
            }
            p.skip_ws();
            if !p.at_end_of_line() {
                return Err(err("unexpected text after table header"));
            }
            // Arrays of tables are not used by manifests: ignore their contents.
            table = if array_table {
                vec!["\u{0}".into()]
            } else {
                path
            };
            continue;
        }
        let mut key = table.clone();
        key.extend(p.parse_key().map_err(|e| err(&e))?);
        p.skip_ws();
        if !p.eat('=') {
            return Err(err("expected '='"));
        }
        p.skip_ws();
        p.parse_value_into(key, &mut out).map_err(|e| err(&e))?;
        p.skip_ws();
        if !p.at_end_of_line() {
            return Err(err("unexpected text after value"));
        }
    }
    Ok(out)
}

/// Whether a line is `key = [...`.
fn starts_array_value(line: &str) -> bool {
    let mut p = Parser::new(line);
    p.skip_ws();
    if p.peek() == Some('[') || p.parse_key().is_err() {
        return false;
    }
    p.skip_ws();
    if !p.eat('=') {
        return false;
    }
    p.skip_ws();
    p.peek() == Some('[')
}

/// Parses the `key = value` of `text`, to see whether the value is complete.
fn skip_value(text: &str) -> Result<(), String> {
    let mut p = Parser::new(text);
    p.skip_ws();
    let key = p.parse_key()?;
    p.skip_ws();
    p.eat('=');
    p.skip_ws();
    p.parse_value_into(key, &mut vec![])
}

struct Parser<'a> {
    chars: std::iter::Peekable<std::str::Chars<'a>>,
}

impl<'a> Parser<'a> {
    fn new(s: &'a str) -> Self {
        Self {
            chars: s.chars().peekable(),
        }
    }

    fn peek(&mut self) -> Option<char> {
        self.chars.peek().copied()
    }

    fn eat(&mut self, c: char) -> bool {
        if self.peek() == Some(c) {
            self.chars.next();
            true
        } else {
            false
        }
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(' ' | '\t' | '\r')) {
            self.chars.next();
        }
    }

    fn at_end_of_line(&mut self) -> bool {
        matches!(self.peek(), None | Some('#'))
    }

    /// A dotted key of bare or quoted parts.
    fn parse_key(&mut self) -> Result<Vec<String>, String> {
        let mut parts = vec![];
        loop {
            self.skip_ws();
            let part = match self.peek() {
                Some('"') => {
                    self.chars.next();
                    self.parse_basic_string()?
                },
                Some('\'') => {
                    self.chars.next();
                    self.parse_literal_string()?
                },
                _ => {
                    let mut s = String::new();
                    while let Some(c) = self.peek() {
                        if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                            s.push(c);
                            self.chars.next();
                        } else {
                            break;
                        }
                    }
                    if s.is_empty() {
                        return Err("expected a key".into());
                    }
                    s
                },
            };
            parts.push(part);
            self.skip_ws();
            if !self.eat('.') {
                return Ok(parts);
            }
        }
    }

    fn parse_value_into(
        &mut self,
        key: Vec<String>,
        out: &mut Vec<(Vec<String>, Value)>,
    ) -> Result<(), String> {
        match self.peek() {
            Some('"') => {
                self.chars.next();
                if self.eat('"') {
                    if self.eat('"') {
                        return Err("multi-line strings are not supported".into());
                    }
                    out.push((key, Value::Str(String::new())));
                } else {
                    let s = self.parse_basic_string()?;
                    out.push((key, Value::Str(s)));
                }
            },
            Some('\'') => {
                self.chars.next();
                let s = self.parse_literal_string()?;
                out.push((key, Value::Str(s)));
            },
            Some('{') => {
                self.chars.next();
                self.skip_ws();
                if self.eat('}') {
                    return Ok(());
                }
                loop {
                    let mut k = key.clone();
                    k.extend(self.parse_key()?);
                    self.skip_ws();
                    if !self.eat('=') {
                        return Err("expected '=' in inline table".into());
                    }
                    self.skip_ws();
                    self.parse_value_into(k, out)?;
                    self.skip_ws();
                    if self.eat('}') {
                        return Ok(());
                    }
                    if !self.eat(',') {
                        return Err("expected ',' or '}' in inline table".into());
                    }
                }
            },
            Some('[') => {
                // Arrays (e.g. `[ADBC.features] supported = [...]`) aren't read: skipped.
                self.skip_array()?;
                out.push((key, Value::Other));
            },
            Some(_) => {
                let mut s = String::new();
                while let Some(c) = self.peek() {
                    if matches!(c, ',' | '}' | '#' | ' ' | '\t' | '\r') {
                        break;
                    }
                    s.push(c);
                    self.chars.next();
                }
                let value = match s.replace('_', "").parse::<i64>() {
                    Ok(i) => Value::Int(i),
                    Err(_) => Value::Other,
                };
                out.push((key, value));
            },
            None => return Err("expected a value".into()),
        }
        Ok(())
    }

    /// Skips an array value, which may span lines and nest, with strings and comments inside.
    fn skip_array(&mut self) -> Result<(), String> {
        let mut depth = 0usize;
        loop {
            match self.chars.next() {
                None => return Err("unterminated array".into()),
                Some('[') => depth += 1,
                Some(']') => {
                    depth -= 1;
                    if depth == 0 {
                        return Ok(());
                    }
                },
                Some('"') => {
                    if self.eat('"') {
                        if self.eat('"') {
                            return Err("multi-line strings are not supported".into());
                        }
                    } else {
                        self.parse_basic_string()?;
                    }
                },
                Some('\'') => {
                    self.parse_literal_string()?;
                },
                Some('#') => {
                    while self.peek().is_some_and(|c| c != '\n') {
                        self.chars.next();
                    }
                },
                Some(_) => {},
            }
        }
    }

    /// After the opening `"`.
    fn parse_basic_string(&mut self) -> Result<String, String> {
        let mut s = String::new();
        loop {
            match self.chars.next() {
                None => return Err("unterminated string".into()),
                Some('"') => return Ok(s),
                Some('\\') => match self.chars.next() {
                    Some('\\') => s.push('\\'),
                    Some('"') => s.push('"'),
                    Some('n') => s.push('\n'),
                    Some('t') => s.push('\t'),
                    Some('r') => s.push('\r'),
                    Some('b') => s.push('\u{8}'),
                    Some('f') => s.push('\u{c}'),
                    Some(u @ ('u' | 'U')) => {
                        let n = if u == 'u' { 4 } else { 8 };
                        let hex: String = (0..n).filter_map(|_| self.chars.next()).collect();
                        let c = u32::from_str_radix(&hex, 16)
                            .ok()
                            .and_then(char::from_u32)
                            .ok_or("invalid unicode escape")?;
                        s.push(c);
                    },
                    _ => return Err("invalid escape in string".into()),
                },
                Some(c) => s.push(c),
            }
        }
    }

    /// After the opening `'`.
    fn parse_literal_string(&mut self) -> Result<String, String> {
        let mut s = String::new();
        loop {
            match self.chars.next() {
                None => return Err("unterminated string".into()),
                Some('\'') => return Ok(s),
                Some(c) => s.push(c),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_manifest() {
        let m = Manifest::parse(
            r#"
# A driver.
manifest_version = 1
name = "PostgreSQL"   # trailing comment
version = "1.12.0"
publisher = "x # not a comment"

[ADBC]
version = "v1.1.0"

[Driver]
entrypoint = "AdbcDriverPostgresqlInit"

[Driver.shared]
linux_amd64 = "/opt/a \"b\"\\libpg.so"
"macos_arm64" = '/opt/c\libpg.dylib'
"#,
        )
        .unwrap();
        assert_eq!(m.name.as_deref(), Some("PostgreSQL"));
        assert_eq!(m.version.as_deref(), Some("1.12.0"));
        assert_eq!(m.manifest_version, Some(1));
        assert_eq!(m.adbc_version.as_deref(), Some("v1.1.0"));
        assert_eq!(m.entrypoint.as_deref(), Some("AdbcDriverPostgresqlInit"));
        assert_eq!(m.library_for("linux_amd64"), Some("/opt/a \"b\"\\libpg.so"));
        assert_eq!(m.library_for("macos_arm64"), Some("/opt/c\\libpg.dylib"));
        assert_eq!(m.library_for("windows_amd64"), None);
    }

    #[test]
    fn skips_arrays() {
        let m = Manifest::parse(
            r#"
name = "SQLite"
[ADBC.features]
supported = []
unsupported = ["bulk ingest", 'a ] b', [1, 2],  # comment ]
  "x"]
[Driver]
shared = "/x/lib.so"
"#,
        )
        .unwrap();
        assert_eq!(m.name.as_deref(), Some("SQLite"));
        assert_eq!(m.library_for("linux_arm64"), Some("/x/lib.so"));
        assert!(Manifest::parse("a = [1, 2").is_err());
    }

    #[test]
    fn parses_shared_string_and_inline_table() {
        let m = Manifest::parse("[Driver]\nshared = \"/x/lib.so\"\n").unwrap();
        assert_eq!(m.library_for("linux_arm64"), Some("/x/lib.so"));

        let m =
            Manifest::parse("Driver.shared = { linux_arm64 = \"/a.so\", macos_arm64 = \"/b\" }")
                .unwrap();
        assert_eq!(m.library_for("linux_arm64"), Some("/a.so"));
        assert_eq!(m.library_for("macos_arm64"), Some("/b"));
    }

    #[test]
    fn rejects_bad_manifests() {
        assert!(Manifest::parse("name = \"x").is_err());
        assert!(Manifest::parse("[Driver\n").is_err());
        assert!(Manifest::parse("manifest_version = \"1\"").is_err());
    }

    #[test]
    fn finds_manifests() {
        let dir = std::env::temp_dir().join(format!("polars-adbc-manifest-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let lib = dir.join(format!("libadbc_driver_fake.{}", library_extension()));
        std::fs::write(&lib, b"").unwrap();
        let platform = current_platform();
        std::fs::write(
            dir.join("fake.toml"),
            format!(
                "version = \"0.1\"\n[Driver.shared]\n{platform} = \"{}\"\n",
                lib.display().to_string().replace('\\', "\\\\")
            ),
        )
        .unwrap();
        std::fs::write(
            dir.join("other_platform.toml"),
            "[Driver.shared]\nnot_a_platform = \"/nonexistent.so\"\n",
        )
        .unwrap();
        std::fs::write(
            dir.join("relative.toml"),
            format!(
                "[Driver]\nshared = \"libadbc_driver_fake.{}\"\n",
                library_extension()
            ),
        )
        .unwrap();

        let dirs = vec![dir.join("missing"), dir.clone()];
        let found = find_driver_impl("fake", &dirs).unwrap();
        assert_eq!(found.library, lib);
        assert_eq!(found.version.as_deref(), Some("0.1"));
        assert_eq!(found.manifest, Some(dir.join("fake.toml")));
        assert!(find_driver_impl("other_platform", &dirs).is_none());
        assert!(find_driver_impl("missing", &dirs).is_none());
        assert_eq!(
            find_driver_impl("relative", &dirs).unwrap().library,
            dir.join(format!("libadbc_driver_fake.{}", library_extension()))
        );
        // Paths.
        let by_path = find_driver_impl(dir.join("fake.toml").to_str().unwrap(), &[]).unwrap();
        assert_eq!(by_path.library, lib);
        let by_stem = find_driver_impl(dir.join("fake").to_str().unwrap(), &[]).unwrap();
        assert_eq!(by_stem.library, lib);
        let by_lib = find_driver_impl(lib.to_str().unwrap(), &[]).unwrap();
        assert_eq!(by_lib.library, lib);
        assert_eq!(by_lib.manifest, None);

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
