//! Project workspace: create/read/write/rename/delete files and dirs,
//! confined under a single root. The UI owns paths as workspace-relative
//! strings; this module resolves them and rejects anything that would
//! escape the root (`..` past top, absolute paths, symlink exits).
//!
//! Note: no secret scanning here on purpose. These are the user's own
//! files (`.env` files legitimately contain keys); the guard watches
//! AI-bound sends, not the filesystem.

use anyhow::{Context, Result};
use std::path::{Component, Path, PathBuf};
use std::process::Stdio;

/// Cap on files opened into the UI buffer (chars). Bigger files refuse
/// with a clear error instead of freezing the frame.
pub const MAX_OPEN_CHARS: usize = 500_000;

#[derive(Debug, Clone)]
pub struct DirEntry {
    pub name: String,
    pub is_dir: bool,
    pub size: u64,
}

pub struct Workspace {
    root: PathBuf,
}

impl Workspace {
    /// Open an existing directory as the workspace root.
    pub fn new(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        anyhow::ensure!(root.is_dir(), "workspace root is not a directory: {}", root.display());
        let root = root.canonicalize().context("canonicalizing workspace root")?;
        Ok(Self { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Resolve a workspace-relative path onto the root. `..` clamps at
    /// the root (never escapes); absolute paths and symlink exits out of
    /// the root are rejected.
    pub fn resolve(&self, rel: &str) -> Result<PathBuf> {
        let mut p = self.root.clone();
        for comp in Path::new(rel).components() {
            match comp {
                Component::Normal(c) => p.push(c),
                Component::CurDir => {}
                // Clamped: pop only while deeper than root.
                Component::ParentDir => {
                    if p != self.root {
                        p.pop();
                    }
                }
                Component::RootDir | Component::Prefix(_) => {
                    anyhow::bail!("absolute paths are not allowed in the workspace");
                }
            }
        }
        self.contained(&p)
    }

    /// Reject symlink exits: the final path (or nearest existing ancestor)
    /// must canonicalize inside the root.
    fn contained(&self, p: &Path) -> Result<PathBuf> {
        let anchor = if p.exists() {
            p.canonicalize().context("canonicalizing workspace path")?
        } else {
            let parent = p.parent().unwrap_or(&self.root);
            let mut anc = parent;
            while !anc.exists() {
                anc = anc.parent().unwrap_or(&self.root);
            }
            let base = anc.canonicalize().context("canonicalizing ancestor")?;
            let rel = p.strip_prefix(anc).unwrap_or(Path::new(""));
            base.join(rel)
        };
        anyhow::ensure!(
            anchor.starts_with(&self.root),
            "path escapes the workspace root"
        );
        Ok(p.to_path_buf())
    }

    /// Entries of a workspace-relative dir, dirs first then files, by name.
    pub fn list(&self, rel: &str) -> Result<Vec<DirEntry>> {
        let dir = self.resolve(rel)?;
        anyhow::ensure!(dir.is_dir(), "not a directory");
        let mut out = Vec::new();
        for ent in std::fs::read_dir(&dir).with_context(|| format!("listing {}", dir.display()))? {
            let ent = ent?;
            let ft = ent.file_type()?;
            let size = if ft.is_file() { ent.metadata().map(|m| m.len()).unwrap_or(0) } else { 0 };
            out.push(DirEntry {
                name: ent.file_name().to_string_lossy().into_owned(),
                is_dir: ft.is_dir(),
                size,
            });
        }
        out.sort_by(|a, b| b.is_dir.cmp(&a.is_dir).then(a.name.cmp(&b.name)));
        Ok(out)
    }

    pub fn read(&self, rel: &str) -> Result<String> {
        let p = self.resolve(rel)?;
        let bytes = std::fs::read(&p).with_context(|| format!("reading {}", p.display()))?;
        let s = String::from_utf8_lossy(&bytes).into_owned();
        anyhow::ensure!(
            s.chars().count() <= MAX_OPEN_CHARS,
            "file too large to open ({} chars, cap {})",
            s.chars().count(),
            MAX_OPEN_CHARS
        );
        Ok(s)
    }

    /// Write (or overwrite) a file, creating parent dirs as needed.
    pub fn write(&self, rel: &str, content: &str) -> Result<()> {
        let p = self.resolve(rel)?;
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating dirs for {}", p.display()))?;
        }
        std::fs::write(&p, content).with_context(|| format!("writing {}", p.display()))?;
        Ok(())
    }

    pub fn create_dir(&self, rel: &str) -> Result<()> {
        let p = self.resolve(rel)?;
        std::fs::create_dir_all(&p).with_context(|| format!("creating dir {}", p.display()))?;
        Ok(())
    }

    pub fn rename(&self, from: &str, to: &str) -> Result<()> {
        let a = self.resolve(from)?;
        let b = self.resolve(to)?;
        if let Some(parent) = b.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::rename(&a, &b)
            .with_context(|| format!("renaming {} -> {}", a.display(), b.display()))?;
        Ok(())
    }

    /// Delete a file or a whole directory tree.
    pub fn delete(&self, rel: &str) -> Result<()> {
        let p = self.resolve(rel)?;
        anyhow::ensure!(p != self.root, "refusing to delete the workspace root");
        if p.is_dir() {
            std::fs::remove_dir_all(&p).with_context(|| format!("removing dir {}", p.display()))?;
        } else if p.exists() {
            std::fs::remove_file(&p).with_context(|| format!("removing {}", p.display()))?;
        } else {
            anyhow::bail!("no such file or directory");
        }
        Ok(())
    }
}

/// Cap on a single shell command (chars). The UI enforces the same.
pub const MAX_CMD_CHARS: usize = 4000;

/// Spawn `sh -c <cmd>` with cwd confined to the workspace. The caller owns
/// the Child (kill on Stop) and drains both pipes; the app polls
/// `try_wait` each frame for the exit code.
pub fn spawn_shell(cmd: &str, cwd: &Path) -> Result<tokio::process::Child> {
    anyhow::ensure!(!cmd.trim().is_empty(), "empty command");
    anyhow::ensure!(
        cmd.len() <= MAX_CMD_CHARS,
        "command too long (cap {MAX_CMD_CHARS})"
    );
    let child = tokio::process::Command::new("sh")
        .arg("-c")
        .arg(cmd)
        .current_dir(cwd)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("spawning command in {}", cwd.display()))?;
    Ok(child)
}

/// Join two workspace-relative paths for the UI breadcrumb.
pub fn join_rel(dir: &str, name: &str) -> String {
    if dir.trim().is_empty() {
        name.to_string()
    } else {
        format!("{}/{}", dir.trim_end_matches('/'), name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_root(name: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("mlab-ws-test-{name}"));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn crud_roundtrip() {
        let ws = Workspace::new(tmp_root("crud")).unwrap();
        ws.write("proj/main.py", "print(1)\n").unwrap();
        assert_eq!(ws.read("proj/main.py").unwrap(), "print(1)\n");
        let names: Vec<_> = ws.list("proj").unwrap().into_iter().map(|e| e.name).collect();
        assert_eq!(names, vec!["main.py"]);
        ws.rename("proj/main.py", "proj/app.py").unwrap();
        assert!(ws.read("proj/app.py").is_ok());
        ws.delete("proj/app.py").unwrap();
        assert!(ws.read("proj/app.py").is_err());
        ws.delete("proj").unwrap();
    }

    #[test]
    fn escapes_rejected_and_root_survives() {
        let root = tmp_root("esc");
        std::fs::write(root.join("secret.txt"), "x").unwrap();
        let ws = Workspace::new(&root).unwrap();
        // `..` clamps at root: this resolves INSIDE, to secret.txt.
        assert_eq!(ws.read("../secret.txt").unwrap(), "x");
        assert!(ws.read("/etc/hostname").is_err());
        assert!(ws.delete("").is_err());
        assert!(root.exists());
    }

    #[test]
    fn resolve_clamps_and_rejects() {
        let ws = Workspace::new(tmp_root("res")).unwrap();
        let a = ws.resolve("a/../../b").unwrap();
        assert!(a.starts_with(ws.root()));
        assert!(a.ends_with("b"));
        assert!(ws.resolve("/etc/hostname").is_err());
    }

    #[tokio::test]
    async fn spawn_shell_validates() {
        let root = tmp_root("sh");
        assert!(spawn_shell("", &root).is_err());
        let too_long = "x".repeat(MAX_CMD_CHARS + 1);
        assert!(spawn_shell(&too_long, &root).is_err());
        // A real run: echo exits 0 with its line on stdout.
        let mut child = spawn_shell("echo hi", &root).unwrap();
        let out = child.wait_with_output().await.unwrap();
        assert!(out.status.success());
        assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "hi");
    }

    #[test]
    fn missing_root_refused() {
        assert!(Workspace::new("/nonexistent-mlab-ws-xyz").is_err());
    }
}
