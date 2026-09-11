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

    /// Join a workspace-relative path onto the root. `..` clamps at the
    /// root (never escapes), absolute paths and prefixes are rejected.
    fn join(&self, rel: &str) -> Result<PathBuf> {
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
        let dir = self.join(rel)?;
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
        let p = self.join(rel)?;
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
        let p = self.join(rel)?;
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating dirs for {}", p.display()))?;
        }
        std::fs::write(&p, content).with_context(|| format!("writing {}", p.display()))?;
        Ok(())
    }

    pub fn create_dir(&self, rel: &str) -> Result<()> {
        let p = self.join(rel)?;
        std::fs::create_dir_all(&p).with_context(|| format!("creating dir {}", p.display()))?;
        Ok(())
    }

    pub fn rename(&self, from: &str, to: &str) -> Result<()> {
        let a = self.join(from)?;
        let b = self.join(to)?;
        if let Some(parent) = b.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::rename(&a, &b)
            .with_context(|| format!("renaming {} -> {}", a.display(), b.display()))?;
        Ok(())
    }

    /// Delete a file or a whole directory tree.
    pub fn delete(&self, rel: &str) -> Result<()> {
        let p = self.join(rel)?;
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
    fn missing_root_refused() {
        assert!(Workspace::new("/nonexistent-mlab-ws-xyz").is_err());
    }
}
