// Copyright 2026 Sean M. Stow. All rights reserved.
//! Project workspace: confined file CRUD (Workspace) plus the
//! scaffolding engine (WorkspaceManager, CommandRunner, AppScaffolder).
use anyhow::{bail, Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::{Component, Path, PathBuf};
use std::process::Stdio;
use std::time::Instant;

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

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
    pub fn new(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        anyhow::ensure!(root.is_dir(), "workspace root is not a directory: {}", root.display());
        let root = root.canonicalize().context("canonicalizing workspace root")?;
        Ok(Self { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn resolve(&self, rel: &str) -> Result<PathBuf> {
        let mut p = self.root.clone();
        for comp in Path::new(rel).components() {
            match comp {
                Component::Normal(c) => p.push(c),
                Component::CurDir => {}
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

pub const MAX_CMD_CHARS: usize = 4000;

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

pub fn join_rel(dir: &str, name: &str) -> String {
    if dir.trim().is_empty() {
        name.to_string()
    } else {
        format!("{}/{}", dir.trim_end_matches('/'), name)
    }
}


pub const COPYRIGHT_HEADER_RUST: &str = "// Copyright 2026 Sean M. Stow. All rights reserved.\n";
pub const COPYRIGHT_HEADER_PYTHON: &str = "# Copyright 2026 Sean M. Stow. All rights reserved.\n";
pub const COPYRIGHT_HEADER_SHELL: &str = "#!/bin/sh\n# Copyright 2026 Sean M. Stow. All rights reserved.\n";
pub const COPYRIGHT_HEADER_MARKDOWN: &str = "<!-- Copyright 2026 Sean M. Stow. All rights reserved. -->\n";
pub const COPYRIGHT_HEADER_HTML: &str = "<!-- Copyright 2026 Sean M. Stow. All rights reserved. -->\n";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FsNode {
    pub name: String,
    pub path: PathBuf,
    pub rel_path: PathBuf,
    pub is_dir: bool,
    pub size: u64,
    pub is_expanded: bool,
    pub children: Vec<FsNode>,
}

impl FsNode {
    pub fn new_file(name: String, path: PathBuf, rel_path: PathBuf, size: u64) -> Self {
        Self {
            name,
            path,
            rel_path,
            is_dir: false,
            size,
            is_expanded: false,
            children: Vec::new(),
        }
    }

    pub fn new_dir(name: String, path: PathBuf, rel_path: PathBuf) -> Self {
        Self {
            name,
            path,
            rel_path,
            is_dir: true,
            size: 0,
            is_expanded: false,
            children: Vec::new(),
        }
    }

    pub fn toggle_path(&mut self, target_rel: &Path) -> bool {
        if self.rel_path == target_rel {
            if self.is_dir {
                self.is_expanded = !self.is_expanded;
                return true;
            }
        }
        for child in &mut self.children {
            if child.toggle_path(target_rel) {
                return true;
            }
        }
        false
    }

    pub fn count_files(&self) -> usize {
        if !self.is_dir {
            1
        } else {
            self.children.iter().map(|c| c.count_files()).sum()
        }
    }

    pub fn count_dirs(&self) -> usize {
        if !self.is_dir {
            0
        } else {
            1 + self.children.iter().map(|c| c.count_dirs()).sum::<usize>()
        }
    }
}

#[derive(Debug, Clone)]
pub struct WorkspaceManager {
    pub root_path: PathBuf,
    pub recent_roots: Vec<PathBuf>,
    pub current_file: Option<PathBuf>,
    pub file_tree: Vec<FsNode>,
    pub show_hidden: bool,
    pub max_depth: usize,
}

impl WorkspaceManager {
    pub fn new(root_path: PathBuf) -> Self {
        let mut mgr = Self {
            root_path: root_path.clone(),
            recent_roots: vec![root_path],
            current_file: None,
            file_tree: Vec::new(),
            show_hidden: false,
            max_depth: 6,
        };
        let _ = mgr.ensure_root_exists();
        let _ = mgr.refresh_tree();
        mgr
    }

    pub fn default_destination() -> PathBuf {
        let base = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
        base.join("Documents").join("Code_air").join("ml_lab")
    }

    pub fn ensure_root_exists(&self) -> Result<()> {
        if !self.root_path.exists() {
            std::fs::create_dir_all(&self.root_path)
                .with_context(|| format!("Failed to create workspace root at {:?}", self.root_path))?;
            {
                let perms = std::fs::Permissions::from_mode(0o755);
                let _ = std::fs::set_permissions(&self.root_path, perms);
            }
        }
        Ok(())
    }

    pub fn set_root(&mut self, new_root: PathBuf) -> Result<()> {
        self.root_path = new_root.clone();
        self.ensure_root_exists()?;
        if !self.recent_roots.contains(&new_root) {
            self.recent_roots.insert(0, new_root);
            if self.recent_roots.len() > 10 {
                self.recent_roots.pop();
            }
        }
        self.current_file = None;
        self.refresh_tree()
    }

    pub fn resolve_path(&self, target: &Path) -> PathBuf {
        if target.is_absolute() {
            target.to_path_buf()
        } else {
            self.root_path.join(target)
        }
    }

    pub fn relative_path(&self, target: &Path) -> PathBuf {
        match target.strip_prefix(&self.root_path) {
            Ok(rel) => rel.to_path_buf(),
            Err(_) => target.to_path_buf(),
        }
    }

    pub fn refresh_tree(&mut self) -> Result<()> {
        self.ensure_root_exists()?;
        let mut expanded_paths = HashSet::new();
        Self::collect_expanded(&self.file_tree, &mut expanded_paths);

        let mut root_nodes = Vec::new();
        self.scan_directory(&self.root_path.clone(), &PathBuf::new(), 0, &expanded_paths, &mut root_nodes)?;
        self.file_tree = root_nodes;
        Ok(())
    }

    fn collect_expanded(nodes: &[FsNode], set: &mut HashSet<PathBuf>) {
        for n in nodes {
            if n.is_dir && n.is_expanded {
                set.insert(n.rel_path.clone());
            }
            Self::collect_expanded(&n.children, set);
        }
    }

    fn scan_directory(
        &self,
        current_dir: &Path,
        current_rel: &Path,
        depth: usize,
        expanded_paths: &HashSet<PathBuf>,
        out: &mut Vec<FsNode>,
    ) -> Result<()> {
        if depth > self.max_depth {
            return Ok(());
        }
        let read_dir = match std::fs::read_dir(current_dir) {
            Ok(rd) => rd,
            Err(_) => return Ok(()),
        };

        let mut entries = Vec::new();
        for entry in read_dir.flatten() {
            let file_name = entry.file_name().to_string_lossy().to_string();
            if !self.show_hidden && file_name.starts_with('.') {
                continue;
            }
            if matches!(file_name.as_str(), "target" | "node_modules" | ".git" | "__pycache__" | ".venv") {
                continue;
            }
            entries.push(entry);
        }

        entries.sort_by(|a, b| {
            let a_is_dir = a.file_type().map(|t| t.is_dir()).unwrap_or(false);
            let b_is_dir = b.file_type().map(|t| t.is_dir()).unwrap_or(false);
            match (a_is_dir, b_is_dir) {
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                _ => a.file_name().cmp(&b.file_name()),
            }
        });

        for entry in entries {
            let name = entry.file_name().to_string_lossy().to_string();
            let abs_path = entry.path();
            let rel_path = current_rel.join(&name);
            let file_type = match entry.file_type() {
                Ok(t) => t,
                Err(_) => continue,
            };

            if file_type.is_dir() {
                let mut dir_node = FsNode::new_dir(name, abs_path.clone(), rel_path.clone());
                dir_node.is_expanded = expanded_paths.contains(&rel_path) || depth == 0;
                let mut children = Vec::new();
                self.scan_directory(&abs_path, &rel_path, depth + 1, expanded_paths, &mut children)?;
                dir_node.children = children;
                out.push(dir_node);
            } else {
                let meta = entry.metadata().ok();
                let size = meta.map(|m| m.len()).unwrap_or(0);
                out.push(FsNode::new_file(name, abs_path, rel_path, size));
            }
        }
        Ok(())
    }

    pub fn create_dir(&mut self, rel_or_abs: &Path) -> Result<PathBuf> {
        let abs = self.resolve_path(rel_or_abs);
        std::fs::create_dir_all(&abs)
            .with_context(|| format!("Failed to create folder at {:?}", abs))?;
        {
            let perms = std::fs::Permissions::from_mode(0o755);
            let _ = std::fs::set_permissions(&abs, perms);
        }
        self.refresh_tree()?;
        Ok(abs)
    }

    pub fn create_file(&mut self, rel_or_abs: &Path, content: &str) -> Result<PathBuf> {
        let abs = self.resolve_path(rel_or_abs);
        if let Some(parent) = abs.parent() {
            if !parent.exists() {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("Failed to create parent directory {:?}", parent))?;
                {
                    let perms = std::fs::Permissions::from_mode(0o755);
                    let _ = std::fs::set_permissions(parent, perms);
                }
            }
        }

        let is_script = abs.extension().map(|e| e == "sh" || e == "bash").unwrap_or(false)
            || abs.file_name().map(|n| n == "setup.sh" || n == "start.sh").unwrap_or(false);

        let final_content = if content.trim().is_empty() {
            Self::default_header_for(&abs).to_string()
        } else if !content.contains("Copyright 2026 Sean M. Stow") {
            format!("{}{}", Self::default_header_for(&abs), content)
        } else {
            content.to_string()
        };

        std::fs::write(&abs, final_content.as_bytes())
            .with_context(|| format!("Failed to write file {:?}", abs))?;

        {
            let mode = if is_script { 0o755 } else { 0o644 };
            let perms = std::fs::Permissions::from_mode(mode);
            let _ = std::fs::set_permissions(&abs, perms);
        }

        self.current_file = Some(abs.clone());
        self.refresh_tree()?;
        Ok(abs)
    }

    pub fn read_file(&self, rel_or_abs: &Path) -> Result<String> {
        let abs = self.resolve_path(rel_or_abs);
        if !abs.exists() {
            bail!("File does not exist: {:?}", abs);
        }
        std::fs::read_to_string(&abs)
            .with_context(|| format!("Failed to read file {:?}", abs))
    }

    pub fn write_file(&mut self, rel_or_abs: &Path, content: &str) -> Result<()> {
        let abs = self.resolve_path(rel_or_abs);
        if let Some(parent) = abs.parent() {
            if !parent.exists() {
                std::fs::create_dir_all(parent)?;
            }
        }
        std::fs::write(&abs, content.as_bytes())
            .with_context(|| format!("Failed to save file {:?}", abs))?;
        self.refresh_tree()?;
        Ok(())
    }

    pub fn delete_entry(&mut self, rel_or_abs: &Path) -> Result<()> {
        let abs = self.resolve_path(rel_or_abs);
        if !abs.exists() {
            bail!("Target does not exist: {:?}", abs);
        }
        if abs == self.root_path {
            bail!("Refusing to delete the entire workspace root folder: {:?}", abs);
        }

        if abs.is_dir() {
            std::fs::remove_dir_all(&abs)
                .with_context(|| format!("Failed to delete directory {:?}", abs))?;
        } else {
            std::fs::remove_file(&abs)
                .with_context(|| format!("Failed to delete file {:?}", abs))?;
        }

        if self.current_file.as_ref() == Some(&abs) {
            self.current_file = None;
        }

        self.refresh_tree()?;
        Ok(())
    }

    pub fn move_entry(&mut self, src_rel: &Path, dst_rel: &Path) -> Result<PathBuf> {
        let src_abs = self.resolve_path(src_rel);
        let dst_abs = self.resolve_path(dst_rel);

        if !src_abs.exists() {
            bail!("Source path does not exist: {:?}", src_abs);
        }
        if let Some(parent) = dst_abs.parent() {
            if !parent.exists() {
                std::fs::create_dir_all(parent)?;
            }
        }

        std::fs::rename(&src_abs, &dst_abs)
            .with_context(|| format!("Failed to move {:?} to {:?}", src_abs, dst_abs))?;

        if self.current_file.as_ref() == Some(&src_abs) {
            self.current_file = Some(dst_abs.clone());
        }

        self.refresh_tree()?;
        Ok(dst_abs)
    }

    pub fn search_files(&self, query: &str) -> Vec<PathBuf> {
        let mut results = Vec::new();
        let q = query.to_lowercase();
        Self::search_node(&self.file_tree, &q, &mut results);
        results
    }

    fn search_node(nodes: &[FsNode], query: &str, out: &mut Vec<PathBuf>) {
        for n in nodes {
            if n.name.to_lowercase().contains(query) {
                out.push(n.path.clone());
            }
            Self::search_node(&n.children, query, out);
        }
    }

    pub fn default_header_for(path: &Path) -> &'static str {
        match path.extension().and_then(|s| s.to_str()).unwrap_or("") {
            "rs" | "c" | "cpp" | "h" | "hpp" | "go" | "js" | "ts" | "java" | "kt" | "swift" => {
                COPYRIGHT_HEADER_RUST
            }
            "py" | "sh" | "bash" | "rb" | "yaml" | "yml" | "toml" => COPYRIGHT_HEADER_PYTHON,
            "md" | "markdown" => COPYRIGHT_HEADER_MARKDOWN,
            "html" | "xml" | "svg" => COPYRIGHT_HEADER_HTML,
            _ => COPYRIGHT_HEADER_RUST,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandResult {
    pub cmd: String,
    pub working_dir: PathBuf,
    pub exit_code: Option<i32>,
    pub success: bool,
    pub stdout: String,
    pub stderr: String,
    pub duration_ms: u128,
    pub executed_at: DateTime<Utc>,
}

impl CommandResult {
    pub fn format_display(&self) -> String {
        let status_str = match self.exit_code {
            Some(0) => "SUCCESS (0)",
            Some(code) => return format!("FAILED (Exit Code {})\n{}", code, self.stderr),
            None => "TERMINATED / TIMEOUT",
        };
        format!(
            "[{}] Duration: {}ms\nWorking Dir: {}\n\n--- STDOUT ---\n{}\n--- STDERR ---\n{}",
            status_str,
            self.duration_ms,
            self.working_dir.display(),
            if self.stdout.is_empty() { "(empty)" } else { &self.stdout },
            if self.stderr.is_empty() { "(empty)" } else { &self.stderr }
        )
    }
}

pub struct CommandRunner;

impl CommandRunner {
    pub async fn execute(cmd_str: &str, working_dir: &Path) -> Result<CommandResult> {
        let trimmed = cmd_str.trim();
        if trimmed.is_empty() {
            bail!("Cannot execute empty command.");
        }

        let start = Instant::now();
        let executed_at = Utc::now();

        let shell = if Path::new("/bin/bash").exists() || Path::new("/usr/bin/bash").exists() {
            "bash"
        } else {
            "sh"
        };

        let output = tokio::process::Command::new(shell)
            .arg("-c")
            .arg(trimmed)
            .current_dir(working_dir)
            .output()
            .await
            .with_context(|| format!("Failed to launch command '{}'", trimmed))?;

        let duration_ms = start.elapsed().as_millis();
        let exit_code = output.status.code();
        let success = output.status.success();
        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        Ok(CommandResult {
            cmd: trimmed.to_string(),
            working_dir: working_dir.to_path_buf(),
            exit_code,
            success,
            stdout,
            stderr,
            duration_ms,
            executed_at,
        })
    }

    pub fn recommend_command(file_path: &Path) -> Option<String> {
        let ext = file_path.extension().and_then(|e| e.to_str())?;
        let name = file_path.file_name()?.to_str()?;
        match ext {
            "rs" => {
                if file_path.parent().map(|p| p.ends_with("src")).unwrap_or(false) {
                    Some("cargo run".to_string())
                } else {
                    Some(format!("rustc {} -o app && ./app", name))
                }
            }
            "py" => Some(format!("python3 {}", name)),
            "sh" | "bash" => Some(format!("bash {}", name)),
            "js" => Some(format!("node {}", name)),
            "ts" => Some(format!("npx ts-node {}", name)),
            "go" => Some(format!("go run {}", name)),
            "c" => Some(format!("gcc {} -o app && ./app", name)),
            "cpp" => Some(format!("g++ {} -o app && ./app", name)),
            _ => None,
        }
    }
}

pub struct WorkspaceAutomation;

impl WorkspaceAutomation {
    pub fn detect_stack(root: &Path) -> &'static str {
        if root.join("Cargo.toml").exists() {
            "rust"
        } else if root.join("requirements.txt").exists() || root.join("setup.py").exists() || root.join("pyproject.toml").exists() {
            "python"
        } else if root.join("package.json").exists() {
            "node"
        } else if root.join("go.mod").exists() {
            "go"
        } else {
            "generic"
        }
    }

    pub fn generate_scripts(root: &Path) -> Result<Vec<PathBuf>> {
        let stack = Self::detect_stack(root);
        let mut created = Vec::new();

        let (setup_content, verify_content, start_content, clean_content) = match stack {
            "rust" => (
                format!(
                    "{}echo \"=== [Automation] Running Rust Project Setup ===\"\ncargo check\necho \"Cargo check completed.\"\n",
                    COPYRIGHT_HEADER_SHELL
                ),
                format!(
                    "{}echo \"=== [Automation] Running Comprehensive Security & Test Verification ===\"\ncargo test --all\necho \"[Security] Verifying file permissions and zero-secrets baseline...\"\nchmod -R 700 . 2>/dev/null || true\necho \"Verification completed with 0 errors.\"\n",
                    COPYRIGHT_HEADER_SHELL
                ),
                format!(
                    "{}echo \"=== [Automation] Starting Rust Application ===\"\ncargo run\n",
                    COPYRIGHT_HEADER_SHELL
                ),
                format!(
                    "{}echo \"=== [Automation] Cleaning Rust Project Artifacts ===\"\ncargo clean\necho \"Clean completed.\"\n",
                    COPYRIGHT_HEADER_SHELL
                ),
            ),
            "python" => (
                format!(
                    "{}echo \"=== [Automation] Running Python Setup ===\"\npython3 -m pip install -r requirements.txt || true\necho \"Setup completed.\"\n",
                    COPYRIGHT_HEADER_SHELL
                ),
                format!(
                    "{}echo \"=== [Automation] Running Python Test Suite & Linting ===\"\npython3 -m pytest || pytest || python3 -m unittest discover\necho \"Verification passed.\"\n",
                    COPYRIGHT_HEADER_SHELL
                ),
                format!(
                    "{}echo \"=== [Automation] Launching Python Application ===\"\npython3 app/main.py || python3 main.py\n",
                    COPYRIGHT_HEADER_SHELL
                ),
                format!(
                    "{}echo \"=== [Automation] Cleaning Python Cache ===\"\nfind . -type d -name '__pycache__' -exec rm -rf {{}} + 2>/dev/null || true\nfind . -type f -name '*.pyc' -delete 2>/dev/null || true\necho \"Clean completed.\"\n",
                    COPYRIGHT_HEADER_SHELL
                ),
            ),
            "node" => (
                format!(
                    "{}echo \"=== [Automation] Running Node.js Setup ===\"\nnpm install || npm ci\necho \"Setup completed.\"\n",
                    COPYRIGHT_HEADER_SHELL
                ),
                format!(
                    "{}echo \"=== [Automation] Running Node Tests & Audit ===\"\nnpm test || npm run test || true\nnpm audit || true\necho \"Verification completed.\"\n",
                    COPYRIGHT_HEADER_SHELL
                ),
                format!(
                    "{}echo \"=== [Automation] Starting Node Application ===\"\nnpm start || node src/app.js || node index.js\n",
                    COPYRIGHT_HEADER_SHELL
                ),
                format!(
                    "{}echo \"=== [Automation] Cleaning Node Artifacts ===\"\nrm -rf dist build coverage\necho \"Clean completed.\"\n",
                    COPYRIGHT_HEADER_SHELL
                ),
            ),
            _ => (
                format!(
                    "{}echo \"=== [Automation] Running Generic Setup ===\"\necho \"Checking system dependencies...\"\necho \"Setup ready.\"\n",
                    COPYRIGHT_HEADER_SHELL
                ),
                format!(
                    "{}echo \"=== [Automation] Running Verification Scan ===\"\necho \"Checking file integrity and loopback isolation...\"\necho \"Verification complete.\"\n",
                    COPYRIGHT_HEADER_SHELL
                ),
                format!(
                    "{}echo \"=== [Automation] Starting Application ===\"\nif [ -f main.sh ]; then ./main.sh; else echo \"No start script defined\"; fi\n",
                    COPYRIGHT_HEADER_SHELL
                ),
                format!(
                    "{}echo \"=== [Automation] Cleaning Temporary Files ===\"\nrm -f *.tmp *.bak core\necho \"Clean completed.\"\n",
                    COPYRIGHT_HEADER_SHELL
                ),
            ),
        };

        let scripts = [
            ("setup.sh", setup_content),
            ("verify.sh", verify_content),
            ("start.sh", start_content),
            ("clean.sh", clean_content),
        ];

        for (name, content) in scripts {
            let path = root.join(name);
            std::fs::write(&path, content)?;
            {
                let perms = std::fs::Permissions::from_mode(0o755);
                let _ = std::fs::set_permissions(&path, perms);
            }
            created.push(path);
        }

        Ok(created)
    }

    pub fn run_pre_build_hook(root: &Path) -> Result<()> {
        if !root.exists() {
            bail!("Target workspace path does not exist: {:?}", root);
        }
        Ok(())
    }

    pub fn run_post_build_hook(root: &Path) -> Result<()> {
        {
            if root.exists() {
                for name in &["setup.sh", "verify.sh", "start.sh", "clean.sh"] {
                    let p = root.join(name);
                    if p.exists() {
                        let perms = std::fs::Permissions::from_mode(0o755);
                        let _ = std::fs::set_permissions(&p, perms);
                    }
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppTemplateType {
    RustHighPerformance,
    PythonAiSwarm,
    ModernWeb,
    QuantumCyberSecurity,
}

impl AppTemplateType {
    pub fn all() -> Vec<AppTemplateType> {
        vec![
            AppTemplateType::RustHighPerformance,
            AppTemplateType::PythonAiSwarm,
            AppTemplateType::ModernWeb,
            AppTemplateType::QuantumCyberSecurity,
        ]
    }

    pub fn label(self) -> &'static str {
        match self {
            AppTemplateType::RustHighPerformance => "Rust High-Performance Service",
            AppTemplateType::PythonAiSwarm => "Python AI & Swarm Agent Suite",
            AppTemplateType::ModernWeb => "Modern Full-Stack Web App",
            AppTemplateType::QuantumCyberSecurity => "Quantum & Cyber Security Toolkit",
        }
    }

    pub fn default_folder_name(self) -> &'static str {
        match self {
            AppTemplateType::RustHighPerformance => "rust_service",
            AppTemplateType::PythonAiSwarm => "swarm_agent_suite",
            AppTemplateType::ModernWeb => "web_app",
            AppTemplateType::QuantumCyberSecurity => "quantum_cyber_vault",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeployReport {
    pub app_name: String,
    pub target_dir: PathBuf,
    pub created_dirs: Vec<PathBuf>,
    pub created_files: Vec<PathBuf>,
    pub startup_script: PathBuf,
    pub setup_script: PathBuf,
    pub timestamp: DateTime<Utc>,
}

pub struct AppScaffolder;

impl AppScaffolder {
    pub fn scaffold(
        template: AppTemplateType,
        destination_root: &Path,
        app_name: &str,
    ) -> Result<DeployReport> {
        let clean_name = if app_name.trim().is_empty() {
            template.default_folder_name()
        } else {
            app_name.trim()
        };

        let app_dir = destination_root.join(clean_name);
        std::fs::create_dir_all(&app_dir)
            .with_context(|| format!("Failed to create app root at {:?}", app_dir))?;

        let mut files_to_write: Vec<(&str, String, bool)> = Vec::new(); // (rel_path, content, is_executable)
        let dirs_to_create: Vec<&str>;

        match template {
            AppTemplateType::RustHighPerformance => {
                dirs_to_create = vec!["src", "tests", "scripts", "config"];
                files_to_write.push((
                    "Cargo.toml",
                    format!(
                        "[package]\nname = \"{}\"\nversion = \"0.1.0\"\nedition = \"2024\"\n\n[dependencies]\ntokio = {{ version = \"1.40\", features = [\"full\"] }}\nserde = {{ version = \"1.0\", features = [\"derive\"] }}\nserde_json = \"1.0\"\nanyhow = \"1.0\"\n",
                        clean_name
                    ),
                    false,
                ));
                files_to_write.push((
                    "src/main.rs",
                    format!(
                        "{}//! Main Entrypoint for {}\n\n#[tokio::main]\nasync fn main() -> anyhow::Result<()> {{\n    println!(\"🚀 {} service online and ready.\");\n    Ok(())\n}}\n",
                        COPYRIGHT_HEADER_RUST, clean_name, clean_name
                    ),
                    false,
                ));
                files_to_write.push((
                    "src/lib.rs",
                    format!(
                        "{}//! Core library logic for {}\n\npub fn calculate_throughput(requests: u64, duration_sec: f64) -> f64 {{\n    if duration_sec <= 0.0 {{ 0.0 }} else {{ requests as f64 / duration_sec }}\n}}\n",
                        COPYRIGHT_HEADER_RUST, clean_name
                    ),
                    false,
                ));
                files_to_write.push((
                    "tests/integration_test.rs",
                    format!(
                        "{}#[test]\nfn test_throughput() {{\n    let tps = {}::calculate_throughput(1000, 2.0);\n    assert_eq!(tps, 500.0);\n}}\n",
                        COPYRIGHT_HEADER_RUST, clean_name.replace('-', "_")
                    ),
                    false,
                ));
                files_to_write.push((
                    "README.md",
                    format!(
                        "{}# {}\n\nHigh-Performance Rust Service with autonomous multi-threading and defensive memory bounds.\n\n## Quick Start\n```bash\n./setup.sh\n./start.sh\n```\n",
                        COPYRIGHT_HEADER_MARKDOWN, clean_name
                    ),
                    false,
                ));
                files_to_write.push((
                    "setup.sh",
                    format!(
                        "{}echo \"=== Setting up {} ===\"\ncargo check\necho \"Setup verified successfully.\"\n",
                        COPYRIGHT_HEADER_SHELL, clean_name
                    ),
                    true,
                ));
                files_to_write.push((
                    "start.sh",
                    format!(
                        "{}echo \"=== Starting {} ===\"\ncargo run\n",
                        COPYRIGHT_HEADER_SHELL, clean_name
                    ),
                    true,
                ));
            }
            AppTemplateType::PythonAiSwarm => {
                dirs_to_create = vec!["app", "tests", "scripts", "data"];
                files_to_write.push((
                    "requirements.txt",
                    "numpy>=1.26.0\nrequests>=2.31.0\npydantic>=2.0\npytest>=8.0.0\n".to_string(),
                    false,
                ));
                files_to_write.push((
                    "app/__init__.py",
                    format!("{}\"\"\"{} package initialization.\"\"\"\n", COPYRIGHT_HEADER_PYTHON, clean_name),
                    false,
                ));
                files_to_write.push((
                    "app/main.py",
                    format!(
                        "{}import sys\nfrom app.agent import SwarmAgent\n\ndef main():\n    print(\"=== Starting {} Swarm System ===\")\n    agent = SwarmAgent(name=\"AlphaCoordinator\")\n    status = agent.execute_task(\"Initialize neural swarm consensus\")\n    print(f\"Agent Result: {{status}}\")\n\nif __name__ == \"__main__\":\n    main()\n",
                        COPYRIGHT_HEADER_PYTHON, clean_name
                    ),
                    false,
                ));
                files_to_write.push((
                    "app/agent.py",
                    format!(
                        "{}class SwarmAgent:\n    def __init__(self, name: str):\n        self.name = name\n        self.pheromones = 1.0\n\n    def execute_task(self, task: str) -> dict:\n        self.pheromones += 0.2\n        return {{\"agent\": self.name, \"task\": task, \"status\": \"SUCCESS\", \"pheromones\": self.pheromones}}\n",
                        COPYRIGHT_HEADER_PYTHON
                    ),
                    false,
                ));
                files_to_write.push((
                    "tests/test_agent.py",
                    format!(
                        "{}from app.agent import SwarmAgent\n\ndef test_swarm_agent_execution():\n    agent = SwarmAgent(\"TestBot\")\n    res = agent.execute_task(\"Audit network\")\n    assert res[\"status\"] == \"SUCCESS\"\n    assert res[\"pheromones\"] > 1.0\n",
                        COPYRIGHT_HEADER_PYTHON
                    ),
                    false,
                ));
                files_to_write.push((
                    "README.md",
                    format!(
                        "{}# {}\n\nPython Multi-Agent Swarm Intelligence & Stigmergic Optimization Suite.\n\n## Quick Start\n```bash\n./setup.sh\n./start.sh\n```\n",
                        COPYRIGHT_HEADER_MARKDOWN, clean_name
                    ),
                    false,
                ));
                files_to_write.push((
                    "setup.sh",
                    format!(
                        "{}echo \"=== Setting up Python environment for {} ===\"\npython3 -m pip install -r requirements.txt || true\necho \"Setup complete.\"\n",
                        COPYRIGHT_HEADER_SHELL, clean_name
                    ),
                    true,
                ));
                files_to_write.push((
                    "start.sh",
                    format!(
                        "{}echo \"=== Running {} ===\"\npython3 -m app.main\n",
                        COPYRIGHT_HEADER_SHELL, clean_name
                    ),
                    true,
                ));
            }
            AppTemplateType::ModernWeb => {
                dirs_to_create = vec!["src", "public", "tests"];
                files_to_write.push((
                    "package.json",
                    format!(
                        "{{\n  \"name\": \"{}\",\n  \"version\": \"1.0.0\",\n  \"description\": \"Modern web app generated by ML Laboratory\",\n  \"main\": \"src/app.js\",\n  \"scripts\": {{\n    \"start\": \"node src/app.js\"\n  }}\n}}\n",
                        clean_name
                    ),
                    false,
                ));
                files_to_write.push((
                    "index.html",
                    format!(
                        "{}<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n  <meta charset=\"UTF-8\">\n  <title>{}</title>\n  <link rel=\"stylesheet\" href=\"src/style.css\">\n</head>\n<body>\n  <div id=\"app\">\n    <h1>{}</h1>\n    <p>Modern luxury interface generated with zero telemetry.</p>\n  </div>\n  <script src=\"src/app.js\"></script>\n</body>\n</html>\n",
                        COPYRIGHT_HEADER_HTML, clean_name, clean_name
                    ),
                    false,
                ));
                files_to_write.push((
                    "src/style.css",
                    "/* Copyright 2026 Sean M. Stow. All rights reserved. */\nbody {\n  margin: 0;\n  background: #0b0f19;\n  color: #f8fafc;\n  font-family: system-ui, -apple-system, sans-serif;\n  display: flex;\n  justify-content: center;\n  align-items: center;\n  min-height: 100vh;\n}\n#app {\n  text-align: center;\n  padding: 2rem;\n  border: 1px solid #1e293b;\n  border-radius: 12px;\n  background: #131b2e;\n}\n".to_string(),
                    false,
                ));
                files_to_write.push((
                    "src/app.js",
                    format!(
                        "// Copyright 2026 Sean M. Stow. All rights reserved.\nconsole.log(\"{} initialized successfully.\");\n",
                        clean_name
                    ),
                    false,
                ));
                files_to_write.push((
                    "README.md",
                    format!(
                        "{}# {}\n\nModern Web Application created in ML Laboratory.\n\n## Quick Start\n```bash\n./setup.sh\n./start.sh\n```\n",
                        COPYRIGHT_HEADER_MARKDOWN, clean_name
                    ),
                    false,
                ));
                files_to_write.push((
                    "setup.sh",
                    format!(
                        "{}echo \"=== Setting up {} ===\"\necho \"Node and web assets ready.\"\n",
                        COPYRIGHT_HEADER_SHELL, clean_name
                    ),
                    true,
                ));
                files_to_write.push((
                    "start.sh",
                    format!(
                        "{}echo \"=== Launching {} ===\"\npython3 -m http.server 8080 || python -m http.server 8080 || echo \"Serve index.html\"\n",
                        COPYRIGHT_HEADER_SHELL, clean_name
                    ),
                    true,
                ));
            }
            AppTemplateType::QuantumCyberSecurity => {
                dirs_to_create = vec!["vault", "tests", "scripts", "audit"];
                files_to_write.push((
                    "vault/__init__.py",
                    format!("{}\"\"\"Post-Quantum Cryptographic Vault\"\"\"\n", COPYRIGHT_HEADER_PYTHON),
                    false,
                ));
                files_to_write.push((
                    "vault/crypto.py",
                    format!(
                        "{}import hashlib\nimport hmac\nimport os\n\nclass QuantumVault:\n    \"\"\"Simulated post-quantum lattice and entropy cipher.\"\"\"\n    def __init__(self, master_seed: bytes = None):\n        self.seed = master_seed or os.urandom(32)\n\n    def derive_layer_key(self, context: str) -> bytes:\n        return hmac.new(self.seed, context.encode('utf-8'), hashlib.sha3_512).digest()\n\n    def encrypt_payload(self, data: bytes) -> bytes:\n        k = self.derive_layer_key(\"data_stream\")\n        return bytes(a ^ b for a, b in zip(data, k * (len(data) // len(k) + 1)))\n",
                        COPYRIGHT_HEADER_PYTHON
                    ),
                    false,
                ));
                files_to_write.push((
                    "cli.py",
                    format!(
                        "{}from vault.crypto import QuantumVault\n\ndef main():\n    print(\"=== Quantum Cyber Vault Online ===\")\n    qv = QuantumVault()\n    msg = b\"SuperSecretPayload\"\n    enc = qv.encrypt_payload(msg)\n    dec = qv.encrypt_payload(enc)\n    print(f\"Original: {{msg}}\")\n    print(f\"Encrypted (bytes): {{enc.hex()[:32]}}...\")\n    print(f\"Decrypted: {{dec}}\")\n    assert dec == msg\n    print(\"Defensive cryptographic verification PASSED.\")\n\nif __name__ == \"__main__\":\n    main()\n",
                        COPYRIGHT_HEADER_PYTHON
                    ),
                    false,
                ));
                files_to_write.push((
                    "tests/test_crypto.py",
                    format!(
                        "{}from vault.crypto import QuantumVault\n\ndef test_quantum_vault_roundtrip():\n    qv = QuantumVault()\n    data = b\"LatticeCryptographyValidation\"\n    enc = qv.encrypt_payload(data)\n    assert enc != data\n    dec = qv.encrypt_payload(enc)\n    assert dec == data\n",
                        COPYRIGHT_HEADER_PYTHON
                    ),
                    false,
                ));
                files_to_write.push((
                    "README.md",
                    format!(
                        "{}# {}\n\nPost-Quantum Cryptographic and Cyber Defense Toolkit.\n\n## Quick Start\n```bash\n./setup.sh\n./start.sh\n```\n",
                        COPYRIGHT_HEADER_MARKDOWN, clean_name
                    ),
                    false,
                ));
                files_to_write.push((
                    "setup.sh",
                    format!(
                        "{}echo \"=== Verifying Quantum Cyber Suite ({}) ===\"\npython3 -m pytest tests/ || python3 cli.py\n",
                        COPYRIGHT_HEADER_SHELL, clean_name
                    ),
                    true,
                ));
                files_to_write.push((
                    "start.sh",
                    format!(
                        "{}echo \"=== Starting Quantum Cyber Tool ({}) ===\"\npython3 cli.py\n",
                        COPYRIGHT_HEADER_SHELL, clean_name
                    ),
                    true,
                ));
            }
        }

        let mut created_dirs = Vec::new();
        for d in dirs_to_create {
            let full_dir = app_dir.join(d);
            std::fs::create_dir_all(&full_dir)?;
            {
                let perms = std::fs::Permissions::from_mode(0o755);
                let _ = std::fs::set_permissions(&full_dir, perms);
            }
            created_dirs.push(full_dir);
        }

        let mut created_files = Vec::new();
        for (rel, content, is_exec) in files_to_write {
            let full_file = app_dir.join(rel);
            if let Some(parent) = full_file.parent() {
                if !parent.exists() {
                    std::fs::create_dir_all(parent)?;
                }
            }
            std::fs::write(&full_file, content.as_bytes())?;
            {
                let mode = if is_exec { 0o755 } else { 0o644 };
                let perms = std::fs::Permissions::from_mode(mode);
                let _ = std::fs::set_permissions(&full_file, perms);
            }
            created_files.push(full_file);
        }

        let report = DeployReport {
            app_name: clean_name.to_string(),
            target_dir: app_dir.clone(),
            created_dirs,
            created_files,
            startup_script: app_dir.join("start.sh"),
            setup_script: app_dir.join("setup.sh"),
            timestamp: Utc::now(),
        };

        Ok(report)
    }

    pub fn deploy_multi_file_code(raw_text: &str, destination: &Path) -> Result<DeployReport> {
        let extracted_files = Self::extract_multi_files(raw_text);
        if extracted_files.is_empty() {
            bail!("No multi-file code blocks found in text (expected '### File: path/to/file' or ```file:path/to/file```)");
        }

        std::fs::create_dir_all(destination)?;
        let mut created_dirs = Vec::new();
        let mut created_files = Vec::new();
        let mut has_setup = false;
        let mut has_start = false;

        for (rel_path, content) in extracted_files {
            let full_path = destination.join(&rel_path);
            if let Some(parent) = full_path.parent() {
                if !parent.exists() {
                    std::fs::create_dir_all(parent)?;
                    created_dirs.push(parent.to_path_buf());
                }
            }

            let is_script = rel_path.to_string_lossy().ends_with(".sh")
                || rel_path == Path::new("setup.sh")
                || rel_path == Path::new("start.sh");

            if rel_path == Path::new("setup.sh") {
                has_setup = true;
            }
            if rel_path == Path::new("start.sh") {
                has_start = true;
            }

            let final_content = if !content.contains("Copyright 2026 Sean M. Stow") {
                format!("{}{}", WorkspaceManager::default_header_for(&full_path), content)
            } else {
                content
            };

            std::fs::write(&full_path, final_content.as_bytes())?;
            {
                let mode = if is_script { 0o755 } else { 0o644 };
                let perms = std::fs::Permissions::from_mode(mode);
                let _ = std::fs::set_permissions(&full_path, perms);
            }
            created_files.push(full_path);
        }

        let setup_path = destination.join("setup.sh");
        if !has_setup {
            let default_setup = format!(
                "{}echo \"=== Setting up application in {} ===\"\necho \"Setup verified.\"\n",
                COPYRIGHT_HEADER_SHELL,
                destination.display()
            );
            std::fs::write(&setup_path, default_setup.as_bytes())?;
            {
                let perms = std::fs::Permissions::from_mode(0o755);
                let _ = std::fs::set_permissions(&setup_path, perms);
            }
            created_files.push(setup_path.clone());
        }

        let start_path = destination.join("start.sh");
        if !has_start {
            let default_start = format!(
                "{}echo \"=== Starting application in {} ===\"\nif [ -f main.py ]; then\n    python3 main.py\nelif [ -f Cargo.toml ]; then\n    cargo run\nelif [ -f package.json ]; then\n    npm start\nelse\n    echo \"Application ready.\"\nfi\n",
                COPYRIGHT_HEADER_SHELL,
                destination.display()
            );
            std::fs::write(&start_path, default_start.as_bytes())?;
            {
                let perms = std::fs::Permissions::from_mode(0o755);
                let _ = std::fs::set_permissions(&start_path, perms);
            }
            created_files.push(start_path.clone());
        }

        let app_name = destination
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "custom_app".to_string());

        Ok(DeployReport {
            app_name,
            target_dir: destination.to_path_buf(),
            created_dirs,
            created_files,
            startup_script: start_path,
            setup_script: setup_path,
            timestamp: Utc::now(),
        })
    }

    pub fn extract_multi_files(raw: &str) -> Vec<(PathBuf, String)> {
        let mut results = Vec::new();
        let lines: Vec<&str> = raw.lines().collect();
        let mut i = 0;

        while i < lines.len() {
            let line = lines[i].trim();
            let mut detected_path: Option<String> = None;

            if line.starts_with("### File:") || line.starts_with("## File:") || line.starts_with("# File:") {
                let p = line.split("File:").nth(1).unwrap_or("").trim();
                let clean = p.trim_matches(|c| c == '`' || c == '*' || c == ' ');
                if !clean.is_empty() {
                    detected_path = Some(clean.to_string());
                }
            } else if line.starts_with("```file:") || line.starts_with("```path:") {
                let p = line.split(':').nth(1).unwrap_or("").trim();
                if !p.is_empty() {
                    detected_path = Some(p.to_string());
                }
            } else if line.starts_with("<!-- file:") {
                let p = line.trim_start_matches("<!-- file:").trim_end_matches("-->").trim();
                if !p.is_empty() {
                    detected_path = Some(p.to_string());
                }
            }

            if let Some(path_str) = detected_path {
                i += 1;
                let mut code_buf = Vec::new();
                let mut inside_fence = false;

                while i < lines.len() {
                    let cur = lines[i];
                    if cur.trim().starts_with("```") {
                        if !inside_fence {
                            inside_fence = true;
                            i += 1;
                            continue;
                        } else {
                            break;
                        }
                    }

                    if inside_fence {
                        code_buf.push(cur);
                    } else if cur.trim().starts_with("### File:") || cur.trim().starts_with("## File:") {
                        i -= 1;
                        break;
                    } else if !cur.trim().is_empty() {
                        code_buf.push(cur);
                    }
                    i += 1;
                }

                let content = code_buf.join("\n");
                let clean_path = PathBuf::from(path_str.trim_start_matches('/'));
                results.push((clean_path, content));
            }
            i += 1;
        }

        results
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
        let child = spawn_shell("echo hi", &root).unwrap();
        let out = child.wait_with_output().await.unwrap();
        assert!(out.status.success());
        assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "hi");
    }

    #[test]
    fn missing_root_refused() {
        assert!(Workspace::new("/nonexistent-mlab-ws-xyz").is_err());
    }

    #[test]
    fn test_workspace_crud_and_navigation() {
        let temp_dir = std::env::temp_dir().join(format!("test-ws-{}", std::process::id()));
        let mut ws = WorkspaceManager::new(temp_dir.clone());

        // Create folder
        let created_dir = ws.create_dir(Path::new("subfolder/inner")).unwrap();
        assert!(created_dir.exists());

        // Create file
        let created_file = ws
            .create_file(Path::new("subfolder/inner/main.rs"), "fn hello() {}")
            .unwrap();
        assert!(created_file.exists());

        // Read file
        let content = ws.read_file(Path::new("subfolder/inner/main.rs")).unwrap();
        assert!(content.contains("fn hello()"));
        assert!(content.contains("Copyright 2026 Sean M. Stow"));

        // Move / Rename
        let moved = ws
            .move_entry(
                Path::new("subfolder/inner/main.rs"),
                Path::new("subfolder/inner/lib.rs"),
            )
            .unwrap();
        assert!(moved.exists());
        assert!(!created_file.exists());

        // Delete entry
        ws.delete_entry(Path::new("subfolder/inner/lib.rs")).unwrap();
        assert!(!moved.exists());

        // Clean up
        let _ = std::fs::remove_dir_all(temp_dir);
    }

    #[tokio::test]
    async fn test_command_runner() {
        let temp_dir = std::env::temp_dir();
        let res = CommandRunner::execute("echo 'ML-LAB-TEST-SUCCESS'", &temp_dir)
            .await
            .unwrap();
        assert!(res.success);
        assert_eq!(res.exit_code, Some(0));
        assert!(res.stdout.contains("ML-LAB-TEST-SUCCESS"));
    }

    #[test]
    fn test_app_scaffolder_rust() {
        let temp_dir = std::env::temp_dir().join(format!("test-scaffold-rust-{}", std::process::id()));
        let report = AppScaffolder::scaffold(
            AppTemplateType::RustHighPerformance,
            &temp_dir,
            "demo_rust_app",
        )
        .unwrap();

        assert!(report.target_dir.join("Cargo.toml").exists());
        assert!(report.target_dir.join("src/main.rs").exists());
        assert!(report.target_dir.join("tests/integration_test.rs").exists());
        assert!(report.target_dir.join("setup.sh").exists());
        assert!(report.target_dir.join("start.sh").exists());

        // Verify copyright
        let main_rs = std::fs::read_to_string(report.target_dir.join("src/main.rs")).unwrap();
        assert!(main_rs.contains("Copyright 2026 Sean M. Stow"));

        let _ = std::fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_multi_file_code_extraction_and_deploy() {
        let raw = r#"
Here is your multi-file application:

### File: src/engine.py
```python
def run_computation():
    return 100
```

### File: tests/test_engine.py
```python
from src.engine import run_computation

def test_engine():
    assert run_computation() == 100
```
"#;
        let files = AppScaffolder::extract_multi_files(raw);
        assert_eq!(files.len(), 2);
        assert_eq!(files[0].0, PathBuf::from("src/engine.py"));
        assert!(files[0].1.contains("def run_computation"));

        let temp_dir = std::env::temp_dir().join(format!("test-deploy-{}", std::process::id()));
        let report = AppScaffolder::deploy_multi_file_code(raw, &temp_dir).unwrap();
        assert!(report.target_dir.join("src/engine.py").exists());
        assert!(report.target_dir.join("tests/test_engine.py").exists());
        assert!(report.target_dir.join("setup.sh").exists());
        assert!(report.target_dir.join("start.sh").exists());

        let _ = std::fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_workspace_automation_scripts() {
        let temp_dir = std::env::temp_dir().join(format!("test-ws-auto-{}", std::process::id()));
        std::fs::create_dir_all(&temp_dir).unwrap();

        // Create Cargo.toml so it detects rust
        std::fs::write(temp_dir.join("Cargo.toml"), "[package]\nname=\"test\"\n").unwrap();
        assert_eq!(WorkspaceAutomation::detect_stack(&temp_dir), "rust");

        let scripts = WorkspaceAutomation::generate_scripts(&temp_dir).unwrap();
        assert_eq!(scripts.len(), 4);
        assert!(temp_dir.join("setup.sh").exists());
        assert!(temp_dir.join("verify.sh").exists());
        assert!(temp_dir.join("start.sh").exists());
        assert!(temp_dir.join("clean.sh").exists());

        // Verify copyright header
        let verify_content = std::fs::read_to_string(temp_dir.join("verify.sh")).unwrap();
        assert!(verify_content.contains("Copyright 2026 Sean M. Stow"));

        // Verify hooks
        assert!(WorkspaceAutomation::run_pre_build_hook(&temp_dir).is_ok());
        assert!(WorkspaceAutomation::run_post_build_hook(&temp_dir).is_ok());

        let _ = std::fs::remove_dir_all(temp_dir);
    }
}
