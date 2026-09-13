// Copyright 2026 Sean M. Stow. All rights reserved.
//! AI Code Studio & Workspace IDE
//!
//! Integrates complete file and folder management (mkdir, touch, delete, move, edit),
//! customizable destination folder selection, an interactive hierarchical file tree,
//! asynchronous terminal command execution dock, and autonomous full-fledged
//! application scaffolding and deployment.

use eframe::egui;
use egui_code_editor::{CodeEditor, Syntax};
use crate::ollama::api::{ChatOptions, ChatRequest, Message, OllamaClient};
use crate::storage::ChatMessage;
use crate::workspace::{
    AppScaffolder, AppTemplateType, CommandResult, CommandRunner, DeployReport, FsNode, WorkspaceManager,
};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use tokio::runtime::Runtime;

#[derive(Clone, Debug, PartialEq)]
pub struct CompilerDiagnostic {
    pub language: String,
    pub error_code: Option<String>,
    pub file: Option<String>,
    pub line: Option<usize>,
    pub message: String,
    pub raw_snippet: String,
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct TerminalSession {
    pub id: usize,
    pub name: String,
    pub working_dir: PathBuf,
    pub logs: Vec<CommandResult>,
    pub history: Vec<String>,
    pub history_idx: Option<usize>,
    pub prompt_input: String,
    pub is_running: bool,
    pub running_cmd: String,
    pub filter_query: String,
}

impl TerminalSession {
    pub fn new(id: usize, name: &str, working_dir: PathBuf) -> Self {
        Self {
            id,
            name: name.to_string(),
            working_dir,
            logs: Vec::new(),
            history: Vec::new(),
            history_idx: None,
            prompt_input: String::new(),
            is_running: false,
            running_cmd: String::new(),
            filter_query: String::new(),
        }
    }
}

pub struct EditorPanel {
    pub code: String,
    pub language: String,
    suggestion_id: usize,
    pending_suggestion: Option<String>,
    suggestion_buf: String,
    show_diff: bool,
    gate: crate::security::ConfirmGate,
    pub file_path: String,
    pub file_status: String,
    // AI Coder Chatbox state
    pub coder_model: Option<String>,
    pub coder_input: String,
    pub coder_messages: Vec<ChatMessage>,
    pub coder_parsed_cache: Vec<Vec<crate::ui::chat::MessageSegment>>,
    pub coder_is_streaming: bool,
    pub coder_stream_buf: String,
    pub show_coder_chat: bool,
    pub include_editor_context: bool,

    // Workspace & File Management State
    pub workspace: WorkspaceManager,
    pub show_workspace_tree: bool,
    pub selected_node_path: Option<PathBuf>,
    pub file_search_query: String,
    pub show_destination_modal: bool,
    pub destination_input: String,
    pub new_file_dialog_open: bool,
    pub new_file_input: String,
    pub new_folder_dialog_open: bool,
    pub new_folder_input: String,
    pub rename_dialog_open: bool,
    pub rename_target: Option<PathBuf>,
    pub rename_input: String,
    pub delete_confirm_open: bool,
    pub delete_target: Option<PathBuf>,

    // Terminal & Command Runner State
    pub show_terminal: bool,
    pub terminal_input: String,
    pub terminal_prompt_input: String,
    pub terminal_history: Vec<String>,
    pub terminal_history_idx: Option<usize>,
    pub terminal_logs: Vec<CommandResult>,
    pub terminal_is_running: bool,
    pub terminal_running_cmd: String,
    pub terminal_height_ratio: f32,
    pub terminal_sessions: Vec<TerminalSession>,
    pub active_terminal_idx: usize,
    pub next_terminal_id: usize,
    pub terminal_filter: String,

    // Application Scaffolding & Deployment State
    pub show_scaffold_modal: bool,
    pub scaffold_selected_template: AppTemplateType,
    pub scaffold_app_name: String,
    pub last_deploy_report: Option<DeployReport>,
}

impl EditorPanel {
    pub fn new() -> Self {
        let ws_dest = WorkspaceManager::default_destination();
        let ws = WorkspaceManager::new(ws_dest.clone());
        let default_file = ws_dest.join("main.rs");

        Self {
            code: "// Start coding with AI assistance\nfn main() {\n    println!(\"Hello, world!\");\n}".to_string(),
            language: "rust".to_string(),
            suggestion_id: 0,
            pending_suggestion: None,
            suggestion_buf: String::new(),
            show_diff: true,
            gate: crate::security::ConfirmGate::new(),
            file_path: default_file.to_string_lossy().to_string(),
            file_status: String::new(),
            coder_model: None,
            coder_input: String::new(),
            coder_messages: Vec::new(),
            coder_parsed_cache: Vec::new(),
            coder_is_streaming: false,
            coder_stream_buf: String::new(),
            show_coder_chat: true,
            include_editor_context: true,

            workspace: ws,
            show_workspace_tree: true,
            selected_node_path: None,
            file_search_query: String::new(),
            show_destination_modal: false,
            destination_input: ws_dest.to_string_lossy().to_string(),
            new_file_dialog_open: false,
            new_file_input: String::new(),
            new_folder_dialog_open: false,
            new_folder_input: String::new(),
            rename_dialog_open: false,
            rename_target: None,
            rename_input: String::new(),
            delete_confirm_open: false,
            delete_target: None,

            show_terminal: true,
            terminal_input: String::new(),
            terminal_prompt_input: String::new(),
            terminal_history: Vec::new(),
            terminal_history_idx: None,
            terminal_logs: Vec::new(),
            terminal_is_running: false,
            terminal_running_cmd: String::new(),
            terminal_height_ratio: 0.55,
            terminal_sessions: vec![TerminalSession::new(1, "bash 1", ws_dest)],
            active_terminal_idx: 0,
            next_terminal_id: 1,
            terminal_filter: String::new(),

            show_scaffold_modal: false,
            scaffold_selected_template: AppTemplateType::RustHighPerformance,
            scaffold_app_name: "my_application".to_string(),
            last_deploy_report: None,
        }
    }

    /// Spawn a new terminal tab session
    pub fn new_terminal_session(&mut self) {
        self.next_terminal_id += 1;
        let id = self.next_terminal_id;
        let name = format!("bash {}", id);
        let ws = self.workspace.root_path.clone();
        let session = TerminalSession::new(id, &name, ws);
        self.terminal_sessions.push(session);
        self.active_terminal_idx = self.terminal_sessions.len().saturating_sub(1);
        self.sync_active_session_fields();
    }

    /// Close a terminal tab session (preserves at least 1 session)
    pub fn close_terminal_session(&mut self, idx: usize) {
        if self.terminal_sessions.len() > 1 && idx < self.terminal_sessions.len() {
            self.terminal_sessions.remove(idx);
            if self.active_terminal_idx >= self.terminal_sessions.len() {
                self.active_terminal_idx = self.terminal_sessions.len().saturating_sub(1);
            } else if self.active_terminal_idx > idx {
                self.active_terminal_idx -= 1;
            }
            self.sync_active_session_fields();
        }
    }

    /// Sync active session data into top-level editor fields
    pub fn sync_active_session_fields(&mut self) {
        let idx = self.active_terminal_idx.min(self.terminal_sessions.len().saturating_sub(1));
        if let Some(s) = self.terminal_sessions.get(idx) {
            self.terminal_logs = s.logs.clone();
            self.terminal_history = s.history.clone();
            self.terminal_prompt_input = s.prompt_input.clone();
            self.terminal_is_running = s.is_running;
            self.terminal_running_cmd = s.running_cmd.clone();
            self.terminal_history_idx = s.history_idx;
            self.terminal_filter = s.filter_query.clone();
        }
    }

    /// Sync top-level fields back into active session struct
    pub fn update_active_session_from_fields(&mut self) {
        let idx = self.active_terminal_idx.min(self.terminal_sessions.len().saturating_sub(1));
        if let Some(s) = self.terminal_sessions.get_mut(idx) {
            s.logs = self.terminal_logs.clone();
            s.history = self.terminal_history.clone();
            s.prompt_input = self.terminal_prompt_input.clone();
            s.is_running = self.terminal_is_running;
            s.running_cmd = self.terminal_running_cmd.clone();
            s.history_idx = self.terminal_history_idx;
            s.filter_query = self.terminal_filter.clone();
        }
    }

    /// Calculate Shannon entropy (bits per byte) for code security auditing
    pub fn calculate_entropy(s: &str) -> f32 {
        if s.is_empty() {
            return 0.0;
        }
        let mut counts = [0usize; 256];
        for &b in s.as_bytes() {
            counts[b as usize] += 1;
        }
        let len = s.len() as f32;
        let mut entropy = 0.0_f32;
        for &c in &counts {
            if c > 0 {
                let p = c as f32 / len;
                entropy -= p * p.log2();
            }
        }
        entropy
    }

    /// Extract structured compiler / runtime diagnostics from terminal output
    pub fn extract_compiler_diagnostic(log: &CommandResult) -> Option<CompilerDiagnostic> {
        if log.success || log.exit_code == Some(0) {
            return None;
        }
        let combined = format!("{}\n{}", log.stdout, log.stderr);
        if combined.trim().is_empty() {
            return None;
        }

        // Rust compiler diagnostic pattern
        if let Some(err_idx) = combined.find("error[E") {
            let snippet = &combined[err_idx..];
            let line_end = snippet.find('\n').unwrap_or(snippet.len());
            let err_line = &snippet[..line_end];
            let code = err_line.split(':').next().map(|s| s.trim().to_string());

            let mut file_found = None;
            let mut line_found = None;
            if let Some(arrow_idx) = snippet.find("--> ") {
                let arrow_sub = &snippet[arrow_idx + 4..];
                let arrow_line = arrow_sub.lines().next().unwrap_or("");
                let parts: Vec<&str> = arrow_line.split(':').collect();
                if parts.len() >= 2 {
                    file_found = Some(parts[0].trim().to_string());
                    line_found = parts[1].trim().parse::<usize>().ok();
                }
            }

            return Some(CompilerDiagnostic {
                language: "rust".to_string(),
                error_code: code,
                file: file_found,
                line: line_found,
                message: err_line.to_string(),
                raw_snippet: snippet.lines().take(12).collect::<Vec<_>>().join("\n"),
            });
        }

        // Python traceback pattern
        if let Some(tb_idx) = combined.find("Traceback (most recent call last):") {
            let snippet = &combined[tb_idx..];
            let last_line = snippet.lines().filter(|l| !l.trim().is_empty()).last().unwrap_or("Python Error");
            return Some(CompilerDiagnostic {
                language: "python".to_string(),
                error_code: None,
                file: None,
                line: None,
                message: last_line.trim().to_string(),
                raw_snippet: snippet.lines().take(14).collect::<Vec<_>>().join("\n"),
            });
        }

        // Generic error snippet
        let first_err = combined
            .lines()
            .find(|l| l.contains("error") || l.contains("Error") || l.contains("FAIL") || l.contains("failed"))
            .unwrap_or_else(|| combined.lines().next().unwrap_or("Process failed with non-zero exit"));

        Some(CompilerDiagnostic {
            language: "generic".to_string(),
            error_code: None,
            file: None,
            line: None,
            message: first_err.trim().to_string(),
            raw_snippet: combined.lines().take(10).collect::<Vec<_>>().join("\n"),
        })
    }

    pub fn push_coder_message(&mut self, msg: ChatMessage) {
        let segs = crate::ui::chat::parse_segments(&msg.content);
        self.coder_parsed_cache.push(segs);
        self.coder_messages.push(msg);
    }

    /// Load code imported from Swarm Relay (extracting code blocks if present).
    pub fn load_imported_code(&mut self, content: &str) {
        self.code = extract_code_or_raw(content);
        self.file_status = "Imported from Swarm Relay".to_string();
    }

    /// Load a file from absolute or relative path into editor
    pub fn load_file_from_path(&mut self, path: &Path) {
        match std::fs::read_to_string(path) {
            Ok(content) => {
                self.code = content;
                self.file_path = path.to_string_lossy().to_string();
                self.selected_node_path = Some(path.to_path_buf());
                self.language = Self::guess_language_for(path);
                self.file_status = format!("Loaded {} ({} bytes)", path.display(), self.code.len());
            }
            Err(e) => {
                self.file_status = format!("Error loading file: {}", e);
            }
        }
    }

    /// Guess syntax highlighting language based on file extension
    pub fn guess_language_for(path: &Path) -> String {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
        match ext.as_str() {
            "rs" => "rust".to_string(),
            "py" => "python".to_string(),
            "sh" | "bash" => "shell".to_string(),
            "js" => "javascript".to_string(),
            "ts" => "typescript".to_string(),
            "c" | "h" => "c".to_string(),
            "cpp" | "hpp" | "cc" => "cpp".to_string(),
            "go" => "go".to_string(),
            "java" => "java".to_string(),
            "sql" => "sql".to_string(),
            "lua" => "lua".to_string(),
            "asm" | "s" => "asm".to_string(),
            _ => "rust".to_string(),
        }
    }

    /// Execute terminal command asynchronously
    pub fn run_terminal_command(
        &mut self,
        cmd: &str,
        tx: &mpsc::Sender<crate::ui::app::AppMessage>,
        rt: &Runtime,
    ) {
        let trimmed = cmd.trim();
        if trimmed.is_empty() || self.terminal_is_running {
            return;
        }

        self.terminal_history.push(trimmed.to_string());
        self.terminal_input.clear();
        self.terminal_prompt_input.clear();
        self.terminal_history_idx = None;

        // Built-in terminal commands for native VS Code terminal behavior
        if trimmed == "clear" || trimmed == "cls" {
            self.terminal_logs.clear();
            let idx = self.active_terminal_idx.min(self.terminal_sessions.len().saturating_sub(1));
            if let Some(s) = self.terminal_sessions.get_mut(idx) {
                s.logs.clear();
            }
            return;
        }

        if trimmed == "pwd" {
            let res = CommandResult {
                cmd: trimmed.to_string(),
                working_dir: self.workspace.root_path.clone(),
                exit_code: Some(0),
                success: true,
                stdout: self.workspace.root_path.display().to_string(),
                stderr: String::new(),
                duration_ms: 1,
                executed_at: chrono::Utc::now(),
            };
            self.on_terminal_finished(res);
            return;
        }

        if trimmed == "cd" || trimmed.starts_with("cd ") {
            let target_str = if trimmed == "cd" {
                "~"
            } else {
                trimmed[3..].trim()
            };

            let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
            let target_path = if target_str == "~" || target_str.starts_with("~/") {
                if target_str == "~" {
                    home
                } else {
                    home.join(&target_str[2..])
                }
            } else {
                let p = Path::new(target_str);
                if p.is_absolute() {
                    p.to_path_buf()
                } else {
                    self.workspace.root_path.join(p)
                }
            };

            let target_canonical = target_path.canonicalize().unwrap_or(target_path);
            if target_canonical.is_dir() {
                let _ = self.workspace.set_root(target_canonical.clone());
                let res = CommandResult {
                    cmd: trimmed.to_string(),
                    working_dir: target_canonical,
                    exit_code: Some(0),
                    success: true,
                    stdout: String::new(),
                    stderr: String::new(),
                    duration_ms: 1,
                    executed_at: chrono::Utc::now(),
                };
                self.on_terminal_finished(res);
            } else {
                let res = CommandResult {
                    cmd: trimmed.to_string(),
                    working_dir: self.workspace.root_path.clone(),
                    exit_code: Some(1),
                    success: false,
                    stdout: String::new(),
                    stderr: format!("cd: no such file or directory: {}", target_str),
                    duration_ms: 1,
                    executed_at: chrono::Utc::now(),
                };
                self.on_terminal_finished(res);
            }
            return;
        }

        self.terminal_is_running = true;
        self.terminal_running_cmd = trimmed.to_string();
        self.update_active_session_from_fields();

        let working_dir = self.workspace.root_path.clone();
        let cmd_str = trimmed.to_string();
        let tx = tx.clone();

        rt.spawn(async move {
            let res = match CommandRunner::execute(&cmd_str, &working_dir).await {
                Ok(r) => r,
                Err(e) => CommandResult {
                    cmd: cmd_str,
                    working_dir,
                    exit_code: Some(1),
                    success: false,
                    stdout: String::new(),
                    stderr: format!("Execution failed: {}", e),
                    duration_ms: 0,
                    executed_at: chrono::Utc::now(),
                },
            };
            let _ = tx.send(crate::ui::app::AppMessage::TerminalFinished(res));
        });
    }

    pub fn on_terminal_finished(&mut self, result: CommandResult) {
        self.terminal_is_running = false;
        self.terminal_running_cmd.clear();
        self.terminal_logs.push(result.clone());
        if self.terminal_logs.len() > 100 {
            self.terminal_logs.remove(0);
        }
        let idx = self.active_terminal_idx.min(self.terminal_sessions.len().saturating_sub(1));
        if let Some(s) = self.terminal_sessions.get_mut(idx) {
            s.is_running = false;
            s.running_cmd.clear();
            s.logs.push(result);
            if s.logs.len() > 100 {
                s.logs.remove(0);
            }
        }
    }

    pub fn run_current_file(&mut self, tx: &mpsc::Sender<crate::ui::app::AppMessage>, rt: &Runtime) {
        let p = Path::new(&self.file_path);
        if let Some(cmd) = CommandRunner::recommend_command(p) {
            self.run_terminal_command(&cmd, tx, rt);
        } else {
            self.file_status = format!("No default run command for {:?}", p.file_name());
        }
    }

    pub fn run_build(&mut self, tx: &mpsc::Sender<crate::ui::app::AppMessage>, rt: &Runtime) {
        let cmd = if self.workspace.root_path.join("Cargo.toml").exists() {
            "cargo build"
        } else if self.workspace.root_path.join("package.json").exists() {
            "npm run build"
        } else {
            "echo 'No build configuration found'"
        };
        self.run_terminal_command(cmd, tx, rt);
    }

    pub fn run_test(&mut self, tx: &mpsc::Sender<crate::ui::app::AppMessage>, rt: &Runtime) {
        let cmd = if self.workspace.root_path.join("Cargo.toml").exists() {
            "cargo test"
        } else if self.workspace.root_path.join("pytest.ini").exists()
            || self.workspace.root_path.join("requirements.txt").exists()
        {
            "python3 -m pytest tests/ || pytest"
        } else if self.workspace.root_path.join("package.json").exists() {
            "npm test"
        } else {
            "echo 'No test configuration found'"
        };
        self.run_terminal_command(cmd, tx, rt);
    }

    pub fn run_setup(&mut self, tx: &mpsc::Sender<crate::ui::app::AppMessage>, rt: &Runtime) {
        self.run_terminal_command("./setup.sh", tx, rt);
    }

    pub fn run_start(&mut self, tx: &mpsc::Sender<crate::ui::app::AppMessage>, rt: &Runtime) {
        self.run_terminal_command("./start.sh", tx, rt);
    }

    pub fn run_verify(&mut self, tx: &mpsc::Sender<crate::ui::app::AppMessage>, rt: &Runtime) {
        let cmd = if self.workspace.root_path.join("verify.sh").exists() {
            "./verify.sh"
        } else if self.workspace.root_path.join("Cargo.toml").exists() {
            "cargo test --all"
        } else if self.workspace.root_path.join("requirements.txt").exists() {
            "python3 -m pytest || pytest"
        } else if self.workspace.root_path.join("package.json").exists() {
            "npm test"
        } else {
            "echo 'No verification script found — click [🛠 Gen Scripts] to generate one'"
        };
        self.run_terminal_command(cmd, tx, rt);
    }

    pub fn run_clean(&mut self, tx: &mpsc::Sender<crate::ui::app::AppMessage>, rt: &Runtime) {
        let cmd = if self.workspace.root_path.join("clean.sh").exists() {
            "./clean.sh"
        } else if self.workspace.root_path.join("Cargo.toml").exists() {
            "cargo clean"
        } else {
            "echo 'No clean script found'"
        };
        self.run_terminal_command(cmd, tx, rt);
    }

    pub fn generate_automation_scripts(&mut self) -> anyhow::Result<usize> {
        let scripts = crate::workspace::WorkspaceAutomation::generate_scripts(&self.workspace.root_path)?;
        let count = scripts.len();
        let _ = self.workspace.refresh_tree();
        self.file_status = format!("Generated {} turnkey automation scripts in workspace", count);
        Ok(count)
    }

    /// Autonomous Full-Fledged Application Scaffolding
    pub fn scaffold_app(
        &mut self,
        template: AppTemplateType,
        app_name: &str,
    ) -> anyhow::Result<DeployReport> {
        let report = AppScaffolder::scaffold(template, &self.workspace.root_path, app_name)?;
        self.last_deploy_report = Some(report.clone());
        let _ = self.workspace.refresh_tree();

        // Load entry file into editor if present
        let entry_candidates = [
            report.target_dir.join("src/main.rs"),
            report.target_dir.join("app/main.py"),
            report.target_dir.join("src/app.js"),
            report.target_dir.join("cli.py"),
        ];
        for candidate in &entry_candidates {
            if candidate.exists() {
                self.load_file_from_path(candidate);
                break;
            }
        }

        self.file_status = format!(
            "Scaffolded full-fledged app '{}' ({} files, {} dirs) in {}",
            report.app_name,
            report.created_files.len(),
            report.created_dirs.len(),
            report.target_dir.display()
        );

        Ok(report)
    }

    /// Deploy multi-file code blocks into destination folder
    pub fn deploy_multi_file_from_text(&mut self, raw_text: &str) -> anyhow::Result<DeployReport> {
        let report = AppScaffolder::deploy_multi_file_code(raw_text, &self.workspace.root_path)?;
        self.last_deploy_report = Some(report.clone());
        let _ = self.workspace.refresh_tree();

        if let Some(first) = report.created_files.first() {
            self.load_file_from_path(first);
        }

        self.file_status = format!(
            "Deployed {} files into destination: {}",
            report.created_files.len(),
            report.target_dir.display()
        );

        Ok(report)
    }

    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        models: &[crate::ollama::api::Model],
        selected_model: &Option<String>,
        system_prompt: &str,
        api_client: &Option<OllamaClient>,
        tx: &mpsc::Sender<crate::ui::app::AppMessage>,
        rt: &Runtime,
        num_threads: u32,
    ) {
        self.render_modals(ui, tx, rt);

        // Top Control Bar
        ui.horizontal(|ui| {
            ui.add_space(4.0);
            ui.heading(
                egui::RichText::new("AI Code Studio & Workspace IDE")
                    .size(19.0)
                    .color(egui::Color32::from_rgb(0x00, 0xaa, 0xff)),
            );

            ui.add_space(8.0);
            let dest_label = format!(
                "📂 Dest: {}",
                self.workspace.root_path.file_name().map(|n| n.to_string_lossy()).unwrap_or_else(|| self.workspace.root_path.to_string_lossy())
            );
            if ui
                .button(egui::RichText::new(dest_label).size(12.0).color(egui::Color32::from_rgb(0x38, 0xbd, 0xf8)))
                .on_hover_text(format!("Current folder destination: {}\nClick to choose or change root folder.", self.workspace.root_path.display()))
                .clicked()
            {
                self.destination_input = self.workspace.root_path.to_string_lossy().to_string();
                self.show_destination_modal = true;
            }

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.add_space(4.0);
                // AI Coder toggle
                let coder_label = if self.show_coder_chat { "🤖 Hide AI Coder" } else { "🤖 AI Coder" };
                let coder_fill = if self.show_coder_chat { egui::Color32::from_rgb(0x00, 0x55, 0xaa) } else { egui::Color32::from_rgb(0x1e, 0x29, 0x3b) };
                if ui
                    .add(egui::Button::new(egui::RichText::new(coder_label).size(12.0).color(egui::Color32::WHITE)).fill(coder_fill).corner_radius(egui::CornerRadius::same(6)))
                    .clicked()
                {
                    self.show_coder_chat = !self.show_coder_chat;
                }

                ui.add_space(4.0);
                // Terminal toggle
                let term_label = if self.show_terminal { "▶ Hide Terminal" } else { "▶ Terminal" };
                let term_fill = if self.show_terminal { egui::Color32::from_rgb(0x05, 0x60, 0x40) } else { egui::Color32::from_rgb(0x1e, 0x29, 0x3b) };
                if ui
                    .add(egui::Button::new(egui::RichText::new(term_label).size(12.0).color(egui::Color32::WHITE)).fill(term_fill).corner_radius(egui::CornerRadius::same(6)))
                    .clicked()
                {
                    self.show_terminal = !self.show_terminal;
                }

                ui.add_space(4.0);
                // Explorer toggle
                let ws_label = if self.show_workspace_tree { "📁 Hide Files" } else { "📁 File Tree" };
                let ws_fill = if self.show_workspace_tree { egui::Color32::from_rgb(0x55, 0x33, 0x88) } else { egui::Color32::from_rgb(0x1e, 0x29, 0x3b) };
                if ui
                    .add(egui::Button::new(egui::RichText::new(ws_label).size(12.0).color(egui::Color32::WHITE)).fill(ws_fill).corner_radius(egui::CornerRadius::same(6)))
                    .clicked()
                {
                    self.show_workspace_tree = !self.show_workspace_tree;
                }

                ui.add_space(4.0);
                // Full-Fledged App Scaffolder button
                if ui
                    .add(egui::Button::new(egui::RichText::new("🏗 Scaffold Full App").size(12.0).strong().color(egui::Color32::WHITE)).fill(egui::Color32::from_rgb(0xd9, 0x77, 0x06)).corner_radius(egui::CornerRadius::same(6)))
                    .on_hover_text("Generate complete multi-file application (Rust, Python AI Swarm, Web, Cyber Security) in destination folder")
                    .clicked()
                {
                    self.show_scaffold_modal = true;
                }

                ui.add_space(4.0);
                if ui.small_button("Clear Editor").clicked() {
                    self.code.clear();
                }

                ui.add_space(4.0);
                if ui.small_button("Send to Chat tab").clicked() {
                    let _ = tx.send(crate::ui::app::AppMessage::EditorToChat(self.code.clone()));
                }
            });
        });

        ui.add_space(4.0);
        ui.separator();
        ui.add_space(4.0);

        // Proportional 3-Column Layout Engine with Perimeter Hardening
        let total_w = ui.available_width();
        let total_h = ui.available_height();
        let spacing = 6.0;

        let show_ws = self.show_workspace_tree;
        let show_coder = self.show_coder_chat;

        let ws_width = if show_ws {
            240.0_f32.min(total_w * 0.25).max(180.0_f32)
        } else {
            0.0
        };

        let coder_width = if show_coder {
            360.0_f32.min(total_w * 0.35).max(260.0_f32)
        } else {
            0.0
        };

        let mut active_sidebars = 0.0;
        if show_ws {
            active_sidebars += 1.0;
        }
        if show_coder {
            active_sidebars += 1.0;
        }

        let center_width = (total_w - ws_width - coder_width - (active_sidebars * spacing)).max(320.0);

        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = spacing;

            if show_ws {
                ui.allocate_ui_with_layout(
                    egui::vec2(ws_width, total_h),
                    egui::Layout::top_down(egui::Align::Min),
                    |ui| {
                        ui.set_width(ws_width);
                        ui.set_height(total_h);
                        self.show_workspace_pane(ui, tx);
                    },
                );
            }

            ui.allocate_ui_with_layout(
                egui::vec2(center_width, total_h),
                egui::Layout::top_down(egui::Align::Min),
                |ui| {
                    ui.set_width(center_width);
                    ui.set_height(total_h);
                    self.show_editor_and_terminal_pane(
                        ui,
                        models,
                        selected_model,
                        system_prompt,
                        api_client,
                        tx,
                        rt,
                        num_threads,
                    );
                },
            );

            if show_coder {
                ui.allocate_ui_with_layout(
                    egui::vec2(coder_width, total_h),
                    egui::Layout::top_down(egui::Align::Min),
                    |ui| {
                        ui.set_width(coder_width);
                        ui.set_height(total_h);
                        self.show_coder_chat_pane(
                            ui,
                            models,
                            selected_model,
                            system_prompt,
                            api_client,
                            tx,
                            rt,
                            num_threads,
                        );
                    },
                );
            }
        });
    }

    /// Left Pane: Workspace File Tree & File System CRUD
    fn show_workspace_pane(&mut self, ui: &mut egui::Ui, tx: &mpsc::Sender<crate::ui::app::AppMessage>) {
        let frame = egui::Frame::NONE
            .fill(egui::Color32::from_rgb(0x0c, 0x11, 0x1c))
            .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(0x1e, 0x29, 0x3b)))
            .corner_radius(egui::CornerRadius::same(8))
            .inner_margin(egui::Margin::same(8));

        frame.show(ui, |ui| {
            ui.set_min_height(ui.available_height());
            ui.set_min_width(ui.available_width());
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new("📁 Workspace Explorer")
                        .size(14.0)
                        .strong()
                        .color(egui::Color32::from_rgb(0x38, 0xbd, 0xf8)),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.small_button("🔄").on_hover_text("Refresh file tree").clicked() {
                        let _ = self.workspace.refresh_tree();
                    }
                    if ui.small_button("📂").on_hover_text("Change destination folder").clicked() {
                        self.destination_input = self.workspace.root_path.to_string_lossy().to_string();
                        self.show_destination_modal = true;
                    }
                });
            });

            ui.add_space(2.0);
            ui.label(
                egui::RichText::new(format!("Root: {}", self.workspace.root_path.display()))
                    .size(10.0)
                    .color(egui::Color32::from_rgb(0x94, 0xa3, 0xb8)),
            );

            ui.add_space(4.0);
            // Action buttons toolbar: + File, + Folder, Rename, Delete
            ui.horizontal_wrapped(|ui| {
                if ui
                    .add(egui::Button::new(egui::RichText::new("+ File").size(11.0)).fill(egui::Color32::from_rgb(0x1e, 0x3a, 0x8a)))
                    .on_hover_text("Create a new file in workspace")
                    .clicked()
                {
                    self.new_file_input.clear();
                    self.new_file_dialog_open = true;
                }
                if ui
                    .add(egui::Button::new(egui::RichText::new("+ Folder").size(11.0)).fill(egui::Color32::from_rgb(0x13, 0x4e, 0x4a)))
                    .on_hover_text("Create a new folder in workspace")
                    .clicked()
                {
                    self.new_folder_input.clear();
                    self.new_folder_dialog_open = true;
                }
                if ui
                    .add(egui::Button::new(egui::RichText::new("✏ Move").size(11.0)).fill(egui::Color32::from_rgb(0x37, 0x30, 0xa3)))
                    .on_hover_text("Rename or move selected file/folder")
                    .clicked()
                {
                    if let Some(target) = &self.selected_node_path {
                        let rel = self.workspace.relative_path(target);
                        self.rename_input = rel.to_string_lossy().to_string();
                        self.rename_target = Some(target.clone());
                        self.rename_dialog_open = true;
                    } else {
                        self.file_status = "Select a file or folder first to move/rename.".to_string();
                    }
                }
                if ui
                    .add(egui::Button::new(egui::RichText::new("🗑 Delete").size(11.0)).fill(egui::Color32::from_rgb(0x99, 0x1b, 0x1b)))
                    .on_hover_text("Delete selected file or folder (with safety confirmation)")
                    .clicked()
                {
                    if let Some(target) = &self.selected_node_path {
                        self.delete_target = Some(target.clone());
                        self.delete_confirm_open = true;
                    } else {
                        self.file_status = "Select a file or folder first to delete.".to_string();
                    }
                }
            });

            ui.add_space(4.0);
            ui.add(
                egui::TextEdit::singleline(&mut self.file_search_query)
                    .hint_text("🔍 Search files...")
                    .desired_width(f32::INFINITY),
            );

            ui.add_space(4.0);
            ui.separator();
            ui.add_space(4.0);

            // File Tree Scroll Area
            let mut file_to_load = None;
            let mut node_to_select = None;
            let mut path_to_toggle = None;

            egui::ScrollArea::vertical()
                .id_salt("workspace_tree_scroll")
                .max_height(ui.available_height() - 10.0)
                .show(ui, |ui| {
                    if !self.file_search_query.trim().is_empty() {
                        let query = self.file_search_query.trim();
                        let matches = self.workspace.search_files(query);
                        if matches.is_empty() {
                            ui.label(egui::RichText::new("No files match query.").size(11.0).color(egui::Color32::GRAY));
                        } else {
                            for m in matches {
                                let name = m.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
                                let is_selected = self.selected_node_path.as_ref() == Some(&m);
                                if ui.selectable_label(is_selected, format!("📄 {}", name)).clicked() {
                                    node_to_select = Some(m.clone());
                                    file_to_load = Some(m.clone());
                                }
                            }
                        }
                    } else if self.workspace.file_tree.is_empty() {
                        ui.label(egui::RichText::new("Destination folder is empty.").size(11.5).color(egui::Color32::GRAY));
                        ui.add_space(4.0);
                        if ui.button("Create Sample Project").clicked() {
                            let _ = self.scaffold_app(AppTemplateType::RustHighPerformance, "app");
                        }
                    } else {
                        for node in &self.workspace.file_tree {
                            Self::render_tree_node(
                                ui,
                                node,
                                0,
                                &self.selected_node_path,
                                &mut node_to_select,
                                &mut file_to_load,
                                &mut path_to_toggle,
                            );
                        }
                    }
                });

            if let Some(target_rel) = path_to_toggle {
                for node in &mut self.workspace.file_tree {
                    node.toggle_path(&target_rel);
                }
            }
            if let Some(sel) = node_to_select {
                self.selected_node_path = Some(sel);
            }
            if let Some(to_load) = file_to_load {
                self.load_file_from_path(&to_load);
                let _ = tx.send(crate::ui::app::AppMessage::Audit(
                    "file.open".to_string(),
                    to_load.to_string_lossy().to_string(),
                ));
            }
        });
    }

    fn render_tree_node(
        ui: &mut egui::Ui,
        node: &FsNode,
        depth: usize,
        selected_path: &Option<PathBuf>,
        node_to_select: &mut Option<PathBuf>,
        file_to_load: &mut Option<PathBuf>,
        path_to_toggle: &mut Option<PathBuf>,
    ) {
        let is_selected = selected_path.as_ref() == Some(&node.path);
        let indent = depth as f32 * 12.0;

        ui.horizontal(|ui| {
            ui.add_space(indent);
            if node.is_dir {
                let arrow = if node.is_expanded { "▼" } else { "▶" };
                if ui.small_button(arrow).clicked() {
                    *path_to_toggle = Some(node.rel_path.clone());
                }
                let dir_text = format!("📁 {}", node.name);
                let lbl = ui.selectable_label(is_selected, egui::RichText::new(dir_text).color(egui::Color32::from_rgb(0xf5, 0x9e, 0x0b)).size(11.5));
                if lbl.clicked() {
                    *node_to_select = Some(node.path.clone());
                    *path_to_toggle = Some(node.rel_path.clone());
                }
            } else {
                let icon = match node.path.extension().and_then(|e| e.to_str()).unwrap_or("") {
                    "rs" => "🦀",
                    "py" => "🐍",
                    "sh" | "bash" => "📜",
                    "js" | "ts" | "html" | "css" => "🌐",
                    "md" => "📝",
                    _ => "📄",
                };
                let size_str = if node.size > 1024 {
                    format!("{} KB", node.size / 1024)
                } else {
                    format!("{} B", node.size)
                };
                let file_label = format!("{} {}", icon, node.name);
                let lbl = ui.selectable_label(is_selected, egui::RichText::new(file_label).size(11.5));
                if lbl.clicked() {
                    *node_to_select = Some(node.path.clone());
                    *file_to_load = Some(node.path.clone());
                }
                lbl.on_hover_text(format!("Path: {}\nSize: {}", node.path.display(), size_str));
            }
        });

        if node.is_dir && node.is_expanded {
            for child in &node.children {
                Self::render_tree_node(
                    ui,
                    child,
                    depth + 1,
                    selected_path,
                    node_to_select,
                    file_to_load,
                    path_to_toggle,
                );
            }
        }
    }

    /// Center Pane: Editor + Dockable Terminal
    fn show_editor_and_terminal_pane(
        &mut self,
        ui: &mut egui::Ui,
        models: &[crate::ollama::api::Model],
        selected_model: &Option<String>,
        system_prompt: &str,
        api_client: &Option<OllamaClient>,
        tx: &mpsc::Sender<crate::ui::app::AppMessage>,
        rt: &Runtime,
        num_threads: u32,
    ) {
        let frame = egui::Frame::NONE
            .fill(egui::Color32::from_rgb(0x0a, 0x0f, 0x18))
            .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(0x1e, 0x29, 0x3b)))
            .corner_radius(egui::CornerRadius::same(8))
            .inner_margin(egui::Margin::same(8));

        frame.show(ui, |ui| {
            ui.set_min_height(ui.available_height());
            ui.set_min_width(ui.available_width());

            // Upper part: Code Editor
            self.show_editor_pane(ui, tx);

            // Lower part: Terminal / Command Runner Dock (if active)
            if self.show_terminal {
                ui.add_space(6.0);
                ui.separator();
                ui.add_space(4.0);
                self.show_terminal_dock(
                    ui,
                    models,
                    selected_model,
                    system_prompt,
                    api_client,
                    tx,
                    rt,
                    num_threads,
                );
            }
        });
    }

    fn show_editor_pane(&mut self, ui: &mut egui::Ui, tx: &mpsc::Sender<crate::ui::app::AppMessage>) {
        // Primary Center Toolbar: Spread apart across the entire width
        ui.horizontal(|ui| {
            // Left Group: Language selector, file path input, Save, Open
            ui.label(
                egui::RichText::new("Language:")
                    .size(12.0)
                    .color(egui::Color32::from_rgb(0x94, 0xa3, 0xb8)),
            );
            egui::ComboBox::from_id_salt("editor_language")
                .selected_text(egui::RichText::new(&self.language).size(12.0))
                .width(105.0)
                .show_ui(ui, |ui| {
                    for lang in [
                        "rust", "python", "javascript", "typescript", "go", "c", "cpp", "java",
                        "sql", "shell", "lua", "asm",
                    ] {
                        ui.selectable_value(
                            &mut self.language,
                            lang.to_string(),
                            egui::RichText::new(lang).size(12.0),
                        );
                    }
                });

            ui.add_space(6.0);
            ui.label(
                egui::RichText::new("File:")
                    .size(12.0)
                    .color(egui::Color32::from_rgb(0x94, 0xa3, 0xb8)),
            );
            ui.add(
                egui::TextEdit::singleline(&mut self.file_path)
                    .desired_width(220.0)
                    .font(egui::TextStyle::Monospace)
                    .hint_text("path/to/file.rs"),
            );

            ui.add_space(4.0);
            let save_btn = ui.add(
                egui::Button::new(
                    egui::RichText::new("💾 Save")
                        .size(12.0)
                        .strong()
                        .color(egui::Color32::WHITE),
                )
                .fill(egui::Color32::from_rgb(0x05, 0x96, 0x69))
                .corner_radius(egui::CornerRadius::same(5)),
            );
            if save_btn
                .on_hover_text("Write editor content to this file")
                .clicked()
            {
                let ok = self.save_to_file();
                if !self.file_path.trim().is_empty() {
                    let _ = tx.send(crate::ui::app::AppMessage::Audit(
                        if ok {
                            "file.save".to_string()
                        } else {
                            "file.save_failed".to_string()
                        },
                        self.file_path.trim().to_string(),
                    ));
                }
            }

            ui.add_space(2.0);
            let open_btn = ui.add(
                egui::Button::new(
                    egui::RichText::new("📂 Open")
                        .size(12.0)
                        .color(egui::Color32::WHITE),
                )
                .fill(egui::Color32::from_rgb(0x1e, 0x29, 0x3b))
                .corner_radius(egui::CornerRadius::same(5)),
            );
            if open_btn
                .on_hover_text("Load this file into the editor")
                .clicked()
            {
                let ok = self.open_from_file();
                if !self.file_path.trim().is_empty() {
                    let _ = tx.send(crate::ui::app::AppMessage::Audit(
                        if ok {
                            "file.open".to_string()
                        } else {
                            "file.open_failed".to_string()
                        },
                        self.file_path.trim().to_string(),
                    ));
                }
            }

            // Right-aligned group: Swarm Audit, Copy, Live file status
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let audit_btn = ui.add(
                    egui::Button::new(
                        egui::RichText::new("🐝 Swarm Audit & Harden")
                            .size(12.0)
                            .strong()
                            .color(egui::Color32::from_rgb(0x0a, 0x0f, 0x18)),
                    )
                    .fill(egui::Color32::from_rgb(0xfb, 0xbf, 0x24))
                    .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(0xf5, 0x9e, 0x0b)))
                    .corner_radius(egui::CornerRadius::same(5)),
                );
                if audit_btn
                    .on_hover_text("Dispatch active code to Swarm Security Auditor to inspect memory safety, logic flaws, and generate an instant patch")
                    .clicked()
                {
                    let _ = tx.send(crate::ui::app::AppMessage::EditorAuditCode);
                }

                ui.add_space(4.0);
                if ui
                    .small_button("📋 Copy")
                    .on_hover_text("Copy active editor code to clipboard")
                    .clicked()
                {
                    ui.ctx().copy_text(self.code.clone());
                    self.file_status = "Code copied to clipboard.".to_string();
                }

                if !self.file_status.is_empty() {
                    ui.add_space(4.0);
                    ui.label(
                        egui::RichText::new(&self.file_status)
                            .size(11.0)
                            .color(egui::Color32::from_rgb(0x34, 0xd3, 0x99)),
                    );
                }
            });
        });

        // Secondary Toolbar Row: Templates & Live Quantum Code Health HUD
        ui.add_space(3.0);
        ui.horizontal(|ui| {
            ui.label(
                egui::RichText::new("Templates:")
                    .size(11.0)
                    .color(egui::Color32::from_rgb(0x64, 0x74, 0x8b)),
            );
            if ui.small_button("Rust").clicked() {
                self.apply_template("rust");
            }
            if ui.small_button("Python").clicked() {
                self.apply_template("python");
            }
            if ui.small_button("Shell").clicked() {
                self.apply_template("shell");
            }
            if ui.small_button("C++").clicked() {
                self.apply_template("cpp");
            }
            if ui.small_button("Go").clicked() {
                self.apply_template("go");
            }
            if ui.small_button("JS").clicked() {
                self.apply_template("javascript");
            }

            ui.add_space(6.0);
            ui.separator();
            ui.add_space(4.0);

            // Quantum Code Health HUD: Shannon Entropy, Memory Safety, Guardrail
            let entropy = Self::calculate_entropy(&self.code);
            let (entropy_text, entropy_color) = if entropy <= 4.7 {
                (format!("🛡 Entropy: {:.2} b/B (Safe)", entropy), egui::Color32::from_rgb(0x10, 0xb9, 0x81))
            } else {
                (format!("⚠️ Entropy: {:.2} b/B (High Secret Risk)", entropy), egui::Color32::from_rgb(0xf5, 0x9e, 0x0b))
            };
            ui.label(egui::RichText::new(entropy_text).size(10.5).monospace().color(entropy_color));

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let hint = self.run_hint();
                if !hint.is_empty() {
                    ui.label(
                        egui::RichText::new(hint)
                            .size(11.0)
                            .monospace()
                            .color(egui::Color32::from_rgb(0x94, 0xa3, 0xb8)),
                    );
                }
                let lines = self.code.lines().count();
                let bytes = self.code.len();
                ui.label(
                    egui::RichText::new(format!("{} lines | {} B", lines, bytes))
                        .size(10.5)
                        .color(egui::Color32::from_rgb(0x64, 0x74, 0x8b)),
                );
            });
        });

        ui.add_space(4.0);

        // Dynamic Height Editor Sizing: Apportioned with Terminal Height Ratio
        let syntax = Self::syntax_for_language(&self.language);
        let avail_h = ui.available_height();
        let reserved_for_diff = if self.pending_suggestion.is_some() { 130.0 } else { 0.0 };
        let term_ratio = self.terminal_height_ratio.clamp(0.20, 0.80);
        let editor_ratio = (1.0 - term_ratio).clamp(0.20, 0.80);
        let target_editor_h = if self.show_terminal {
            (avail_h * editor_ratio - reserved_for_diff).max(180.0)
        } else {
            (avail_h - 10.0 - reserved_for_diff).max(250.0)
        };
        let editor_rows = ((target_editor_h / 17.5) as usize).max(10);

        let mut editor = CodeEditor::default()
            .id_source("code_editor_main")
            .with_rows(editor_rows)
            .with_fontsize(13.0);

        let _ = editor.show(ui, &mut self.code, &syntax);

        if let Some(suggestion) = self.pending_suggestion.clone() {
            ui.add_space(6.0);
            ui.separator();
            ui.add_space(4.0);
            ui.label(
                egui::RichText::new("AI Suggestion Diff:")
                    .size(12.5)
                    .color(egui::Color32::from_rgb(0x00, 0xcc, 0x88)),
            );
            ui.checkbox(&mut self.show_diff, "Show side-by-side / line diff");
            if self.show_diff {
                let diff = Self::diff_lines(&self.code, &suggestion);
                egui::ScrollArea::vertical()
                    .max_height(100.0)
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        for (sign, line) in diff.iter().take(300) {
                            let col = match sign {
                                '+' => egui::Color32::from_rgb(0x44, 0xdd, 0x77),
                                '-' => egui::Color32::from_rgb(0xff, 0x66, 0x66),
                                _ => egui::Color32::from_rgb(0x88, 0x88, 0x88),
                            };
                            ui.label(
                                egui::RichText::new(format!("{sign} {line}"))
                                    .size(11.0)
                                    .monospace()
                                    .color(col),
                            );
                        }
                    });
            }

            ui.add_space(4.0);
            ui.horizontal(|ui| {
                if ui
                    .add(
                        egui::Button::new(egui::RichText::new("Apply Suggestion").size(12.0))
                            .fill(egui::Color32::from_rgb(0x00, 0x77, 0x55))
                            .corner_radius(egui::CornerRadius::same(6)),
                    )
                    .clicked()
                {
                    self.code = suggestion;
                    self.pending_suggestion = None;
                    self.file_status = "Suggestion applied to editor.".to_string();
                }

                ui.add_space(6.0);
                if ui
                    .add(
                        egui::Button::new(egui::RichText::new("Dismiss").size(12.0))
                            .fill(egui::Color32::from_rgb(0xaa, 0x33, 0x33))
                            .corner_radius(egui::CornerRadius::same(6)),
                    )
                    .clicked()
                {
                    self.pending_suggestion = None;
                }
            });
        }
    }

    /// Terminal Dock: Run All Executable Commands, Compilers, & Scripts
    fn show_terminal_dock(
        &mut self,
        ui: &mut egui::Ui,
        models: &[crate::ollama::api::Model],
        selected_model: &Option<String>,
        system_prompt: &str,
        api_client: &Option<OllamaClient>,
        tx: &mpsc::Sender<crate::ui::app::AppMessage>,
        rt: &Runtime,
        num_threads: u32,
    ) {
        let frame = egui::Frame::NONE
            .fill(egui::Color32::from_rgb(0x08, 0x0c, 0x14))
            .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(0x1e, 0x29, 0x3b)))
            .corner_radius(egui::CornerRadius::same(6))
            .inner_margin(egui::Margin::symmetric(8, 6));

        frame.show(ui, |ui| {
            ui.set_min_width(ui.available_width());
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new("⚡ Command Runner")
                        .size(12.5)
                        .strong()
                        .color(egui::Color32::from_rgb(0x10, 0xb9, 0x81)),
                );

                if self.terminal_is_running {
                    ui.label(
                        egui::RichText::new(format!("⏳ Running '{}'...", self.terminal_running_cmd))
                            .size(11.0)
                            .color(egui::Color32::from_rgb(0xfb, 0xbf, 0x24)),
                    );
                } else {
                    ui.label(
                        egui::RichText::new("● Terminal Console Ready")
                            .size(11.0)
                            .color(egui::Color32::from_rgb(0x34, 0xd3, 0x99)),
                    );
                }

                // Quick preset execution buttons integrated in header row
                if ui.small_button("▶ Run").on_hover_text("Execute active file in editor").clicked() {
                    self.run_current_file(tx, rt);
                }
                if ui.small_button("🔨 Build").on_hover_text("Run project build (cargo build / npm build)").clicked() {
                    self.run_build(tx, rt);
                }
                if ui.small_button("🧪 Test").on_hover_text("Run test suite (cargo test / pytest)").clicked() {
                    self.run_test(tx, rt);
                }
                if ui.small_button("🛡 Audit").on_hover_text("Run ./verify.sh or comprehensive security verification").clicked() {
                    self.run_verify(tx, rt);
                }
                if ui.small_button("⚙ setup").on_hover_text("Execute ./setup.sh in workspace root").clicked() {
                    self.run_setup(tx, rt);
                }
                if ui.small_button("🚀 start").on_hover_text("Execute ./start.sh in workspace root").clicked() {
                    self.run_start(tx, rt);
                }
                if ui.small_button("🧹 clean").on_hover_text("Execute ./clean.sh or project cleanup").clicked() {
                    self.run_clean(tx, rt);
                }
                if ui.small_button("🛠 Gen").on_hover_text("Generate turnkey setup.sh, verify.sh, start.sh, and clean.sh in workspace").clicked() {
                    let _ = self.generate_automation_scripts();
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.small_button("Clear").clicked() {
                        self.terminal_logs.clear();
                    }
                });
            });

            ui.add_space(3.0);
            // Custom command input bar
            let mut execute_now = false;
            ui.horizontal(|ui| {
                let resp = ui.add(
                    egui::TextEdit::singleline(&mut self.terminal_input)
                        .id_salt("ide_terminal_top_cmd_input")
                        .hint_text("Enter command... (e.g. cargo check, git status, python3 main.py)")
                        .font(egui::TextStyle::Monospace)
                        .desired_width(ui.available_width() - 75.0),
                );
                if resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                    execute_now = true;
                }

                let btn_text = if self.terminal_is_running { "..." } else { "Run ↵" };
                let btn = ui.add_enabled(
                    !self.terminal_is_running && !self.terminal_input.trim().is_empty(),
                    egui::Button::new(egui::RichText::new(btn_text).size(11.5).color(egui::Color32::WHITE))
                        .fill(egui::Color32::from_rgb(0x05, 0x96, 0x69))
                        .corner_radius(egui::CornerRadius::same(4)),
                );
                if btn.clicked() {
                    execute_now = true;
                }
            });

            if execute_now {
                let cmd = self.terminal_input.clone();
                self.run_terminal_command(&cmd, tx, rt);
            }

            // Divider line below command runner part
            ui.add_space(4.0);
            ui.separator();
            ui.add_space(2.0);

            // Integrated Terminal Section (VS Code Style)
            let ws_display = if let Some(home) = std::env::var_os("HOME").and_then(|h| h.into_string().ok()) {
                let p_str = self.workspace.root_path.to_string_lossy();
                if let Some(rel) = p_str.strip_prefix(&home) {
                    format!("~{}", rel)
                } else {
                    p_str.to_string()
                }
            } else {
                self.workspace.root_path.to_string_lossy().to_string()
            };

            // Multi-Terminal Sub-header Bar (Session Tabs, Filter, Working Dir, & Controls)
            let mut switch_to_idx = None;
            let mut close_session_idx = None;

            ui.horizontal(|ui| {
                for (idx, sess) in self.terminal_sessions.iter().enumerate() {
                    let is_active = idx == self.active_terminal_idx;
                    let (fill, stroke_col, text_col) = if is_active {
                        (
                            egui::Color32::from_rgb(0x0f, 0x27, 0x44),
                            egui::Color32::from_rgb(0x38, 0xbd, 0xf8),
                            egui::Color32::from_rgb(0x38, 0xbd, 0xf8),
                        )
                    } else {
                        (
                            egui::Color32::from_rgb(0x0b, 0x11, 0x20),
                            egui::Color32::from_rgb(0x1e, 0x29, 0x3b),
                            egui::Color32::from_rgb(0x94, 0xa3, 0xb8),
                        )
                    };

                    egui::Frame::NONE
                        .fill(fill)
                        .stroke(egui::Stroke::new(1.0, stroke_col))
                        .corner_radius(egui::CornerRadius::same(4))
                        .inner_margin(egui::Margin::symmetric(6, 2))
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                let tab_label = format!(" {}", sess.name);
                                if ui.add(
                                    egui::Label::new(
                                        egui::RichText::new(tab_label)
                                            .size(11.0)
                                            .monospace()
                                            .color(text_col),
                                    ).sense(egui::Sense::click())
                                ).clicked() {
                                    switch_to_idx = Some(idx);
                                }
                                if self.terminal_sessions.len() > 1 {
                                    if ui.add(
                                        egui::Label::new(
                                            egui::RichText::new("✕")
                                                .size(9.5)
                                                .color(egui::Color32::from_rgb(0x94, 0xa3, 0xb8)),
                                        ).sense(egui::Sense::click())
                                    ).on_hover_text("Close terminal tab").clicked() {
                                        close_session_idx = Some(idx);
                                    }
                                }
                            });
                        });
                    ui.add_space(2.0);
                }

                if let Some(idx) = close_session_idx {
                    self.close_terminal_session(idx);
                } else if let Some(idx) = switch_to_idx {
                    self.update_active_session_from_fields();
                    self.active_terminal_idx = idx;
                    self.sync_active_session_fields();
                }

                // [+] Add new terminal session tab
                if ui.small_button("＋").on_hover_text("Spawn new integrated terminal session").clicked() {
                    self.update_active_session_from_fields();
                    self.new_terminal_session();
                }

                ui.add_space(4.0);
                ui.separator();
                ui.add_space(4.0);

                // Inline log filter
                ui.add(
                    egui::TextEdit::singleline(&mut self.terminal_filter)
                        .id_salt("ide_terminal_filter_input")
                        .hint_text("🔍 Filter logs...")
                        .font(egui::TextStyle::Monospace)
                        .desired_width(110.0),
                );
                if !self.terminal_filter.is_empty() {
                    if ui.small_button("✕").on_hover_text("Clear filter").clicked() {
                        self.terminal_filter.clear();
                    }
                }

                ui.add_space(4.0);
                ui.label(
                    egui::RichText::new(format!("📁 {}", ws_display))
                        .size(11.0)
                        .monospace()
                        .color(egui::Color32::from_rgb(0x94, 0xa3, 0xb8)),
                );

                if self.terminal_is_running {
                    ui.label(
                        egui::RichText::new(format!("● RUNNING: {}", self.terminal_running_cmd))
                            .size(10.5)
                            .strong()
                            .color(egui::Color32::from_rgb(0xfb, 0xbf, 0x24)),
                    );
                } else {
                    ui.label(
                        egui::RichText::new("● ONLINE")
                            .size(10.5)
                            .strong()
                            .color(egui::Color32::from_rgb(0x10, 0xb9, 0x81)),
                    );
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.small_button("📋 Copy Terminal").on_hover_text("Copy terminal buffer to clipboard").clicked() {
                        let mut full_term = String::new();
                        for log in &self.terminal_logs {
                            full_term.push_str(&format!("$ {}\n", log.cmd));
                            if !log.stdout.is_empty() { full_term.push_str(&log.stdout); full_term.push('\n'); }
                            if !log.stderr.is_empty() { full_term.push_str(&log.stderr); full_term.push('\n'); }
                        }
                        ui.ctx().copy_text(full_term);
                    }

                    ui.add_space(6.0);
                    // Height presets & expand toggle
                    if self.terminal_height_ratio < 0.62 {
                        if ui
                            .add(egui::Button::new(egui::RichText::new("⤢ Expand Height").size(11.0).color(egui::Color32::WHITE)).fill(egui::Color32::from_rgb(0x1e, 0x3a, 0x8a)))
                            .on_hover_text("Expand terminal dock to commanding 70% height")
                            .clicked()
                        {
                            self.terminal_height_ratio = 0.70;
                        }
                    } else {
                        if ui
                            .add(egui::Button::new(egui::RichText::new("⤡ Balance Height").size(11.0).color(egui::Color32::WHITE)).fill(egui::Color32::from_rgb(0x1e, 0x29, 0x3b)))
                            .on_hover_text("Return terminal dock to balanced 55% height")
                            .clicked()
                        {
                            self.terminal_height_ratio = 0.55;
                        }
                    }

                    if ui.selectable_label((self.terminal_height_ratio - 0.70).abs() < 0.05, "70%").on_hover_text("70% Terminal / 30% Editor (Maximized)").clicked() {
                        self.terminal_height_ratio = 0.70;
                    }
                    if ui.selectable_label((self.terminal_height_ratio - 0.55).abs() < 0.05, "55%").on_hover_text("55% Terminal / 45% Editor (Expanded)").clicked() {
                        self.terminal_height_ratio = 0.55;
                    }
                    if ui.selectable_label((self.terminal_height_ratio - 0.35).abs() < 0.05, "35%").on_hover_text("35% Terminal / 65% Editor (Compact)").clicked() {
                        self.terminal_height_ratio = 0.35;
                    }
                    ui.label(egui::RichText::new("Dock Height:").size(10.5).color(egui::Color32::from_rgb(0x94, 0xa3, 0xb8)));
                });
            });

            ui.add_space(2.0);

            // Integrated Terminal Screen Box spanning cleanly to dock borders
            let user_name = std::env::var("USER").unwrap_or_else(|_| "user".to_string());
            let prompt_prefix = format!("{}@krovyx:{}$", user_name, ws_display);

            let term_frame = egui::Frame::NONE
                .fill(egui::Color32::from_rgb(0x05, 0x09, 0x14))
                .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(0x1e, 0x29, 0x3b)))
                .corner_radius(egui::CornerRadius::same(5))
                .inner_margin(egui::Margin::symmetric(8, 6));

            let mut auto_fix_trigger: Option<(String, Option<i32>, CompilerDiagnostic)> = None;
            let filter_term = self.terminal_filter.trim().to_lowercase();

            term_frame.show(ui, |ui| {
                ui.set_min_width(ui.available_width());
                let term_screen_h = (ui.available_height() - 40.0).max(120.0);

                egui::ScrollArea::vertical()
                    .id_salt("ide_terminal_screen_scroll")
                    .auto_shrink([false, false])
                    .min_scrolled_height(term_screen_h)
                    .max_height(term_screen_h)
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        if self.terminal_logs.is_empty() && !self.terminal_is_running {
                            ui.label(
                                egui::RichText::new(format!(
                                    "⚡ Krovyx Terminal v1.0 [Ready]\nWorking directory: {}\nEnter commands at prompt below or use quick actions above.",
                                    ws_display
                                ))
                                .size(11.0)
                                .monospace()
                                .color(egui::Color32::from_rgb(0x64, 0x74, 0x8b)),
                            );
                        } else {
                            for log in &self.terminal_logs {
                                if !filter_term.is_empty() {
                                    let c_match = log.cmd.to_lowercase().contains(&filter_term);
                                    let o_match = log.stdout.to_lowercase().contains(&filter_term);
                                    let e_match = log.stderr.to_lowercase().contains(&filter_term);
                                    if !c_match && !o_match && !e_match {
                                        continue;
                                    }
                                }

                                // Prompt line
                                ui.horizontal_wrapped(|ui| {
                                    ui.label(
                                        egui::RichText::new(&prompt_prefix)
                                            .size(11.0)
                                            .monospace()
                                            .strong()
                                            .color(egui::Color32::from_rgb(0x10, 0xb9, 0x81)),
                                    );
                                    ui.label(
                                        egui::RichText::new(&log.cmd)
                                            .size(11.0)
                                            .monospace()
                                            .strong()
                                            .color(egui::Color32::from_rgb(0x38, 0xbd, 0xf8)),
                                    );
                                });

                                // Stdout
                                if !log.stdout.is_empty() {
                                    ui.label(
                                        egui::RichText::new(&log.stdout)
                                            .size(10.5)
                                            .monospace()
                                            .color(egui::Color32::from_rgb(0xec, 0xfd, 0xf5)),
                                    );
                                }

                                // Stderr
                                if !log.stderr.is_empty() {
                                    ui.label(
                                        egui::RichText::new(&log.stderr)
                                            .size(10.5)
                                            .monospace()
                                            .color(egui::Color32::from_rgb(0xf8, 0x71, 0x71)),
                                    );
                                }

                                // Status line & Auto-Fix Trigger
                                let is_err = !log.success || log.exit_code != Some(0);
                                let (status_str, status_col) = match log.exit_code {
                                    Some(0) => (format!("➜ Process exited with code 0 ({}ms)", log.duration_ms), egui::Color32::from_rgb(0x05, 0x96, 0x69)),
                                    Some(c) => (format!("✖ Process exited with code {} ({}ms)", c, log.duration_ms), egui::Color32::from_rgb(0xef, 0x44, 0x44)),
                                    None => (format!("⚠ Process terminated ({}ms)", log.duration_ms), egui::Color32::from_rgb(0xf5, 0x9e, 0x0b)),
                                };

                                ui.horizontal(|ui| {
                                    ui.label(
                                        egui::RichText::new(status_str)
                                            .size(10.0)
                                            .monospace()
                                            .color(status_col),
                                    );

                                    if is_err {
                                        if let Some(diag) = Self::extract_compiler_diagnostic(log) {
                                            let btn_label = if let Some(code) = &diag.error_code {
                                                format!("⚡ Auto-Fix [{}]", code)
                                            } else {
                                                "⚡ Swarm Auto-Fix & Patch".to_string()
                                            };
                                            let btn = ui.add(
                                                egui::Button::new(
                                                    egui::RichText::new(btn_label)
                                                        .size(10.0)
                                                        .strong()
                                                        .color(egui::Color32::WHITE),
                                                )
                                                .fill(egui::Color32::from_rgb(0x99, 0x1b, 0x1b))
                                                .corner_radius(egui::CornerRadius::same(3)),
                                            ).on_hover_text(format!("Send diagnostic '{}' directly to AI Coder to generate an autonomous fix patch", diag.message));

                                            if btn.clicked() {
                                                auto_fix_trigger = Some((log.cmd.clone(), log.exit_code, diag));
                                            }
                                        }
                                    }
                                });
                                ui.add_space(4.0);
                            }
                        }

                        if self.terminal_is_running {
                            ui.horizontal(|ui| {
                                ui.label(
                                    egui::RichText::new(&prompt_prefix)
                                        .size(11.0)
                                        .monospace()
                                        .color(egui::Color32::from_rgb(0x10, 0xb9, 0x81)),
                                );
                                ui.label(
                                    egui::RichText::new(&self.terminal_running_cmd)
                                        .size(11.0)
                                        .monospace()
                                        .color(egui::Color32::from_rgb(0x38, 0xbd, 0xf8)),
                                );
                                ui.spinner();
                            });
                        }
                    });

                // Trigger autonomous auto-fix if requested
                if let Some((cmd, code, diag)) = auto_fix_trigger {
                    let loc_str = match (&diag.file, &diag.line) {
                        (Some(f), Some(l)) => format!("{}:{}", f, l),
                        (Some(f), None) => f.clone(),
                        _ => "unknown".to_string(),
                    };
                    let fix_prompt = format!(
                        "Autonomous Compiler Fix Request:\n\
                        The command `{}` failed with exit code {:?}.\n\
                        Diagnostic: {}\n\
                        Location: {}\n\n\
                        Compiler / Runtime Output Snippet:\n```\n{}\n```\n\n\
                        Please analyze this failure, explain the root cause, and provide the complete fixed replacement code in a clean code block.",
                        cmd,
                        code.unwrap_or(1),
                        diag.message,
                        loc_str,
                        diag.raw_snippet
                    );
                    self.coder_input = fix_prompt.clone();
                    self.show_coder_chat = true;
                    self.send_coder_message(
                        &fix_prompt,
                        models,
                        selected_model,
                        system_prompt,
                        api_client,
                        tx,
                        rt,
                        num_threads,
                    );
                }

                // Interactive Bottom Terminal Prompt
                ui.separator();
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new(&prompt_prefix)
                            .size(11.0)
                            .monospace()
                            .strong()
                            .color(egui::Color32::from_rgb(0x10, 0xb9, 0x81)),
                    );

                    let mut exec_term_prompt = false;
                    let resp = ui.add(
                        egui::TextEdit::singleline(&mut self.terminal_prompt_input)
                            .id_salt("ide_terminal_interactive_prompt")
                            .hint_text("Enter shell command... (Enter to run, ↑/↓ history)")
                            .font(egui::TextStyle::Monospace)
                            .desired_width(ui.available_width() - 50.0),
                    );

                    // Up / Down arrow history recall
                    if resp.has_focus() {
                        if ui.input(|i| i.key_pressed(egui::Key::ArrowUp)) && !self.terminal_history.is_empty() {
                            let next_idx = match self.terminal_history_idx {
                                None => self.terminal_history.len().saturating_sub(1),
                                Some(0) => 0,
                                Some(i) => i.saturating_sub(1),
                            };
                            self.terminal_history_idx = Some(next_idx);
                            if let Some(cmd) = self.terminal_history.get(next_idx) {
                                self.terminal_prompt_input = cmd.clone();
                            }
                        } else if ui.input(|i| i.key_pressed(egui::Key::ArrowDown)) {
                            if let Some(i) = self.terminal_history_idx {
                                if i + 1 < self.terminal_history.len() {
                                    self.terminal_history_idx = Some(i + 1);
                                    if let Some(cmd) = self.terminal_history.get(i + 1) {
                                        self.terminal_prompt_input = cmd.clone();
                                    }
                                } else {
                                    self.terminal_history_idx = None;
                                    self.terminal_prompt_input.clear();
                                }
                            }
                        }
                    }

                    if (resp.lost_focus() || resp.has_focus()) && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                        exec_term_prompt = true;
                    }

                    if ui.add_enabled(!self.terminal_is_running && !self.terminal_prompt_input.trim().is_empty(),
                        egui::Button::new(egui::RichText::new("↵").size(11.0).color(egui::Color32::WHITE))
                            .fill(egui::Color32::from_rgb(0x05, 0x96, 0x69))
                            .corner_radius(egui::CornerRadius::same(3))
                    ).clicked() {
                        exec_term_prompt = true;
                    }

                    if exec_term_prompt {
                        let cmd = self.terminal_prompt_input.clone();
                        self.terminal_prompt_input.clear();
                        self.terminal_history_idx = None;
                        self.run_terminal_command(&cmd, tx, rt);
                        resp.request_focus();
                    }
                });
            });
        });
    }

    /// Right Pane: AI Coder & Full-Fledged Application Scaffolding
    fn show_coder_chat_pane(
        &mut self,
        ui: &mut egui::Ui,
        models: &[crate::ollama::api::Model],
        selected_model: &Option<String>,
        system_prompt: &str,
        api_client: &Option<OllamaClient>,
        tx: &mpsc::Sender<crate::ui::app::AppMessage>,
        rt: &Runtime,
        num_threads: u32,
    ) {
        let frame = egui::Frame::NONE
            .fill(egui::Color32::from_rgb(0x0f, 0x14, 0x22))
            .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(0x1e, 0x29, 0x3b)))
            .corner_radius(egui::CornerRadius::same(8))
            .inner_margin(egui::Margin::same(10));

        frame.show(ui, |ui| {
            ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Wrap);
            ui.set_min_height(ui.available_height());
            ui.set_min_width(ui.available_width());
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new("🤖 AI Coder & Architect")
                        .size(15.0)
                        .strong()
                        .color(egui::Color32::from_rgb(0x38, 0xbd, 0xf8)),
                );

                let cur_model = self
                    .coder_model
                    .clone()
                    .or_else(|| selected_model.clone())
                    .unwrap_or_else(|| "(no model)".to_string());

                egui::ComboBox::from_id_salt("ide_coder_model_selector")
                    .selected_text(egui::RichText::new(&cur_model).size(12.0))
                    .width(150.0)
                    .show_ui(ui, |ui| {
                        for m in models {
                            let is_sel = self.coder_model.as_deref() == Some(&m.name)
                                || (self.coder_model.is_none() && selected_model.as_deref() == Some(&m.name));
                            if ui.selectable_label(is_sel, &m.name).clicked() {
                                self.coder_model = Some(m.name.clone());
                            }
                        }
                    });

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.small_button("Clear").on_hover_text("Clear AI Coder conversation").clicked() {
                        self.coder_messages.clear();
                        self.coder_parsed_cache.clear();
                        self.coder_stream_buf.clear();
                    }
                });
            });

            ui.add_space(4.0);

            // Quick Prompt Suggestions
            ui.horizontal_wrapped(|ui| {
                ui.label(egui::RichText::new("Prompt:").size(11.0).color(egui::Color32::from_rgb(0x94, 0xa3, 0xb8)));
                if ui.small_button("✨ Create").clicked() {
                    self.coder_input = "Create a complete, robust implementation of: ".to_string();
                }
                if ui.small_button("⚡ Optimize").clicked() {
                    self.send_coder_message("Optimize the current code for peak throughput, memory bounds, and zero allocations.", models, selected_model, system_prompt, api_client, tx, rt, num_threads);
                }
                if ui.small_button("🛡 Audit").clicked() {
                    self.send_coder_message("Perform a rigorous cybersecurity and memory safety audit on this code. Highlight all vulnerabilities.", models, selected_model, system_prompt, api_client, tx, rt, num_threads);
                }
                if ui.small_button(egui::RichText::new("🐝 Swarm Assist").color(egui::Color32::from_rgb(0xfb, 0xbf, 0x24))).on_hover_text("Trigger Swarm Security Auditor to inspect and patch active code").clicked() {
                    let _ = tx.send(crate::ui::app::AppMessage::EditorAuditCode);
                }
                if ui.small_button("🏗 Scaffold App").clicked() {
                    self.show_scaffold_modal = true;
                }
            });

            ui.add_space(4.0);
            ui.separator();
            ui.add_space(4.0);

            // Messages Viewport
            let mut apply_code_req = None;
            let mut deploy_multi_req = None;

            // Reserve room for bottom docked chatbox (field 2 rows ≈ 36px + actions ≈ 26px + spacing/separators ≈ 14px = ~76px)
            let input_reserve = 88.0 * ui.ctx().zoom_factor();
            let scroll_h = (ui.available_height() - input_reserve).max(60.0);
            egui::ScrollArea::vertical()
                .id_salt("ide_coder_scroll_area")
                .max_height(scroll_h)
                .auto_shrink([false, false])
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    if self.coder_messages.is_empty() && !self.coder_is_streaming {
                        ui.vertical_centered(|ui| {
                            ui.add_space(20.0);
                            ui.label(
                                egui::RichText::new("AI Coder ready.")
                                    .size(14.0)
                                    .color(egui::Color32::from_rgb(0x64, 0x74, 0x8b)),
                            );
                            ui.label(
                                egui::RichText::new("Ask for full application architectures, modules, bug fixes, or optimizations.")
                                    .size(11.5)
                                    .color(egui::Color32::from_rgb(0x47, 0x55, 0x69)),
                            );
                        });
                    }

                    for (m_idx, msg) in self.coder_messages.iter().enumerate() {
                        let segs = self.coder_parsed_cache.get(m_idx).map(|s| &s[..]).unwrap_or(&[]);
                        Self::render_coder_message_with_deploy(
                            ui,
                            msg,
                            segs,
                            m_idx,
                            false,
                            &mut apply_code_req,
                            &mut deploy_multi_req,
                            tx,
                        );
                    }

                    if self.coder_is_streaming {
                        if !self.coder_stream_buf.is_empty() {
                            let stream_content = format!("{}▍", self.coder_stream_buf);
                            let stream_segs = crate::ui::chat::parse_segments(&stream_content);
                            let tmp_msg = ChatMessage {
                                role: "assistant".to_string(),
                                content: stream_content,
                                timestamp: chrono::Utc::now(),
                            };
                            Self::render_coder_message_with_deploy(
                                ui,
                                &tmp_msg,
                                &stream_segs,
                                self.coder_messages.len(),
                                true,
                                &mut apply_code_req,
                                &mut deploy_multi_req,
                                tx,
                            );
                        } else {
                            ui.add_space(6.0);
                            ui.horizontal(|ui| {
                                ui.add_space(4.0);
                                ui.spinner();
                                ui.label(
                                    egui::RichText::new("Thinking & architecting...")
                                        .size(12.0)
                                        .color(egui::Color32::from_rgb(0x38, 0xbd, 0xf8)),
                                );
                            });
                        }
                    }
                });

            if let Some(code) = apply_code_req {
                self.code = code;
                self.file_status = "Applied code from AI Coder to editor.".to_string();
            }
            if let Some(text) = deploy_multi_req {
                let _ = self.deploy_multi_file_from_text(&text);
            }

            ui.add_space(2.0);
            ui.separator();
            ui.add_space(2.0);

            // Docked Chatbox Input Area: Aligned and placed right down to the border
            let input_resp = ui.add(
                egui::TextEdit::multiline(&mut self.coder_input)
                    .id_salt("ide_coder_input_box")
                    .desired_rows(2)
                    .desired_width(f32::INFINITY)
                    .hint_text("Ask AI Coder to write, scaffold, or refactor... (Enter to send, Shift+Enter for newline)")
                    .font(egui::TextStyle::Body),
            );

            ui.add_space(3.0);
            ui.horizontal(|ui| {
                ui.checkbox(&mut self.include_editor_context, "Include editor code");

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let can_send = !self.coder_input.trim().is_empty() && !self.coder_is_streaming;
                    let send_btn = ui.add_enabled(
                        can_send,
                        egui::Button::new(
                            egui::RichText::new(if self.coder_is_streaming { "Generating..." } else { "Send ↵" })
                                .size(12.0)
                                .color(egui::Color32::WHITE),
                        )
                        .fill(if can_send { egui::Color32::from_rgb(0x00, 0x66, 0xcc) } else { egui::Color32::from_rgb(0x1a, 0x3a, 0x66) })
                        .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(0x38, 0xbd, 0xf8)))
                        .corner_radius(egui::CornerRadius::same(5))
                        .min_size(egui::vec2(85.0, 26.0)),
                    );

                    if send_btn.clicked() {
                        let prompt = self.coder_input.clone();
                        self.send_coder_message(&prompt, models, selected_model, system_prompt, api_client, tx, rt, num_threads);
                    }

                    if self.coder_is_streaming {
                        ui.add_space(4.0);
                        let stop_btn = ui.add(
                            egui::Button::new(egui::RichText::new("Stop").size(12.0).color(egui::Color32::WHITE))
                                .fill(egui::Color32::from_rgb(0xaa, 0x33, 0x33))
                                .corner_radius(egui::CornerRadius::same(5)),
                        );
                        if stop_btn.clicked() {
                            self.stop_coder_stream();
                        }
                    }
                });
            });

            let send_triggered = ui.input(|i| i.key_pressed(egui::Key::Enter) && !i.modifiers.shift)
                && input_resp.has_focus()
                && !self.coder_input.trim().is_empty()
                && !self.coder_is_streaming;
            if send_triggered {
                let prompt = self.coder_input.clone();
                self.send_coder_message(&prompt, models, selected_model, system_prompt, api_client, tx, rt, num_threads);
            }
        });
    }

    fn render_coder_message_with_deploy(
        ui: &mut egui::Ui,
        msg: &ChatMessage,
        segments: &[crate::ui::chat::MessageSegment],
        msg_idx: usize,
        is_streaming: bool,
        apply_req: &mut Option<String>,
        deploy_multi_req: &mut Option<String>,
        tx: &mpsc::Sender<crate::ui::app::AppMessage>,
    ) {
        let is_user = msg.role == "user";
        let is_sys = msg.role == "system";
        let (bg, who, tag_color) = if is_user {
            (egui::Color32::from_rgb(0x13, 0x2f, 0x4c), "You", egui::Color32::from_rgb(0x38, 0xbd, 0xf8))
        } else if is_sys {
            (egui::Color32::from_rgb(0x33, 0x22, 0x11), "System", egui::Color32::from_rgb(0xf5, 0x9e, 0x0b))
        } else if is_streaming {
            (egui::Color32::from_rgb(0x16, 0x1e, 0x2e), "AI Coder (generating...)", egui::Color32::from_rgb(0x4a, 0xde, 0x80))
        } else {
            (egui::Color32::from_rgb(0x16, 0x1e, 0x2e), "AI Coder", egui::Color32::from_rgb(0x4a, 0xde, 0x80))
        };

        ui.add_space(3.0);
        egui::Frame::NONE
            .fill(bg)
            .corner_radius(egui::CornerRadius::same(6))
            .inner_margin(egui::Margin::same(8))
            .show(ui, |ui| {
                ui.set_max_width(ui.available_width());
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new(who).size(11.0).strong().color(tag_color));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(
                            egui::RichText::new(msg.timestamp.format("%H:%M").to_string())
                                .size(10.0)
                                .color(egui::Color32::from_rgb(0x94, 0xa3, 0xb8)),
                        );
                    });
                });

                ui.add_space(3.0);
                let fallback_segs;
                let actual_segs = if segments.is_empty() && !msg.content.is_empty() {
                    fallback_segs = crate::ui::chat::parse_segments(&msg.content);
                    &fallback_segs[..]
                } else {
                    segments
                };

                for (seg_idx, seg) in actual_segs.iter().enumerate() {
                    match seg {
                        crate::ui::chat::MessageSegment::Text(t) => {
                            if !t.is_empty() {
                                ui.add(
                                    egui::Label::new(
                                        egui::RichText::new(t)
                                            .size(12.0)
                                            .color(egui::Color32::WHITE),
                                    )
                                    .wrap(),
                                );
                            }
                        }
                        crate::ui::chat::MessageSegment::Think(th) => {
                            let word_count = th.split_whitespace().count();
                            let is_active_stream_think = is_streaming && seg_idx == actual_segs.len() - 1;
                            if is_active_stream_think {
                                // Actively streaming thoughts in real-time
                                egui::Frame::NONE
                                    .fill(egui::Color32::from_rgb(0x0a, 0x10, 0x1f))
                                    .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(0x25, 0x63, 0xeb)))
                                    .corner_radius(egui::CornerRadius::same(6))
                                    .inner_margin(egui::Margin::symmetric(8, 6))
                                    .show(ui, |ui| {
                                        ui.horizontal(|ui| {
                                            ui.spinner();
                                            ui.label(
                                                egui::RichText::new("💭 Reasoning Stream (Thinking...)")
                                                    .size(11.0)
                                                    .strong()
                                                    .color(egui::Color32::from_rgb(0x38, 0xbd, 0xf8)),
                                            );
                                            ui.label(
                                                egui::RichText::new(format!("({} words)", word_count))
                                                    .size(10.5)
                                                    .color(egui::Color32::from_rgb(0x64, 0x74, 0x8b)),
                                            );
                                        });
                                        ui.add_space(2.0);
                                        ui.add(
                                            egui::Label::new(
                                                egui::RichText::new(th)
                                                    .size(11.0)
                                                    .color(egui::Color32::from_rgb(0x94, 0xa3, 0xb8))
                                                    .italics(),
                                            )
                                            .wrap(),
                                        );
                                    });
                            } else {
                                egui::CollapsingHeader::new(
                                    egui::RichText::new(format!("💭 Thought Process ({} words)", word_count))
                                        .size(10.5)
                                        .color(egui::Color32::from_rgb(0x94, 0xa3, 0xb8)),
                                )
                                .id_salt(format!("coder_think_{}_{}", msg_idx, seg_idx))
                                .default_open(false)
                                .show(ui, |ui| {
                                    ui.add(
                                        egui::Label::new(
                                            egui::RichText::new(th)
                                                .size(10.5)
                                                .color(egui::Color32::from_rgb(0x94, 0xa3, 0xb8))
                                                .italics(),
                                        )
                                        .wrap(),
                                    );
                                });
                            }
                        }
                        crate::ui::chat::MessageSegment::Code { lang, code } => {
                            ui.add_space(3.0);
                            let cb_frame = egui::Frame::NONE
                                .fill(egui::Color32::from_rgb(0x0a, 0x0f, 0x1d))
                                .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(0x1e, 0x2d, 0x48)))
                                .corner_radius(egui::CornerRadius::same(6))
                                .inner_margin(egui::Margin::same(6));

                            cb_frame.show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    let display_lang = if lang.is_empty() { "code" } else { lang };
                                    ui.label(
                                        egui::RichText::new(display_lang)
                                            .size(10.0)
                                            .strong()
                                            .color(egui::Color32::from_rgb(0x38, 0xbd, 0xf8)),
                                    );
                                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                        if ui.small_button("📋 Copy").clicked() {
                                            ui.ctx().copy_text(code.clone());
                                        }
                                        if ui
                                            .small_button("⚡ Replace Editor")
                                            .on_hover_text("Load this code block directly into the editor")
                                            .clicked()
                                        {
                                            *apply_req = Some(code.clone());
                                        }
                                    });
                                });

                                ui.add_space(2.0);
                                egui::ScrollArea::horizontal()
                                    .id_salt(format!("code_blk_{}_{}_{}", msg_idx, seg_idx, code.len()))
                                    .show(ui, |ui| {
                                        ui.label(
                                            egui::RichText::new(code)
                                                .size(11.0)
                                                .monospace()
                                                .color(egui::Color32::from_rgb(0xe2, 0xe8, 0xf0)),
                                        );
                                    });
                            });
                        }
                    }
                }

                if !is_streaming {
                    ui.add_space(2.0);
                    ui.horizontal(|ui| {
                        if ui.small_button("📋 Copy").on_hover_text("Copy message text").clicked() {
                            ui.ctx().copy_text(msg.content.clone());
                        }
                        let think_blocks: Vec<&str> = actual_segs
                            .iter()
                            .filter_map(|s| match s {
                                crate::ui::chat::MessageSegment::Think(th) => Some(th.as_str()),
                                _ => None,
                            })
                            .collect();
                        if !think_blocks.is_empty() {
                            if ui.small_button("💭 Copy thoughts").on_hover_text("Copy reasoning trace").clicked() {
                                ui.ctx().copy_text(think_blocks.join("\n\n"));
                            }
                        }
                    });
                }

                // Check for multi-file application pattern
                if !is_user && (msg.content.contains("### File:") || msg.content.contains("```file:")) {
                    let files = AppScaffolder::extract_multi_files(&msg.content);
                    if !files.is_empty() {
                        ui.add_space(4.0);
                        egui::Frame::NONE
                            .fill(egui::Color32::from_rgb(0x1e, 0x3a, 0x8a))
                            .corner_radius(egui::CornerRadius::same(6))
                            .inner_margin(egui::Margin::same(6))
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.label(
                                        egui::RichText::new(format!("📦 Multi-File Application Detected ({} files)", files.len()))
                                            .size(11.5)
                                            .strong()
                                            .color(egui::Color32::WHITE),
                                    );
                                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                        if ui
                                            .add(egui::Button::new(egui::RichText::new("🚀 Deploy Full App to Destination").size(11.0).strong().color(egui::Color32::WHITE)).fill(egui::Color32::from_rgb(0x05, 0x96, 0x69)))
                                            .on_hover_text("Automatically creates all folders, writes all files, and adds setup.sh and start.sh")
                                            .clicked()
                                        {
                                            *deploy_multi_req = Some(msg.content.clone());
                                            let _ = tx.send(crate::ui::app::AppMessage::Notice(format!(
                                                "Deploying {} files into selected folder destination...",
                                                files.len()
                                            )));
                                        }
                                    });
                                });
                            });
                    }
                }
            });
    }

    /// Modals for Destination Picker, New File, New Folder, Rename, Delete, and Full App Scaffolding
    fn render_modals(
        &mut self,
        ui: &mut egui::Ui,
        tx: &mpsc::Sender<crate::ui::app::AppMessage>,
        _rt: &Runtime,
    ) {
        // 1. Destination Folder Modal
        if self.show_destination_modal {
            egui::Window::new("📂 Select Workspace Folder Destination")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ui.ctx(), |ui| {
                    ui.label("Configure the target root destination for all files, folders, and applications:");
                    ui.add_space(4.0);
                    ui.add(
                        egui::TextEdit::singleline(&mut self.destination_input)
                            .desired_width(360.0)
                            .font(egui::TextStyle::Monospace),
                    );

                    ui.add_space(6.0);
                    ui.label("Quick Presets:");
                    ui.horizontal_wrapped(|ui| {
                        if ui.small_button("Documents/Code_air/ml_lab").clicked() {
                            let p = WorkspaceManager::default_destination();
                            self.destination_input = p.to_string_lossy().to_string();
                        }
                        if ui.small_button("Current ML-Laboratory").clicked() {
                            if let Ok(p) = std::env::current_dir() {
                                self.destination_input = p.to_string_lossy().to_string();
                            }
                        }
                        if ui.small_button("Home Directory").clicked() {
                            if let Some(h) = dirs::home_dir() {
                                self.destination_input = h.to_string_lossy().to_string();
                            }
                        }
                    });

                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        if ui.button("Select & Apply Destination").clicked() {
                            let p = PathBuf::from(self.destination_input.trim());
                            let _ = self.workspace.set_root(p);
                            self.show_destination_modal = false;
                        }
                        if ui.button("Cancel").clicked() {
                            self.show_destination_modal = false;
                        }
                    });
                });
        }

        // 2. New File Modal
        if self.new_file_dialog_open {
            egui::Window::new("📄 Create New File")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ui.ctx(), |ui| {
                    ui.label("Enter relative file path within workspace:");
                    ui.add(
                        egui::TextEdit::singleline(&mut self.new_file_input)
                            .hint_text("e.g. src/neural/quantum.rs or tests/test_main.py")
                            .desired_width(320.0),
                    );
                    ui.add_space(6.0);
                    ui.horizontal(|ui| {
                        if ui.button("Create File").clicked() {
                            let rel = Path::new(self.new_file_input.trim());
                            if !rel.as_os_str().is_empty() {
                                match self.workspace.create_file(rel, "") {
                                    Ok(p) => {
                                        self.load_file_from_path(&p);
                                        let _ = tx.send(crate::ui::app::AppMessage::Audit(
                                            "file.create".to_string(),
                                            p.to_string_lossy().to_string(),
                                        ));
                                    }
                                    Err(e) => {
                                        self.file_status = format!("File create failed: {}", e);
                                    }
                                }
                            }
                            self.new_file_dialog_open = false;
                        }
                        if ui.button("Cancel").clicked() {
                            self.new_file_dialog_open = false;
                        }
                    });
                });
        }

        // 3. New Folder Modal
        if self.new_folder_dialog_open {
            egui::Window::new("📁 Create New Folder")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ui.ctx(), |ui| {
                    ui.label("Enter relative folder path within workspace:");
                    ui.add(
                        egui::TextEdit::singleline(&mut self.new_folder_input)
                            .hint_text("e.g. src/components or docs/api")
                            .desired_width(320.0),
                    );
                    ui.add_space(6.0);
                    ui.horizontal(|ui| {
                        if ui.button("Create Folder").clicked() {
                            let rel = Path::new(self.new_folder_input.trim());
                            if !rel.as_os_str().is_empty() {
                                match self.workspace.create_dir(rel) {
                                    Ok(p) => {
                                        self.file_status = format!("Created folder: {}", p.display());
                                        let _ = tx.send(crate::ui::app::AppMessage::Audit(
                                            "folder.create".to_string(),
                                            p.to_string_lossy().to_string(),
                                        ));
                                    }
                                    Err(e) => {
                                        self.file_status = format!("Folder create failed: {}", e);
                                    }
                                }
                            }
                            self.new_folder_dialog_open = false;
                        }
                        if ui.button("Cancel").clicked() {
                            self.new_folder_dialog_open = false;
                        }
                    });
                });
        }

        // 4. Rename / Move Modal
        if self.rename_dialog_open {
            egui::Window::new("✏ Rename / Move Target")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ui.ctx(), |ui| {
                    if let Some(target) = &self.rename_target {
                        ui.label(format!("Source: {}", target.display()));
                        ui.add_space(4.0);
                        ui.label("Enter new relative destination path:");
                        ui.add(
                            egui::TextEdit::singleline(&mut self.rename_input)
                                .desired_width(320.0),
                        );
                        ui.add_space(6.0);
                        ui.horizontal(|ui| {
                            if ui.button("Confirm Move").clicked() {
                                let src_rel = self.workspace.relative_path(target);
                                let dst_rel = Path::new(self.rename_input.trim());
                                match self.workspace.move_entry(&src_rel, dst_rel) {
                                    Ok(d) => {
                                        self.file_status = format!("Moved to {}", d.display());
                                        let _ = tx.send(crate::ui::app::AppMessage::Audit(
                                            "file.move".to_string(),
                                            format!("{} -> {}", target.display(), d.display()),
                                        ));
                                    }
                                    Err(e) => {
                                        self.file_status = format!("Move failed: {}", e);
                                    }
                                }
                                self.rename_dialog_open = false;
                            }
                            if ui.button("Cancel").clicked() {
                                self.rename_dialog_open = false;
                            }
                        });
                    }
                });
        }

        // 5. Delete Confirmation Modal (Safety First)
        if self.delete_confirm_open {
            egui::Window::new("⚠ Confirm Delete")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ui.ctx(), |ui| {
                    if let Some(target) = &self.delete_target {
                        ui.label(
                            egui::RichText::new("Are you sure you want to permanently delete this target?")
                                .color(egui::Color32::from_rgb(0xff, 0x66, 0x66))
                                .strong(),
                        );
                        ui.label(format!("Target: {}", target.display()));
                        ui.add_space(6.0);
                        ui.horizontal(|ui| {
                            if ui
                                .add(egui::Button::new(egui::RichText::new("Delete Forever").color(egui::Color32::WHITE)).fill(egui::Color32::from_rgb(0xaa, 0x22, 0x22)))
                                .clicked()
                            {
                                let rel = self.workspace.relative_path(target);
                                match self.workspace.delete_entry(&rel) {
                                    Ok(()) => {
                                        self.file_status = format!("Deleted {}", target.display());
                                        let _ = tx.send(crate::ui::app::AppMessage::Audit(
                                            "file.delete".to_string(),
                                            target.to_string_lossy().to_string(),
                                        ));
                                    }
                                    Err(e) => {
                                        self.file_status = format!("Delete failed: {}", e);
                                    }
                                }
                                self.delete_confirm_open = false;
                            }
                            if ui.button("Cancel").clicked() {
                                self.delete_confirm_open = false;
                            }
                        });
                    }
                });
        }

        // 6. Autonomous Full-Fledged Application Scaffolder Modal
        if self.show_scaffold_modal {
            egui::Window::new("🏗 Autonomous Full-Fledged Application Scaffolder")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ui.ctx(), |ui| {
                    ui.label("Generate a complete, production-grade application in the destination folder:");
                    ui.add_space(4.0);

                    ui.label("Application Architecture / Template:");
                    for tmpl in AppTemplateType::all() {
                        let is_sel = self.scaffold_selected_template == tmpl;
                        if ui.selectable_label(is_sel, tmpl.label()).clicked() {
                            self.scaffold_selected_template = tmpl;
                            if self.scaffold_app_name == "my_application" || self.scaffold_app_name.is_empty() {
                                self.scaffold_app_name = tmpl.default_folder_name().to_string();
                            }
                        }
                    }

                    ui.add_space(6.0);
                    ui.label("Application Name / Folder:");
                    ui.add(
                        egui::TextEdit::singleline(&mut self.scaffold_app_name)
                            .desired_width(280.0),
                    );

                    let target_preview = self.workspace.root_path.join(&self.scaffold_app_name);
                    ui.add_space(2.0);
                    ui.label(
                        egui::RichText::new(format!("Target: {}", target_preview.display()))
                            .size(10.5)
                            .color(egui::Color32::from_rgb(0x38, 0xbd, 0xf8)),
                    );

                    ui.add_space(4.0);
                    ui.label(
                        egui::RichText::new("✓ Includes src/, tests/, README.md, and executable setup.sh & start.sh")
                            .size(10.5)
                            .color(egui::Color32::from_rgb(0x4a, 0xde, 0x80)),
                    );

                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        if ui
                            .add(egui::Button::new(egui::RichText::new("⚡ Generate & Deploy Full App").size(12.5).strong().color(egui::Color32::WHITE)).fill(egui::Color32::from_rgb(0xd9, 0x77, 0x06)))
                            .clicked()
                        {
                            let tmpl = self.scaffold_selected_template;
                            let name = self.scaffold_app_name.clone();
                            match self.scaffold_app(tmpl, &name) {
                                Ok(rep) => {
                                    let _ = tx.send(crate::ui::app::AppMessage::Audit(
                                        "app.scaffold".to_string(),
                                        rep.target_dir.display().to_string(),
                                    ));
                                }
                                Err(e) => {
                                    self.file_status = format!("Scaffold failed: {}", e);
                                }
                            }
                            self.show_scaffold_modal = false;
                        }
                        if ui.button("Cancel").clicked() {
                            self.show_scaffold_modal = false;
                        }
                    });
                });
        }
    }

    fn save_to_file(&mut self) -> bool {
        let path = PathBuf::from(self.file_path.trim());
        if path.as_os_str().is_empty() {
            self.file_status = "Pick a file path first.".to_string();
            return false;
        }
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                if let Err(e) = std::fs::create_dir_all(parent) {
                    self.file_status = format!("Can't create folder: {}", e);
                    return false;
                }
            }
        }
        match std::fs::write(&path, &self.code) {
            Ok(()) => {
                self.file_status = format!("Saved {} bytes to {}", self.code.len(), path.display());
                let _ = self.workspace.refresh_tree();
                true
            }
            Err(e) => {
                self.file_status = format!("Save failed: {}", e);
                false
            }
        }
    }

    fn open_from_file(&mut self) -> bool {
        let path = self.file_path.trim().to_string();
        if path.is_empty() {
            self.file_status = "Pick a file path first.".to_string();
            return false;
        }
        match std::fs::read_to_string(&path) {
            Ok(text) => {
                self.code = text;
                self.file_status = format!("Opened {} ({} bytes)", path, self.code.len());
                self.language = Self::guess_language_for(Path::new(&path));
                true
            }
            Err(e) => {
                self.file_status = format!("Open failed: {}", e);
                false
            }
        }
    }

    fn apply_template(&mut self, kind: &str) {
        match kind {
            "rust" => {
                self.language = "rust".to_string();
                self.code = "// Copyright 2026 Sean M. Stow. All rights reserved.\n// New Rust app\n// Run: cargo run or rustc main.rs -o app && ./app\n\nfn main() {\n    println!(\"Hello from ML Lab\");\n}\n".to_string();
            }
            "python" => {
                self.language = "python".to_string();
                self.code = "# Copyright 2026 Sean M. Stow. All rights reserved.\n# New Python app\n# Run: python3 main.py\n\ndef main():\n    print(\"Hello from ML Lab\")\n\nif __name__ == \"__main__\":\n    main()\n".to_string();
            }
            "shell" => {
                self.language = "shell".to_string();
                self.code = "#!/bin/sh\n# Copyright 2026 Sean M. Stow. All rights reserved.\n# New shell script\n# Run: sh script.sh\n\necho \"Hello from ML Lab\"\n".to_string();
            }
            _ => {}
        }
        self.file_status = format!("{} template loaded — Save to keep it.", kind);
    }

    fn run_hint(&self) -> &str {
        match self.language.as_str() {
            "rust" => "Run: cargo run or rustc <file> -o app && ./app",
            "python" => "Run: python3 <file>",
            "shell" => "Run: bash <file>",
            "javascript" => "Run: node <file>",
            _ => "Run with your toolchain",
        }
    }

    pub fn push_chunk(&mut self, suggestion_id: usize, piece: &str) {
        if suggestion_id + 1 != self.suggestion_id || piece.is_empty() {
            return;
        }
        let clean = crate::ui::chat::sanitize_text_cow(piece);
        if clean.is_empty() {
            return;
        }
        if self.suggestion_buf.len() < 50_000 {
            self.suggestion_buf.push_str(&clean);
        }
        if self.coder_stream_buf.len() < 50_000 {
            self.coder_is_streaming = true;
            self.coder_stream_buf.push_str(&clean);
        }
    }

    pub fn handle_ai_suggestion(&mut self, suggestion_id: usize, response: Result<crate::ollama::api::ChatResponse, anyhow::Error>) {
        if suggestion_id + 1 != self.suggestion_id {
            return;
        }
        self.suggestion_buf.clear();
        self.coder_is_streaming = false;
        self.coder_stream_buf.clear();

        match response {
            Ok(resp) => {
                let clean = crate::ui::chat::sanitize_text(&resp.message.content);
                self.pending_suggestion = Some(clean.clone());
                self.push_coder_message(ChatMessage {
                    role: "assistant".to_string(),
                    content: clean,
                    timestamp: chrono::Utc::now(),
                });
            }
            Err(e) => {
                let err_msg = crate::ui::chat::sanitize_text(&format!("Error: {}", e));
                self.pending_suggestion = Some(err_msg.clone());
                self.push_coder_message(ChatMessage {
                    role: "system".to_string(),
                    content: err_msg,
                    timestamp: chrono::Utc::now(),
                });
            }
        }
    }

    pub fn send_coder_message(
        &mut self,
        prompt: &str,
        _models: &[crate::ollama::api::Model],
        selected_model: &Option<String>,
        system_prompt: &str,
        api_client: &Option<OllamaClient>,
        tx: &mpsc::Sender<crate::ui::app::AppMessage>,
        rt: &Runtime,
        num_threads: u32,
    ) {
        let active_model = self.coder_model.clone().or_else(|| selected_model.clone());
        let Some(model_name) = active_model else {
            self.push_coder_message(ChatMessage {
                role: "system".to_string(),
                content: "Select an AI model first using the dropdown above.".to_string(),
                timestamp: chrono::Utc::now(),
            });
            return;
        };
        let Some(client) = api_client else {
            self.push_coder_message(ChatMessage {
                role: "system".to_string(),
                content: "Ollama is not connected on loopback 127.0.0.1:11434.".to_string(),
                timestamp: chrono::Utc::now(),
            });
            return;
        };

        let p = prompt.trim();
        if p.is_empty() || self.coder_is_streaming {
            return;
        }

        if let Err(hits) = self.gate.check(p) {
            let kinds: Vec<String> = hits
                .iter()
                .map(|h| format!("{} ({})", h.kind, h.preview))
                .collect();
            self.push_coder_message(ChatMessage {
                role: "system".to_string(),
                content: format!(
                    "Blocked: possible secret ({}). Send again within 60s to override.",
                    kinds.join(", ")
                ),
                timestamp: chrono::Utc::now(),
            });
            let _ = tx.send(crate::ui::app::AppMessage::Audit(
                "secret.blocked".to_string(),
                "coder chat".to_string(),
            ));
            return;
        }

        self.push_coder_message(ChatMessage {
            role: "user".to_string(),
            content: p.to_string(),
            timestamp: chrono::Utc::now(),
        });
        self.coder_input.clear();

        let mut full_prompt = String::new();
        if self.include_editor_context && !self.code.trim().is_empty() {
            full_prompt.push_str(&format!(
                "Workspace Destination: {}\nCurrent File ({}, {}):\n```{}\n{}\n```\n\nTask / Request:\n{}",
                self.workspace.root_path.display(), self.file_path, self.language, self.language, self.code, p
            ));
        } else {
            full_prompt.push_str(&format!("Workspace Destination: {}\n\nTask:\n{}", self.workspace.root_path.display(), p));
        }

        let suggestion_id = self.suggestion_id;
        self.suggestion_id += 1;
        self.coder_is_streaming = true;
        self.coder_stream_buf.clear();

        let client = client.clone();
        let tx = tx.clone();
        let sys_prompt = if !system_prompt.trim().is_empty() {
            system_prompt.to_string()
        } else {
            "You are an expert software engineer and code architect. When generating full applications, specify files using '### File: relative/path/to/file' followed by code fences. Always include setup.sh and start.sh scripts.".to_string()
        };

        let history: Vec<(String, String)> = self
            .coder_messages
            .iter()
            .rev()
            .take(10)
            .rev()
            .filter(|m| m.role == "user" || m.role == "assistant")
            .map(|m| (m.role.clone(), m.content.clone()))
            .collect();

        rt.spawn(async move {
            let mut messages = Vec::new();
            messages.push(Message {
                role: "system".to_string(),
                content: sys_prompt,
            });
            for (role, content) in history {
                messages.push(Message { role, content });
            }
            if let Some(last) = messages.last_mut() {
                if last.role == "user" {
                    last.content = full_prompt;
                }
            }

            let options = if num_threads > 0 {
                ChatOptions::lowram_with_threads(Some(num_threads))
            } else {
                let mut opts = ChatOptions::lowram();
                opts.num_thread = Some(ChatOptions::optimal_threads());
                opts
            };

            let req = ChatRequest {
                model: model_name,
                messages,
                stream: true,
                options: Some(options),
                keep_alive: Some("30m".to_string()),
            };

            let result = client
                .chat_stream(req, |piece| {
                    let _ = tx.send(crate::ui::app::AppMessage::EditorChunk(
                        suggestion_id,
                        piece.to_string(),
                    ));
                })
                .await;
            let _ = tx.send(crate::ui::app::AppMessage::EditorSuggestion(
                suggestion_id,
                result,
            ));
        });
    }

    pub fn stop_coder_stream(&mut self) {
        self.suggestion_id += 1;
        self.coder_is_streaming = false;
        self.coder_stream_buf.clear();
        self.suggestion_buf.clear();
    }

    pub fn set_code_from_chat(&mut self, code: String, lang: String) {
        let l = lang.to_lowercase();
        let mapped = match l.as_str() {
            "py" => "python",
            "js" => "javascript",
            "ts" => "typescript",
            "sh" | "bash" => "shell",
            "c++" => "cpp",
            _ => l.as_str(),
        };
        if ["rust", "python", "javascript", "typescript", "go", "c", "cpp", "java", "sql", "shell", "lua", "asm"]
            .contains(&mapped)
        {
            self.language = mapped.to_string();
        }
        self.code = code;
        self.file_status = "Code from chat — pick a file path and Save.".to_string();
    }

    fn syntax_for_language(lang: &str) -> Syntax {
        match lang {
            "rust" => Syntax::rust(),
            "shell" => Syntax::shell(),
            _ => Syntax::rust(),
        }
    }

    pub fn diff_lines<'a>(old: &'a str, new: &'a str) -> Vec<(char, &'a str)> {
        let diff = similar::TextDiff::from_lines(old, new);
        let mut out = Vec::new();
        for change in diff.iter_all_changes() {
            let sign = match change.tag() {
                similar::ChangeTag::Delete => '-',
                similar::ChangeTag::Insert => '+',
                similar::ChangeTag::Equal => ' ',
            };
            out.push((sign, change.value()));
        }
        out
    }
}

pub fn extract_code_or_raw(content: &str) -> String {
    if let Some(start) = content.find("```") {
        let after_fence = &content[start + 3..];
        let code_body = match after_fence.find('\n') {
            Some(nl) => &after_fence[nl + 1..],
            None => after_fence,
        };
        if let Some(end) = code_body.find("```") {
            return code_body[..end].trim().to_string();
        }
    }
    content.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suggestion_chunk_currency() {
        let mut e = EditorPanel::new();
        e.push_chunk(0, "x");
        assert!(e.suggestion_buf.is_empty());
        e.suggestion_id = 1;
        e.push_chunk(0, "hello ");
        e.push_chunk(0, "world");
        assert_eq!(e.suggestion_buf, "hello world");
        assert_eq!(e.coder_stream_buf, "hello world");
        e.push_chunk(5, "stale");
        e.push_chunk(7, "stale");
        assert_eq!(e.suggestion_buf, "hello world");
    }

    #[test]
    fn diff_marks_changes() {
        let d = EditorPanel::diff_lines("a\nb\nc\n", "a\nB\nc\nd\n");
        let signs: String = d.iter().map(|(s, _)| *s).collect();
        assert!(signs.contains(' '), "context lines kept");
        assert!(signs.contains('-'), "removal marked");
        assert!(signs.contains('+'), "addition marked");
        assert_eq!(EditorPanel::diff_lines("", "").len(), 0);
        let same = EditorPanel::diff_lines("x\n", "x\n");
        assert!(same.iter().all(|(s, _)| *s == ' '));
    }

    #[test]
    fn coder_chat_message_flow() {
        let mut e = EditorPanel::new();
        assert!(e.coder_messages.is_empty());
        assert!(e.coder_parsed_cache.is_empty());
        e.push_coder_message(ChatMessage {
            role: "user".to_string(),
            content: "Write hello".to_string(),
            timestamp: chrono::Utc::now(),
        });
        assert_eq!(e.coder_parsed_cache.len(), 1);
        e.suggestion_id = 1;
        e.push_chunk(0, "```rust\nfn hello() {}\n```");
        assert_eq!(e.coder_stream_buf, "```rust\nfn hello() {}\n```");
        e.handle_ai_suggestion(0, Ok(crate::ollama::api::ChatResponse {
            model: "qwen2.5-coder:7b".to_string(),
            created_at: "now".to_string(),
            message: crate::ollama::api::Message {
                role: "assistant".to_string(),
                content: "```rust\nfn hello() {}\n```".to_string(),
            },
            done: true,
            total_duration: None,
            load_duration: None,
            prompt_eval_count: None,
            prompt_eval_duration: None,
            eval_count: None,
            eval_duration: None,
        }));
        assert_eq!(e.coder_messages.len(), 2);
        assert_eq!(e.coder_messages[1].role, "assistant");
        assert!(e.coder_messages[1].content.contains("fn hello"));
    }

    #[test]
    fn test_extract_code_or_raw() {
        let markdown = "Here is the implementation:\n```rust\nfn calculate(x: i32) -> i32 {\n    x * 2\n}\n```\nHope that helps!";
        let extracted = extract_code_or_raw(markdown);
        assert_eq!(extracted, "fn calculate(x: i32) -> i32 {\n    x * 2\n}");

        let raw_code = "fn raw() { let y = 10; }";
        assert_eq!(extract_code_or_raw(raw_code), raw_code);
    }

    #[test]
    fn test_load_imported_code() {
        let mut e = EditorPanel::new();
        let relay_output = "## Consensual Synthesis\n```python\ndef solve_quantum_circuit():\n    return 42\n```";
        e.load_imported_code(relay_output);
        assert_eq!(e.code, "def solve_quantum_circuit():\n    return 42");
        assert_eq!(e.file_status, "Imported from Swarm Relay");
    }

    #[test]
    fn test_editor_workspace_and_scaffold() {
        let temp_dir = std::env::temp_dir().join(format!("test-editor-ws-{}", std::process::id()));
        let mut e = EditorPanel::new();
        e.workspace.set_root(temp_dir.clone()).unwrap();

        // Scaffold Python AI Swarm app
        let report = e.scaffold_app(AppTemplateType::PythonAiSwarm, "test_swarm").unwrap();
        assert!(report.target_dir.join("app/main.py").exists());
        assert!(report.target_dir.join("setup.sh").exists());
        assert!(report.target_dir.join("start.sh").exists());

        // Verify loaded file
        assert!(e.file_path.contains("main.py"));
        assert_eq!(e.language, "python");
        assert!(e.code.contains("SwarmAgent"));

        let _ = std::fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_editor_proportional_layout_and_dynamic_rows() {
        // Test wide display layout allocation
        let total_w = 1600.0_f32;
        let spacing = 6.0_f32;
        let ws_w = 240.0_f32.min(total_w * 0.25).max(180.0_f32);
        let coder_w = 360.0_f32.min(total_w * 0.35).max(260.0_f32);
        let center_w = (total_w - ws_w - coder_w - (2.0 * spacing)).max(320.0);

        assert_eq!(ws_w, 240.0);
        assert_eq!(coder_w, 360.0);
        assert_eq!(center_w, 988.0);
        assert!(center_w > ws_w * 4.0, "Center editor must command majority of width");

        // Test narrow display minimum clamping
        let narrow_w = 700.0_f32;
        let ws_narrow = 240.0_f32.min(narrow_w * 0.25).max(180.0_f32);
        let coder_narrow = 360.0_f32.min(narrow_w * 0.35).max(260.0_f32);
        let center_narrow = (narrow_w - ws_narrow - coder_narrow - (2.0 * spacing)).max(320.0);

        assert_eq!(ws_narrow, 180.0);
        assert_eq!(coder_narrow, 260.0);
        assert_eq!(center_narrow, 320.0);

        // Test dynamic row calculation filling viewport across terminal height ratios
        let avail_h = 900.0_f32;
        let target_h_no_term = (avail_h - 10.0).max(250.0);
        let rows_no_term = ((target_h_no_term / 17.5) as usize).max(10);
        assert!(rows_no_term >= 50, "Without terminal, editor should fill 50+ rows on 900px height");

        // Expanded default terminal ratio (55% terminal, 45% editor)
        let ratio_default = 0.55_f32;
        let target_h_default = (avail_h * (1.0 - ratio_default)).max(180.0);
        let rows_default = ((target_h_default / 17.5) as usize).max(10);
        assert!(rows_default >= 23, "With 55% terminal, editor should comfortably display 23+ rows");

        // Maximized terminal ratio (70% terminal, 30% editor)
        let ratio_max = 0.70_f32;
        let target_h_max = (avail_h * (1.0 - ratio_max)).max(180.0);
        let rows_max = ((target_h_max / 17.5) as usize).max(10);
        assert!(rows_max >= 15, "With 70% terminal, editor should allocate 15+ rows");

        // Compact terminal ratio (35% terminal, 65% editor)
        let ratio_compact = 0.35_f32;
        let target_h_compact = (avail_h * (1.0 - ratio_compact)).max(180.0);
        let rows_compact = ((target_h_compact / 17.5) as usize).max(10);
        assert!(rows_compact >= 33, "With 35% terminal, editor should allocate 33+ rows");
    }

    #[test]
    fn test_terminal_builtins_and_integrated_session() {
        let (tx, _rx) = mpsc::channel();
        let rt = Runtime::new().unwrap();
        let mut e = EditorPanel::new();

        // 1. Test pwd command
        e.run_terminal_command("pwd", &tx, &rt);
        assert_eq!(e.terminal_logs.len(), 1);
        assert_eq!(e.terminal_logs[0].cmd, "pwd");
        assert!(e.terminal_logs[0].success);
        assert_eq!(e.terminal_logs[0].stdout, e.workspace.root_path.display().to_string());

        // 2. Test clear command
        e.run_terminal_command("clear", &tx, &rt);
        assert!(e.terminal_logs.is_empty(), "clear must empty the terminal buffer");

        // 3. Test cd command
        let initial_root = e.workspace.root_path.clone();
        let temp_dir = std::env::temp_dir();
        e.run_terminal_command(&format!("cd {}", temp_dir.display()), &tx, &rt);
        assert_eq!(e.terminal_logs.len(), 1);
        assert!(e.terminal_logs[0].success);
        let canonical_temp = temp_dir.canonicalize().unwrap_or(temp_dir);
        assert_eq!(e.workspace.root_path, canonical_temp);

        // Restore initial
        let _ = e.workspace.set_root(initial_root);
    }

    #[test]
    fn test_terminal_multi_sessions() {
        let mut e = EditorPanel::new();
        assert_eq!(e.terminal_sessions.len(), 1);
        assert_eq!(e.active_terminal_idx, 0);

        // Spawn a new session tab
        e.new_terminal_session();
        assert_eq!(e.terminal_sessions.len(), 2);
        assert_eq!(e.active_terminal_idx, 1);
        assert_eq!(e.terminal_sessions[1].name, "bash 2");

        // Write log in active session 1
        e.terminal_logs.push(CommandResult {
            cmd: "echo 'in session 2'".to_string(),
            working_dir: PathBuf::from("/tmp"),
            exit_code: Some(0),
            success: true,
            stdout: "in session 2\n".to_string(),
            stderr: String::new(),
            duration_ms: 5,
            executed_at: chrono::Utc::now(),
        });
        e.update_active_session_from_fields();

        // Switch to session 0
        e.active_terminal_idx = 0;
        e.sync_active_session_fields();
        assert!(e.terminal_logs.is_empty(), "Session 0 should have no logs yet");

        // Switch back to session 1
        e.active_terminal_idx = 1;
        e.sync_active_session_fields();
        assert_eq!(e.terminal_logs.len(), 1);
        assert_eq!(e.terminal_logs[0].cmd, "echo 'in session 2'");

        // Close session 0
        e.close_terminal_session(0);
        assert_eq!(e.terminal_sessions.len(), 1);
        assert_eq!(e.active_terminal_idx, 0);
        assert_eq!(e.terminal_sessions[0].name, "bash 2");

        // Closing the only remaining session should be a no-op
        e.close_terminal_session(0);
        assert_eq!(e.terminal_sessions.len(), 1);
    }

    #[test]
    fn test_shannon_entropy_calculation() {
        // Empty string
        assert_eq!(EditorPanel::calculate_entropy(""), 0.0);

        // Single repetitive character -> 0 entropy
        let low = EditorPanel::calculate_entropy("aaaaaaaaaaaaaaaaaaaa");
        assert_eq!(low, 0.0);

        // Moderate natural code
        let code = "fn main() { println!(\"Hello, world!\"); }";
        let code_entropy = EditorPanel::calculate_entropy(code);
        assert!(code_entropy > 3.0 && code_entropy < 4.8);

        // High entropy (e.g. base64 / encrypted secret / token)
        let high_secret = "VjhzKzFhMmJjZDNlZjQ1Njc4OTAqJiZeJSQjQCExMjM0NTY3ODkw";
        let secret_entropy = EditorPanel::calculate_entropy(high_secret);
        assert!(secret_entropy > 4.5, "High random tokens have high Shannon entropy");
    }

    #[test]
    fn test_compiler_diagnostic_extraction() {
        // 1. Successful result returns None
        let ok_res = CommandResult {
            cmd: "cargo check".to_string(),
            working_dir: PathBuf::from("/test"),
            exit_code: Some(0),
            success: true,
            stdout: "Finished dev profile".to_string(),
            stderr: String::new(),
            duration_ms: 100,
            executed_at: chrono::Utc::now(),
        };
        assert!(EditorPanel::extract_compiler_diagnostic(&ok_res).is_none());

        // 2. Rust compiler diagnostic
        let rust_err = CommandResult {
            cmd: "cargo check".to_string(),
            working_dir: PathBuf::from("/test"),
            exit_code: Some(101),
            success: false,
            stdout: String::new(),
            stderr: "error[E0308]: mismatched types\n  --> src/main.rs:42:15\n   |\n42 |     let x: u32 = \"str\";\n".to_string(),
            duration_ms: 250,
            executed_at: chrono::Utc::now(),
        };
        let diag = EditorPanel::extract_compiler_diagnostic(&rust_err).expect("Must extract Rust error");
        assert_eq!(diag.language, "rust");
        assert_eq!(diag.error_code, Some("error[E0308]".to_string()));
        assert_eq!(diag.file, Some("src/main.rs".to_string()));
        assert_eq!(diag.line, Some(42));

        // 3. Python traceback
        let py_err = CommandResult {
            cmd: "python3 test.py".to_string(),
            working_dir: PathBuf::from("/test"),
            exit_code: Some(1),
            success: false,
            stdout: String::new(),
            stderr: "Traceback (most recent call last):\n  File \"test.py\", line 12, in <module>\nZeroDivisionError: division by zero\n".to_string(),
            duration_ms: 80,
            executed_at: chrono::Utc::now(),
        };
        let py_diag = EditorPanel::extract_compiler_diagnostic(&py_err).expect("Must extract Python error");
        assert_eq!(py_diag.language, "python");
        assert!(py_diag.message.contains("ZeroDivisionError"));

        // 4. Generic error snippet
        let gen_err = CommandResult {
            cmd: "./verify.sh".to_string(),
            working_dir: PathBuf::from("/test"),
            exit_code: Some(2),
            success: false,
            stdout: "Execution failed due to missing dependency".to_string(),
            stderr: String::new(),
            duration_ms: 30,
            executed_at: chrono::Utc::now(),
        };
        let gen_diag = EditorPanel::extract_compiler_diagnostic(&gen_err).expect("Must extract generic error");
        assert_eq!(gen_diag.language, "generic");
        assert!(gen_diag.message.contains("Execution failed"));
    }
}
