use eframe::egui;
use egui_code_editor::{CodeEditor, Syntax};
use crate::ollama::api::{ChatOptions, ChatRequest, Message, OllamaClient};
use std::sync::mpsc;
use tokio::runtime::Runtime;

pub struct EditorPanel {
    code: String,
    language: String,
    suggestion_id: usize,
    pending_suggestion: Option<String>,
    suggestion_buf: String,
    show_diff: bool,
    gate: crate::security::ConfirmGate,
    file_path: String,
    file_status: String,
}

impl EditorPanel {
    pub fn new() -> Self {
        Self {
            code: "// Start coding with AI assistance\nfn main() {\n    println!(\"Hello, world!\");\n}".to_string(),
            language: "rust".to_string(),
            suggestion_id: 0,
            pending_suggestion: None,
            suggestion_buf: String::new(),
            show_diff: true,
            gate: crate::security::ConfirmGate::new(),
            file_path: Self::default_path(),
            file_status: String::new(),
        }
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
    ) {
        ui.horizontal(|ui| {
            ui.add_space(4.0);
            ui.heading(egui::RichText::new("AI Code Editor").size(22.0).color(egui::Color32::from_rgb(0x00, 0xaa, 0xff)));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.add_space(8.0);
                let clear_btn = ui.add(
                    egui::Button::new(egui::RichText::new("Clear").size(13.0))
                        .fill(egui::Color32::from_rgb(0xaa, 0x33, 0x33))
                        .corner_radius(egui::CornerRadius::same(6))
                );
                if clear_btn.clicked() {
                    self.code.clear();
                }
                ui.add_space(8.0);
                if ui
                    .small_button("Send to chat")
                    .on_hover_text("Copy the editor code into the focused chat input")
                    .clicked()
                {
                    let _ = tx.send(crate::ui::app::AppMessage::EditorToChat(self.code.clone()));
                }
            });
        });

        ui.add_space(8.0);
        ui.separator();
        ui.add_space(8.0);

        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Language:").size(13.0).color(egui::Color32::from_rgb(0xcc, 0xcc, 0xcc)));
            ui.add_space(8.0);
            egui::ComboBox::from_id_salt("editor_language")
                .selected_text(egui::RichText::new(&self.language).size(13.0))
                .width(150.0)
                .show_ui(ui, |ui| {
                    for lang in ["rust", "python", "javascript", "typescript", "go", "c", "cpp", "java", "sql", "shell", "lua", "asm"] {
                        ui.selectable_value(&mut self.language, lang.to_string(), egui::RichText::new(lang).size(13.0));
                    }
                });
        });

        ui.add_space(12.0);

        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("File:").size(13.0).color(egui::Color32::from_rgb(0xcc, 0xcc, 0xcc)));
            ui.add_space(8.0);
            ui.add(
                egui::TextEdit::singleline(&mut self.file_path)
                    .desired_width(300.0)
                    .font(egui::TextStyle::Monospace)
                    .hint_text("path/to/main.py"),
            );
            ui.add_space(8.0);
            if ui
                .button(egui::RichText::new("Save").size(13.0))
                .on_hover_text("Write the editor to this file (folders created as needed)")
                .clicked()
            {
                let ok = self.save_to_file();
                if !self.file_path.trim().is_empty() {
                    let _ = tx.send(crate::ui::app::AppMessage::Audit(
                        if ok { "file.save".to_string() } else { "file.save_failed".to_string() },
                        self.file_path.trim().to_string(),
                    ));
                }
            }
            if ui
                .button(egui::RichText::new("Open").size(13.0))
                .on_hover_text("Load this file into the editor")
                .clicked()
            {
                let ok = self.open_from_file();
                if !self.file_path.trim().is_empty() {
                    let _ = tx.send(crate::ui::app::AppMessage::Audit(
                        if ok { "file.open".to_string() } else { "file.open_failed".to_string() },
                        self.file_path.trim().to_string(),
                    ));
                }
            }
        });
        if !self.file_status.is_empty() {
            ui.label(
                egui::RichText::new(&self.file_status)
                    .size(11.0)
                    .color(egui::Color32::from_rgb(0x99, 0x99, 0x99)),
            );
        }
        ui.add_space(8.0);
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("New:").size(13.0).color(egui::Color32::from_rgb(0xcc, 0xcc, 0xcc)));
            ui.add_space(8.0);
            if ui.button("Rust app").clicked() {
                self.apply_template("rust");
            }
            if ui.button("Python app").clicked() {
                self.apply_template("python");
            }
            if ui.button("Shell script").clicked() {
                self.apply_template("shell");
            }
            ui.add_space(8.0);
            ui.label(
                egui::RichText::new(self.run_hint())
                    .size(11.0)
                    .color(egui::Color32::from_rgb(0x88, 0x88, 0x88)),
            );
        });
        ui.add_space(8.0);

        let syntax = Self::syntax_for_language(&self.language);
        let mut editor = CodeEditor::default()
            .id_source("code_editor")
            .with_rows(24)
            .with_fontsize(14.0);

        let _response = editor.show(ui, &mut self.code, &syntax);

        ui.add_space(12.0);
        ui.separator();
        ui.add_space(8.0);

        ui.horizontal(|ui| {
            ui.add_space(4.0);
            ui.label(egui::RichText::new("AI Assist:").size(14.0).color(egui::Color32::from_rgb(0x00, 0xaa, 0xff)));
            ui.add_space(16.0);

            let can_ai = selected_model.is_some() && api_client.is_some();
            
            let explain_btn = ui.add_enabled(can_ai,
                egui::Button::new(egui::RichText::new("Explain Code").size(13.0))
                    .fill(egui::Color32::from_rgb(0x00, 0x55, 0xaa))
                    .corner_radius(egui::CornerRadius::same(6))
            );
            if explain_btn.clicked() {
                self.request_ai_help("Explain this code in detail:", models, selected_model, system_prompt, api_client, tx, rt);
            }

            ui.add_space(8.0);
            let improve_btn = ui.add_enabled(can_ai,
                egui::Button::new(egui::RichText::new("Improve Code").size(13.0))
                    .fill(egui::Color32::from_rgb(0x00, 0x77, 0x55))
                    .corner_radius(egui::CornerRadius::same(6))
            );
            if improve_btn.clicked() {
                self.request_ai_help("Improve this code (add comments, fix bugs, optimize):", models, selected_model, system_prompt, api_client, tx, rt);
            }

            ui.add_space(8.0);
            let complete_btn = ui.add_enabled(can_ai,
                egui::Button::new(egui::RichText::new("Complete Code").size(13.0))
                    .fill(egui::Color32::from_rgb(0xaa, 0x55, 0x00))
                    .corner_radius(egui::CornerRadius::same(6))
            );
            if complete_btn.clicked() {
                self.request_ai_help("Complete this code snippet:", models, selected_model, system_prompt, api_client, tx, rt);
            }
        });

        if !self.suggestion_buf.is_empty() {
            ui.add_space(12.0);
            ui.separator();
            ui.add_space(8.0);
            ui.label(egui::RichText::new("AI Suggestion (streaming):").size(14.0).color(egui::Color32::from_rgb(0x00, 0xcc, 0x88)));
            ui.add_space(8.0);

            egui::ScrollArea::vertical().max_height(200.0).stick_to_bottom(true).show(ui, |ui| {
                ui.code(format!("{}▍", self.suggestion_buf));
            });
        }

        if let Some(suggestion) = self.pending_suggestion.clone() {
            ui.add_space(12.0);
            ui.separator();
            ui.add_space(8.0);
            ui.label(egui::RichText::new("AI Suggestion:").size(14.0).color(egui::Color32::from_rgb(0x00, 0xcc, 0x88)));
            ui.add_space(8.0);
            
            egui::ScrollArea::vertical().max_height(200.0).show(ui, |ui| {
                ui.code(suggestion.clone());
            });

            ui.add_space(8.0);
            ui.horizontal(|ui| {
                ui.checkbox(&mut self.show_diff, "Show diff vs current code");
            });
            if self.show_diff {
                let diff = Self::diff_lines(&self.code, &suggestion);
                egui::ScrollArea::vertical()
                    .max_height(200.0)
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        for (sign, line) in diff.iter().take(400) {
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

            ui.add_space(8.0);
            ui.horizontal(|ui| {
                let apply_btn = ui.add(
                    egui::Button::new(egui::RichText::new("Apply Suggestion").size(13.0))
                        .fill(egui::Color32::from_rgb(0x00, 0x77, 0x55))
                        .corner_radius(egui::CornerRadius::same(6))
                );
                if apply_btn.clicked() {
                    self.code = suggestion.clone();
                    self.pending_suggestion = None;
                }
                
                ui.add_space(8.0);
                let dismiss_btn = ui.add(
                    egui::Button::new(egui::RichText::new("Dismiss").size(13.0))
                        .fill(egui::Color32::from_rgb(0xaa, 0x33, 0x33))
                        .corner_radius(egui::CornerRadius::same(6))
                );
                if dismiss_btn.clicked() {
                    self.pending_suggestion = None;
                }
            });
        }
    }

    /// Line diff of current code (old) vs AI suggestion (new).
    /// Returns (sign, text): '+' added, '-' removed, ' ' context.
    pub fn diff_lines(old: &str, new: &str) -> Vec<(char, String)> {
        use similar::{ChangeTag, TextDiff};
        let diff = TextDiff::from_lines(old, new);
        let mut out = Vec::new();
        for op in diff.ops() {
            for change in diff.iter_changes(op) {
                let sign = match change.tag() {
                    ChangeTag::Delete => '-',
                    ChangeTag::Insert => '+',
                    ChangeTag::Equal => ' ',
                };
                out.push((
                    sign,
                    change.value().trim_end_matches('\n').to_string(),
                ));
            }
        }
        out
    }

    fn syntax_for_language(lang: &str) -> Syntax {
        match lang {
            "rust" => Syntax::rust(),
            "python" => Syntax::python(),
            "javascript" | "typescript" => Syntax::rust(),
            "go" => Syntax::rust(),
            "c" | "cpp" => Syntax::rust(),
            "java" => Syntax::rust(),
            "sql" => Syntax::sql(),
            "shell" => Syntax::shell(),
            "lua" => Syntax::lua(),
            "asm" => Syntax::asm(),
            _ => Syntax::rust(),
        }
    }

    fn request_ai_help(
        &mut self,
        prompt_prefix: &str,
        _models: &[crate::ollama::api::Model],
        selected_model: &Option<String>,
        system_prompt: &str,
        api_client: &Option<OllamaClient>,
        tx: &mpsc::Sender<crate::ui::app::AppMessage>,
        rt: &Runtime,
    ) {
        let Some(model_name) = selected_model else { return };
        let Some(client) = api_client else { return };

        let full_prompt = format!("{}\n\n```{}\n{}\n```", prompt_prefix, self.language, self.code);
        if let Err(hits) = self.gate.check(&full_prompt) {
            let _ = tx.send(crate::ui::app::AppMessage::Audit(
                "secret.blocked".to_string(),
                "editor assist".to_string(),
            ));
            let kinds: Vec<String> = hits
                .iter()
                .map(|h| format!("{} ({})", h.kind, h.preview))
                .collect();
            self.file_status = format!(
                "Blocked: possible secret ({}). Send again within 60s to override.",
                kinds.join(", ")
            );
            return;
        }
        let model_name = model_name.clone();
        let client = client.clone();
        let tx = tx.clone();
        let suggestion_id = self.suggestion_id;
        self.suggestion_id += 1;
        let system_prompt = system_prompt.to_string();

        rt.spawn(async move {
            let mut messages = Vec::new();
            if !system_prompt.trim().is_empty() {
                messages.push(Message {
                    role: "system".to_string(),
                    content: system_prompt,
                });
            }
            messages.push(Message {
                role: "user".to_string(),
                content: full_prompt,
            });
            let req = ChatRequest {
                model: model_name,
                messages,
                stream: true,
                options: Some(ChatOptions::lowram()),
                keep_alive: Some("10m".to_string()),
            };

            let result = client
                .chat_stream(req, |piece| {
                    let _ = tx.send(crate::ui::app::AppMessage::EditorChunk(
                        suggestion_id,
                        piece.to_string(),
                    ));
                })
                .await;
            let _ = tx.send(crate::ui::app::AppMessage::EditorSuggestion(suggestion_id, result));
        });
    }

    /// Code arriving from chat: adopt it and guess its language.
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

    fn default_path() -> String {
        let base = dirs::home_dir().unwrap_or_else(|| std::path::PathBuf::from("."));
        base.join("Documents")
            .join("Code_air")
            .join("ml_lab")
            .join("main.py")
            .to_string_lossy()
            .to_string()
    }

    fn save_to_file(&mut self) -> bool {
        let path = std::path::PathBuf::from(self.file_path.trim());
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
                self.code = "// New Rust app\n// Run: rustc main.rs -o app && ./app\n\nfn main() {\n    println!(\"Hello from ML Lab\");\n}\n".to_string();
            }
            "python" => {
                self.language = "python".to_string();
                self.code = "# New Python app\n# Run: python3 main.py\n\ndef main():\n    print(\"Hello from ML Lab\")\n\n\nif __name__ == \"__main__\":\n    main()\n".to_string();
            }
            "shell" => {
                self.language = "shell".to_string();
                self.code = "#!/bin/sh\n# New shell script\n# Run: sh script.sh\n\necho \"Hello from ML Lab\"\n".to_string();
            }
            _ => {}
        }
        self.file_status = format!("{} template loaded — Save to keep it.", kind);
    }

    fn run_hint(&self) -> &str {
        match self.language.as_str() {
            "rust" => "Run: rustc <file> -o app && ./app",
            "python" => "Run: python3 <file>",
            "shell" => "Run: sh <file>",
            "javascript" => "Run: node <file>",
            _ => "Run with your toolchain",
        }
    }

    /// Live token piece for the in-flight suggestion; stale ids ignored.
    pub fn push_chunk(&mut self, suggestion_id: usize, piece: &str) {
        if suggestion_id + 1 != self.suggestion_id || piece.is_empty() {
            return;
        }
        if self.suggestion_buf.len() < 50_000 {
            self.suggestion_buf.push_str(piece);
        }
    }

    pub fn handle_ai_suggestion(&mut self, suggestion_id: usize, response: Result<crate::ollama::api::ChatResponse, anyhow::Error>) {
        if suggestion_id + 1 != self.suggestion_id {
            return;
        }
        self.suggestion_buf.clear();

        match response {
            Ok(resp) => {
                self.pending_suggestion = Some(resp.message.content);
            }
            Err(e) => {
                self.pending_suggestion = Some(format!("Error: {}", e));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suggestion_chunk_currency() {
        let mut e = EditorPanel::new();
        // No request in flight: stale ids ignored.
        e.push_chunk(0, "x");
        assert!(e.suggestion_buf.is_empty());
        // Simulate an issued request (id 0 -> next id 1).
        e.suggestion_id = 1;
        e.push_chunk(0, "hello ");
        e.push_chunk(0, "world");
        assert_eq!(e.suggestion_buf, "hello world");
        // Stale and future ids ignored.
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
}
