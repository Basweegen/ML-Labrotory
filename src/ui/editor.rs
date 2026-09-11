// Copyright 2026 Sean M. Stow. All rights reserved.
use eframe::egui;
use egui_code_editor::{CodeEditor, Syntax};
use crate::ollama::api::{ChatOptions, ChatRequest, Message, OllamaClient};
use crate::storage::ChatMessage;
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
    // AI Coder Chatbox state
    pub coder_model: Option<String>,
    pub coder_input: String,
    pub coder_messages: Vec<ChatMessage>,
    pub coder_is_streaming: bool,
    pub coder_stream_buf: String,
    pub show_coder_chat: bool,
    pub include_editor_context: bool,
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
            coder_model: None,
            coder_input: String::new(),
            coder_messages: Vec::new(),
            coder_is_streaming: false,
            coder_stream_buf: String::new(),
            show_coder_chat: true,
            include_editor_context: true,
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
            ui.heading(
                egui::RichText::new("AI Code Studio & IDE")
                    .size(20.0)
                    .color(egui::Color32::from_rgb(0x00, 0xaa, 0xff)),
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.add_space(8.0);
                let toggle_label = if self.show_coder_chat {
                    "🤖 Hide AI Coder"
                } else {
                    "🤖 Show AI Coder"
                };
                let toggle_fill = if self.show_coder_chat {
                    egui::Color32::from_rgb(0x00, 0x55, 0xaa)
                } else {
                    egui::Color32::from_rgb(0x22, 0x33, 0x55)
                };
                if ui
                    .add(
                        egui::Button::new(egui::RichText::new(toggle_label).size(12.5).color(egui::Color32::WHITE))
                            .fill(toggle_fill)
                            .corner_radius(egui::CornerRadius::same(6)),
                    )
                    .on_hover_text("Toggle the integrated AI Coder Chatbox sidebar")
                    .clicked()
                {
                    self.show_coder_chat = !self.show_coder_chat;
                }

                ui.add_space(8.0);
                let clear_btn = ui.add(
                    egui::Button::new(egui::RichText::new("Clear Editor").size(12.0))
                        .fill(egui::Color32::from_rgb(0x88, 0x22, 0x22))
                        .corner_radius(egui::CornerRadius::same(6)),
                );
                if clear_btn.clicked() {
                    self.code.clear();
                }

                ui.add_space(8.0);
                if ui
                    .small_button("Send to Chat tab")
                    .on_hover_text("Copy the editor code into the main chat tab input")
                    .clicked()
                {
                    let _ = tx.send(crate::ui::app::AppMessage::EditorToChat(self.code.clone()));
                }
            });
        });

        ui.add_space(6.0);
        ui.separator();
        ui.add_space(6.0);

        if self.show_coder_chat {
            ui.columns(2, |cols| {
                self.show_editor_pane(&mut cols[0], tx);
                self.show_coder_chat_pane(&mut cols[1], models, selected_model, system_prompt, api_client, tx, rt);
            });
        } else {
            self.show_editor_pane(ui, tx);
        }
    }

    fn show_editor_pane(&mut self, ui: &mut egui::Ui, tx: &mpsc::Sender<crate::ui::app::AppMessage>) {
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Language:").size(13.0).color(egui::Color32::from_rgb(0xcc, 0xcc, 0xcc)));
            ui.add_space(8.0);
            egui::ComboBox::from_id_salt("editor_language")
                .selected_text(egui::RichText::new(&self.language).size(13.0))
                .width(130.0)
                .show_ui(ui, |ui| {
                    for lang in ["rust", "python", "javascript", "typescript", "go", "c", "cpp", "java", "sql", "shell", "lua", "asm"] {
                        ui.selectable_value(&mut self.language, lang.to_string(), egui::RichText::new(lang).size(13.0));
                    }
                });
        });

        ui.add_space(6.0);

        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("File:").size(13.0).color(egui::Color32::from_rgb(0xcc, 0xcc, 0xcc)));
            ui.add_space(8.0);
            ui.add(
                egui::TextEdit::singleline(&mut self.file_path)
                    .desired_width(200.0)
                    .font(egui::TextStyle::Monospace)
                    .hint_text("path/to/main.rs"),
            );
            ui.add_space(6.0);
            if ui
                .button(egui::RichText::new("Save").size(12.5))
                .on_hover_text("Write editor content to this file")
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
                .button(egui::RichText::new("Open").size(12.5))
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
            ui.add_space(2.0);
            ui.label(
                egui::RichText::new(&self.file_status)
                    .size(11.0)
                    .color(egui::Color32::from_rgb(0x00, 0xcc, 0x88)),
            );
        }

        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Templates:").size(12.0).color(egui::Color32::from_rgb(0xaa, 0xaa, 0xaa)));
            ui.add_space(4.0);
            if ui.small_button("Rust").clicked() {
                self.apply_template("rust");
            }
            if ui.small_button("Python").clicked() {
                self.apply_template("python");
            }
            if ui.small_button("Shell").clicked() {
                self.apply_template("shell");
            }
            ui.add_space(8.0);
            ui.label(
                egui::RichText::new(self.run_hint())
                    .size(11.0)
                    .color(egui::Color32::from_rgb(0x88, 0x88, 0x88)),
            );
        });

        ui.add_space(6.0);

        let syntax = Self::syntax_for_language(&self.language);
        let mut editor = CodeEditor::default()
            .id_source("code_editor_main")
            .with_rows(24)
            .with_fontsize(13.5);

        let _ = editor.show(ui, &mut self.code, &syntax);

        if let Some(suggestion) = self.pending_suggestion.clone() {
            ui.add_space(8.0);
            ui.separator();
            ui.add_space(4.0);
            ui.label(
                egui::RichText::new("AI Suggestion Diff:")
                    .size(13.0)
                    .color(egui::Color32::from_rgb(0x00, 0xcc, 0x88)),
            );
            ui.checkbox(&mut self.show_diff, "Show side-by-side / line diff");
            if self.show_diff {
                let diff = Self::diff_lines(&self.code, &suggestion);
                egui::ScrollArea::vertical()
                    .max_height(140.0)
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
                        egui::Button::new(egui::RichText::new("Apply Suggestion").size(12.5))
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
                        egui::Button::new(egui::RichText::new("Dismiss").size(12.5))
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

    fn show_coder_chat_pane(
        &mut self,
        ui: &mut egui::Ui,
        models: &[crate::ollama::api::Model],
        selected_model: &Option<String>,
        system_prompt: &str,
        api_client: &Option<OllamaClient>,
        tx: &mpsc::Sender<crate::ui::app::AppMessage>,
        rt: &Runtime,
    ) {
        let frame = egui::Frame::NONE
            .fill(egui::Color32::from_rgb(0x0f, 0x14, 0x22))
            .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(0x1e, 0x29, 0x3b)))
            .corner_radius(egui::CornerRadius::same(8))
            .inner_margin(egui::Margin::same(10));

        frame.show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new("🤖 AI Coder")
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
                    .selected_text(egui::RichText::new(&cur_model).size(12.5))
                    .width(160.0)
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
                        self.coder_stream_buf.clear();
                    }
                });
            });

            ui.add_space(6.0);

            ui.horizontal_wrapped(|ui| {
                ui.label(egui::RichText::new("Prompt:").size(11.0).color(egui::Color32::from_rgb(0x94, 0xa3, 0xb8)));
                if ui.small_button("✨ Create").on_hover_text("Write prompt to create new code").clicked() {
                    self.coder_input = "Create a complete, robust implementation of: ".to_string();
                }
                if ui.small_button("⚡ Optimize").on_hover_text("Ask AI to optimize current code").clicked() {
                    self.send_coder_message(
                        "Refactor and optimize the current editor code for maximum performance, clean idiomatic style, and security.",
                        models, selected_model, system_prompt, api_client, tx, rt
                    );
                }
                if ui.small_button("🐛 Fix Bugs").on_hover_text("Ask AI to identify and fix bugs").clicked() {
                    self.send_coder_message(
                        "Analyze the current editor code for bugs, logic errors, or memory leaks, and provide the corrected code.",
                        models, selected_model, system_prompt, api_client, tx, rt
                    );
                }
                if ui.small_button("🧪 Tests").on_hover_text("Ask AI to generate tests").clicked() {
                    self.send_coder_message(
                        "Generate comprehensive unit tests covering edge cases for this code.",
                        models, selected_model, system_prompt, api_client, tx, rt
                    );
                }
                if ui.small_button("📖 Explain").on_hover_text("Ask AI to explain this code").clicked() {
                    self.send_coder_message(
                        "Explain the architecture, design patterns, and algorithmic flow of this code in detail.",
                        models, selected_model, system_prompt, api_client, tx, rt
                    );
                }
            });

            ui.add_space(6.0);
            ui.separator();
            ui.add_space(4.0);

            let reserve_h = 135.0 * ui.ctx().zoom_factor();
            let chat_h = (ui.available_height() - reserve_h).max(120.0);

            let mut apply_code_request: Option<String> = None;
            let mut append_code_request: Option<String> = None;

            egui::ScrollArea::vertical()
                .id_salt("ide_coder_chat_scroll")
                .max_height(chat_h)
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    if self.coder_messages.is_empty() && self.coder_stream_buf.is_empty() {
                        ui.add_space(8.0);
                        ui.label(
                            egui::RichText::new("💬 AI Coder is ready. Ask to generate new code, refactor, or fix bugs. Use 'Apply' on any code snippet to test it immediately.")
                                .size(12.0)
                                .color(egui::Color32::from_rgb(0x64, 0x74, 0x8b)),
                        );
                    }

                    for (m_idx, msg) in self.coder_messages.iter().enumerate() {
                        Self::render_coder_message(
                            ui,
                            msg,
                            m_idx,
                            &mut apply_code_request,
                            &mut append_code_request,
                            tx,
                        );
                    }

                    if !self.coder_stream_buf.is_empty() {
                        let tmp = ChatMessage {
                            role: "assistant".to_string(),
                            content: format!("{}▍", self.coder_stream_buf),
                            timestamp: chrono::Utc::now(),
                        };
                        Self::render_coder_message(
                            ui,
                            &tmp,
                            self.coder_messages.len(),
                            &mut apply_code_request,
                            &mut append_code_request,
                            tx,
                        );
                    }

                    if self.coder_is_streaming && self.coder_stream_buf.is_empty() {
                        ui.horizontal(|ui| {
                            ui.spinner();
                            ui.label(
                                egui::RichText::new("Coding...")
                                    .size(12.0)
                                    .color(egui::Color32::from_rgb(0x38, 0xbd, 0xf8)),
                            );
                        });
                    }
                });

            if let Some(code) = apply_code_request {
                self.code = code;
                self.file_status = "Code applied to Editor from AI Coder.".to_string();
            }
            if let Some(code) = append_code_request {
                if !self.code.trim().is_empty() {
                    self.code.push_str("\n\n");
                }
                self.code.push_str(&code);
                self.file_status = "Code appended to Editor from AI Coder.".to_string();
            }

            ui.add_space(4.0);
            ui.separator();
            ui.add_space(4.0);

            let input_resp = ui.add(
                egui::TextEdit::multiline(&mut self.coder_input)
                    .desired_rows(2)
                    .desired_width(f32::INFINITY)
                    .hint_text("Ask AI coder to write, refactor, or fix code... (Enter to send)")
                    .font(egui::TextStyle::Body),
            );

            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.checkbox(&mut self.include_editor_context, "Include editor code in context");

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let can_send = !self.coder_input.trim().is_empty() && !self.coder_is_streaming;
                    let send_btn = ui.add_enabled(
                        can_send,
                        egui::Button::new(
                            egui::RichText::new(if self.coder_is_streaming { "Generating..." } else { "Send (Enter)" })
                                .size(12.5)
                                .color(egui::Color32::WHITE),
                        )
                        .fill(egui::Color32::from_rgb(0x00, 0x66, 0xcc))
                        .corner_radius(egui::CornerRadius::same(6)),
                    );

                    if send_btn.clicked() {
                        let prompt = self.coder_input.clone();
                        self.send_coder_message(&prompt, models, selected_model, system_prompt, api_client, tx, rt);
                    }

                    if self.coder_is_streaming {
                        ui.add_space(4.0);
                        let stop_btn = ui.add(
                            egui::Button::new(egui::RichText::new("Stop").size(12.5).color(egui::Color32::WHITE))
                                .fill(egui::Color32::from_rgb(0xaa, 0x33, 0x33))
                                .corner_radius(egui::CornerRadius::same(6)),
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
                self.send_coder_message(&prompt, models, selected_model, system_prompt, api_client, tx, rt);
            }
        });
    }

    fn render_coder_message(
        ui: &mut egui::Ui,
        msg: &ChatMessage,
        m_idx: usize,
        apply_req: &mut Option<String>,
        append_req: &mut Option<String>,
        _tx: &mpsc::Sender<crate::ui::app::AppMessage>,
    ) {
        let is_user = msg.role == "user";
        let is_sys = msg.role == "system";
        let (bg, who, tag_color) = if is_user {
            (egui::Color32::from_rgb(0x13, 0x2f, 0x4c), "You", egui::Color32::from_rgb(0x38, 0xbd, 0xf8))
        } else if is_sys {
            (egui::Color32::from_rgb(0x33, 0x22, 0x11), "System", egui::Color32::from_rgb(0xf5, 0x9e, 0x0b))
        } else {
            (egui::Color32::from_rgb(0x16, 0x1e, 0x2e), "AI Coder", egui::Color32::from_rgb(0x4a, 0xde, 0x80))
        };

        ui.add_space(3.0);
        egui::Frame::NONE
            .fill(bg)
            .corner_radius(egui::CornerRadius::same(6))
            .inner_margin(egui::Margin::same(8))
            .show(ui, |ui| {
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
                let segments = crate::ui::chat::parse_segments(&msg.content);

                for (s_idx, seg) in segments.iter().enumerate() {
                    match seg {
                        crate::ui::chat::MessageSegment::Text(t) => {
                            if !t.is_empty() {
                                ui.label(egui::RichText::new(t).size(12.5).color(egui::Color32::WHITE));
                            }
                        }
                        crate::ui::chat::MessageSegment::Think(th) => {
                            egui::CollapsingHeader::new(
                                egui::RichText::new("💭 Thought Process")
                                    .size(11.0)
                                    .color(egui::Color32::from_rgb(0x94, 0xa3, 0xb8)),
                            )
                            .default_open(false)
                            .show(ui, |ui| {
                                ui.label(
                                    egui::RichText::new(th)
                                        .size(11.0)
                                        .color(egui::Color32::from_rgb(0x94, 0xa3, 0xb8))
                                        .italics(),
                                );
                            });
                        }
                        crate::ui::chat::MessageSegment::Code { lang, code } => {
                            ui.add_space(4.0);
                            let cb_frame = egui::Frame::NONE
                                .fill(egui::Color32::from_rgb(0x0a, 0x0f, 0x1d))
                                .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(0x1e, 0x2d, 0x48)))
                                .corner_radius(egui::CornerRadius::same(6))
                                .inner_margin(egui::Margin::symmetric(8, 6));

                            cb_frame.show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    let display_lang = if lang.trim().is_empty() { "CODE" } else { lang.trim() };
                                    ui.label(
                                        egui::RichText::new(format!("💻 {}", display_lang.to_uppercase()))
                                            .size(11.0)
                                            .monospace()
                                            .color(egui::Color32::from_rgb(0x38, 0xbd, 0xf8)),
                                    );

                                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                        if ui.small_button("📥 Apply").on_hover_text("Replace editor code with this snippet").clicked() {
                                            *apply_req = Some(code.clone());
                                        }
                                        ui.add_space(4.0);
                                        if ui.small_button("➕ Append").on_hover_text("Append this snippet to the editor").clicked() {
                                            *append_req = Some(code.clone());
                                        }
                                        ui.add_space(4.0);
                                        if ui.small_button("📋 Copy").on_hover_text("Copy snippet to clipboard").clicked() {
                                            ui.ctx().copy_text(code.clone());
                                        }
                                    });
                                });

                                ui.add_space(2.0);
                                ui.separator();
                                ui.add_space(2.0);

                                let scroll_id = format!("coder_cb_{}_{}_{}", m_idx, s_idx, code.len());
                                egui::ScrollArea::horizontal()
                                    .id_salt(scroll_id)
                                    .auto_shrink([false, false])
                                    .show(ui, |ui| {
                                        ui.add(
                                            egui::Label::new(
                                                egui::RichText::new(code)
                                                    .monospace()
                                                    .size(12.0)
                                                    .color(egui::Color32::from_rgb(0xec, 0xf0, 0xf8)),
                                            )
                                        );
                                    });
                            });
                            ui.add_space(4.0);
                        }
                    }
                }
            });
        ui.add_space(3.0);
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

    fn send_coder_message(
        &mut self,
        prompt: &str,
        _models: &[crate::ollama::api::Model],
        selected_model: &Option<String>,
        system_prompt: &str,
        api_client: &Option<OllamaClient>,
        tx: &mpsc::Sender<crate::ui::app::AppMessage>,
        rt: &Runtime,
    ) {
        let active_model = self.coder_model.clone().or_else(|| selected_model.clone());
        let Some(model_name) = active_model else {
            self.coder_messages.push(ChatMessage {
                role: "system".to_string(),
                content: "Select an AI model first using the dropdown above.".to_string(),
                timestamp: chrono::Utc::now(),
            });
            return;
        };
        let Some(client) = api_client else {
            self.coder_messages.push(ChatMessage {
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
            self.coder_messages.push(ChatMessage {
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

        self.coder_messages.push(ChatMessage {
            role: "user".to_string(),
            content: p.to_string(),
            timestamp: chrono::Utc::now(),
        });
        self.coder_input.clear();

        let mut full_prompt = String::new();
        if self.include_editor_context && !self.code.trim().is_empty() {
            full_prompt.push_str(&format!(
                "Current File ({}, {}):\n```{}\n{}\n```\n\nTask / Request:\n{}",
                self.file_path, self.language, self.language, self.code, p
            ));
        } else {
            full_prompt.push_str(p);
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
            "You are an expert software engineer and code architect. Produce correct, clean, idiomatic code in standard markdown code blocks with language specifiers.".to_string()
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
        let clean = crate::ui::chat::sanitize_text(piece);
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
                self.coder_messages.push(ChatMessage {
                    role: "assistant".to_string(),
                    content: clean,
                    timestamp: chrono::Utc::now(),
                });
            }
            Err(e) => {
                let err_msg = crate::ui::chat::sanitize_text(&format!("Error: {}", e));
                self.pending_suggestion = Some(err_msg.clone());
                self.coder_messages.push(ChatMessage {
                    role: "system".to_string(),
                    content: err_msg,
                    timestamp: chrono::Utc::now(),
                });
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
        assert_eq!(e.coder_stream_buf, "hello world");
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

    #[test]
    fn coder_chat_message_flow() {
        let mut e = EditorPanel::new();
        assert!(e.coder_messages.is_empty());
        e.coder_messages.push(ChatMessage {
            role: "user".to_string(),
            content: "Write hello".to_string(),
            timestamp: chrono::Utc::now(),
        });
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
}
