use eframe::egui;
use egui_code_editor::{CodeEditor, Syntax};
use crate::ollama::api::{OllamaClient, ChatRequest, Message};
use std::sync::mpsc;
use tokio::runtime::Runtime;

pub struct EditorPanel {
    code: String,
    language: String,
    suggestion_id: usize,
    pending_suggestion: Option<String>,
}

impl EditorPanel {
    pub fn new() -> Self {
        Self {
            code: "// Start coding with AI assistance\nfn main() {\n    println!(\"Hello, world!\");\n}".to_string(),
            language: "rust".to_string(),
            suggestion_id: 0,
            pending_suggestion: None,
        }
    }

    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        models: &[crate::ollama::api::Model],
        selected_model: &Option<String>,
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

        let syntax = Self::syntax_for_language(&self.language);
        let mut editor = CodeEditor::default()
            .id_source("code_editor")
            .with_rows(30)
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
                self.request_ai_help("Explain this code in detail:", models, selected_model, api_client, tx, rt);
            }

            ui.add_space(8.0);
            let improve_btn = ui.add_enabled(can_ai,
                egui::Button::new(egui::RichText::new("Improve Code").size(13.0))
                    .fill(egui::Color32::from_rgb(0x00, 0x77, 0x55))
                    .corner_radius(egui::CornerRadius::same(6))
            );
            if improve_btn.clicked() {
                self.request_ai_help("Improve this code (add comments, fix bugs, optimize):", models, selected_model, api_client, tx, rt);
            }

            ui.add_space(8.0);
            let complete_btn = ui.add_enabled(can_ai,
                egui::Button::new(egui::RichText::new("Complete Code").size(13.0))
                    .fill(egui::Color32::from_rgb(0xaa, 0x55, 0x00))
                    .corner_radius(egui::CornerRadius::same(6))
            );
            if complete_btn.clicked() {
                self.request_ai_help("Complete this code snippet:", models, selected_model, api_client, tx, rt);
            }
        });

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
        api_client: &Option<OllamaClient>,
        tx: &mpsc::Sender<crate::ui::app::AppMessage>,
        rt: &Runtime,
    ) {
        let Some(model_name) = selected_model else { return };
        let Some(client) = api_client else { return };

        let full_prompt = format!("{}\n\n```{}\n{}\n```", prompt_prefix, self.language, self.code);
        let model_name = model_name.clone();
        let client = client.clone();
        let tx = tx.clone();
        let suggestion_id = self.suggestion_id;
        self.suggestion_id += 1;

        rt.spawn(async move {
            let req = ChatRequest {
                model: model_name,
                messages: vec![Message {
                    role: "user".to_string(),
                    content: full_prompt,
                }],
                stream: false,
                options: None,
                keep_alive: Some("10m".to_string()),
            };

            let result = client.chat(req).await;
            let _ = tx.send(crate::ui::app::AppMessage::EditorSuggestion(suggestion_id, result));
        });
    }

    pub fn handle_ai_suggestion(&mut self, suggestion_id: usize, response: Result<crate::ollama::api::ChatResponse, anyhow::Error>) {
        if suggestion_id + 1 != self.suggestion_id {
            return;
        }

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
