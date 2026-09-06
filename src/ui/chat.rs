use eframe::egui;
use anyhow::Result;
use crate::ollama::api::{ChatRequest, Message, OllamaClient, ChatResponse};
use crate::storage::ChatMessage;
use std::sync::mpsc;
use tokio::runtime::Runtime;

pub struct ChatPanel {
    input: String,
    messages: Vec<ChatMessage>,
    is_streaming: bool,
    voice_enabled: bool,
}

impl ChatPanel {
    pub fn new() -> Self {
        Self {
            input: String::new(),
            messages: Vec::new(),
            is_streaming: false,
            voice_enabled: false,
        }
    }

    pub fn clear_chat(&mut self) {
        self.messages.clear();
        self.input.clear();
    }

    pub fn messages(&self) -> &[ChatMessage] {
        &self.messages
    }

    pub fn set_voice_enabled(&mut self, enabled: bool) {
        self.voice_enabled = enabled;
    }

    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        _models: &[crate::ollama::api::Model],
        selected_model: &Option<String>,
        role_prompt: &str,
        api_client: &Option<OllamaClient>,
        slot_idx: usize,
        tx: &mpsc::Sender<crate::ui::app::AppMessage>,
        rt: &Runtime,
    ) {
        ui.horizontal(|ui| {
            ui.add_space(4.0);
            let voice_btn = ui.add(
                egui::Button::new(if self.voice_enabled { "Voice ON" } else { "Voice OFF" })
                    .corner_radius(egui::CornerRadius::same(6)),
            )
            .on_hover_text("Toggle voice input/output (local, free)");
            if voice_btn.clicked() {
                self.voice_enabled = !self.voice_enabled;
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.add_space(4.0);
                if ui
                    .small_button("Clear chat")
                    .on_hover_text("Clear this slot's chat history")
                    .clicked()
                {
                    self.clear_chat();
                }
            });
        });

        ui.add_space(4.0);
        ui.separator();
        ui.add_space(4.0);

        // Message history; reserve room below for the input row.
        let list_h = (ui.available_height() - 150.0).max(120.0);
        egui::ScrollArea::vertical()
            .max_height(list_h)
            .show(ui, |ui| {
                if self.messages.is_empty() {
                    ui.horizontal(|ui| {
                        ui.add_space(4.0);
                        ui.label(
                            egui::RichText::new("No messages yet - pick a model above, then type below.")
                                .size(13.0)
                                .color(egui::Color32::from_rgb(0x99, 0x99, 0x99)),
                        );
                    });
                }
                for i in 0..self.messages.len() {
                    let msg = self.messages[i].clone();
                    self.show_message(ui, &msg);
                }
                if self.is_streaming {
                    ui.horizontal(|ui| {
                        ui.add_space(4.0);
                        ui.spinner();
                        ui.label(
                            egui::RichText::new("Thinking...")
                                .size(13.0)
                                .color(egui::Color32::from_rgb(0x99, 0x99, 0x99)),
                        );
                    });
                }
            });

        ui.separator();

        ui.add_space(4.0);
        ui.horizontal(|ui| {
            // Reserve a fixed column for Send/Stop: an INFINITY-width text
            // field starves its horizontal siblings, hiding the buttons.
            let input_w = (ui.available_width() - 118.0).max(140.0);
            let response = ui.add(
                egui::TextEdit::multiline(&mut self.input)
                    .desired_rows(3)
                    .desired_width(input_w)
                    .hint_text("Type your message... (Enter to send, Shift+Enter for newline)")
                    .font(egui::TextStyle::Body),
            );

            // Enter sends while the field has focus (multiline keeps focus on
            // Enter, so lost_focus() never fires for it). Shift+Enter = newline.
            let send_triggered = ui.input(|i| {
                i.key_pressed(egui::Key::Enter) && !i.modifiers.shift
            }) && response.has_focus()
                && !self.input.trim().is_empty()
                && !self.is_streaming;
            if send_triggered {
                self.send_message(_models, selected_model, role_prompt, slot_idx, api_client, tx, rt);
            }

            ui.add_space(8.0);
            ui.vertical(|ui| {
                let send_enabled = !self.input.trim().is_empty() && selected_model.is_some() && api_client.is_some() && !self.is_streaming;
                let send_btn = ui.add_enabled(send_enabled,
                    egui::Button::new(egui::RichText::new("Send (Enter)").size(13.0))
                        .fill(if send_enabled { egui::Color32::from_rgb(0x00, 0x55, 0xaa) } else { egui::Color32::from_rgb(0x33, 0x33, 0x33) })
                        .corner_radius(egui::CornerRadius::same(6))
                ).on_hover_text("Send message (Enter to send, Shift+Enter for newline)");
                if send_btn.clicked() {
                    self.send_message(_models, selected_model, role_prompt, slot_idx, api_client, tx, rt);
                }

                if self.is_streaming {
                    ui.add_space(4.0);
                    let stop_btn = ui.add(
                        egui::Button::new(egui::RichText::new("Stop").size(13.0))
                                .fill(egui::Color32::from_rgb(0xaa, 0x33, 0x33))
                                .corner_radius(egui::CornerRadius::same(6))
                    ).on_hover_text("Stop generation");
                    if stop_btn.clicked() {
                        self.is_streaming = false;
                    }
                }
            });
        });
        ui.add_space(8.0);
    }

    fn show_message(&self, ui: &mut egui::Ui, msg: &ChatMessage) {
        let is_user = msg.role == "user";
        let (bg_color, align) = if is_user {
            (egui::Color32::from_rgb(0x00, 0x44, 0x88), egui::Align::RIGHT)
        } else if msg.role == "system" {
            (egui::Color32::from_rgb(0x44, 0x33, 0x00), egui::Align::LEFT)
        } else {
            (egui::Color32::from_rgb(0x2d, 0x2d, 0x2d), egui::Align::LEFT)
        };
        ui.with_layout(egui::Layout::top_down(align), |ui| {
            ui.add_space(4.0);
            egui::Frame::NONE
                .fill(bg_color)
                .corner_radius(egui::CornerRadius::same(6))
                .inner_margin(egui::Margin::same(8))
                .show(ui, |ui| {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(
                            egui::RichText::new(msg.timestamp.format("%H:%M").to_string())
                                .size(11.0)
                                .color(egui::Color32::from_rgb(0xaa, 0xaa, 0xaa)),
                        );
                    });
                    ui.label(
                        egui::RichText::new(&msg.content)
                            .size(13.0)
                            .color(egui::Color32::WHITE),
                    );
                });
        });
        ui.add_space(4.0);
    }

    fn send_message(
        &mut self,
        _models: &[crate::ollama::api::Model],
        selected_model: &Option<String>,
        role_prompt: &str,
        slot_idx: usize,
        api_client: &Option<OllamaClient>,
        tx: &mpsc::Sender<crate::ui::app::AppMessage>,
        rt: &Runtime,
    ) {
        let Some(model_name) = selected_model else { return };
        let Some(client) = api_client else { return };

        // Trim the newline Enter inserts before the send triggers, and refuse
        // empty sends (the Enter path bypasses the button's enabled guard).
        let prompt = self.input.trim().to_string();
        if prompt.is_empty() || self.is_streaming {
            return;
        }

        let user_msg = ChatMessage {
            role: "user".to_string(),
            content: prompt.clone(),
            timestamp: chrono::Utc::now(),
        };
        self.messages.push(user_msg);
        self.input.clear();
        let model_name = model_name.clone();
        let client = client.clone();
        let tx = tx.clone();

        self.is_streaming = true;

        let role_prompt = role_prompt.to_string();
        rt.spawn(async move {
            let mut messages = Vec::new();
            if !role_prompt.trim().is_empty() {
                messages.push(Message {
                    role: "system".to_string(),
                    content: role_prompt,
                });
            }
            messages.push(Message {
                role: "user".to_string(),
                content: prompt,
            });
            let req = ChatRequest {
                model: model_name,
                messages,
                stream: false,
                options: None,
                keep_alive: Some("10m".to_string()),
            };

            let result = client.chat(req).await;
            let _ = tx.send(crate::ui::app::AppMessage::ChatResponse(slot_idx, result));
        });
    }

    pub fn handle_response(&mut self, response: Result<ChatResponse>) {
        self.is_streaming = false;
        match response {
            Ok(resp) => {
                let msg = ChatMessage {
                    role: "assistant".to_string(),
                    content: resp.message.content,
                    timestamp: chrono::Utc::now(),
                };
                self.messages.push(msg);
            }
            Err(e) => {
                let msg = ChatMessage {
                    role: "system".to_string(),
                    content: format!("Request failed: {e}"),
                    timestamp: chrono::Utc::now(),
                };
                self.messages.push(msg);
            }
        }
    }
}
