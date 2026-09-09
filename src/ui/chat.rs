use eframe::egui;
use anyhow::Result;
use crate::ollama::api::{ChatOptions, ChatRequest, Message, OllamaClient, ChatResponse};
use crate::security::{ConfirmGate, SecretHit};
use crate::storage::ChatMessage;
use std::sync::mpsc;
use std::time::Instant;
use tokio::runtime::Runtime;

pub struct ChatPanel {
    input: String,
    messages: Vec<ChatMessage>,
    is_streaming: bool,
    voice_enabled: bool,
    revision: u64,
    send_started: Option<Instant>,
    stream_buf: String,
    stream_seq: u64,
    gate: ConfirmGate,
}

impl ChatPanel {
    pub fn new() -> Self {
        Self {
            input: String::new(),
            messages: Vec::new(),
            is_streaming: false,
            voice_enabled: false,
            revision: 0,
            send_started: None,
            stream_buf: String::new(),
            stream_seq: 0,
            gate: ConfirmGate::new(),
        }
    }

    /// In-memory bounds: render cap is 100, but the Vec itself must also be
    /// bounded or long sessions grow RAM + sled blobs without limit.
    pub const MAX_MESSAGES: usize = 500;
    /// Single-message content cap (chars). Guards against a runaway model
    /// streaming megabytes into RAM and into the sled blob on save.
    pub const MAX_CONTENT_CHARS: usize = 50_000;

    pub fn clear_chat(&mut self) {
        self.messages.clear();
        self.messages.shrink_to_fit();
        self.input.clear();
        self.revision = self.revision.saturating_add(1);
    }

    fn push_capped(&mut self, msg: ChatMessage) {
        if self.messages.len() >= Self::MAX_MESSAGES {
            let overflow = self.messages.len() - Self::MAX_MESSAGES + 1;
            self.messages.drain(0..overflow);
        }
        self.messages.push(msg);
        self.revision = self.revision.saturating_add(1);
    }

    fn cap_content(s: &str) -> String {
        if s.len() > Self::MAX_CONTENT_CHARS {
            // char-boundary safe truncation + marker
            let mut end = Self::MAX_CONTENT_CHARS;
            while !s.is_char_boundary(end) { end -= 1; }
            format!("{}…[truncated {} chars]", &s[..end], s.len() - end)
        } else {
            s.to_string()
        }
    }

    pub fn messages(&self) -> &[ChatMessage] {
        &self.messages
    }

    /// Bumped on every local change; the app persists the chat when it differs.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// System notice bubble (guard blocks, etc.).
    pub fn push_system_note(&mut self, content: String) {
        self.push_capped(ChatMessage {
            role: "system".to_string(),
            content,
            timestamp: chrono::Utc::now(),
        });
    }

    fn secret_warning(hits: &[SecretHit]) -> String {
        let mut lines = vec![format!(
            "Blocked: looks like {} secret{} — resend unchanged within 60s to override:",
            if hits.len() == 1 { "a" } else { "several" },
            if hits.len() == 1 { "" } else { "s" }
        )];
        for h in hits {
            lines.push(format!("  \u{2022} {} ({})", h.kind, h.preview));
        }
        lines.join("\n")
    }

    /// Broadcast pre-check on this panel's gate. Err = formatted warning.
    pub fn broadcast_check(&mut self, prompt: &str) -> Result<(), String> {
        match self.gate.check(prompt) {
            Ok(()) => Ok(()),
            Err(hits) => Err(Self::secret_warning(&hits)),
        }
    }

    /// Current stream generation; the send task tags its chunks with this.
    pub fn stream_seq(&self) -> u64 {
        self.stream_seq
    }

    /// Live token piece from the streaming task. Stale generations (after
    /// Stop / resend) are ignored via the sequence number.
    pub fn push_chunk(&mut self, seq: u64, piece: &str) {
        if seq != self.stream_seq || piece.is_empty() {
            return;
        }
        self.is_streaming = true;
        if self.stream_buf.len() < Self::MAX_CONTENT_CHARS {
            let room = Self::MAX_CONTENT_CHARS - self.stream_buf.len();
            let mut end = piece.len().min(room);
            while !piece.is_char_boundary(end) {
                end -= 1;
            }
            self.stream_buf.push_str(&piece[..end]);
        }
    }

    /// User-pressed Stop: invalidate in-flight chunks, drop the partial.
    pub fn stop_stream(&mut self) {
        self.stream_seq = self.stream_seq.saturating_add(1);
        self.is_streaming = false;
        self.stream_buf.clear();
    }

    /// Replace chat contents with a stored session (History -> Open in chat).
    pub fn load_messages(&mut self, msgs: Vec<ChatMessage>) {
        let start = msgs.len().saturating_sub(Self::MAX_MESSAGES);
        self.messages = msgs[start..].to_vec();
        self.revision = self.revision.saturating_add(1);
    }

    pub fn set_voice_enabled(&mut self, enabled: bool) {
        self.voice_enabled = enabled;
    }

    /// Dictated/transcribed text lands in the input box for review before send.
    pub fn append_input(&mut self, text: &str) {
        let t = text.trim();
        if t.is_empty() {
            return;
        }
        if !self.input.trim().is_empty() {
            self.input.push(' ');
        }
        self.input.push_str(t);
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
                let _ = tx.send(crate::ui::app::AppMessage::VoiceToggled(
                    !self.voice_enabled,
                ));
            }
            let mic_btn = ui
                .add(
                    egui::Button::new("Dictate")
                        .corner_radius(egui::CornerRadius::same(6)),
                )
                .on_hover_text("Record 5s from mic and transcribe into the input box");
            if mic_btn.clicked() {
                let _ = tx.send(crate::ui::app::AppMessage::VoiceListen(slot_idx));
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

        // Message history. Reserve room for the input block below (field +
        // Send row + status + separators ≈ 170px at 100%): the list may use
        // everything ABOVE the reserve but never more, so Send always has
        // room. Scales with the Settings zoom factor so large text can't
        // overflow the reserve.
        let input_reserve = 170.0 * ui.ctx().zoom_factor();
        let list_h = (ui.available_height() - input_reserve).max(80.0);
        egui::ScrollArea::vertical()
            .max_height(list_h)
            .stick_to_bottom(true)
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
                let total = self.messages.len();
                if total > 100 {
                    ui.horizontal(|ui| {
                        ui.add_space(4.0);
                        ui.label(
                            egui::RichText::new(format!(
                                "Showing last 100 of {} messages (older kept for the model, hidden for speed).",
                                total
                            ))
                            .size(11.0)
                            .color(egui::Color32::from_rgb(0x99, 0x99, 0x99)),
                        );
                    });
                }
                let start = total.saturating_sub(100);
                for i in start..total {
                    self.show_message(ui, &self.messages[i], tx);
                }
                if !self.stream_buf.is_empty() {
                    let tmp = ChatMessage {
                        role: "assistant".to_string(),
                        content: format!("{}▍", self.stream_buf),
                        timestamp: chrono::Utc::now(),
                    };
                    self.show_message(ui, &tmp, tx);
                }
                if self.is_streaming && self.stream_buf.is_empty() {
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
        // Full-width field on its own row; Send/Stop share a right-aligned
        // row beneath it. (Side by side proved unworkable: in a horizontal
        // row both the multiline field and the button column expand to full
        // width, so they wrapped unpredictably.)
        let response = ui.add(
            egui::TextEdit::multiline(&mut self.input)
                .desired_rows(3)
                .desired_width(f32::INFINITY)
                .hint_text("Type your message... (Enter to send, Shift+Enter for newline)")
                .font(egui::TextStyle::Body),
        );
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let need_model = selected_model.is_none();
                let need_link = api_client.is_none();
                let send_enabled = !self.input.trim().is_empty()
                    && !need_model
                    && !need_link
                    && !self.is_streaming;
                // The button is ALWAYS drawn bright enough to spot. When it
                // cannot send yet, the label names the missing piece.
                let label = if self.is_streaming {
                    "Working..."
                } else if need_model {
                    "Send (pick model ^)"
                } else if need_link {
                    "Send (no Ollama link)"
                } else {
                    "Send (Enter)"
                };
                let fill = if send_enabled {
                    egui::Color32::from_rgb(0x00, 0x66, 0xcc)
                } else {
                    egui::Color32::from_rgb(0x1a, 0x3a, 0x66)
                };
                let send_btn = ui.add_enabled(send_enabled,
                    egui::Button::new(egui::RichText::new(label).size(13.0).color(egui::Color32::WHITE))
                        .fill(fill)
                        .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(0x44, 0xaa, 0xff)))
                        .corner_radius(egui::CornerRadius::same(6))
                        .min_size(egui::vec2(112.0, 34.0))
                ).on_hover_text("Send message (Enter to send, Shift+Enter for newline)");
                if send_btn.clicked() {
                self.send_message(_models, selected_model, role_prompt, slot_idx, api_client, tx, rt);
                }
                ui.add_space(4.0);
                let ask_btn = ui
                    .add(
                        egui::Button::new(egui::RichText::new("Ask all").size(13.0))
                            .fill(egui::Color32::from_rgb(0x00, 0x55, 0x55))
                            .corner_radius(egui::CornerRadius::same(6)),
                    )
                    .on_hover_text("Send this input to every slot with a model");
                if ask_btn.clicked() {
                    if let Some(prompt) = self.take_broadcast() {
                        let _ = tx.send(crate::ui::app::AppMessage::Broadcast(prompt));
                    }
                }

                if self.is_streaming {
                    ui.add_space(4.0);
                    let stop_btn = ui.add(
                        egui::Button::new(egui::RichText::new("Stop").size(13.0))
                                .fill(egui::Color32::from_rgb(0xaa, 0x33, 0x33))
                                .corner_radius(egui::CornerRadius::same(6))
                    ).on_hover_text("Stop generation");
                    if stop_btn.clicked() {
                        self.stop_stream();
                        let _ = tx.send(crate::ui::app::AppMessage::StopStream(slot_idx));
                    }
                }
            });
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
        });
        ui.add_space(4.0);
        // Status line: always shows what this chat needs to work.
        ui.horizontal(|ui| {
            ui.add_space(4.0);
            let model_txt = selected_model.clone().unwrap_or("(no model)".to_string());
            let link_txt = if api_client.is_some() { "Ollama: linked" } else { "Ollama: NOT linked" };
            ui.label(
                egui::RichText::new(format!(
                    "Slot {} -> {} | {} | {} model(s) listed",
                    slot_idx + 1,
                    model_txt,
                    link_txt,
                    _models.len(),
                ))
                .size(11.0)
                .color(egui::Color32::from_rgb(0x99, 0x99, 0x99)),
            );
        });
        ui.add_space(8.0);
    }

    /// Split ```fences into (lang, code) blocks.
    fn code_blocks(content: &str) -> Vec<(String, String)> {
        let mut out = Vec::new();
        let mut rest = content;
        while let Some(s) = rest.find("```") {
            let after = &rest[s + 3..];
            let (lang, code_start) = match after.find('\n') {
                Some(i) => (after[..i].trim().to_string(), &after[i + 1..]),
                None => (String::new(), after),
            };
            match code_start.find("```") {
                Some(e) => {
                    out.push((lang, code_start[..e].trim_matches('\n').to_string()));
                    rest = &code_start[e + 3..];
                }
                None => break,
            }
        }
        out
    }

    fn show_message(
        &self,
        ui: &mut egui::Ui,
        msg: &ChatMessage,
        tx: &mpsc::Sender<crate::ui::app::AppMessage>,
    ) {
        let is_user = msg.role == "user";
        let (bg_color, align) = if is_user {
            (egui::Color32::from_rgb(0x00, 0x44, 0x88), egui::Align::RIGHT)
        } else if msg.role == "system" {
            (egui::Color32::from_rgb(0x44, 0x33, 0x00), egui::Align::LEFT)
        } else {
            (egui::Color32::from_rgb(0x2d, 0x2d, 0x2d), egui::Align::LEFT)
        };
        let who = if is_user {
            "You"
        } else if msg.role == "system" {
            "System"
        } else {
            "Assistant"
        };
        let blocks = Self::code_blocks(&msg.content);
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
                        ui.label(
                            egui::RichText::new(who)
                                .size(11.0)
                                .color(egui::Color32::from_rgb(0x00, 0xaa, 0xff)),
                        );
                    });
                    if blocks.is_empty() {
                        ui.label(
                            egui::RichText::new(&msg.content)
                                .size(13.0)
                                .color(egui::Color32::WHITE),
                        );
                    } else {
                        let mut rest = msg.content.as_str();
                        let mut bi = 0usize;
                        while let Some(s) = rest.find("```") {
                            let before = rest[..s].trim();
                            if !before.is_empty() {
                                ui.label(
                                    egui::RichText::new(before)
                                        .size(13.0)
                                        .color(egui::Color32::WHITE),
                                );
                            }
                            if bi < blocks.len() {
                                let (lang, code) = &blocks[bi];
                                if !lang.is_empty() {
                                    ui.label(
                                        egui::RichText::new(lang.clone())
                                            .size(11.0)
                                            .color(egui::Color32::from_rgb(0x00, 0xcc, 0x88)),
                                    );
                                }
                                egui::ScrollArea::horizontal().show(ui, |ui| {
                                    ui.code(code.clone());
                                });
                                bi += 1;
                            }
                            let after = &rest[s + 3..];
                            match after.find("```") {
                                Some(e) => rest = &after[e + 3..],
                                None => {
                                    rest = "";
                                    break;
                                }
                            }
                        }
                        let tail = rest.trim();
                        if !tail.is_empty() {
                            ui.label(
                                egui::RichText::new(tail)
                                    .size(13.0)
                                    .color(egui::Color32::WHITE),
                            );
                        }
                    }
                    if !is_user && msg.role == "assistant" && !blocks.is_empty() {
                        ui.add_space(4.0);
                        ui.horizontal(|ui| {
                            if ui.small_button("Copy code").clicked() {
                                if let Some((_, code)) = blocks.iter().max_by_key(|(_, c)| c.len()) {
                                    ui.ctx().copy_text(code.clone());
                                }
                            }
                            if ui
                                .small_button("Send to Editor")
                                .on_hover_text("Open the biggest code block in the Editor tab")
                                .clicked()
                            {
                                if let Some((lang, code)) =
                                    blocks.iter().max_by_key(|(_, c)| c.len())
                                {
                                    let _ = tx.send(crate::ui::app::AppMessage::ChatToEditor(
                                        code.clone(),
                                        lang.clone(),
                                    ));
                                }
                            }
                        });
                    }
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
        // Trim the newline Enter inserts before the send triggers (the Enter
        // path bypasses the button's enabled guard).
        let prompt = self.input.trim().to_string();
        if prompt.is_empty() {
            return;
        }
        if let Err(hits) = self.gate.check(&prompt) {
            self.push_system_note(Self::secret_warning(&hits));
            let _ = tx.send(crate::ui::app::AppMessage::Audit(
                "secret.blocked".to_string(),
                format!("chat slot {}", slot_idx + 1),
            ));
            return;
        }
        if self.send_prompt(
            prompt,
            _models,
            selected_model,
            role_prompt,
            slot_idx,
            api_client,
            tx,
            rt,
        ) {
            self.input.clear();
        }
    }

    /// Drain the input for a broadcast. None when empty/streaming.
    pub fn take_broadcast(&mut self) -> Option<String> {
        let prompt = self.input.trim().to_string();
        if prompt.is_empty() || self.is_streaming {
            return None;
        }
        self.input.clear();
        Some(prompt)
    }

    /// Send an explicit prompt (single Sends and broadcasts share this).
    /// False when skipped (no model/link, already streaming).
    pub fn send_prompt(
        &mut self,
        prompt: String,
        _models: &[crate::ollama::api::Model],
        selected_model: &Option<String>,
        role_prompt: &str,
        slot_idx: usize,
        api_client: &Option<OllamaClient>,
        tx: &mpsc::Sender<crate::ui::app::AppMessage>,
        rt: &Runtime,
    ) -> bool {
        let Some(model_name) = selected_model else { return false };
        let Some(client) = api_client else { return false };

        let prompt = prompt.trim().to_string();
        if prompt.is_empty() || self.is_streaming {
            return false;
        }

        let prompt = if prompt.len() > Self::MAX_CONTENT_CHARS {
            let mut end = Self::MAX_CONTENT_CHARS;
            while !prompt.is_char_boundary(end) { end -= 1; }
            prompt[..end].to_string()
        } else { prompt };
        let user_msg = ChatMessage {
            role: "user".to_string(),
            content: prompt.clone(),
            timestamp: chrono::Utc::now(),
        };
        self.push_capped(user_msg);
        self.input.clear();
        let model_name = model_name.clone();
        let client = client.clone();
        let tx = tx.clone();

        // Keep continuity: resend recent turns so the model sees persona +
        // memory (in role_prompt) AND the conversation so far.
        let history: Vec<(String, String)> = self.messages.iter().rev().take(20).rev()
            .map(|m| (m.role.clone(), m.content.clone())).collect();
        self.is_streaming = true;
        self.stream_buf.clear();
        self.stream_seq = self.stream_seq.saturating_add(1);
        let seq = self.stream_seq;
        self.send_started = Some(Instant::now());

        let role_prompt = role_prompt.to_string();
        let tx_handle = tx.clone();
        let h = rt.spawn(async move {
            let mut messages = Vec::new();
            if !role_prompt.trim().is_empty() {
                messages.push(Message {
                    role: "system".to_string(),
                    content: role_prompt,
                });
            }
            for (role, content) in history {
                // Only user/assistant turns; skip old error lines.
                if role == "user" || role == "assistant" {
                    messages.push(Message { role, content });
                }
            }
            messages.push(Message {
                role: "user".to_string(),
                content: prompt,
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
                    let _ = tx.send(crate::ui::app::AppMessage::ChatChunk(
                        slot_idx,
                        seq,
                        piece.to_string(),
                    ));
                })
                .await;
            let _ = tx.send(crate::ui::app::AppMessage::ChatResponse(
                slot_idx, seq, result,
            ));
        });
        let _ = tx_handle.send(crate::ui::app::AppMessage::StreamHandle(slot_idx, h));
        true
    }

    /// Returns (success, elapsed_secs, response_chars) as a training signal.
    pub fn handle_response(&mut self, response: Result<ChatResponse>) -> (bool, f32, usize) {
        self.is_streaming = false;
        self.stream_buf.clear();
        let elapsed = self
            .send_started
            .map(|t| t.elapsed().as_secs_f32())
            .unwrap_or(0.0);
        self.send_started = None;
        match response {
            Ok(resp) => {
                let n = resp.message.content.len();
                let msg = ChatMessage {
                    role: "assistant".to_string(),
                    content: Self::cap_content(&resp.message.content),
                    timestamp: chrono::Utc::now(),
                };
                self.push_capped(msg);
                (true, elapsed, n)
            }
            Err(e) => {
                // No prefix: OllamaError already describes itself
                // ("API error: ...", "Request failed: ...").
                let em = format!("{e}");
                let msg = ChatMessage {
                    role: "system".to_string(),
                    content: Self::cap_content(&em),
                    timestamp: chrono::Utc::now(),
                };
                self.push_capped(msg);
                (false, elapsed, 0)
            }
        }
    }
}
