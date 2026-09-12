// Copyright 2026 Sean M. Stow. All rights reserved.
use eframe::egui;
use anyhow::Result;
use crate::ollama::api::{ChatOptions, ChatRequest, Message, OllamaClient, ChatResponse};
use crate::security::{ConfirmGate, SecretHit};
use crate::storage::ChatMessage;
use std::borrow::Cow;
use std::sync::mpsc;
use std::time::Instant;
use tokio::runtime::Runtime;

/// Test if text contains characters that require sanitization.
/// Scans ASCII control characters (< 0x20, 0x7f) excluding \n, \r, \t,
/// ANSI escape prefixes (0x1b), and Unicode replacement characters (\u{FFFD}).
pub fn needs_sanitization(s: &str) -> bool {
    s.as_bytes().iter().any(|&b| b == 0x1b || (b < 0x20 && b != b'\n' && b != b'\r' && b != b'\t') || b == 0x7f)
        || s.contains('\u{FFFD}')
}

/// Zero-copy sanitization wrapper: returns Cow::Borrowed if no invalid characters exist,
/// eliminating heap allocation churn for >99% of streaming tokens.
pub fn sanitize_text_cow<'a>(s: &'a str) -> Cow<'a, str> {
    if !needs_sanitization(s) {
        return Cow::Borrowed(s);
    }
    Cow::Owned(sanitize_text(s))
}

/// Strip ANSI escape sequences, replacement characters (\u{FFFD}),
/// and unprintable control characters, preserving \n, \r, \t.
pub fn sanitize_text(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\x1b' {
            // ANSI escape sequence: \x1b[ ... letter
            if let Some(&'[') = chars.peek() {
                chars.next(); // consume '['
                while let Some(&next_ch) = chars.peek() {
                    chars.next();
                    if next_ch.is_ascii_alphabetic() {
                        break;
                    }
                }
                continue;
            }
        }
        if ch == '\u{FFFD}' {
            continue;
        }
        if ch.is_control() && ch != '\n' && ch != '\r' && ch != '\t' {
            continue;
        }
        out.push(ch);
    }
    out
}

#[derive(Debug, Clone, PartialEq)]
pub enum MessageSegment {
    Text(String),
    Think(String),
    Code { lang: String, code: String },
}

/// Helper that splits text with markdown code blocks into Text and Code segments.
fn parse_text_and_code(input: &str) -> Vec<MessageSegment> {
    let mut segments = Vec::new();
    let mut rest = input;

    while let Some(start_idx) = rest.find("```") {
        let before = &rest[..start_idx];
        if !before.is_empty() {
            segments.push(MessageSegment::Text(before.to_string()));
        }

        let after_fence = &rest[start_idx + 3..];
        if let Some(nl_pos) = after_fence.find('\n') {
            let lang = after_fence[..nl_pos].trim().to_string();
            let code_content = &after_fence[nl_pos + 1..];
            if let Some(end_idx) = code_content.find("```") {
                let code = code_content[..end_idx].trim_matches('\n').to_string();
                segments.push(MessageSegment::Code { lang, code });
                rest = &code_content[end_idx + 3..];
            } else {
                // In-progress / unclosed code block (e.g. while streaming)
                let code = code_content.trim_matches('\n').to_string();
                segments.push(MessageSegment::Code { lang, code });
                rest = "";
                break;
            }
        } else {
            // Started fence and language name, but no newline yet (e.g. "```rust")
            let lang = after_fence.trim().to_string();
            segments.push(MessageSegment::Code { lang, code: String::new() });
            rest = "";
            break;
        }
    }

    if !rest.is_empty() {
        segments.push(MessageSegment::Text(rest.to_string()));
    }

    segments
}

/// Parse full message into Text, Think, and Code segments.
pub fn parse_segments(content: &str) -> Vec<MessageSegment> {
    let mut segments = Vec::new();
    let mut rest = content;

    while let Some(think_start) = rest.find("<think>") {
        let before_think = &rest[..think_start];
        if !before_think.is_empty() {
            segments.extend(parse_text_and_code(before_think));
        }

        let after_think_tag = &rest[think_start + 7..];
        if let Some(think_end) = after_think_tag.find("</think>") {
            let think_content = after_think_tag[..think_end].trim();
            if !think_content.is_empty() {
                segments.push(MessageSegment::Think(think_content.to_string()));
            }
            rest = &after_think_tag[think_end + 8..];
        } else {
            // Active unclosed think block (streaming)
            let think_content = after_think_tag.trim();
            if !think_content.is_empty() {
                segments.push(MessageSegment::Think(think_content.to_string()));
            }
            rest = "";
            break;
        }
    }

    if !rest.is_empty() {
        segments.extend(parse_text_and_code(rest));
    }

    segments
}

/// High-contrast, beautifully styled code block frame with language badge,
/// copy to clipboard, and one-click transfer to the IDE Editor tab.
pub fn render_code_block(
    ui: &mut egui::Ui,
    lang: &str,
    code: &str,
    block_id: &str,
    tx: &mpsc::Sender<crate::ui::app::AppMessage>,
) {
    ui.add_space(4.0);
    let frame = egui::Frame::NONE
        .fill(egui::Color32::from_rgb(0x0a, 0x0f, 0x1d))
        .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(0x1e, 0x2d, 0x48)))
        .corner_radius(egui::CornerRadius::same(6))
        .inner_margin(egui::Margin::symmetric(10, 8));

    frame.show(ui, |ui| {
        ui.horizontal(|ui| {
            let display_lang = if lang.trim().is_empty() {
                "CODE".to_string()
            } else {
                lang.trim().to_uppercase()
            };
            ui.label(
                egui::RichText::new(format!("💻 {}", display_lang))
                    .size(11.0)
                    .monospace()
                    .color(egui::Color32::from_rgb(0x38, 0xbd, 0xf8)),
            );

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let is_shell = matches!(lang.trim().to_lowercase().as_str(), "bash" | "sh" | "shell" | "zsh");
                if is_shell {
                    let run_btn = ui.small_button(
                        egui::RichText::new("⚡ Run")
                            .size(11.0)
                            .color(egui::Color32::from_rgb(0x34, 0xd3, 0x99)),
                    ).on_hover_text("Execute shell command directly in the IDE Terminal Dock");
                    if run_btn.clicked() {
                        let _ = tx.send(crate::ui::app::AppMessage::TerminalRun(code.to_string()));
                    }
                    ui.add_space(6.0);
                }

                let send_btn = ui.small_button(
                    egui::RichText::new("📝 Send to Editor")
                        .size(11.0)
                        .color(egui::Color32::from_rgb(0xa5, 0xb4, 0xfc)),
                ).on_hover_text("Open this code block in the IDE / Editor tab");
                if send_btn.clicked() {
                    let _ = tx.send(crate::ui::app::AppMessage::ChatToEditor(
                        code.to_string(),
                        lang.to_string(),
                    ));
                }

                ui.add_space(6.0);
                let copy_btn = ui.small_button(
                    egui::RichText::new("📋 Copy")
                        .size(11.0)
                        .color(egui::Color32::from_rgb(0x94, 0xa3, 0xb8)),
                ).on_hover_text("Copy code to clipboard");
                if copy_btn.clicked() {
                    ui.ctx().copy_text(code.to_string());
                }
            });
        });

        ui.add_space(4.0);
        ui.separator();
        ui.add_space(4.0);

        egui::ScrollArea::horizontal()
            .id_salt(block_id)
            .auto_shrink([false, false])
            .show(ui, |ui| {
                ui.add(
                    egui::Label::new(
                        egui::RichText::new(code)
                            .monospace()
                            .size(12.5)
                            .color(egui::Color32::from_rgb(0xec, 0xf0, 0xf8)),
                    )
                );
            });

        ui.add_space(2.0);
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("🛠 Quick Tools:").size(10.0).color(egui::Color32::from_rgb(0x77, 0x88, 0x99)));
            let edit_chip = ui.small_button(egui::RichText::new("Gedit").size(10.0));
            if edit_chip.clicked() {
                let _ = tx.send(crate::ui::app::AppMessage::RunTool {
                    id: "editor".to_string(),
                    args: None,
                });
            }
            let code_chip = ui.small_button(egui::RichText::new("VS Code").size(10.0));
            if code_chip.clicked() {
                let _ = tx.send(crate::ui::app::AppMessage::RunTool {
                    id: "code".to_string(),
                    args: None,
                });
            }
            let hex_chip = ui.small_button(egui::RichText::new("Hexdump").size(10.0));
            if hex_chip.clicked() {
                let _ = tx.send(crate::ui::app::AppMessage::RunTool {
                    id: "hexdump".to_string(),
                    args: None,
                });
            }
        });
    });
    ui.add_space(4.0);
}

pub struct ChatPanel {
    pub input: String,
    messages: Vec<ChatMessage>,
    parsed_cache: Vec<Vec<MessageSegment>>,
    is_streaming: bool,
    voice_enabled: bool,
    revision: u64,
    send_started: Option<Instant>,
    stream_buf: String,
    stream_seq: u64,
    gate: ConfirmGate,
    last_reply_secs: Option<f32>,
    reply_secs_total: f32,
    reply_count: u32,
    pub history_depth: usize,
    pub num_threads: u32,
    cached_stream_segs: Vec<MessageSegment>,
    stream_dirty: bool,
}

impl ChatPanel {
    pub fn new() -> Self {
        Self {
            input: String::new(),
            messages: Vec::new(),
            parsed_cache: Vec::new(),
            is_streaming: false,
            voice_enabled: false,
            revision: 0,
            send_started: None,
            stream_buf: String::new(),
            stream_seq: 0,
            gate: ConfirmGate::new(),
            last_reply_secs: None,
            reply_secs_total: 0.0,
            reply_count: 0,
            history_depth: 20,
            num_threads: 0,
            cached_stream_segs: Vec::new(),
            stream_dirty: false,
        }
    }

    /// In-memory bounds: render cap is 100, but the Vec itself must also be
    /// bounded or long sessions grow RAM + sled blobs without limit.
    pub const MAX_MESSAGES: usize = 500;
    /// Single-message content cap (chars). Guards against a runaway model
    /// streaming megabytes into RAM and into the sled blob on save.
    pub const MAX_CONTENT_CHARS: usize = 50_000;

    /// Copy conversation history into a fresh panel for slot cloning:
    /// messages (and the latency chip) carry over, live stream/gate/input do not.
    pub fn carry_messages(&self) -> Self {
        Self {
            input: String::new(),
            messages: self.messages.clone(),
            parsed_cache: if self.parsed_cache.len() == self.messages.len() {
                self.parsed_cache.clone()
            } else {
                self.messages.iter().map(|m| parse_segments(&m.content)).collect()
            },
            is_streaming: false,
            voice_enabled: self.voice_enabled,
            revision: 0,
            send_started: None,
            stream_buf: String::new(),
            stream_seq: 0,
            gate: ConfirmGate::new(),
            last_reply_secs: self.last_reply_secs,
            reply_secs_total: self.reply_secs_total,
            reply_count: self.reply_count,
            history_depth: self.history_depth,
            num_threads: self.num_threads,
            cached_stream_segs: Vec::new(),
            stream_dirty: false,
        }
    }

    pub fn clear_chat(&mut self) {
        self.last_reply_secs = None;
        self.reply_secs_total = 0.0;
        self.reply_count = 0;
        self.messages.clear();
        self.messages.shrink_to_fit();
        self.parsed_cache.clear();
        self.parsed_cache.shrink_to_fit();
        self.cached_stream_segs.clear();
        self.stream_dirty = false;
        self.input.clear();
        self.revision = self.revision.saturating_add(1);
    }

    fn push_capped(&mut self, msg: ChatMessage) {
        if self.messages.len() >= Self::MAX_MESSAGES {
            let overflow = self.messages.len() - Self::MAX_MESSAGES + 1;
            self.messages.drain(0..overflow);
            if self.parsed_cache.len() >= overflow {
                self.parsed_cache.drain(0..overflow);
            } else {
                self.parsed_cache.clear();
            }
        }
        let segs = parse_segments(&msg.content);
        self.parsed_cache.push(segs);
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
    pub fn push_system_note(&mut self, content: impl Into<String>) {
        self.push_capped(ChatMessage {
            role: "system".to_string(),
            content: content.into(),
            timestamp: chrono::Utc::now(),
        });
    }

    /// Push an assistant message (e.g. imported from Swarm Relay).
    pub fn push_assistant_message(&mut self, content: String) {
        self.push_capped(ChatMessage {
            role: "assistant".to_string(),
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
    pub fn is_streaming(&self) -> bool {
        self.is_streaming
    }

    pub fn stream_seq(&self) -> u64 {
        self.stream_seq
    }

    /// Live token piece from the streaming task. Stale generations (after
    /// Stop / resend) are ignored via the sequence number.
    pub fn push_chunk(&mut self, seq: u64, piece: &str) {
        if seq != self.stream_seq || piece.is_empty() {
            return;
        }
        let clean = sanitize_text_cow(piece);
        if clean.is_empty() {
            return;
        }
        self.is_streaming = true;
        self.stream_dirty = true;
        if self.stream_buf.len() < Self::MAX_CONTENT_CHARS {
            let room = Self::MAX_CONTENT_CHARS - self.stream_buf.len();
            let mut end = clean.len().min(room);
            while !clean.is_char_boundary(end) {
                end -= 1;
            }
            self.stream_buf.push_str(&clean[..end]);
        }
    }

    /// User-pressed Stop: invalidate in-flight chunks, drop the partial.
    pub fn stop_stream(&mut self) {
        self.stream_seq = self.stream_seq.saturating_add(1);
        self.is_streaming = false;
        self.stream_buf.clear();
        self.cached_stream_segs.clear();
        self.stream_dirty = false;
    }

    /// Replace chat contents with a stored session (History -> Open in chat).
    pub fn load_messages(&mut self, msgs: Vec<ChatMessage>) {
        let start = msgs.len().saturating_sub(Self::MAX_MESSAGES);
        self.messages = msgs[start..].to_vec();
        self.parsed_cache = self.messages.iter().map(|m| parse_segments(&m.content)).collect();
        self.revision = self.revision.saturating_add(1);
    }

    pub fn voice_enabled(&self) -> bool {
        self.voice_enabled
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

    /// Message list view that can be embedded in split view columns or full view.
    pub fn show_message_list(
        &mut self,
        ui: &mut egui::Ui,
        tx: &mpsc::Sender<crate::ui::app::AppMessage>,
        max_height: f32,
    ) {
        egui::ScrollArea::vertical()
            .max_height(max_height)
            .auto_shrink([false, false])
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
                if total > 50 {
                    ui.horizontal(|ui| {
                        ui.add_space(4.0);
                        ui.label(
                            egui::RichText::new(format!(
                                "Showing last 50 of {} messages (older kept for the model, hidden for speed).",
                                total
                            ))
                            .size(11.0)
                            .color(egui::Color32::from_rgb(0x99, 0x99, 0x99)),
                        );
                    });
                }
                let start = total.saturating_sub(50);
                for i in start..total {
                    let segs: &[MessageSegment] = if i < self.parsed_cache.len() {
                        &self.parsed_cache[i]
                    } else {
                        &[]
                    };
                    self.show_message(ui, &self.messages[i], segs, tx);
                }
                if !self.stream_buf.is_empty() {
                    if self.stream_dirty || self.cached_stream_segs.is_empty() {
                        let tmp_content = format!("{}▍", self.stream_buf);
                        self.cached_stream_segs = parse_segments(&tmp_content);
                        self.stream_dirty = false;
                    }
                    let tmp = ChatMessage {
                        role: "assistant".to_string(),
                        content: format!("{}▍", self.stream_buf),
                        timestamp: chrono::Utc::now(),
                    };
                    self.show_message(ui, &tmp, &self.cached_stream_segs, tx);
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
        dual_run_mode: bool,
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
            if ui
                .small_button("Stop voice")
                .on_hover_text("Stop read-aloud playback")
                .clicked()
            {
                let _ = tx.send(crate::ui::app::AppMessage::StopSpeak);
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
        self.show_message_list(ui, tx, list_h);

        ui.separator();
        ui.add_space(4.0);

        // Explicit ID salt prevents widget ID shifts during live typing
        let response = ui.add(
            egui::TextEdit::multiline(&mut self.input)
                .id_salt(format!("chat_input_textedit_slot_{}", slot_idx))
                .desired_rows(3)
                .desired_width(f32::INFINITY)
                .hint_text(if dual_run_mode {
                    "Type your message for ALL active models... (Enter to send to both, Shift+Enter for newline)"
                } else {
                    "Type your message... (Enter to send, Shift+Enter for newline)"
                })
                .font(egui::TextStyle::Body),
        );

        let trimmed_input = self.input.trim().to_string();
        let domain_info = if !trimmed_input.is_empty() && trimmed_input.len() >= 6 {
            Some(crate::neural::classify_prompt_domain(&trimmed_input))
        } else {
            None
        };

        ui.add_space(4.0);
        ui.horizontal(|ui| {
            if let Some((domain_idx, domain_name)) = domain_info {
                ui.label(
                    egui::RichText::new("🐝 Swarm Router:")
                        .size(11.0)
                        .color(egui::Color32::from_rgb(0xff, 0xcc, 0x00))
                );
                let badge = match domain_idx {
                    1 => "💻 Coder",
                    2 => "🔬 Researcher",
                    3 => "🛡️ Cyber / Critic",
                    4 => "📋 Planner",
                    5 => "✍️ Writer",
                    _ => "🌐 General",
                };
                ui.label(
                    egui::RichText::new(format!("{badge} ({domain_name})"))
                        .size(11.0)
                        .color(egui::Color32::from_rgb(0x00, 0xee, 0xff))
                );
            }

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
                } else if dual_run_mode {
                    "⚡ Send to Both (Enter)"
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
                        .min_size(egui::vec2(130.0, 34.0))
                ).on_hover_text("Send message (Enter to send, Shift+Enter for newline)");
                if send_btn.clicked() {
                    if dual_run_mode {
                        if let Some(prompt) = self.take_broadcast() {
                            let _ = tx.send(crate::ui::app::AppMessage::Broadcast(prompt));
                        }
                    } else {
                        self.send_message(_models, selected_model, role_prompt, slot_idx, api_client, tx, rt);
                    }
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
                ui.add_space(4.0);
                let can_retry = !self.is_streaming
                    && selected_model.is_some()
                    && api_client.is_some()
                    && self.last_user_prompt().is_some();
                let retry_btn = ui
                    .add_enabled(
                        can_retry,
                        egui::Button::new(egui::RichText::new("Retry").size(13.0))
                            .fill(egui::Color32::from_rgb(0x55, 0x33, 0x00))
                            .corner_radius(egui::CornerRadius::same(6)),
                    )
                    .on_hover_text("Resend the last prompt as a new turn");
                if retry_btn.clicked() {
                    if let Some(prompt) = self.last_user_prompt() {
                        self.input = prompt;
                        self.send_message(
                            _models,
                            selected_model,
                            role_prompt,
                            slot_idx,
                            api_client,
                            tx,
                            rt,
                        );
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
            // Exclude ctrl modifier so Ctrl+Enter triggers broadcast!
            let send_triggered = ui.input(|i| {
                i.key_pressed(egui::Key::Enter) && !i.modifiers.shift && !i.modifiers.ctrl
            }) && response.has_focus()
                && !self.input.trim().is_empty()
                && !self.is_streaming;
            if send_triggered {
                if dual_run_mode {
                    if let Some(prompt) = self.take_broadcast() {
                        let _ = tx.send(crate::ui::app::AppMessage::Broadcast(prompt));
                    }
                } else {
                    self.send_message(_models, selected_model, role_prompt, slot_idx, api_client, tx, rt);
                }
            }
            // Ctrl+Enter broadcasts the field to every slot (Enter alone sends here).
            let broadcast_triggered = ui.input(|i| {
                i.key_pressed(egui::Key::Enter) && i.modifiers.ctrl
            }) && response.has_focus()
                && !self.input.trim().is_empty()
                && !self.is_streaming;
            if broadcast_triggered {
                if let Some(prompt) = self.take_broadcast() {
                    let _ = tx.send(crate::ui::app::AppMessage::Broadcast(prompt));
                }
            }
        });
        ui.add_space(4.0);
        // Status line: always shows what this chat needs to work.
        ui.horizontal(|ui| {
            ui.add_space(4.0);
            let model_txt = selected_model.clone().unwrap_or("(no model)".to_string());
            let link_txt = if api_client.is_some() { "Ollama: linked" } else { "Ollama: NOT linked" };
            let turns = self.messages.len();
            let chars: usize = self.messages.iter().map(|m| m.content.chars().count()).sum();
            let tok = chars / 4;
            ui.label(
                egui::RichText::new(format!(
                    "Slot {} -> {} | {} | {} model(s) | {} turns \u{00B7} {} chars \u{00B7} ~{} tok \u{00B7} {}",
                    slot_idx + 1,
                    model_txt,
                    link_txt,
                    _models.len(),
                    turns,
                    chars,
                    tok,
                    Self::fmt_latency(self.last_reply_secs, self.reply_secs_total, self.reply_count),
                ))
                .size(11.0)
                .color(egui::Color32::from_rgb(0x99, 0x99, 0x99)),
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let bar_tok = self.messages.iter().map(|m| m.content.chars().count()).sum::<usize>() / 4;
                ui.add(
                    egui::ProgressBar::new((bar_tok as f32 / 8000.0).clamp(0.0, 1.0))
                        .desired_width(70.0)
                        .show_percentage(),
                )
                .on_hover_text("Rough context use: chars/4 vs an 8k-token budget");
                if ui
                    .small_button("Copy all")
                    .on_hover_text("Copy this slot transcript to the clipboard")
                    .clicked()
                {
                    let md = Self::slot_markdown(selected_model, &self.messages);
                    ui.ctx().copy_text(md.clone());
                    let _ = tx.send(crate::ui::app::AppMessage::Notice(format!(
                        "Copied {} chars",
                        md.chars().count()
                    )));
                }
                if ui
                    .small_button("Save .md")
                    .on_hover_text("Export this slot transcript to a markdown file")
                    .clicked()
                {
                    let stem = format!(
                        "slot{}-{}",
                        slot_idx + 1,
                        selected_model.clone().unwrap_or("chat".to_string()),
                    );
                    let note = match crate::ui::history::HistoryPanel::export_path(&stem) {
                        Some(path) => {
                            if let Some(parent) = path.parent() {
                                let _ = std::fs::create_dir_all(parent);
                            }
                            match std::fs::write(&path, Self::slot_markdown(selected_model, &self.messages)) {
                                Ok(()) => format!("Saved {}", path.display()),
                                Err(e) => format!("Export failed: {e}"),
                            }
                        }
                        None => "Export failed: bad slot name.".to_string(),
                    };
                    let _ = tx.send(crate::ui::app::AppMessage::Notice(note));
                }
            });
        });
        ui.add_space(8.0);
    }

    /// Render this slot's transcript as markdown (same shape as History export).
    pub(crate) fn slot_markdown(model: &Option<String>, messages: &[ChatMessage]) -> String {
        let mut md = format!("# {}\n\n", model.clone().unwrap_or("chat".to_string()));
        for m in messages {
            let who = match m.role.as_str() {
                "user" => "You",
                "assistant" => "Assistant",
                "system" => "System",
                _ => "Note",
            };
            md.push_str(&format!(
                "## {who} ({})\n\n{}\n\n",
                m.timestamp.format("%H:%M"),
                m.content
            ));
        }
        md
    }

    /// Latest user prompt, if any (powers Retry).
    fn last_user_prompt(&self) -> Option<String> {
        self.messages
            .iter()
            .rev()
            .find(|m| m.role == "user")
            .map(|m| m.content.clone())
    }

    /// Short latency chip for the status line (last + running average).
    fn fmt_latency(secs: Option<f32>, total: f32, count: u32) -> String {
        match secs {
            Some(s) if count > 1 => {
                format!("{s:.1}s reply \u{00B7} avg {:.1}s", total / count as f32)
            }
            Some(s) => format!("{s:.1}s reply"),
            None => "\u{2014}".to_string(),
        }
    }

    fn show_message(
        &self,
        ui: &mut egui::Ui,
        msg: &ChatMessage,
        segments: &[MessageSegment],
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
        let fallback_segs;
        let actual_segs = if segments.is_empty() && !msg.content.is_empty() {
            fallback_segs = parse_segments(&msg.content);
            &fallback_segs[..]
        } else {
            segments
        };
        let code_blocks: Vec<(&str, &str)> = actual_segs
            .iter()
            .filter_map(|s| match s {
                MessageSegment::Code { lang, code } => Some((lang.as_str(), code.as_str())),
                _ => None,
            })
            .collect();

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

                    for (seg_idx, seg) in actual_segs.iter().enumerate() {
                        match seg {
                            MessageSegment::Text(t) => {
                                if !t.is_empty() {
                                    ui.label(
                                        egui::RichText::new(t)
                                            .size(13.0)
                                            .color(egui::Color32::WHITE),
                                    );
                                }
                            }
                            MessageSegment::Think(th) => {
                                egui::CollapsingHeader::new(
                                    egui::RichText::new("💭 Thought Process")
                                        .size(11.0)
                                        .color(egui::Color32::from_rgb(0x94, 0xa3, 0xb8)),
                                )
                                .default_open(false)
                                .show(ui, |ui| {
                                    egui::Frame::NONE
                                        .fill(egui::Color32::from_rgb(0x13, 0x18, 0x24))
                                        .corner_radius(egui::CornerRadius::same(4))
                                        .inner_margin(egui::Margin::same(6))
                                        .show(ui, |ui| {
                                            ui.label(
                                                egui::RichText::new(th)
                                                    .size(11.0)
                                                    .color(egui::Color32::from_rgb(0x94, 0xa3, 0xb8))
                                                    .italics(),
                                            );
                                        });
                                });
                            }
                            MessageSegment::Code { lang, code } => {
                                let block_id = format!("chat_cb_{}", seg_idx);
                                render_code_block(ui, lang, code, &block_id, tx);
                            }
                        }
                    }

                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        if ui
                            .small_button("Copy")
                            .on_hover_text("Copy message text")
                            .clicked()
                        {
                            ui.ctx().copy_text(msg.content.clone());
                        }
                        if !is_user && msg.role == "assistant" {
                            let speak = ui.small_button("\u{1F50A}");
                            if speak.on_hover_text("Read this message aloud").clicked()
                            {
                                let _ = tx.send(
                                    crate::ui::app::AppMessage::SpeakText(msg.content.clone()),
                                );
                            }
                        }
                        if !is_user && msg.role == "assistant" && !code_blocks.is_empty() {
                            if ui.small_button("Copy code").clicked() {
                                if let Some((_, code)) = code_blocks.iter().max_by_key(|(_, c)| c.len()) {
                                    ui.ctx().copy_text((*code).to_string());
                                }
                            }
                            if ui
                                .small_button("Send to Editor")
                                .on_hover_text("Open the biggest code block in the Editor tab")
                                .clicked()
                            {
                                if let Some((lang, code)) =
                                    code_blocks.iter().max_by_key(|(_, c)| c.len())
                                {
                                    let _ = tx.send(crate::ui::app::AppMessage::ChatToEditor(
                                        (*code).to_string(),
                                        (*lang).to_string(),
                                    ));
                                }
                            }
                        }
                    });
                });
        });
        ui.add_space(4.0);
    }

    pub fn send_message(
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
        let mut prompt = self.input.trim().to_string();
        if prompt.is_empty() {
            return;
        }
        if prompt.len() > Self::MAX_CONTENT_CHARS {
            let mut end = Self::MAX_CONTENT_CHARS;
            while !prompt.is_char_boundary(end) { end -= 1; }
            prompt.truncate(end);
        }

        // Unified In-Chat Slash Command Interception
        if prompt.starts_with('/') {
            match crate::commands::SlashCommand::parse(&prompt) {
                Ok(Some(cmd)) => {
                    self.input.clear();
                    match cmd {
                        crate::commands::SlashCommand::Help => {
                            self.push_system_note(crate::commands::SlashCommand::help_manual());
                        }
                        crate::commands::SlashCommand::Clear => {
                            self.clear_chat();
                            self.push_system_note("⚡ **Chat cleared from viewport.**");
                        }
                        crate::commands::SlashCommand::New => {
                            let _ = tx.send(crate::ui::app::AppMessage::NewChat);
                            self.push_system_note("⚡ **Started fresh chat session.**");
                        }
                        crate::commands::SlashCommand::Model(name) => {
                            let _ = tx.send(crate::ui::app::AppMessage::ModelSelected(name.clone()));
                            self.push_system_note(&format!("⚡ **Switching model to `{}`** (pre-warming into RAM)...", name));
                        }
                        crate::commands::SlashCommand::Swarm(cmd) => match cmd {
                            crate::commands::SwarmCommand::Dispatch(task) => {
                                let _ = tx.send(crate::ui::app::AppMessage::LaunchSwarmTask(task));
                                self.push_system_note("🧬 **Swarm Relay dispatched.** Switching to Swarm Relay tab...");
                            }
                            crate::commands::SwarmCommand::Preset { template, prompt } => {
                                if let Some(tmpl) = crate::ui::relay::SwarmTemplate::from_id(&template) {
                                    let _ = tx.send(crate::ui::app::AppMessage::LaunchSwarmPreset {
                                        template: tmpl,
                                        prompt,
                                    });
                                    self.push_system_note(&format!("🧬 **Swarm preset `{}` activated.** Switching to Swarm Relay tab...", tmpl.label()));
                                } else {
                                    self.push_system_note(&format!("❌ **Unknown swarm preset `{}`.** Use `/swarm presets` to list presets.", template));
                                }
                            }
                            crate::commands::SwarmCommand::ListPresets => {
                                let mut note = String::from("### 🧬 **Available Multi-Agent Swarm Presets**\n\n| Short ID | Preset Description |\n|---|---|\n");
                                for t in crate::ui::relay::SwarmTemplate::all() {
                                    if t != crate::ui::relay::SwarmTemplate::Custom {
                                        note.push_str(&format!("| `{}` | {} |\n", t.short_id(), t.label()));
                                    }
                                }
                                note.push_str("\n*Usage:* `/swarm preset <id> [prompt]` (e.g. `/swarm preset triad Build an auth system`)");
                                self.push_system_note(&note);
                            }
                            crate::commands::SwarmCommand::Dag { preset, prompt } => {
                                let dag_preset = preset.as_deref().and_then(crate::swarm::DagPreset::from_id);
                                let p_label = dag_preset.map(|p| p.label()).unwrap_or("Diamond Swarm");
                                let _ = tx.send(crate::ui::app::AppMessage::LaunchSwarmDag {
                                    preset: dag_preset,
                                    prompt,
                                });
                                self.push_system_note(&format!("🕸 **Autonomous Swarm DAG `{}` activated.** Switching to Relay tab...", p_label));
                            }
                            crate::commands::SwarmCommand::Blackboard => {
                                let _ = tx.send(crate::ui::app::AppMessage::ShowBlackboard);
                                self.push_system_note("🐝 **Opening Stigmergic Blackboard Vault in Relay tab...**");
                            }
                            crate::commands::SwarmCommand::Abort => {
                                let _ = tx.send(crate::ui::app::AppMessage::AbortSwarm);
                                self.push_system_note("🛑 **Aborting active swarm operations across all nodes.**");
                            }
                        }
                        crate::commands::SlashCommand::Skills(sub) => {
                            let _ = tx.send(crate::ui::app::AppMessage::SkillsCommand(slot_idx, sub));
                        }
                        crate::commands::SlashCommand::Audit => {
                            let _ = tx.send(crate::ui::app::AppMessage::ShowAudit(slot_idx));
                        }
                        crate::commands::SlashCommand::Threads(n) => {
                            let _ = tx.send(crate::ui::app::AppMessage::SetThreads(n));
                            self.push_system_note(&format!("⚡ **Inference CPU threads set to {}.**", n));
                        }
                        crate::commands::SlashCommand::Status => {
                            let _ = tx.send(crate::ui::app::AppMessage::ShowStatus(slot_idx));
                        }
                        crate::commands::SlashCommand::Exec(cmd_line) => {
                            let _ = tx.send(crate::ui::app::AppMessage::TerminalRun(cmd_line));
                            self.push_system_note("⚡ **Command sent to IDE Terminal Dock.**");
                        }
                        crate::commands::SlashCommand::Mkdir(dir_path) => {
                            let _ = tx.send(crate::ui::app::AppMessage::WorkspaceMkdir(dir_path));
                        }
                        crate::commands::SlashCommand::Touch(file_path) => {
                            let _ = tx.send(crate::ui::app::AppMessage::WorkspaceTouch(file_path));
                        }
                        crate::commands::SlashCommand::Rm(target) => {
                            let _ = tx.send(crate::ui::app::AppMessage::WorkspaceRm(target));
                        }
                        crate::commands::SlashCommand::Mv { src, dst } => {
                            let _ = tx.send(crate::ui::app::AppMessage::WorkspaceMv { src, dst });
                        }
                        crate::commands::SlashCommand::Ls(path_opt) => {
                            let _ = tx.send(crate::ui::app::AppMessage::WorkspaceLs(path_opt));
                        }
                        crate::commands::SlashCommand::Project(proj) => {
                            let _ = tx.send(crate::ui::app::AppMessage::WorkspaceProject(proj));
                        }
                        crate::commands::SlashCommand::Tools(sub) => {
                            let _ = tx.send(crate::ui::app::AppMessage::ToolsCommand(slot_idx, sub));
                        }
                        crate::commands::SlashCommand::Guardrail(sub) => {
                            let _ = tx.send(crate::ui::app::AppMessage::GuardrailCommand(slot_idx, sub));
                        }
                    }
                    return;
                }
                Err(err_msg) => {
                    self.input.clear();
                    self.push_system_note(&format!("⚠ **Command Error:**\n{}", err_msg));
                    return;
                }
                Ok(None) => {}
            }
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
        // Keep continuity: capture prior conversation turns (excluding the
        // newly initiated prompt, which is appended to the payload below).
        let history: Vec<(String, String)> = self
            .messages
            .iter()
            .rev()
            .take(self.history_depth.max(1))
            .rev()
            .map(|m| (m.role.clone(), m.content.clone()))
            .collect();

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
        self.is_streaming = true;
        self.stream_buf.clear();
        self.stream_seq = self.stream_seq.saturating_add(1);
        let seq = self.stream_seq;
        self.send_started = Some(Instant::now());

        let role_prompt = role_prompt.to_string();
        let tx_handle = tx.clone();
        let num_threads = self.num_threads;
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
            let threads_opt = if num_threads > 0 { Some(num_threads) } else { None };
            let req = ChatRequest {
                model: model_name,
                messages,
                stream: true,
                options: Some(ChatOptions::lowram_with_threads(threads_opt)),
                keep_alive: Some("30m".to_string()),
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
        self.cached_stream_segs.clear();
        self.stream_dirty = false;
        let elapsed = self
            .send_started
            .map(|t| t.elapsed().as_secs_f32())
            .unwrap_or(0.0);
        self.send_started = None;
        match response {
            Ok(resp) => {
                self.last_reply_secs = Some(elapsed);
                self.reply_secs_total += elapsed;
                self.reply_count += 1;
                let clean = sanitize_text(&resp.message.content);
                let n = clean.len();
                let msg = ChatMessage {
                    role: "assistant".to_string(),
                    content: Self::cap_content(&clean),
                    timestamp: chrono::Utc::now(),
                };
                self.push_capped(msg);
                (true, elapsed, n)
            }
            Err(e) => {
                // No prefix: OllamaError already describes itself
                // ("API error: ...", "Request failed: ...").
                let em = sanitize_text(&format!("{e}"));
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

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: role.to_string(),
            content: content.to_string(),
            timestamp: chrono::Utc::now(),
        }
    }

    #[test]
    fn slot_markdown_shapes() {
        let md = ChatPanel::slot_markdown(
            &Some("llama3.2:1b".to_string()),
            &[msg("user", "hi"), msg("assistant", "Hello.")],
        );
        assert!(md.starts_with("# llama3.2:1b\n"));
        assert!(md.contains("## You ("));
        assert!(md.contains("## Assistant ("));
        assert!(md.contains("Hello."));
    }

    #[test]
    fn slot_markdown_empty_model() {
        let md = ChatPanel::slot_markdown(&None, &[]);
        assert_eq!(md, "# chat\n\n");
    }
}

#[cfg(test)]
mod panel_tests {
    use super::*;

    #[test]
    fn latency_chip_shapes() {
        assert_eq!(ChatPanel::fmt_latency(None, 0.0, 0), "\u{2014}");
        assert_eq!(ChatPanel::fmt_latency(Some(12.345), 12.345, 1), "12.3s reply");
        assert_eq!(
            ChatPanel::fmt_latency(Some(10.0), 30.0, 3),
            "10.0s reply \u{00B7} avg 10.0s"
        );
    }

    #[test]
    fn carry_keeps_history_not_live_state() {
        let mut p = ChatPanel::new();
        p.input = "draft".to_string();
        p.messages.push(ChatMessage {
            role: "user".to_string(),
            content: "hello".to_string(),
            timestamp: chrono::Utc::now(),
        });
        p.is_streaming = true;
        p.last_reply_secs = Some(3.0);
        let q = p.carry_messages();
        assert_eq!(q.messages.len(), 1);
        assert_eq!(q.last_reply_secs, Some(3.0));
        assert!(q.input.is_empty());
        assert!(!q.is_streaming);
    }

    #[test]
    fn last_prompt_picks_newest_user() {
        let mut p = ChatPanel::new();
        assert!(p.last_user_prompt().is_none());
        for (role, content) in [("user", "first"), ("assistant", "hi"), ("user", "second")] {
            p.messages.push(ChatMessage {
                role: role.to_string(),
                content: content.to_string(),
                timestamp: chrono::Utc::now(),
            });
        }
        assert_eq!(p.last_user_prompt().as_deref(), Some("second"));
    }

    #[test]
    fn history_continuity_excludes_new_user_prompt() {
        let mut p = ChatPanel::new();
        p.messages.push(ChatMessage {
            role: "user".to_string(),
            content: "turn 1".to_string(),
            timestamp: chrono::Utc::now(),
        });
        p.messages.push(ChatMessage {
            role: "assistant".to_string(),
            content: "reply 1".to_string(),
            timestamp: chrono::Utc::now(),
        });
        let history: Vec<(String, String)> = p
            .messages
            .iter()
            .rev()
            .take(p.history_depth.max(1))
            .rev()
            .map(|m| (m.role.clone(), m.content.clone()))
            .collect();
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].1, "turn 1");
        assert_eq!(history[1].1, "reply 1");
    }

    #[test]
    fn sanitize_text_cleans_ansi_and_replacement_artifacts() {
        let dirty = "\x1b[31mRed Alert\x1b[0m \u{FFFD}clean\x07text\twith\nnewlines";
        let clean = sanitize_text(dirty);
        assert_eq!(clean, "Red Alert cleantext\twith\nnewlines");
    }

    #[test]
    fn parse_segments_splits_code_and_text() {
        let content = "Here is the code:\n```rust\nfn main() {\n    println!(\"hello\");\n}\n```\nEnjoy!";
        let segs = parse_segments(content);
        assert_eq!(segs.len(), 3);
        assert_eq!(segs[0], MessageSegment::Text("Here is the code:\n".to_string()));
        assert_eq!(segs[1], MessageSegment::Code {
            lang: "rust".to_string(),
            code: "fn main() {\n    println!(\"hello\");\n}".to_string(),
        });
        assert_eq!(segs[2], MessageSegment::Text("\nEnjoy!".to_string()));
    }

    #[test]
    fn parse_segments_handles_unclosed_streaming_code() {
        let streaming = "Streaming snippet:\n```python\ndef compute(x):\n    return x * 2";
        let segs = parse_segments(streaming);
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0], MessageSegment::Text("Streaming snippet:\n".to_string()));
        assert_eq!(segs[1], MessageSegment::Code {
            lang: "python".to_string(),
            code: "def compute(x):\n    return x * 2".to_string(),
        });
    }

    #[test]
    fn parse_segments_isolates_think_blocks() {
        let msg = "<think>\nAnalyzing problem\nStep 1: Check constraints\n</think>\nResult:\n```sh\necho done\n```";
        let segs = parse_segments(msg);
        assert_eq!(segs.len(), 3);
        assert_eq!(segs[0], MessageSegment::Think("Analyzing problem\nStep 1: Check constraints".to_string()));
        assert_eq!(segs[1], MessageSegment::Text("\nResult:\n".to_string()));
        assert_eq!(segs[2], MessageSegment::Code {
            lang: "sh".to_string(),
            code: "echo done".to_string(),
        });
    }

    #[test]
    fn test_sanitize_text_cow_zero_alloc() {
        use std::borrow::Cow;
        let clean_text = "This is a clean response with newlines\nand tabs\twithout escapes.";
        let res = sanitize_text_cow(clean_text);
        assert!(matches!(res, Cow::Borrowed(_)));
        assert_eq!(res, clean_text);

        let dirty_text = "Hello\x1b[31mRed\x1b[0m world\u{FFFD}!\x07";
        let res2 = sanitize_text_cow(dirty_text);
        assert!(matches!(res2, Cow::Owned(_)));
        assert_eq!(res2, "HelloRed world!");
    }

    #[test]
    fn test_take_broadcast_and_clear_chat() {
        let mut p = ChatPanel::new();
        p.input = "  broadcast message to all models  ".to_string();
        let prompt = p.take_broadcast();
        assert_eq!(prompt.as_deref(), Some("broadcast message to all models"));
        assert!(p.input.is_empty());

        // When streaming, take_broadcast must return None
        p.input = "next message".to_string();
        p.is_streaming = true;
        assert!(p.take_broadcast().is_none());
        assert_eq!(p.input, "next message");

        p.is_streaming = false;
        p.messages.push(ChatMessage {
            role: "user".to_string(),
            content: "hello".to_string(),
            timestamp: chrono::Utc::now(),
        });
        assert_eq!(p.messages.len(), 1);
        p.clear_chat();
        assert_eq!(p.messages.len(), 0);
    }
}

