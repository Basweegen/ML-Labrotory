use eframe::egui;
use crate::storage::{AuditEntry, ChatSession, Storage};
use serde_json;
use std::sync::mpsc;

pub struct HistoryPanel {
    pub sessions: Vec<ChatSession>,
    selected_session: Option<usize>,
    show_session_detail: bool,
    loaded: bool,
    audit: Vec<AuditEntry>,
    search: String,
    rename_buf: String,
    delete_all_armed: bool,
    notice: String,
}

impl HistoryPanel {
    pub fn new() -> Self {
        Self {
            sessions: Vec::new(),
            selected_session: None,
            show_session_detail: false,
            loaded: false,
            audit: Vec::new(),
            search: String::new(),
            rename_buf: String::new(),
            delete_all_armed: false,
            notice: String::new(),
        }
    }

    fn matches(session: &ChatSession, query: &str) -> bool {
        let q = query.trim().to_lowercase();
        if q.is_empty() {
            return true;
        }
        session.name.to_lowercase().contains(&q)
            || session.model.to_lowercase().contains(&q)
            || session
                .messages
                .iter()
                .any(|m| m.content.to_lowercase().contains(&q))
    }

    /// Markdown export of one session (file + clipboard share this).
    pub fn session_markdown(session: &ChatSession) -> String {
        let mut out = format!(
            "# {}\n\nModel: {} \u{00B7} Created: {} \u{00B7} Updated: {}\n\n",
            session.name,
            session.model,
            session.created_at.format("%Y-%m-%d %H:%M"),
            session.updated_at.format("%Y-%m-%d %H:%M")
        );
        for m in &session.messages {
            let who = match m.role.as_str() {
                "user" => "You",
                "system" => "System",
                _ => "Assistant",
            };
            out.push_str(&format!(
                "## {} ({})\n\n{}\n\n",
                who,
                m.timestamp.format("%H:%M"),
                m.content
            ));
        }
        out
    }

    fn export_path(name: &str) -> Option<std::path::PathBuf> {
        let stem: String = name
            .chars()
            .filter(|c| c.is_alphanumeric() || matches!(c, '_' | '-' | ' '))
            .collect::<String>()
            .trim()
            .replace(' ', "_");
        let stem = stem.chars().take(60).collect::<String>();
        if stem.is_empty() {
            return None;
        }
        let base = dirs::document_dir().unwrap_or_else(|| std::path::PathBuf::from("."));
        Some(base.join("Code_air").join("ml_lab_exports").join(format!("{stem}.md")))
    }

    /// Force a reload next time the tab is shown (called after an autosave).
    pub fn mark_dirty(&mut self) {
        self.loaded = false;
    }

    fn reload(&mut self, storage: &Storage) {
        if let Ok(sessions) = storage.load_sessions() {
            self.sessions = sessions;
        }
        if let Ok(audit) = storage.load_audit() {
            self.audit = audit;
        }
        self.loaded = true;
    }

    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        storage: &Storage,
        tx: &mpsc::Sender<crate::ui::app::AppMessage>,
    ) {
        if !self.loaded {
            self.reload(storage);
        }
        // Audit rows arrive from app actions at any time; refresh cheaply on
        // every show so the trail never looks stale.
        if let Ok(audit) = storage.load_audit() {
            self.audit = audit;
        }

        egui::CollapsingHeader::new(format!("Security activity ({})", self.audit.len()))
            .default_open(true)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().max_height(150.0).show(ui, |ui| {
                    if self.audit.is_empty() {
                        ui.label(
                            egui::RichText::new("No security events yet.")
                                .size(12.0)
                                .color(egui::Color32::from_rgb(0x88, 0x88, 0x88)),
                        );
                    }
                    for a in &self.audit {
                        // Test scaffolding stays in the store but out of the UI.
                        if a.kind.starts_with("test.") {
                            continue;
                        }
                        ui.label(
                            egui::RichText::new(format!(
                                "{}  {:<16}  {}",
                                a.ts.format("%m-%d %H:%M"),
                                a.kind,
                                a.detail
                            ))
                            .size(11.0)
                            .monospace()
                            .color(egui::Color32::from_rgb(0xaa, 0xaa, 0xaa)),
                        );
                    }
                });
            });
        ui.add_space(8.0);
        ui.horizontal(|ui| {
            ui.add_space(4.0);
            ui.heading(egui::RichText::new("📜 Chat History").size(22.0).color(egui::Color32::from_rgb(0x00, 0xaa, 0xff)));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.add_space(8.0);
                let refresh_btn = ui.add(
                    egui::Button::new(egui::RichText::new("🔄 Refresh").size(13.0))
                        .fill(egui::Color32::from_rgb(0x33, 0x33, 0x33))
                        .corner_radius(egui::CornerRadius::same(6))
                );
                if refresh_btn.clicked() {
                    self.reload(storage);
                    self.selected_session = None;
                    self.show_session_detail = false;
                    self.delete_all_armed = false;
                }
                ui.add_space(8.0);
                if self.delete_all_armed {
                    if ui.small_button("Confirm wipe").clicked() {
                        for s in &self.sessions {
                            let _ = storage.delete_session(s.id);
                        }
                        self.sessions.clear();
                        self.selected_session = None;
                        self.show_session_detail = false;
                        self.delete_all_armed = false;
                        self.notice = "All sessions deleted.".to_string();
                    }
                    if ui.small_button("Keep").clicked() {
                        self.delete_all_armed = false;
                    }
                } else if ui
                    .small_button("\u{1F5D1} All")
                    .on_hover_text("Delete ALL sessions (asks first)")
                    .clicked()
                {
                    self.delete_all_armed = true;
                }
                ui.add_space(8.0);
                let new_btn = ui.add(
                    egui::Button::new(egui::RichText::new("➕ New Chat").size(13.0))
                        .fill(egui::Color32::from_rgb(0x00, 0x55, 0xaa))
                        .corner_radius(egui::CornerRadius::same(6))
                );
                if new_btn.clicked() {
                    let _ = tx.send(crate::ui::app::AppMessage::NewChat);
                    self.selected_session = None;
                    self.show_session_detail = false;
                }
            });
        });

        if !self.notice.is_empty() {
            ui.label(
                egui::RichText::new(&self.notice)
                    .size(11.0)
                    .color(egui::Color32::from_rgb(0x99, 0x99, 0x99)),
            );
        }
        ui.add_space(8.0);
        ui.separator();
        ui.add_space(8.0);

        ui.horizontal(|ui| {
            ui.add_space(4.0);
            ui.label(
                egui::RichText::new("Search:")
                    .size(13.0)
                    .color(egui::Color32::from_rgb(0xcc, 0xcc, 0xcc)),
            );
            ui.add(
                egui::TextEdit::singleline(&mut self.search)
                    .desired_width(280.0)
                    .hint_text("name, model, or message text…"),
            );
        });
        ui.add_space(4.0);

        let mut clicked_name: Option<(usize, String)> = None;
        egui::ScrollArea::vertical().show(ui, |ui| {
            for (idx, session) in self
                .sessions
                .iter()
                .enumerate()
                .filter(|(_, s)| Self::matches(s, &self.search)) {
                let is_selected = self.selected_session == Some(idx);
                let response = ui.selectable_label(
                    is_selected,
                    egui::RichText::new(format!(
                        "{}  •  {}  •  {} messages  •  {}",
                        session.model,
                        session.name,
                        session.messages.len(),
                        session.updated_at.format("%Y-%m-%d %H:%M")
                    )).size(13.0).color(if is_selected { egui::Color32::WHITE } else { egui::Color32::from_rgb(0xcc, 0xcc, 0xcc) })
                );

                if response.clicked() {
                    self.selected_session = Some(idx);
                    self.show_session_detail = true;
                    clicked_name = Some((idx, session.name.clone()));
                }

                ui.add_space(4.0);
            }
        });
        if let Some((idx, name)) = clicked_name {
            if self.selected_session == Some(idx) {
                self.rename_buf = name;
            }
        }

        // Session detail view
        if self.show_session_detail {
            if let Some(idx) = self.selected_session {
                if let Some(mut session) = self.sessions.get(idx).cloned() {
                    ui.add_space(12.0);
                    ui.separator();
                    ui.add_space(8.0);

                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new("Rename:")
                                .size(13.0)
                                .color(egui::Color32::from_rgb(0xcc, 0xcc, 0xcc)),
                        );
                        ui.add(
                            egui::TextEdit::singleline(&mut self.rename_buf)
                                .desired_width(240.0),
                        );
                        if ui.small_button("Save name").clicked() {
                            let name = self.rename_buf.trim().to_string();
                            if !name.is_empty() {
                                session.name = name.clone();
                                session.updated_at = chrono::Utc::now();
                                if let Some(s) = self.sessions.get_mut(idx) {
                                    *s = session.clone();
                                    if let Err(e) = storage.save_session(s) {
                                        self.notice = format!("Rename failed: {e}");
                                    } else {
                                        self.notice = format!("Renamed to '{}'.", name);
                                    }
                                }
                            }
                        }
                    });
                    ui.add_space(4.0);
                    ui.heading(egui::RichText::new(&session.name).size(20.0).color(egui::Color32::WHITE));
                    ui.add_space(4.0);
                    ui.label(egui::RichText::new(format!("Model: {}", session.model)).size(13.0).color(egui::Color32::from_rgb(0xaa, 0xaa, 0xaa)));
                    ui.label(egui::RichText::new(format!("Created: {}", session.created_at.format("%Y-%m-%d %H:%M"))).size(13.0).color(egui::Color32::from_rgb(0xaa, 0xaa, 0xaa)));
                    ui.label(egui::RichText::new(format!("Updated: {}", session.updated_at.format("%Y-%m-%d %H:%M"))).size(13.0).color(egui::Color32::from_rgb(0xaa, 0xaa, 0xaa)));

                    ui.add_space(8.0);
                    ui.separator();
                    ui.add_space(8.0);

                    egui::ScrollArea::vertical().show(ui, |ui| {
                        for msg in &session.messages {
                            let is_user = msg.role == "user";
                            let (bg_color, align) = if is_user {
                                (egui::Color32::from_rgb(0x00, 0x44, 0x88), egui::Align::RIGHT)
                            } else {
                                (egui::Color32::from_rgb(0x2a, 0x2a, 0x2a), egui::Align::LEFT)
                            };

                            ui.add_space(8.0);
                            ui.with_layout(egui::Layout::top_down(align), |ui| {
                                egui::Frame::group(&ui.style())
                                    .fill(bg_color)
                                    .corner_radius(egui::CornerRadius::same(10))
                                    .inner_margin(egui::Margin::symmetric(14, 10))
                                    .show(ui, |ui| {
                                        ui.set_max_width(ui.available_width() * 0.85);
                                        ui.label(egui::RichText::new(&msg.content).size(14.0).color(egui::Color32::WHITE));
                                        ui.add_space(4.0);
                                        ui.with_layout(
                                            egui::Layout::right_to_left(egui::Align::Center),
                                            |ui| {
                                                ui.small(egui::RichText::new(msg.timestamp.format("%H:%M").to_string()).color(egui::Color32::from_rgb(0x99, 0x99, 0x99)));
                                            },
                                        );
                                    });
                            });
                        }
                    });

                    ui.add_space(12.0);
                    ui.separator();
                    ui.add_space(8.0);
                    
                    let session_id = session.id;
                    ui.horizontal(|ui| {
                        let open_btn = ui.add(
                            egui::Button::new(egui::RichText::new("Open in chat").size(13.0))
                                .fill(egui::Color32::from_rgb(0x00, 0x55, 0xaa))
                                .corner_radius(egui::CornerRadius::same(6))
                        );
                        if open_btn.on_hover_text("Load this session into the focused chat slot").clicked() {
                            let _ = tx.send(crate::ui::app::AppMessage::LoadSession(session.clone()));
                        }
                        ui.add_space(8.0);
                        let delete_btn = ui.add(
                            egui::Button::new(egui::RichText::new("🗑 Delete Session").size(13.0))
                                .fill(egui::Color32::from_rgb(0xaa, 0x33, 0x33))
                                .corner_radius(egui::CornerRadius::same(6))
                        );
                        if delete_btn.clicked() {
                            let _ = storage.delete_session(session_id);
                            self.sessions.remove(idx);
                            self.selected_session = None;
                            self.show_session_detail = false;
                        }
                        ui.add_space(8.0);
                        let export_btn = ui.add(
                            egui::Button::new(egui::RichText::new("📋 Export JSON").size(13.0))
                                .fill(egui::Color32::from_rgb(0x33, 0x44, 0x66))
                                .corner_radius(egui::CornerRadius::same(6))
                        );
                        if export_btn.clicked() {
                            if let Ok(json) = serde_json::to_string_pretty(&session) {
                                ui.ctx().copy_text(json);
                                self.notice = "JSON copied to clipboard.".to_string();
                            }
                        }
                        ui.add_space(8.0);
                        let md_btn = ui.add(
                            egui::Button::new(egui::RichText::new("Export .md").size(13.0))
                                .fill(egui::Color32::from_rgb(0x33, 0x44, 0x66))
                                .corner_radius(egui::CornerRadius::same(6))
                        );
                        if md_btn
                            .on_hover_text("Write this session as Markdown under Documents/Code_air/ml_lab_exports/")
                            .clicked()
                        {
                            match Self::export_path(&session.name) {
                                Some(path) => {
                                    if let Some(parent) = path.parent() {
                                        let _ = std::fs::create_dir_all(parent);
                                    }
                                    match std::fs::write(&path, Self::session_markdown(&session)) {
                                        Ok(()) => {
                                            self.notice =
                                                format!("Exported to {}", path.display())
                                        }
                                        Err(e) => {
                                            self.notice = format!("Export failed: {e}")
                                        }
                                    }
                                }
                                None => {
                                    self.notice =
                                        "Export failed: bad session name.".to_string()
                                }
                            }
                        }
                    });
                }
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::ChatMessage;

    fn sample() -> ChatSession {
        ChatSession {
            id: uuid::Uuid::new_v4(),
            name: "Rust help".to_string(),
            model: "llama3.2:1b".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: "how do I borrow?".to_string(),
                timestamp: chrono::Utc::now(),
            }],
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        }
    }

    #[test]
    fn search_matches() {
        let s = sample();
        assert!(HistoryPanel::matches(&s, ""));
        assert!(HistoryPanel::matches(&s, "rust"));
        assert!(HistoryPanel::matches(&s, "LLAMA3"));
        assert!(HistoryPanel::matches(&s, "borrow"));
        assert!(!HistoryPanel::matches(&s, "python"));
    }

    #[test]
    fn markdown_shapes() {
        let md = HistoryPanel::session_markdown(&sample());
        assert!(md.starts_with("# Rust help\n"));
        assert!(md.contains("## You"));
        assert!(md.contains("borrow"));
    }

    #[test]
    fn export_path_sanitizes() {
        assert!(HistoryPanel::export_path("   ").is_none());
        let p = HistoryPanel::export_path("My chat: v2!").unwrap();
        assert!(p.to_string_lossy().ends_with("My_chat_v2.md"));
    }
}
