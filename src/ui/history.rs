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
}

impl HistoryPanel {
    pub fn new() -> Self {
        Self {
            sessions: Vec::new(),
            selected_session: None,
            show_session_detail: false,
            loaded: false,
            audit: Vec::new(),
        }
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

        ui.add_space(8.0);
        ui.separator();
        ui.add_space(8.0);

        egui::ScrollArea::vertical().show(ui, |ui| {
            for (idx, session) in self.sessions.iter().enumerate() {
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
                }

                ui.add_space(4.0);
            }
        });

        // Session detail view
        if self.show_session_detail {
            if let Some(idx) = self.selected_session {
                if let Some(session) = self.sessions.get(idx).cloned() {
                    ui.add_space(12.0);
                    ui.separator();
                    ui.add_space(8.0);

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
                            }
                        }
                    });
                }
            }
        }
    }
}