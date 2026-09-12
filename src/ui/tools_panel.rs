// Copyright 2026 Sean M. Stow. All rights reserved.
//! External Tool Execution & BYOT Registry UI
//! Visual interface for developers, programmers, and SOC specialists to manage,
//! configure, and execute custom CLI utilities, terminal dock scripts, and GUI editors.
//! Integrates with multi-tier Guardrails and cryptographic legal waiver verification.

use chrono::Utc;
use eframe::egui;
use std::sync::mpsc;
use std::time::Instant;

use crate::guardrails::{
    compute_waiver_hash, GuardrailTier, LEGAL_DISCLAIMER_TEXT, REQUIRED_WAIVER_CONFIRMATION,
};
use crate::storage::{AppSettings, Storage};
use crate::tools::{SensitivityLevel, ToolExecutionType, ToolRegistry, UserTool};
use crate::ui::app::AppMessage;

pub struct ToolsPanel {
    pub registry: ToolRegistry,
    pub search_query: String,
    pub is_creating: bool,
    pub editing_id: Option<String>,
    pub form_id: String,
    pub form_name: String,
    pub form_command: String,
    pub form_args: String,
    pub form_execution: ToolExecutionType,
    pub form_sensitivity: SensitivityLevel,
    pub form_guardrail: GuardrailTier,
    pub form_description: String,
    pub show_waiver_modal: bool,
    pub waiver_input: String,
    pub confirm_tool_run: Option<(String, Option<String>)>,
    pub last_refresh: Option<Instant>,
    pub status_banner: Option<(String, bool)>, // (message, is_success)
}

impl Default for ToolsPanel {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolsPanel {
    pub fn new() -> Self {
        Self {
            registry: ToolRegistry::with_defaults(),
            search_query: String::new(),
            is_creating: false,
            editing_id: None,
            form_id: String::new(),
            form_name: String::new(),
            form_command: String::new(),
            form_args: String::new(),
            form_execution: ToolExecutionType::TerminalDock,
            form_sensitivity: SensitivityLevel::Low,
            form_guardrail: GuardrailTier::Heavy,
            form_description: String::new(),
            show_waiver_modal: false,
            waiver_input: String::new(),
            confirm_tool_run: None,
            last_refresh: None,
            status_banner: None,
        }
    }

    pub fn refresh(&mut self, storage: &Option<Storage>) {
        if let Some(st) = storage.as_ref() {
            if let Ok(reg) = st.load_tools() {
                self.registry = reg;
                self.last_refresh = Some(Instant::now());
            }
        }
    }

    pub fn save(&self, storage: &Option<Storage>) {
        if let Some(st) = storage.as_ref() {
            let _ = st.save_tools(&self.registry);
        }
    }

    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        storage: &Option<Storage>,
        settings: &mut AppSettings,
        tx: &mpsc::Sender<AppMessage>,
        _focused_slot: usize,
        workspace_root: Option<&str>,
    ) {
        if self.last_refresh.is_none() {
            self.refresh(storage);
        }

        ui.add_space(8.0);

        // Header and Actions
        ui.horizontal(|ui| {
            ui.heading(
                egui::RichText::new("🛠 External Tools Registry (BYOT)")
                    .size(20.0)
                    .color(egui::Color32::from_rgb(0x00, 0xdd, 0xbb)),
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("➕ Register Tool").clicked() {
                    self.open_create_form();
                }
                if ui.button("↺ Reset Defaults").clicked() {
                    self.registry = ToolRegistry::with_defaults();
                    self.save(storage);
                    self.status_banner = Some(("Restored default starter tools.".to_string(), true));
                }
                if ui.button("🔄 Refresh").clicked() {
                    self.refresh(storage);
                    self.status_banner = Some(("Tools registry reloaded from vault.".to_string(), true));
                }
            });
        });

        ui.label(
            "Bring-Your-Own-Tool harness. Establish custom CLI utilities, scripts, GUI editors, or \
             reverse engineering analyzers. Tools execute locally under active safety guardrails.",
        );

        ui.add_space(8.0);

        // Status banner if present
        if let Some((msg, ok)) = &self.status_banner {
            let color = if *ok {
                egui::Color32::from_rgb(0x88, 0xee, 0x88)
            } else {
                egui::Color32::from_rgb(0xff, 0x77, 0x77)
            };
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(format!("• {}", msg)).color(color));
            });
            ui.add_space(4.0);
        }

        // Guardrails Control Banner
        self.show_guardrails_banner(ui, storage, settings, tx);

        ui.add_space(10.0);
        ui.separator();
        ui.add_space(10.0);

        // Modal for Legal Waiver
        if self.show_waiver_modal {
            self.show_legal_waiver_modal(ui, storage, settings, tx);
        }

        // Modal for High Sensitivity Execution Confirmation
        if self.confirm_tool_run.is_some() {
            self.show_high_sensitivity_confirmation(ui, tx, workspace_root);
        }

        // Modal for Add/Edit Tool
        if self.is_creating {
            self.show_tool_form(ui, storage);
        }

        // Filter and Tool Cards
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Search Tools:").strong());
            ui.add(egui::TextEdit::singleline(&mut self.search_query).desired_width(220.0));
            if !self.search_query.is_empty() && ui.small_button("✖").clicked() {
                self.search_query.clear();
            }

            ui.separator();
            ui.label(
                egui::RichText::new(format!("Total: {} tool(s)", self.registry.tools.len()))
                    .color(egui::Color32::from_rgb(0xaa, 0xaa, 0xaa)),
            );
        });

        ui.add_space(8.0);

        // Render Tool Cards
        self.show_tool_cards(ui, tx, settings.guardrail_tier, workspace_root);
    }

    fn show_guardrails_banner(
        &mut self,
        ui: &mut egui::Ui,
        storage: &Option<Storage>,
        settings: &mut AppSettings,
        _tx: &mpsc::Sender<AppMessage>,
    ) {
        let (r, g, b) = settings.guardrail_tier.badge_rgb();
        let border_color = egui::Color32::from_rgb(r, g, b);

        egui::Frame::group(ui.style())
            .fill(egui::Color32::from_rgba_premultiplied(r / 6, g / 6, b / 6, 25))
            .stroke(egui::Stroke::new(1.5, border_color))
            .inner_margin(egui::Margin::same(10))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("🛡 Active Safety Guardrail Tier:")
                            .strong()
                            .size(14.0),
                    );

                    let tier_text = format!("[ {} ]", settings.guardrail_tier.label());
                    ui.label(
                        egui::RichText::new(tier_text)
                            .strong()
                            .size(14.0)
                            .color(border_color),
                    );

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        // Unrestricted / None
                        let is_none = settings.guardrail_tier == GuardrailTier::None;
                        let none_btn = ui.selectable_label(
                            is_none,
                            egui::RichText::new("None (SOC)").color(egui::Color32::from_rgb(0xff, 0x44, 0x55)),
                        );
                        if none_btn.clicked() && !is_none {
                            if settings.unrestricted_waiver_accepted {
                                settings.guardrail_tier = GuardrailTier::None;
                                if let Some(st) = storage.as_ref() {
                                    let _ = st.save_settings(settings);
                                }
                                self.status_banner = Some((
                                    "Switched to Unrestricted Guardrails.".to_string(),
                                    true,
                                ));
                            } else {
                                self.show_waiver_modal = true;
                                self.waiver_input.clear();
                            }
                        }

                        // Medium / Balanced
                        let is_medium = settings.guardrail_tier == GuardrailTier::Medium;
                        let med_btn = ui.selectable_label(
                            is_medium,
                            egui::RichText::new("Medium (Balanced)").color(egui::Color32::from_rgb(0xff, 0xaa, 0x22)),
                        );
                        if med_btn.clicked() && !is_medium {
                            settings.guardrail_tier = GuardrailTier::Medium;
                            if let Some(st) = storage.as_ref() {
                                let _ = st.save_settings(settings);
                            }
                            self.status_banner = Some((
                                "Switched to Medium (Balanced) Guardrails.".to_string(),
                                true,
                            ));
                        }

                        // Heavy / Sandboxed
                        let is_heavy = settings.guardrail_tier == GuardrailTier::Heavy;
                        let heavy_btn = ui.selectable_label(
                            is_heavy,
                            egui::RichText::new("Heavy (Sandboxed)").color(egui::Color32::from_rgb(0x00, 0xcc, 0x66)),
                        );
                        if heavy_btn.clicked() && !is_heavy {
                            settings.guardrail_tier = GuardrailTier::Heavy;
                            if let Some(st) = storage.as_ref() {
                                let _ = st.save_settings(settings);
                            }
                            self.status_banner = Some((
                                "Switched to Heavy (Sandboxed) Guardrails.".to_string(),
                                true,
                            ));
                        }
                    });
                });

                ui.add_space(4.0);
                ui.label(
                    egui::RichText::new(settings.guardrail_tier.description())
                        .size(12.0)
                        .color(egui::Color32::from_rgb(0xbb, 0xbb, 0xbb)),
                );

                if settings.guardrail_tier == GuardrailTier::None {
                    ui.add_space(4.0);
                    ui.label(
                        egui::RichText::new(
                            "⚠ OPERATING IN UNRESTRICTED MODE: Raw prompt passthrough active. \
                             Ensure all command and tool executions adhere to authorized scope.",
                        )
                        .strong()
                        .size(12.0)
                        .color(egui::Color32::from_rgb(0xff, 0x55, 0x66)),
                    );
                }
            });
    }

    fn show_legal_waiver_modal(
        &mut self,
        ui: &mut egui::Ui,
        storage: &Option<Storage>,
        settings: &mut AppSettings,
        tx: &mpsc::Sender<AppMessage>,
    ) {
        egui::Window::new("⚖ Legal Waiver & Authorization Required")
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .fixed_size(egui::vec2(540.0, 360.0))
            .show(ui.ctx(), |ui| {
                ui.heading(
                    egui::RichText::new("LEGAL ACKNOWLEDGMENT & LIABILITY WAIVER")
                        .size(16.0)
                        .color(egui::Color32::from_rgb(0xff, 0x44, 0x44)),
                );
                ui.separator();
                ui.add_space(6.0);

                egui::ScrollArea::vertical()
                    .max_height(170.0)
                    .show(ui, |ui| {
                        ui.label(
                            egui::RichText::new(LEGAL_DISCLAIMER_TEXT)
                                .size(12.0)
                                .color(egui::Color32::from_rgb(0xdd, 0xdd, 0xdd)),
                        );
                    });

                ui.add_space(8.0);
                ui.separator();
                ui.add_space(6.0);

                ui.label(
                    egui::RichText::new(format!(
                        "To verify and accept, type '{}' below:",
                        REQUIRED_WAIVER_CONFIRMATION
                    ))
                    .strong(),
                );

                ui.horizontal(|ui| {
                    ui.add(
                        egui::TextEdit::singleline(&mut self.waiver_input)
                            .desired_width(200.0)
                            .hint_text(REQUIRED_WAIVER_CONFIRMATION),
                    );

                    let can_accept = self.waiver_input.trim() == REQUIRED_WAIVER_CONFIRMATION;

                    if ui
                        .add_enabled(can_accept, egui::Button::new("✔ Confirm & Unlock"))
                        .clicked()
                    {
                        settings.guardrail_tier = GuardrailTier::None;
                        settings.unrestricted_waiver_accepted = true;
                        let now = Utc::now();
                        settings.unrestricted_waiver_timestamp = Some(now);

                        if let Some(st) = storage.as_ref() {
                            let _ = st.save_settings(settings);
                        }

                        let waiver_hash = compute_waiver_hash();
                        let _ = tx.send(AppMessage::Audit(
                            "waiver.unrestricted_accepted".to_string(),
                            format!("hash: {}", waiver_hash),
                        ));

                        self.show_waiver_modal = false;
                        self.status_banner = Some((
                            "Unrestricted tier unlocked and logged to encrypted audit log.".to_string(),
                            true,
                        ));
                    }

                    if ui.button("Cancel").clicked() {
                        self.show_waiver_modal = false;
                    }
                });
            });
    }

    fn show_high_sensitivity_confirmation(
        &mut self,
        ui: &mut egui::Ui,
        tx: &mpsc::Sender<AppMessage>,
        workspace_root: Option<&str>,
    ) {
        let (id, args) = match &self.confirm_tool_run {
            Some(pair) => pair.clone(),
            None => return,
        };

        let tool_opt = self.registry.get(&id).cloned();

        egui::Window::new("⚠ High-Sensitivity Tool Execution")
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .fixed_size(egui::vec2(480.0, 220.0))
            .show(ui.ctx(), |ui| {
                if let Some(tool) = tool_opt {
                    ui.heading(
                        egui::RichText::new("Confirm High-Sensitivity Operation")
                            .color(egui::Color32::from_rgb(0xff, 0xaa, 0x22)),
                    );
                    ui.add_space(4.0);
                    ui.label(format!("Tool: {} ({})", tool.name, tool.id));
                    ui.label(format!("Execution Mode: {}", tool.execution_type.label()));
                    let cmd_line = tool.format_command_line(None, workspace_root, args.as_deref());
                    ui.add_space(4.0);
                    ui.label("Command to execute:");
                    ui.monospace(&cmd_line);

                    ui.add_space(10.0);
                    ui.horizontal(|ui| {
                        if ui.button("✔ Confirm & Run").clicked() {
                            let _ = tx.send(AppMessage::RunTool { id: tool.id.clone(), args });
                            self.confirm_tool_run = None;
                        }
                        if ui.button("Cancel").clicked() {
                            self.confirm_tool_run = None;
                        }
                    });
                } else {
                    self.confirm_tool_run = None;
                }
            });
    }

    fn open_create_form(&mut self) {
        self.is_creating = true;
        self.editing_id = None;
        self.form_id.clear();
        self.form_name.clear();
        self.form_command.clear();
        self.form_args = "{file}".to_string();
        self.form_execution = ToolExecutionType::TerminalDock;
        self.form_sensitivity = SensitivityLevel::Low;
        self.form_guardrail = GuardrailTier::Heavy;
        self.form_description.clear();
    }

    fn open_edit_form(&mut self, tool: &UserTool) {
        self.is_creating = true;
        self.editing_id = Some(tool.id.clone());
        self.form_id = tool.id.clone();
        self.form_name = tool.name.clone();
        self.form_command = tool.command.clone();
        self.form_args = tool.args_template.clone();
        self.form_execution = tool.execution_type;
        self.form_sensitivity = tool.sensitivity;
        self.form_guardrail = tool.min_guardrail;
        self.form_description = tool.description.clone();
    }

    fn show_tool_form(&mut self, ui: &mut egui::Ui, storage: &Option<Storage>) {
        let is_edit = self.editing_id.is_some();
        let title = if is_edit { "✏ Edit Tool" } else { "➕ Register New Tool" };

        egui::Window::new(title)
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .fixed_size(egui::vec2(520.0, 440.0))
            .show(ui.ctx(), |ui| {
                ui.horizontal(|ui| {
                    ui.label("Tool ID (identifier):");
                    ui.add_enabled(!is_edit, egui::TextEdit::singleline(&mut self.form_id).hint_text("e.g. wireshark"));
                });

                ui.horizontal(|ui| {
                    ui.label("Display Name:");
                    ui.text_edit_singleline(&mut self.form_name);
                });

                ui.horizontal(|ui| {
                    ui.label("Command Binary/Script:");
                    ui.text_edit_singleline(&mut self.form_command);
                });

                ui.horizontal(|ui| {
                    ui.label("Arguments Template:");
                    ui.text_edit_singleline(&mut self.form_args);
                });
                ui.label(
                    egui::RichText::new("Variables: {file} for active file, {workspace} for folder, {input} for prompt text")
                        .size(11.0)
                        .color(egui::Color32::from_rgb(0x88, 0xaa, 0xcc)),
                );

                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    ui.label("Execution Mode:");
                    ui.radio_value(&mut self.form_execution, ToolExecutionType::TerminalDock, "Terminal Dock (CLI)");
                    ui.radio_value(&mut self.form_execution, ToolExecutionType::Detached, "Detached (GUI/App)");
                });

                ui.horizontal(|ui| {
                    ui.label("Sensitivity Level:");
                    ui.radio_value(&mut self.form_sensitivity, SensitivityLevel::Low, "Low (1-Click Run)");
                    ui.radio_value(&mut self.form_sensitivity, SensitivityLevel::High, "High (Confirm Required)");
                });

                ui.horizontal(|ui| {
                    ui.label("Min Guardrail Tier:");
                    ui.radio_value(&mut self.form_guardrail, GuardrailTier::Heavy, "Heavy");
                    ui.radio_value(&mut self.form_guardrail, GuardrailTier::Medium, "Medium");
                    ui.radio_value(&mut self.form_guardrail, GuardrailTier::None, "Unrestricted");
                });

                ui.add_space(6.0);
                ui.label("Description:");
                ui.add(egui::TextEdit::multiline(&mut self.form_description).desired_rows(3).desired_width(480.0));

                ui.add_space(10.0);
                ui.horizontal(|ui| {
                    let can_save = !self.form_id.trim().is_empty() && !self.form_command.trim().is_empty();
                    if ui.add_enabled(can_save, egui::Button::new("💾 Save Tool")).clicked() {
                        let tool = UserTool {
                            id: self.form_id.trim().to_lowercase(),
                            name: if self.form_name.trim().is_empty() { self.form_id.trim().to_string() } else { self.form_name.trim().to_string() },
                            command: self.form_command.trim().to_string(),
                            args_template: self.form_args.trim().to_string(),
                            execution_type: self.form_execution,
                            sensitivity: self.form_sensitivity,
                            min_guardrail: self.form_guardrail,
                            description: self.form_description.trim().to_string(),
                        };
                        self.registry.add_or_update(tool);
                        self.save(storage);
                        self.is_creating = false;
                        self.editing_id = None;
                        self.status_banner = Some(("Tool saved and encrypted into vault.".to_string(), true));
                    }

                    if ui.button("Cancel").clicked() {
                        self.is_creating = false;
                        self.editing_id = None;
                    }
                });
            });
    }

    fn show_tool_cards(
        &mut self,
        ui: &mut egui::Ui,
        tx: &mpsc::Sender<AppMessage>,
        _active_tier: GuardrailTier,
        workspace_root: Option<&str>,
    ) {
        let q = self.search_query.to_lowercase();
        let tools: Vec<UserTool> = self
            .registry
            .tools
            .iter()
            .filter(|t| {
                if q.is_empty() {
                    true
                } else {
                    t.id.to_lowercase().contains(&q)
                        || t.name.to_lowercase().contains(&q)
                        || t.description.to_lowercase().contains(&q)
                        || t.command.to_lowercase().contains(&q)
                }
            })
            .cloned()
            .collect();

        if tools.is_empty() {
            ui.add_space(20.0);
            ui.label(egui::RichText::new("No tools matched your search or registry is empty.").italics());
            return;
        }

        let mut tool_to_delete: Option<String> = None;
        let mut tool_to_edit: Option<UserTool> = None;

        for tool in &tools {
            egui::Frame::group(ui.style())
                .inner_margin(egui::Margin::same(10))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new(&tool.name).strong().size(15.0));

                        ui.label(
                            egui::RichText::new(format!("[{}]", tool.id))
                                .size(12.0)
                                .color(egui::Color32::from_rgb(0x00, 0xbb, 0xee)),
                        );

                        // Mode badge
                        let mode_badge = match tool.execution_type {
                            ToolExecutionType::TerminalDock => "CLI Dock",
                            ToolExecutionType::Detached => "GUI Detached",
                        };
                        ui.label(
                            egui::RichText::new(format!("[{}]", mode_badge))
                                .size(11.0)
                                .color(egui::Color32::from_rgb(0xaa, 0xdd, 0xaa)),
                        );

                        // Sensitivity badge
                        let (sens_text, sens_color) = match tool.sensitivity {
                            SensitivityLevel::Low => ("1-Click Run", egui::Color32::from_rgb(0x88, 0xcc, 0x88)),
                            SensitivityLevel::High => ("Requires Confirm", egui::Color32::from_rgb(0xee, 0xaa, 0x44)),
                        };
                        ui.label(egui::RichText::new(format!("[{}]", sens_text)).size(11.0).color(sens_color));

                        // Min Guardrail tier
                        let (r, g, b) = tool.min_guardrail.badge_rgb();
                        ui.label(
                            egui::RichText::new(format!("Min: {}", tool.min_guardrail.short_label()))
                                .size(11.0)
                                .color(egui::Color32::from_rgb(r, g, b)),
                        );

                        // Actions on the right
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.small_button("🗑").on_hover_text("Delete tool").clicked() {
                                tool_to_delete = Some(tool.id.clone());
                            }
                            if ui.small_button("✏").on_hover_text("Edit tool").clicked() {
                                tool_to_edit = Some(tool.clone());
                            }

                            // Run button
                            let run_btn = ui.button("▶ Run Tool");
                            if run_btn.clicked() {
                                if tool.sensitivity == SensitivityLevel::High {
                                    self.confirm_tool_run = Some((tool.id.clone(), None));
                                } else {
                                    let _ = tx.send(AppMessage::RunTool {
                                        id: tool.id.clone(),
                                        args: None,
                                    });
                                }
                            }
                        });
                    });

                    ui.add_space(4.0);
                    let cmd_preview = tool.format_command_line(None, workspace_root, None);
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("Command:").size(12.0).strong());
                        ui.monospace(egui::RichText::new(&cmd_preview).size(12.0));
                    });

                    if !tool.description.is_empty() {
                        ui.add_space(2.0);
                        ui.label(
                            egui::RichText::new(&tool.description)
                                .size(12.0)
                                .color(egui::Color32::from_rgb(0xaa, 0xaa, 0xaa)),
                        );
                    }
                });

            ui.add_space(6.0);
        }

        if let Some(id) = tool_to_delete {
            self.registry.remove(&id);
            self.status_banner = Some((format!("Removed tool '{}'.", id), true));
        }

        if let Some(tool) = tool_to_edit {
            self.open_edit_form(&tool);
        }
    }
}
