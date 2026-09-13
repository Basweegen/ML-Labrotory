// Copyright 2026 Sean M. Stow. All rights reserved.
use eframe::egui;
use crate::storage::{AppSettings, Theme};
use crate::storage::Storage;
use std::sync::mpsc;

pub struct SettingsPanel;

impl SettingsPanel {
    pub fn new() -> Self {
        Self
    }

    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        settings: &mut AppSettings,
        storage: &Storage,
        tx: &mpsc::Sender<crate::ui::app::AppMessage>,
    ) {
        ui.add_space(16.0);
        ui.heading(egui::RichText::new("Settings").size(22.0).color(egui::Color32::from_rgb(0x00, 0xaa, 0xff)));
        ui.add_space(8.0);
        ui.separator();
        ui.add_space(16.0);

        // Theme section
        ui.label(egui::RichText::new("Appearance").size(16.0).color(egui::Color32::from_rgb(0xcc, 0xcc, 0xcc)));
        ui.add_space(8.0);

        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Theme:").size(13.0).color(egui::Color32::from_rgb(0xcc, 0xcc, 0xcc)));
            ui.add_space(16.0);

            egui::ComboBox::from_id_salt("theme_selector")
                .selected_text(match settings.theme {
                    Theme::Dark => "🌙 Dark",
                    Theme::Light => "☀️ Light",
                    Theme::System => "💻 System",
                })
                .width(180.0)
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut settings.theme, Theme::Dark, "🌙 Dark");
                    ui.selectable_value(&mut settings.theme, Theme::Light, "☀️ Light");
                    ui.selectable_value(&mut settings.theme, Theme::System, "💻 System");
                });
        });

        ui.add_space(8.0);
        ui.horizontal(|ui| {
            ui.checkbox(&mut settings.show_avatar, "Show assistant face (reactive avatar)");
        });
        ui.label(egui::RichText::new("Reactive companion face (Stack-chan style) atop navigation tabs + minis on slot cards.").size(11.0).color(egui::Color32::from_rgb(0x88, 0x88, 0x88)));
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Face size:").size(13.0).color(egui::Color32::from_rgb(0xcc, 0xcc, 0xcc)));
            ui.add_space(8.0);
            let mut sz = settings.avatar_size.clamp(40.0, 80.0);
            if ui.add(egui::Slider::new(&mut sz, 40.0..=80.0).suffix(" px")).changed() {
                settings.avatar_size = sz;
            }
        });

        ui.add_space(12.0);
        ui.separator();
        ui.add_space(12.0);

        // Custom Page Layout section
        ui.label(egui::RichText::new("Custom Page Layout").size(16.0).color(egui::Color32::from_rgb(0xcc, 0xcc, 0xcc)));
        ui.add_space(8.0);
        ui.label(
            egui::RichText::new("Personalize dashboard navigation: choose startup tab, toggle panel visibility, arrange tab sequence, and configure split views.")
                .size(11.0)
                .color(egui::Color32::from_rgb(0x88, 0x88, 0x88)),
        );
        ui.add_space(8.0);

        // 1. Default Landing Tab
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Startup Landing Tab:").size(13.0).color(egui::Color32::from_rgb(0xcc, 0xcc, 0xcc)));
            ui.add_space(16.0);

            let cur_default = settings.default_tab.clone();
            egui::ComboBox::from_id_salt("default_tab_selector")
                .selected_text(&cur_default)
                .width(200.0)
                .show_ui(ui, |ui| {
                    for tab in crate::ui::app::Tab::all() {
                        let label_str = tab.label().to_string();
                        ui.selectable_value(&mut settings.default_tab, label_str, format!("{} {}", tab.icon(), tab.label()));
                    }
                });
        });
        ui.add_space(6.0);

        // 2. Chat Split View Default
        ui.horizontal(|ui| {
            ui.checkbox(
                &mut settings.chat_split_view_default,
                "Default Chat to Side-by-Side Dual Split View (Multi-Model)",
            );
        });
        ui.add_space(10.0);

        // 3. Tab Visibility Matrix
        ui.label(egui::RichText::new("Tab Visibility:").size(13.0).color(egui::Color32::from_rgb(0xcc, 0xcc, 0xcc)));
        ui.add_space(4.0);
        egui::Grid::new("tab_visibility_grid").spacing([16.0, 6.0]).show(ui, |ui| {
            let all_tabs = crate::ui::app::Tab::all();
            for (idx, tab) in all_tabs.iter().enumerate() {
                let label = tab.label();
                let is_locked = !tab.is_closable();
                let mut visible = is_locked || settings.visible_tabs.iter().any(|v| v == label || (v == "Relay" && *tab == crate::ui::app::Tab::Relay));

                let checkbox = ui.add_enabled(!is_locked, egui::Checkbox::new(&mut visible, format!("{} {}", tab.icon(), label)));
                if !is_locked && checkbox.changed() {
                    if visible {
                        if !settings.visible_tabs.iter().any(|v| v == label) {
                            settings.visible_tabs.push(label.to_string());
                        }
                    } else {
                        settings.visible_tabs.retain(|v| v != label && !(v == "Relay" && *tab == crate::ui::app::Tab::Relay));
                    }
                }
                if (idx + 1) % 4 == 0 {
                    ui.end_row();
                }
            }
        });
        ui.label(
            egui::RichText::new("Note: Chat and Settings are permanently enabled core anchors to prevent navigation lockouts.")
                .size(11.0)
                .italics()
                .color(egui::Color32::from_rgb(0x88, 0x88, 0x88)),
        );
        ui.add_space(10.0);

        // 4. Tab Display Order
        ui.label(egui::RichText::new("Tab Display Order:").size(13.0).color(egui::Color32::from_rgb(0xcc, 0xcc, 0xcc)));
        ui.add_space(4.0);

        let mut move_up: Option<usize> = None;
        let mut move_down: Option<usize> = None;
        let order_len = settings.tab_order.len();

        ui.vertical(|ui| {
            for i in 0..order_len {
                let tab_name = settings.tab_order[i].clone();
                let icon = crate::ui::app::Tab::from_label(&tab_name).map(|t| t.icon()).unwrap_or("📌");
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new(format!("{}. {} {}", i + 1, icon, tab_name)).size(12.0));
                    ui.add_space(8.0);
                    if i > 0 {
                        if ui.small_button("▲").on_hover_text("Move tab up").clicked() {
                            move_up = Some(i);
                        }
                    }
                    if i + 1 < order_len {
                        if ui.small_button("▼").on_hover_text("Move tab down").clicked() {
                            move_down = Some(i);
                        }
                    }
                });
            }
        });

        if let Some(i) = move_up {
            settings.tab_order.swap(i, i - 1);
        }
        if let Some(i) = move_down {
            settings.tab_order.swap(i, i + 1);
        }

        ui.add_space(6.0);
        if ui.button("↩ Reset Tab Order & Visibility").clicked() {
            settings.tab_order = crate::storage::default_tab_order();
            settings.visible_tabs = crate::storage::default_visible_tabs();
            settings.default_tab = crate::storage::default_tab_name();
        }

        ui.add_space(12.0);
        ui.separator();
        ui.add_space(12.0);

        // Font size section
        ui.label(egui::RichText::new("Editor").size(16.0).color(egui::Color32::from_rgb(0xcc, 0xcc, 0xcc)));
        ui.add_space(8.0);

        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Font size:").size(13.0).color(egui::Color32::from_rgb(0xcc, 0xcc, 0xcc)));
            ui.add_space(16.0);

            let mut font_size = settings.font_size;
            let response = ui.add(
                egui::DragValue::new(&mut font_size)
                    .speed(0.5)
                    .range(10.0..=32.0)
                    .suffix(" pt"),
            );
            if response.changed() {
                settings.font_size = font_size;
            }

            ui.add_space(8.0);
            ui.label(egui::RichText::new(format!("{:.0} pt", font_size)).size(12.0).color(egui::Color32::from_rgb(0x88, 0x88, 0x88)));
        });

        ui.add_space(12.0);
        ui.separator();
        ui.add_space(12.0);

        // Ollama URL section
        ui.label(egui::RichText::new("Ollama").size(16.0).color(egui::Color32::from_rgb(0xcc, 0xcc, 0xcc)));
        ui.add_space(8.0);

        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Ollama URL:").size(13.0).color(egui::Color32::from_rgb(0xcc, 0xcc, 0xcc)));
            ui.add_space(8.0);

            let mut url = settings.ollama_url.clone();
            let response = ui.add(
                egui::TextEdit::singleline(&mut url)
                    .desired_width(320.0)
                    .font(egui::TextStyle::Monospace)
                    .hint_text("http://localhost:11434"),
            );
            if response.changed() {
                settings.ollama_url = url.clone();
            }

            ui.add_space(8.0);
            if ui.button("Test Connection").clicked() {
                let _ = tx.send(crate::ui::app::AppMessage::RefreshModels);
            }
        });

        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.checkbox(
                &mut settings.allow_remote,
                "Allow remote (non-localhost) Ollama server",
            );
        });
        ui.label(egui::RichText::new("Off = localhost only (recommended). Only enable for a LAN server you trust; the app will relay your prompts there.").size(11.0).color(egui::Color32::from_rgb(0x88, 0x88, 0x88)));

        ui.add_space(8.0);
        ui.label(egui::RichText::new("Changes to URL take effect immediately").size(11.0).color(egui::Color32::from_rgb(0x88, 0x88, 0x88)));

        ui.add_space(12.0);
        ui.separator();
        ui.add_space(12.0);

        // Persona section: one identity shared by every model/slot.
        ui.label(egui::RichText::new("Persona (same for every model)").size(16.0).color(egui::Color32::from_rgb(0xcc, 0xcc, 0xcc)));
        ui.add_space(8.0);
        ui.label(egui::RichText::new("Who the assistant is, no matter which model is loaded. Saved across restarts.").size(11.0).color(egui::Color32::from_rgb(0x88, 0x88, 0x88)));
        ui.add_space(4.0);
        {
            let mut persona = settings.persona.clone();
            let resp = ui.add(
                egui::TextEdit::multiline(&mut persona)
                    .desired_rows(3)
                    .desired_width(f32::INFINITY)
                    .hint_text("e.g. You are ML Lab, calm and direct..."),
            );
            if resp.changed() {
                settings.persona = persona;
            }
        }
        ui.add_space(12.0);
        ui.separator();
        ui.add_space(12.0);

        // Memory section: long-term facts injected into every chat.
        ui.label(egui::RichText::new("Memory (facts kept across models + restarts)").size(16.0).color(egui::Color32::from_rgb(0xcc, 0xcc, 0xcc)));
        ui.add_space(8.0);
        ui.label(egui::RichText::new("Names, preferences, project facts. Sent to every model with each message.").size(11.0).color(egui::Color32::from_rgb(0x88, 0x88, 0x88)));
        ui.add_space(4.0);
        {
            let mut mem = settings.memory.clone();
            let resp = ui.add(
                egui::TextEdit::multiline(&mut mem)
                    .desired_rows(4)
                    .desired_width(f32::INFINITY)
                    .hint_text("e.g. User is Daddy. MacBook Air 2017, 8GB RAM. Project: ml_lab..."),
            );
            if resp.changed() {
                settings.memory = mem;
            }
        }
        ui.add_space(12.0);
        ui.separator();
        ui.add_space(12.0);

        // Chat & Model Rules section: global operational rules and multi-model collaboration directives.
        ui.label(egui::RichText::new("Chat & Model Rules").size(16.0).color(egui::Color32::from_rgb(0xcc, 0xcc, 0xcc)));
        ui.add_space(8.0);
        ui.label(
            egui::RichText::new("Global operational guidelines and collaborative rules enforced across all models and chat sessions. Injected into every model's system prompt.")
                .size(11.0)
                .color(egui::Color32::from_rgb(0x88, 0x88, 0x88)),
        );
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            ui.checkbox(
                &mut settings.auto_assist_rules,
                "Enable Intelligent Multi-Model Auto-Assist (Complementary Gap-Filling)",
            );
        });
        ui.horizontal(|ui| {
            ui.checkbox(
                &mut settings.swarm_auto_assist,
                "Enable Autonomous Swarm Handoff (Slot 1 completes -> Slot 2 automatically assists)",
            );
        });
        ui.label(
            egui::RichText::new("When enabled, models in multi-model chats inspect prior outputs, avoid duplicating identical code, and seamlessly hand off the task between slots.")
                .size(11.0)
                .color(egui::Color32::from_rgb(0x88, 0x88, 0x88)),
        );
        ui.add_space(8.0);
        ui.horizontal_wrapped(|ui| {
            ui.label(egui::RichText::new("Rule Presets:").size(12.0).color(egui::Color32::from_rgb(0xcc, 0xcc, 0xcc)));
            ui.add_space(4.0);
            if ui.small_button("🤝 Auto-Assist").on_hover_text("Add complementary collaboration rule").clicked() {
                if !settings.chat_rules.contains("Complementary Cooperation") {
                    if !settings.chat_rules.is_empty() && !settings.chat_rules.ends_with('\n') {
                        settings.chat_rules.push('\n');
                    }
                    settings.chat_rules.push_str("• Complementary Cooperation: Never duplicate prior model output. Complete missing parts if partial; concur and provide enhancements if complete.\n");
                }
            }
            if ui.small_button("⚡ No Placeholders").on_hover_text("Add production-grade code rule").clicked() {
                if !settings.chat_rules.contains("Production-Grade") {
                    if !settings.chat_rules.is_empty() && !settings.chat_rules.ends_with('\n') {
                        settings.chat_rules.push('\n');
                    }
                    settings.chat_rules.push_str("• Production-Grade Code: Never output placeholders, ellipses (...), or 'rest of code here'. Output 100% complete, runnable code.\n");
                }
            }
            if ui.small_button("🛡 Cyber Defense").on_hover_text("Add security-first rule").clicked() {
                if !settings.chat_rules.contains("Security-First") {
                    if !settings.chat_rules.is_empty() && !settings.chat_rules.ends_with('\n') {
                        settings.chat_rules.push('\n');
                    }
                    settings.chat_rules.push_str("• Security-First Architecture: Prioritize input sanitization, memory safety, least privilege, and defensive validation.\n");
                }
            }
            if ui.small_button("🔄 Reset Defaults").on_hover_text("Reset rules to turnkey default").clicked() {
                settings.chat_rules = crate::storage::default_chat_rules();
            }
        });
        ui.add_space(4.0);
        {
            let mut rules = settings.chat_rules.clone();
            let resp = ui.add(
                egui::TextEdit::multiline(&mut rules)
                    .desired_rows(5)
                    .desired_width(f32::INFINITY)
                    .hint_text("Enter operational rules enforced on all chat models (e.g. 1. Never output placeholders. 2. Complement other models instead of repeating...)"),
            );
            if resp.changed() {
                settings.chat_rules = rules;
            }
        }
        ui.add_space(12.0);
        ui.separator();
        ui.add_space(12.0);

        // Context section: how much history rides along with each prompt.
        ui.label(egui::RichText::new("Context (history sent with each prompt)").size(16.0).color(egui::Color32::from_rgb(0xcc, 0xcc, 0xcc)));
        ui.add_space(8.0);
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Recent turns:").size(13.0).color(egui::Color32::from_rgb(0xcc, 0xcc, 0xcc)));
            ui.add_space(8.0);
            let mut depth = settings.history_depth;
            let resp = ui.add(
                egui::Slider::new(&mut depth, 1..=50).text("turns"),
            );
            if resp.changed() {
                settings.history_depth = depth;
            }
        });
        ui.label(egui::RichText::new("Higher = better continuity, slower + pricier. Saved with the rest.").size(11.0).color(egui::Color32::from_rgb(0x88, 0x88, 0x88)));

        ui.add_space(12.0);
        ui.separator();
        ui.add_space(12.0);

        // Inference Performance (CPU Threading)
        ui.label(egui::RichText::new("Inference Performance (CPU Threads)").size(16.0).color(egui::Color32::from_rgb(0xcc, 0xcc, 0xcc)));
        ui.add_space(8.0);
        let optimal = crate::ollama::api::ChatOptions::optimal_threads();
        let total_avail = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4) as u32;
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Ollama CPU Threads:").size(13.0).color(egui::Color32::from_rgb(0xcc, 0xcc, 0xcc)));
            ui.add_space(8.0);
            let mut threads = settings.num_threads;
            let thread_label = if threads == 0 { format!("0 (Auto: {optimal} P-threads)") } else { format!("{threads} threads") };
            let resp = ui.add(
                egui::Slider::new(&mut threads, 0..=total_avail)
                    .text(thread_label),
            );
            if resp.changed() {
                settings.num_threads = threads;
            }
        });
        ui.label(
            egui::RichText::new(format!(
                "0 = Auto (detected {total_avail} logical threads; auto-clamps to {optimal} P-core threads to prevent hybrid spinlock contention, speeding up inference by up to 3x)."
            ))
            .size(11.0)
            .color(egui::Color32::from_rgb(0x88, 0x88, 0x88)),
        );

        ui.add_space(12.0);
        ui.separator();
        ui.add_space(12.0);

        // Voice section: piper voice + whisper model names (binaries live in ~/.local/bin).
        ui.label(egui::RichText::new("Voice").size(16.0).color(egui::Color32::from_rgb(0xcc, 0xcc, 0xcc)));
        ui.add_space(8.0);
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("TTS voice:").size(13.0).color(egui::Color32::from_rgb(0xcc, 0xcc, 0xcc)));
            ui.add_space(8.0);
            let mut v = settings.tts_voice.clone();
            let resp = ui.add(
                egui::TextEdit::singleline(&mut v)
                    .desired_width(280.0)
                    .hint_text("en_US-lessac-medium"),
            );
            if resp.changed() {
                settings.tts_voice = v;
            }
        });
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("STT model:").size(13.0).color(egui::Color32::from_rgb(0xcc, 0xcc, 0xcc)));
            ui.add_space(8.0);
            let mut m = settings.stt_model.clone();
            let resp = ui.add(
                egui::TextEdit::singleline(&mut m)
                    .desired_width(280.0)
                    .hint_text("ggml-base.en.bin"),
            );
            if resp.changed() {
                settings.stt_model = m;
            }
        });
        ui.label(egui::RichText::new("Takes effect on restart. Saved with the rest.").size(11.0).color(egui::Color32::from_rgb(0x88, 0x88, 0x88)));

        ui.add_space(12.0);
        ui.separator();
        ui.add_space(12.0);

        // Safety Guardrails section
        ui.label(egui::RichText::new("Safety Guardrails & Operational Scope").size(16.0).color(egui::Color32::from_rgb(0xcc, 0xcc, 0xcc)));
        ui.add_space(8.0);

        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Guardrail Tier:").size(13.0).color(egui::Color32::from_rgb(0xcc, 0xcc, 0xcc)));
            ui.add_space(16.0);

            let (r, g, b) = settings.guardrail_tier.badge_rgb();
            egui::ComboBox::from_id_salt("guardrail_tier_selector")
                .selected_text(
                    egui::RichText::new(settings.guardrail_tier.label())
                        .color(egui::Color32::from_rgb(r, g, b))
                )
                .width(220.0)
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut settings.guardrail_tier, crate::guardrails::GuardrailTier::Heavy, "Heavy (Sandboxed)");
                    ui.selectable_value(&mut settings.guardrail_tier, crate::guardrails::GuardrailTier::Medium, "Medium (Balanced)");
                    if settings.unrestricted_waiver_accepted {
                        ui.selectable_value(&mut settings.guardrail_tier, crate::guardrails::GuardrailTier::None, "None (Unrestricted / SOC)");
                    }
                });
        });

        ui.add_space(4.0);
        ui.label(
            egui::RichText::new(settings.guardrail_tier.description())
                .size(11.0)
                .color(egui::Color32::from_rgb(0x88, 0x88, 0x88)),
        );

        if settings.guardrail_tier == crate::guardrails::GuardrailTier::None {
            ui.add_space(4.0);
            ui.label(
                egui::RichText::new(format!(
                    "⚖ Legal Waiver Acknowledged: {}",
                    settings.unrestricted_waiver_timestamp.map(|t| t.to_rfc3339()).unwrap_or_else(|| "Yes".to_string())
                ))
                .size(11.0)
                .color(egui::Color32::from_rgb(0xee, 0x66, 0x66)),
            );
        } else if !settings.unrestricted_waiver_accepted {
            ui.add_space(4.0);
            ui.label(
                egui::RichText::new("Note: Switching to Unrestricted mode requires reviewing and typing 'I ACCEPT' in the Tools tab.")
                    .size(11.0)
                    .italics()
                    .color(egui::Color32::from_rgb(0xaa, 0xaa, 0xaa)),
            );
        }

        ui.add_space(12.0);
        ui.separator();
        ui.add_space(12.0);

        ui.label(
            egui::RichText::new(format!(
                "ML Lab v{} \u{00B7} AGPL-3.0-or-later; commercial licenses: see README",
                env!("CARGO_PKG_VERSION")
            ))
            .size(11.0)
            .color(egui::Color32::from_rgb(0x88, 0x88, 0x88)),
        );
        ui.add_space(8.0);

        // Save settings button
        ui.horizontal(|ui| {
            if ui.button(egui::RichText::new("💾 Save Settings").size(13.0)).clicked() {
                let _ = storage.save_settings(settings);
            }
            ui.add_space(8.0);
            if ui.button(egui::RichText::new("↩ Reset to Defaults").size(13.0)).clicked() {
                *settings = AppSettings::default();
            }
        });
    }
}
