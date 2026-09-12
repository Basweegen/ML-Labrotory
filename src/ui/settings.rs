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
        ui.label(egui::RichText::new("Stack-chan style face above chat + minis on slot cards.").size(11.0).color(egui::Color32::from_rgb(0x88, 0x88, 0x88)));
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
