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
                let _ = storage.save_ollama_url(&url);
            }

            ui.add_space(8.0);
            if ui.button("Test Connection").clicked() {
                let _ = tx.send(crate::ui::app::AppMessage::RefreshModels);
            }
        });

        ui.add_space(8.0);
        ui.label(egui::RichText::new("Changes to URL take effect immediately").size(11.0).color(egui::Color32::from_rgb(0x88, 0x88, 0x88)));

        ui.add_space(12.0);
        ui.separator();
        ui.add_space(12.0);

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
