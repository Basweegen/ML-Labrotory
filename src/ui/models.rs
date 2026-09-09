use eframe::egui;
use crate::ollama::api::{OllamaClient, Model};
use crate::ollama::cli::OllamaCli;
use std::sync::mpsc;
use tokio::runtime::Runtime;

pub struct ModelsPanel {
    pull_input: String,
    pulling: bool,
    deleting: Option<String>,
    pull_lines: Vec<String>,
}

impl ModelsPanel {
    pub fn new() -> Self {
        Self {
            pull_input: String::new(),
            pulling: false,
            deleting: None,
            pull_lines: Vec::new(),
        }
    }

    /// A progress line from `ollama pull`. Bounded; overlong lines trimmed.
    pub fn push_progress(&mut self, line: String) {
        let mut line = line.replace('\r', " ");
        if line.len() > 160 {
            line.truncate(160);
            line.push('…');
        }
        if line.trim().is_empty() {
            return;
        }
        self.pull_lines.push(line);
        const CAP: usize = 30;
        if self.pull_lines.len() > CAP {
            let overflow = self.pull_lines.len() - CAP;
            self.pull_lines.drain(0..overflow);
        }
    }

    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        models: &mut Vec<Model>,
        api_client: &Option<OllamaClient>,
        cli_client: &Option<OllamaCli>,
        tx: &mpsc::Sender<crate::ui::app::AppMessage>,
        rt: &Runtime,
    ) {
        ui.horizontal(|ui| {
            ui.add_space(4.0);
            ui.heading(egui::RichText::new("🤖 Model Management").size(22.0).color(egui::Color32::from_rgb(0x00, 0xaa, 0xff)));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.add_space(8.0);
                let refresh_btn = ui.add(
                    egui::Button::new(egui::RichText::new("⟳ Refresh").size(13.0))
                        .fill(egui::Color32::from_rgb(0x33, 0x33, 0x33))
                        .corner_radius(egui::CornerRadius::same(6))
                );
                if refresh_btn.clicked() {
                    let _ = tx.send(crate::ui::app::AppMessage::RefreshModels);
                }
            });
        });

        ui.add_space(8.0);
        ui.separator();
        ui.add_space(8.0);

        // Pull new model
        ui.group(|ui| {
            ui.label(egui::RichText::new("Pull New Model").size(15.0).color(egui::Color32::from_rgb(0x00, 0xaa, 0xff)));
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                let _text_response = ui.add(
                    egui::TextEdit::singleline(&mut self.pull_input)
                        .desired_width(300.0)
                        .font(egui::TextStyle::Monospace)
                        .hint_text("e.g., llama3.2, codellama, mistral, hermes3"),
                );
                if self.pull_input.trim().is_empty() {
                    ui.label(egui::RichText::new("Enter model name…").weak());
                }
                ui.add_space(8.0);
                let pull_enabled = !self.pull_input.trim().is_empty() && !self.pulling;
                let pull_btn = ui.add_enabled(pull_enabled,
                    egui::Button::new(egui::RichText::new("Pull").size(13.0))
                        .fill(if pull_enabled { egui::Color32::from_rgb(0x00, 0x55, 0xaa) } else { egui::Color32::from_rgb(0x33, 0x33, 0x33) })
                        .corner_radius(egui::CornerRadius::same(6))
                );
                if pull_btn.clicked() {
                    self.pull_model(&self.pull_input.clone(), api_client, cli_client, tx, rt);
                }
                if self.pulling {
                    ui.add_space(8.0);
                    ui.spinner();
                    ui.label(egui::RichText::new("Pulling…").color(egui::Color32::from_rgb(0x88, 0xaa, 0xcc)));
                }
                if !self.pull_lines.is_empty() {
                    ui.add_space(4.0);
                    egui::ScrollArea::vertical()
                        .max_height(120.0)
                        .stick_to_bottom(true)
                        .show(ui, |ui| {
                            for line in &self.pull_lines {
                                ui.label(
                                    egui::RichText::new(line)
                                        .size(11.0)
                                        .monospace()
                                        .color(egui::Color32::from_rgb(0x99, 0x99, 0x99)),
                                );
                            }
                        });
                }
            });
        });

        ui.add_space(12.0);
        ui.separator();
        ui.add_space(8.0);

        // Model list
        ui.horizontal(|ui| {
            ui.add_space(4.0);
            let total_disk: u64 = models.iter().map(|m| m.size).sum();
            ui.label(egui::RichText::new(format!("Installed Models ({}) \u{00B7} {} on disk", models.len(), crate::resources::format_bytes(total_disk))).size(15.0).color(egui::Color32::from_rgb(0xcc, 0xcc, 0xcc)));
        });

        ui.add_space(8.0);
        egui::ScrollArea::vertical().show(ui, |ui| {
            if models.is_empty() {
                ui.centered_and_justified(|ui| {
                    ui.add_space(40.0);
                    ui.label(egui::RichText::new("No models installed").size(16.0).color(egui::Color32::from_rgb(0x88, 0x88, 0x88)));
                    ui.add_space(8.0);
                    ui.label(egui::RichText::new("Pull a model to get started").size(13.0).color(egui::Color32::from_rgb(0x66, 0x66, 0x66)));
                });
            } else {
                let mem = crate::resources::system_memory();
                let free = crate::resources::ResourceGuard::evaluate(&mem, &[]).free_for_models_bytes;
                ui.horizontal(|ui| {
                    ui.add_space(4.0);
                    ui.label(egui::RichText::new(format!(
                        "This box: {} ({} free for models)",
                        crate::resources::hardware_tier(free),
                        crate::resources::format_bytes(free),
                    )).size(12.0).color(egui::Color32::from_rgb(0x88, 0xcc, 0x88)));
                });
                ui.add_space(4.0);
                egui::Grid::new("models_grid")
                    .num_columns(7)
                    .spacing([16.0, 12.0])
                    .striped(true)
                    .show(ui, |ui| {
                        ui.strong(egui::RichText::new("Name").size(13.0).color(egui::Color32::from_rgb(0x00, 0xaa, 0xff)));
                        ui.strong(egui::RichText::new("Size").size(13.0).color(egui::Color32::from_rgb(0x00, 0xaa, 0xff)));
                        ui.strong(egui::RichText::new("Params").size(13.0).color(egui::Color32::from_rgb(0x00, 0xaa, 0xff)));
                        ui.strong(egui::RichText::new("Quant").size(13.0).color(egui::Color32::from_rgb(0x00, 0xaa, 0xff)));
                        ui.strong(egui::RichText::new("Fit").size(13.0).color(egui::Color32::from_rgb(0x00, 0xaa, 0xff)));
                        ui.strong(egui::RichText::new("Modified").size(13.0).color(egui::Color32::from_rgb(0x00, 0xaa, 0xff)));
                        ui.strong(egui::RichText::new("Actions").size(13.0).color(egui::Color32::from_rgb(0x00, 0xaa, 0xff)));
                        ui.end_row();

                        for model in models.iter() {
                            ui.label(egui::RichText::new(&model.name).size(13.0).color(egui::Color32::WHITE));
                            ui.label(egui::RichText::new(crate::resources::format_bytes(model.size)).size(13.0).color(egui::Color32::from_rgb(0xaa, 0xaa, 0xaa)));
                            {
                                let (params, quant) = match &model.details {
                                    Some(d) => (
                                        if d.parameter_size.is_empty() { "\u{2014}".to_string() } else { d.parameter_size.clone() },
                                        if d.quantization_level.is_empty() { "\u{2014}".to_string() } else { d.quantization_level.clone() },
                                    ),
                                    None => ("\u{2014}".to_string(), "\u{2014}".to_string()),
                                };
                                ui.label(egui::RichText::new(params).size(13.0).color(egui::Color32::from_rgb(0xaa, 0xaa, 0xaa)));
                                ui.label(egui::RichText::new(quant).size(13.0).color(egui::Color32::from_rgb(0xaa, 0xaa, 0xaa)));
                            }
                            {
                                let (txt, (rr, gg, bb)) = crate::resources::fit_label(model.size, free);
                                ui.label(egui::RichText::new(txt).size(13.0).color(egui::Color32::from_rgb(rr, gg, bb)));
                            }
                            ui.label(egui::RichText::new(&model.modified_at[..19.min(model.modified_at.len())].replace('T', " ")).size(13.0).color(egui::Color32::from_rgb(0xaa, 0xaa, 0xaa)));

                            ui.horizontal(|ui| {
                                let select_btn = ui.add(
                                    egui::Button::new(egui::RichText::new("Select").size(13.0))
                                        .fill(egui::Color32::from_rgb(0x00, 0x55, 0xaa))
                                        .corner_radius(egui::CornerRadius::same(6))
                                        .frame(false)
                                );
                                if select_btn.on_hover_text("Set as active model").clicked() {
                                    let _ = tx.send(crate::ui::app::AppMessage::ModelSelected(model.name.clone()));
                                }
                                let delete_btn = ui.add(
                                    egui::Button::new(egui::RichText::new("🗑").size(14.0))
                                        .fill(egui::Color32::from_rgb(0x44, 0x22, 0x22))
                                        .corner_radius(egui::CornerRadius::same(6))
                                        .frame(false)
                                );
                                if self.deleting.as_deref() == Some(&model.name) {
                                    if ui.small_button("Confirm").clicked() {
                                        self.delete_model(&model.name, api_client, cli_client, tx, rt);
                                        self.deleting = None;
                                    }
                                    if ui.small_button("Keep").clicked() {
                                        self.deleting = None;
                                    }
                                } else if delete_btn.on_hover_text("Delete").clicked() {
                                    self.deleting = Some(model.name.clone());
                                }
                            });
                            ui.end_row();
                        }
                    });
            }
            ui.add_space(4.0);
        });
    }

    fn pull_model(
        &mut self,
        name: &str,
        api_client: &Option<OllamaClient>,
        cli_client: &Option<OllamaCli>,
        tx: &mpsc::Sender<crate::ui::app::AppMessage>,
        rt: &Runtime,
    ) {
        self.pulling = true;
        self.pull_lines.clear();
        let name = name.to_string();
        let tx = tx.clone();

        // Prefer CLI for pull (shows progress)
        if let Some(client) = cli_client {
            let client = client.clone();
            let tx_done = tx.clone();
            let (ptx, mut prx) = tokio::sync::mpsc::channel::<String>(100);
            rt.spawn(async move {
                while let Some(line) = prx.recv().await {
                    let _ = tx.send(crate::ui::app::AppMessage::PullProgress(line));
                }
            });
            rt.spawn(async move {
                let result = client.pull_model(&name, Some(ptx)).await;
                let _ = tx_done.send(crate::ui::app::AppMessage::ModelPulled(
                    result.map(|_| name),
                ));
            });
        } else if let Some(client) = api_client {
            let client = client.clone();
            rt.spawn(async move {
                let result = client.pull_model(&name).await;
                let _ = tx.send(crate::ui::app::AppMessage::ModelPulled(result.map(|_| name)));
            });
        }
    }

    pub fn note_transfer_finished(&mut self) {
        self.note_pull_finished(true);
    }

    /// On failure the CLI output stays visible (it holds the error);
    /// on success the lines clear back to a clean panel.
    pub fn note_pull_finished(&mut self, ok: bool) {
        self.pulling = false;
        if ok {
            self.pull_lines.clear();
        }
    }

    fn delete_model(
        &mut self,
        name: &str,
        api_client: &Option<OllamaClient>,
        cli_client: &Option<OllamaCli>,
        tx: &mpsc::Sender<crate::ui::app::AppMessage>,
        rt: &Runtime,
    ) {
        self.deleting = Some(name.to_string());
        let name = name.to_string();
        let tx = tx.clone();

        if let Some(client) = cli_client {
            let client = client.clone();
            rt.spawn(async move {
                let result = client.delete_model(&name).await;
                let _ = tx.send(crate::ui::app::AppMessage::ModelDeleted(result.map(|_| name)));
            });
        } else if let Some(client) = api_client {
            let client = client.clone();
            rt.spawn(async move {
                let result = client.delete_model(&name).await;
                let _ = tx.send(crate::ui::app::AppMessage::ModelDeleted(result.map(|_| name)));
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn progress_caps_and_trims() {
        let mut p = ModelsPanel::new();
        for i in 0..40 {
            p.push_progress(format!("line {i}"));
        }
        assert_eq!(p.pull_lines.len(), 30);
        assert_eq!(p.pull_lines[0], "line 10");
        p.push_progress("   ".to_string());
        assert_eq!(p.pull_lines.len(), 30);
        p.push_progress("x".repeat(200));
        assert!(p.pull_lines.last().unwrap().len() <= 165);
        p.note_transfer_finished();
        assert!(p.pull_lines.is_empty());
        assert!(!p.pulling);
    }
}
