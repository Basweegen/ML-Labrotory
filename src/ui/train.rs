//! Train tab: the self-learning center. Past chats already feed the
//! profiler network as experiences; here that learning is visible and
//! controllable: dataset size, exploration/exploitation knobs, on-demand
//! training, and reset.

use eframe::egui;
use std::sync::mpsc;

use crate::neural::ModelProfileNetwork;
use crate::storage::Storage;
use crate::ui::app::AppMessage;
use crate::ui::neural_viz::NeuralVizPanel;

pub struct TrainPanel {
    status: String,
    dirty: bool,
    reset_requested: bool,
}

impl TrainPanel {
    pub fn new() -> Self {
        Self {
            status: String::new(),
            dirty: false,
            reset_requested: false,
        }
    }

    pub fn take_dirty(&mut self) -> bool {
        std::mem::replace(&mut self.dirty, false)
    }

    pub fn take_reset(&mut self) -> bool {
        std::mem::replace(&mut self.reset_requested, false)
    }

    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        network: &mut ModelProfileNetwork,
        neural_panel: &mut NeuralVizPanel,
        _storage: &Storage,
        _tx: &mpsc::Sender<AppMessage>,
    ) {
        ui.horizontal(|ui| {
            ui.add_space(4.0);
            ui.heading(
                egui::RichText::new("🎓 Train")
                    .size(22.0)
                    .color(egui::Color32::from_rgb(0x00, 0xaa, 0xff)),
            );
        });
        ui.add_space(4.0);
        ui.label(
            egui::RichText::new(
                "Every finished chat turn trains the profiler (8 signal features: \
                 sizes, latency, outcome, role, model). Nothing leaves this machine.",
            )
            .size(12.0)
            .color(egui::Color32::from_rgb(0x88, 0x88, 0x88)),
        );
        ui.add_space(8.0);
        ui.separator();
        ui.add_space(8.0);

        // ---- dataset ----
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Dataset").size(15.0).strong());
        });
        ui.label(
            egui::RichText::new(format!(
                "{} experiences (cap 10,000) \u{00B7} avg reward {:.2} over {} turns",
                network.experience_buffer.len(),
                network.avg_performance(),
                network.performance_history.len(),
            ))
            .size(13.0),
        );
        ui.add_space(8.0);

        // ---- knobs ----
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Learning").size(15.0).strong());
        });
        if ui
            .add(
                egui::Slider::new(&mut network.epsilon, 0.0..=0.5)
                    .text("Explore (random pick chance)")
                    .step_by(0.01),
            )
            .changed()
        {
            network.epsilon = network.epsilon.clamp(0.0, 0.5);
            self.dirty = true;
        }
        if ui
            .add(
                egui::Slider::new(&mut network.learning_rate, 0.0001..=0.1)
                    .logarithmic(true)
                    .text("Learning rate"),
            )
            .changed()
        {
            network.learning_rate = network.learning_rate.clamp(1e-6, 0.1);
            self.dirty = true;
        }
        if self.dirty {
            ui.label(
                egui::RichText::new("Knobs changed - saved on this frame")
                    .size(11.0)
                    .color(egui::Color32::from_rgb(0x88, 0xcc, 0x88)),
            );
        }
        ui.add_space(8.0);

        // ---- actions ----
        ui.horizontal(|ui| {
            if ui
                .small_button("Train 10 steps")
                .on_hover_text("Run ten training batches right now")
                .clicked()
            {
                let mut first = None;
                let mut last = 0.0;
                let mut ran = 0;
                for _ in 0..10 {
                    match network.train_step() {
                        Ok(loss) if loss.is_finite() && loss > 0.0 => {
                            if first.is_none() {
                                first = Some(loss);
                            }
                            last = loss;
                            ran += 1;
                            neural_panel.add_training_loss(loss);
                            neural_panel.last_loss = Some(loss);
                        }
                        _ => {}
                    }
                }
                self.status = if ran == 0 {
                    "Need 4+ experiences before training starts (chat first)".to_string()
                } else {
                    format!(
                        "Trained {ran} batches: loss {:.4} \u{2192} {:.4}",
                        first.unwrap_or(last),
                        last
                    )
                };
                self.dirty = true;
            }
            if ui
                .small_button("Reset network")
                .on_hover_text("Discard all training and start fresh")
                .clicked()
            {
                self.reset_requested = true;
            }
        });
        ui.add_space(4.0);
        if !self.status.is_empty() {
            ui.label(
                egui::RichText::new(self.status.clone())
                    .size(12.0)
                    .color(egui::Color32::from_rgb(0x99, 0x99, 0x99)),
            );
        }
    }
}
