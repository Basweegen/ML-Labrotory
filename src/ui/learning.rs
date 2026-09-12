// Copyright 2026 Sean M. Stow. All rights reserved.
use crate::neural::{
    extract_probe_features, BenchmarkMetrics,
    ModelProfileNetwork, OptimizerType, MAX_EXPERIENCE_BUFFER,
};
use crate::ollama::api::Model;
use crate::ui::app::AppMessage;
use egui::{Align, Color32, Layout, RichText, Stroke, Ui};
use egui_plot::{Line, Plot};
use std::sync::mpsc;
use std::time::Instant;

/// Active sub-tab in the Learning Algorithm panel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LearningSubtab {
    Overview,
    Optimizer,
    MemoryNetwork,
    SynapticStats,
    Benchmark,
    Playground,
}

impl LearningSubtab {
    pub fn label(self) -> &'static str {
        match self {
            LearningSubtab::Overview => "📊 Overview & Loss",
            LearningSubtab::Optimizer => "⚙️ Optimizer & Tuning",
            LearningSubtab::MemoryNetwork => "🧠 Associative Memory",
            LearningSubtab::SynapticStats => "🔬 Synaptic Inspector",
            LearningSubtab::Benchmark => "🎯 Benchmark Suite",
            LearningSubtab::Playground => "🧪 Live Playground",
        }
    }

    pub fn all() -> [LearningSubtab; 6] {
        [
            LearningSubtab::Overview,
            LearningSubtab::Optimizer,
            LearningSubtab::MemoryNetwork,
            LearningSubtab::SynapticStats,
            LearningSubtab::Benchmark,
            LearningSubtab::Playground,
        ]
    }
}

/// Output from running a test prediction in the Live Playground.
#[derive(Debug, Clone)]
pub struct TestPredictionResult {
    pub prompt: String,
    pub probe_features: Vec<f32>,
    pub memory_attention: Vec<(usize, String, String, f32)>, // (slot_id, domain, label, weight)
    pub retrieved_memory_vec: Vec<f32>,
    pub quantum_entropy: f32,
    pub quantum_probs: Vec<f32>,
    pub best_slot: usize,
    pub best_domain: String,
    pub confidence: f32,
    pub distribution: Vec<f32>,
}

/// Comprehensive Learning Algorithm & ML Optimization Panel.
pub struct LearningPanel {
    pub active_subtab: LearningSubtab,
    pub benchmark_results: Option<BenchmarkMetrics>,
    pub last_benchmark_time: Option<Instant>,
    pub status_message: Option<String>,
    pub test_prompt: String,
    pub test_result: Option<TestPredictionResult>,
    pub memory_filter: String,
    pub selected_memory_slot: Option<usize>,
    pub inject_domain: String,
    pub inject_label: String,
    pub inject_reward: f32,
    pub show_manual_inject: bool,
}

impl Default for LearningPanel {
    fn default() -> Self {
        Self {
            active_subtab: LearningSubtab::Overview,
            benchmark_results: None,
            last_benchmark_time: None,
            status_message: None,
            test_prompt: "fn quicksort<T: Ord>(arr: &mut [T]) { /* recursive partition */ }".to_string(),
            test_result: None,
            memory_filter: String::new(),
            selected_memory_slot: None,
            inject_domain: "Coder".to_string(),
            inject_label: "Custom algorithm pattern".to_string(),
            inject_reward: 1.0,
            show_manual_inject: false,
        }
    }
}

impl LearningPanel {
    pub fn new() -> Self {
        Self::default()
    }

    /// Primary render method invoked from App CentralPanel.
    pub fn show(
        &mut self,
        ui: &mut Ui,
        network: &mut ModelProfileNetwork,
        models: &[Model],
        tx: &mpsc::Sender<AppMessage>,
    ) {
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                self.render_header(ui, network);
                ui.add_space(8.0);
                self.render_toolbar(ui, network, tx);
                ui.add_space(10.0);
                self.render_subtabs(ui);
                ui.separator();
                ui.add_space(8.0);

                match self.active_subtab {
                    LearningSubtab::Overview => self.render_overview(ui, network),
                    LearningSubtab::Optimizer => self.render_optimizer_controls(ui, network),
                    LearningSubtab::MemoryNetwork => self.render_memory_network(ui, network),
                    LearningSubtab::SynapticStats => self.render_synaptic_stats(ui, network),
                    LearningSubtab::Benchmark => self.render_benchmark_suite(ui, network),
                    LearningSubtab::Playground => self.render_playground(ui, network, models),
                }
            });
    }

    /// Renders high-level system identity and telemetry badges.
    fn render_header(&self, ui: &mut Ui, network: &ModelProfileNetwork) {
        let (param_count, _) = network.compute_synapse_stats();
        let buffer_len = network.experience_buffer.len();
        let mem_slots = network.memory_network.slots.len();
        let mem_cap = network.memory_network.capacity;
        let opt = network.optimizer_config.optimizer_type;

        ui.horizontal(|ui| {
            ui.heading(
                RichText::new("📈 Learning Algorithm & Optimization Suite")
                    .size(20.0)
                    .color(Color32::from_rgb(0x00, 0xee, 0xff))
                    .strong(),
            );
            ui.add_space(8.0);
            ui.label(
                RichText::new("Multi-Layer Backprop · Associative Memory · Quantum Entropy · Swarm Stigmergy")
                    .size(11.0)
                    .color(Color32::from_rgb(0x88, 0xaa, 0xcc)),
            );
        });

        ui.add_space(6.0);
        ui.horizontal_wrapped(|ui| {
            // Optimizer badge
            let opt_color = match opt {
                OptimizerType::Adam => Color32::from_rgb(0x00, 0xee, 0xaa),
                OptimizerType::Momentum => Color32::from_rgb(0x00, 0xbb, 0xff),
                OptimizerType::RmsProp => Color32::from_rgb(0xff, 0xaa, 0x33),
                OptimizerType::Sgd => Color32::from_rgb(0xaa, 0xaa, 0xaa),
            };
            self.badge(ui, "Optimizer", opt.label(), opt_color);

            // Learning Rate badge
            self.badge(
                ui,
                "LR (η)",
                &format!("{:.5}", network.optimizer_config.learning_rate),
                Color32::from_rgb(0xff, 0xcc, 0x00),
            );

            // Replay buffer badge
            let buf_color = if buffer_len >= 32 {
                Color32::from_rgb(0x00, 0xff, 0x88)
            } else if buffer_len >= 4 {
                Color32::from_rgb(0xee, 0xdd, 0x33)
            } else {
                Color32::from_rgb(0x88, 0x88, 0x88)
            };
            self.badge(
                ui,
                "Replay Buffer",
                &format!("{}/{}", buffer_len, MAX_EXPERIENCE_BUFFER),
                buf_color,
            );

            // Associative memory badge
            self.badge(
                ui,
                "Memory Slots",
                &format!("{}/{} ({} Archetypes)", mem_slots, mem_cap, 6.min(mem_slots)),
                Color32::from_rgb(0xcc, 0x66, 0xff),
            );

            // Synapse count badge
            self.badge(
                ui,
                "Synapses",
                &format!("{} params", param_count),
                Color32::from_rgb(0x00, 0xbb, 0xee),
            );

            // Auto-train status badge
            if network.optimizer_config.auto_train {
                self.badge(ui, "Auto-Train", "ENABLED", Color32::from_rgb(0x00, 0xff, 0x66));
            } else {
                self.badge(ui, "Auto-Train", "PAUSED", Color32::from_rgb(0xff, 0x66, 0x66));
            }
        });
    }

    /// Renders an informational metric badge.
    fn badge(&self, ui: &mut Ui, label: &str, value: &str, color: Color32) {
        ui.scope(|ui| {
            let bg = Color32::from_rgba_unmultiplied(color.r() / 6, color.g() / 6, color.b() / 6, 220);
            let frame = egui::Frame::NONE
                .fill(bg)
                .stroke(Stroke::new(1.0, color))
                .corner_radius(4.0)
                .inner_margin(egui::Margin::symmetric(6, 3));
            frame.show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new(label).size(10.0).color(Color32::from_rgb(0xbb, 0xcc, 0xdd)));
                    ui.label(RichText::new(value).size(11.0).color(color).strong());
                });
            });
        });
    }

    /// Renders quick action controls toolbar.
    fn render_toolbar(&mut self, ui: &mut Ui, network: &mut ModelProfileNetwork, tx: &mpsc::Sender<AppMessage>) {
        ui.horizontal(|ui| {
            if ui
                .button(RichText::new("⚡ Train 1 Step").color(Color32::from_rgb(0x00, 0xff, 0xcc)).strong())
                .on_hover_text("Execute one multi-layer backpropagation step through the replay buffer using the active optimizer")
                .clicked()
            {
                match network.train_step() {
                    Ok(loss) => {
                        self.status_message = Some(format!("✅ Step completed. MSE Loss: {:.6}", loss));
                    }
                    Err(e) => {
                        self.status_message = Some(format!("❌ Step error: {}", e));
                    }
                }
            }

            if ui
                .button(RichText::new("🚀 Run Synthetic Epoch").color(Color32::from_rgb(0x00, 0xdd, 0xff)).strong())
                .on_hover_text("Generate multi-domain synthetic experiences, train synaptic weights, update quantum phases, and evaporate pheromone matrix")
                .clicked()
            {
                match network.train_synthetic_epoch() {
                    Ok((loss, entropy)) => {
                        self.status_message = Some(format!(
                            "🚀 Epoch complete: MSE Loss = {:.6} | Quantum Entropy S = {:.4} nats",
                            loss, entropy
                        ));
                    }
                    Err(e) => {
                        self.status_message = Some(format!("❌ Epoch error: {}", e));
                    }
                }
            }

            if ui
                .button(RichText::new("🎯 Evaluate Benchmark").color(Color32::from_rgb(0xff, 0xaa, 0x00)).strong())
                .on_hover_text("Evaluate neural router against canonical domain archetypes and test performance")
                .clicked()
            {
                let metrics = network.evaluate_benchmark();
                self.benchmark_results = Some(metrics);
                self.last_benchmark_time = Some(Instant::now());
                self.status_message = Some("🎯 Benchmark evaluation complete.".to_string());
            }

            if ui
                .button(RichText::new("🧠 Consolidate Memory").color(Color32::from_rgb(0xdd, 0x88, 0xff)).strong())
                .on_hover_text("Apply bio-inspired memory consolidation: decays stale retention scores while strengthening reinforced archetypes")
                .clicked()
            {
                network.memory_network.consolidate();
                self.status_message = Some("🧠 Memory network consolidated. Stale retention scores decayed.".to_string());
            }

            let auto_text = if network.optimizer_config.auto_train {
                "🔄 Auto-Train: ON"
            } else {
                "⏸ Auto-Train: OFF"
            };
            let auto_color = if network.optimizer_config.auto_train {
                Color32::from_rgb(0x00, 0xff, 0x88)
            } else {
                Color32::from_rgb(0xff, 0x88, 0x88)
            };

            if ui
                .button(RichText::new(auto_text).color(auto_color).strong())
                .on_hover_text("Toggle whether background learning steps run automatically whenever chat sessions or DAG nodes complete")
                .clicked()
            {
                network.optimizer_config.auto_train = !network.optimizer_config.auto_train;
            }

            if ui
                .button(RichText::new("💾 Save Weights").color(Color32::from_rgb(0x88, 0xcc, 0xff)))
                .on_hover_text("Persist current synaptic weights and associative memory slots encrypted at rest")
                .clicked()
            {
                let dir = dirs::data_dir()
                    .unwrap_or_else(|| std::path::PathBuf::from("."))
                    .join("ai-dashboard");
                let _ = std::fs::create_dir_all(&dir);
                let path = dir.join("neural.bin").to_string_lossy().to_string();
                match network.save(&path) {
                    Ok(_) => {
                        let _ = tx.send(AppMessage::Audit("neural.save".to_string(), "Neural weights saved".to_string()));
                        self.status_message = Some("💾 Network and memory state saved encrypted at rest.".to_string());
                    }
                    Err(e) => {
                        self.status_message = Some(format!("❌ Save failed: {}", e));
                    }
                }
            }
        });

        let mut clear_status = false;
        if let Some(msg) = &self.status_message {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.label(RichText::new(msg).size(12.0).color(Color32::from_rgb(0xaa, 0xee, 0xff)));
                if ui.small_button("✖").clicked() {
                    clear_status = true;
                }
            });
        }
        if clear_status {
            self.status_message = None;
        }
    }

    /// Renders sub-tab selection navigation.
    fn render_subtabs(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            for tab in LearningSubtab::all() {
                let is_selected = self.active_subtab == tab;
                let text = if is_selected {
                    RichText::new(tab.label())
                        .color(Color32::from_rgb(0x00, 0xee, 0xff))
                        .strong()
                } else {
                    RichText::new(tab.label()).color(Color32::from_rgb(0xaa, 0xbb, 0xcc))
                };

                if ui.selectable_label(is_selected, text).clicked() {
                    self.active_subtab = tab;
                }
            }
        });
    }

    // =========================================================================
    // SUBTAB 1: OVERVIEW & TRAINING LOSS
    // =========================================================================

    fn render_overview(&mut self, ui: &mut Ui, network: &ModelProfileNetwork) {
        ui.horizontal(|ui| {
            ui.label(RichText::new("Mean Squared Error (MSE) Training Loss").size(14.0).strong());
            ui.add_space(12.0);
            ui.label(
                RichText::new(format!("Recorded Steps: {}", network.performance_history.len()))
                    .size(11.0)
                    .color(Color32::from_rgb(0x88, 0x99, 0xaa)),
            );
        });
        ui.add_space(4.0);

        if network.performance_history.is_empty() {
            ui.scope(|ui| {
                let frame = egui::Frame::NONE
                    .fill(Color32::from_rgb(0x15, 0x1c, 0x24))
                    .corner_radius(6.0)
                    .inner_margin(egui::Margin::symmetric(16, 24));
                frame.show(ui, |ui| {
                    ui.vertical_centered(|ui| {
                        ui.label(RichText::new("No training steps recorded yet.").size(13.0).color(Color32::from_rgb(0x88, 0xaa, 0xcc)));
                        ui.label(RichText::new("Click [⚡ Train 1 Step] or [🚀 Run Synthetic Epoch] above to initiate learning.").size(11.0).color(Color32::from_rgb(0x66, 0x77, 0x88)));
                    });
                });
            });
        } else {
            let pts: Vec<[f64; 2]> = network
                .performance_history
                .iter()
                .enumerate()
                .map(|(i, &v)| [i as f64, v as f64])
                .collect();

            let loss_line = Line::new("MSE Loss", pts.clone())
                .color(Color32::from_rgb(0x00, 0xaa, 0xff))
                .width(2.0);

            // Compute exponential moving average
            let mut ema_pts = Vec::with_capacity(pts.len());
            let alpha = 0.15;
            let mut current_ema = pts.first().map(|p| p[1]).unwrap_or(0.0);
            for p in &pts {
                current_ema = alpha * p[1] + (1.0 - alpha) * current_ema;
                ema_pts.push([p[0], current_ema]);
            }
            let ema_line = Line::new("Moving Avg (α=0.15)", ema_pts)
                .color(Color32::from_rgb(0xff, 0xaa, 0x00))
                .width(1.5);

            let plot = Plot::new("learning_loss_plot")
                .view_aspect(3.2)
                .height(220.0)
                .show_axes([true, true])
                .show_grid([true, true])
                .y_axis_label("Loss")
                .x_axis_label("Training Iteration")
                .allow_zoom(true)
                .allow_drag(true);

            plot.show(ui, |plot_ui| {
                plot_ui.line(loss_line);
                plot_ui.line(ema_line);
            });

            // Loss statistics summary line
            let min_loss = network.performance_history.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let max_loss = network.performance_history.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let avg_loss = network.performance_history.iter().sum::<f32>() / network.performance_history.len() as f32;
            let last_loss = *network.performance_history.last().unwrap_or(&0.0);

            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.label(RichText::new(format!("Current: {:.6}", last_loss)).size(11.0).color(Color32::from_rgb(0x00, 0xee, 0xff)));
                ui.add_space(12.0);
                ui.label(RichText::new(format!("Min: {:.6}", min_loss)).size(11.0).color(Color32::from_rgb(0x00, 0xff, 0x88)));
                ui.add_space(12.0);
                ui.label(RichText::new(format!("Avg: {:.6}", avg_loss)).size(11.0).color(Color32::from_rgb(0xff, 0xcc, 0x00)));
                ui.add_space(12.0);
                ui.label(RichText::new(format!("Max: {:.6}", max_loss)).size(11.0).color(Color32::from_rgb(0xff, 0x66, 0x66)));
            });
        }

        ui.add_space(16.0);
        ui.separator();
        ui.add_space(8.0);

        // Replay buffer preview
        ui.horizontal(|ui| {
            ui.label(RichText::new("Replay Experience Buffer Telemetry").size(14.0).strong());
            ui.add_space(12.0);
            let pct = (network.experience_buffer.len() as f32 / MAX_EXPERIENCE_BUFFER as f32) * 100.0;
            ui.label(
                RichText::new(format!("Capacity: {:.1}% ({}/{})", pct, network.experience_buffer.len(), MAX_EXPERIENCE_BUFFER))
                    .size(11.0)
                    .color(Color32::from_rgb(0x88, 0xaa, 0xcc)),
            );
        });

        ui.add_space(6.0);
        if network.experience_buffer.is_empty() {
            ui.label(RichText::new("Replay buffer is empty. Complete chats or DAG nodes to accumulate learning experiences.").color(Color32::from_rgb(0x88, 0x88, 0x88)));
        } else {
            egui::Grid::new("replay_buffer_preview_grid")
                .striped(true)
                .min_col_width(80.0)
                .show(ui, |ui| {
                    ui.label(RichText::new("Index").strong());
                    ui.label(RichText::new("Assigned Slot / Action").strong());
                    ui.label(RichText::new("Scalar Reward").strong());
                    ui.label(RichText::new("Feature Vector Norm").strong());
                    ui.label(RichText::new("Terminal").strong());
                    ui.end_row();

                    let recent = network.experience_buffer.iter().rev().take(8);
                    for (i, exp) in recent.enumerate() {
                        ui.label(format!("#{}", i));
                        ui.label(format!("Slot #{}", exp.action));
                        let rew_color = if exp.reward >= 0.8 {
                            Color32::from_rgb(0x00, 0xff, 0x88)
                        } else if exp.reward >= 0.0 {
                            Color32::from_rgb(0xff, 0xcc, 0x00)
                        } else {
                            Color32::from_rgb(0xff, 0x66, 0x66)
                        };
                        ui.label(RichText::new(format!("{:.3}", exp.reward)).color(rew_color));
                        let norm: f32 = exp.state.data.iter().map(|v| v * v).sum::<f32>().sqrt();
                        ui.label(format!("{:.4}", norm));
                        ui.label(if exp.done { "true" } else { "false" });
                        ui.end_row();
                    }
                });
        }
    }

    // =========================================================================
    // SUBTAB 2: OPTIMIZER & HYPERPARAMETERS
    // =========================================================================

    fn render_optimizer_controls(&mut self, ui: &mut Ui, network: &mut ModelProfileNetwork) {
        ui.label(RichText::new("Optimizer Architecture & Hyperparameter Configuration").size(14.0).strong());
        ui.label(
            RichText::new("Fine-tune multi-layer backpropagation gradient dynamics, learning rate decay, and L2 regularization.")
                .size(11.0)
                .color(Color32::from_rgb(0x88, 0xaa, 0xcc)),
        );
        ui.add_space(10.0);

        egui::Grid::new("optimizer_tuning_grid")
            .num_columns(2)
            .spacing([24.0, 12.0])
            .show(ui, |ui| {
                // Optimizer selector
                ui.label(RichText::new("Optimization Algorithm:").strong());
                ui.horizontal(|ui| {
                    egui::ComboBox::from_id_salt("optimizer_type_combo")
                        .selected_text(network.optimizer_config.optimizer_type.label())
                        .show_ui(ui, |ui| {
                            for opt in OptimizerType::all() {
                                ui.selectable_value(
                                    &mut network.optimizer_config.optimizer_type,
                                    opt,
                                    opt.label(),
                                );
                            }
                        });
                });
                ui.end_row();

                // Learning Rate
                ui.label(RichText::new("Learning Rate (η):").strong());
                ui.horizontal(|ui| {
                    ui.add(
                        egui::Slider::new(&mut network.optimizer_config.learning_rate, 0.0001..=0.1)
                            .logarithmic(true)
                            .step_by(0.0001)
                            .text("step size"),
                    );
                    network.learning_rate = network.optimizer_config.learning_rate;
                });
                ui.end_row();

                // L2 Weight Decay
                ui.label(RichText::new("L2 Weight Decay (λ):").strong());
                ui.horizontal(|ui| {
                    ui.add(
                        egui::Slider::new(&mut network.optimizer_config.weight_decay, 0.0..=0.01)
                            .step_by(0.0001)
                            .text("regularization penalty"),
                    );
                });
                ui.end_row();

                // Batch Size
                ui.label(RichText::new("Replay Batch Size:").strong());
                ui.horizontal(|ui| {
                    ui.add(
                        egui::Slider::new(&mut network.optimizer_config.batch_size, 4..=64)
                            .step_by(2.0)
                            .text("samples / step"),
                    );
                });
                ui.end_row();

                // Momentum Beta1
                ui.label(RichText::new("Momentum / Beta 1 (β₁):").strong());
                ui.horizontal(|ui| {
                    ui.add(
                        egui::Slider::new(&mut network.optimizer_config.beta1, 0.5..=0.999)
                            .step_by(0.01)
                            .text("first moment decay"),
                    );
                    network.optimizer_config.momentum = network.optimizer_config.beta1;
                });
                ui.end_row();

                // Beta2 for Adam
                ui.label(RichText::new("Adam / RMSProp Beta 2 (β₂):").strong());
                ui.horizontal(|ui| {
                    ui.add(
                        egui::Slider::new(&mut network.optimizer_config.beta2, 0.8..=0.9999)
                            .step_by(0.001)
                            .text("second moment decay"),
                    );
                });
                ui.end_row();

                // Memory Softmax Temperature
                ui.label(RichText::new("Memory Attention Temperature (τ):").strong());
                ui.horizontal(|ui| {
                    ui.add(
                        egui::Slider::new(&mut network.memory_network.temperature, 0.05..=2.0)
                            .step_by(0.05)
                            .text("sharper <-> broader attention"),
                    );
                });
                ui.end_row();

                // Memory Consolidation Decay Rate
                ui.label(RichText::new("Memory Decay Rate (δ):").strong());
                ui.horizontal(|ui| {
                    ui.add(
                        egui::Slider::new(&mut network.memory_network.decay_rate, 0.01..=0.3)
                            .step_by(0.01)
                            .text("temporal forgetting speed"),
                    );
                });
                ui.end_row();

                // Auto-train Toggle
                ui.label(RichText::new("Autonomous Auto-Training:").strong());
                ui.horizontal(|ui| {
                    ui.checkbox(&mut network.optimizer_config.auto_train, "Trigger train_step() on successful turns");
                });
                ui.end_row();
            });

        ui.add_space(16.0);
        if ui.button("🔄 Reset Hyperparameters to Defaults").clicked() {
            network.optimizer_config = crate::neural::OptimizerConfig::default();
            network.memory_network.temperature = 0.5;
            network.memory_network.decay_rate = 0.05;
            self.status_message = Some("Optimizer hyperparameters reset to defaults.".to_string());
        }
    }

    // =========================================================================
    // SUBTAB 3: ASSOCIATIVE MEMORY NETWORK
    // =========================================================================

    fn render_memory_network(&mut self, ui: &mut Ui, network: &mut ModelProfileNetwork) {
        ui.horizontal(|ui| {
            ui.label(RichText::new("Bio-Inspired Associative Content-Addressable Memory").size(14.0).strong());
            ui.add_space(12.0);
            ui.label(
                RichText::new(format!(
                    "Slots: {}/{} | Total Reads: {} | Total Writes: {}",
                    network.memory_network.slots.len(),
                    network.memory_network.capacity,
                    network.memory_network.total_reads,
                    network.memory_network.total_writes
                ))
                .size(11.0)
                .color(Color32::from_rgb(0x88, 0xaa, 0xcc)),
            );
        });
        ui.label(
            RichText::new("Soft attention matching retrieves contextual experience vectors. Slots with low retention are recycled under capacity pressure.")
                .size(11.0)
                .color(Color32::from_rgb(0x88, 0x88, 0x99)),
        );
        ui.add_space(8.0);

        ui.horizontal(|ui| {
            ui.label("🔍 Filter:");
            ui.text_edit_singleline(&mut self.memory_filter);
            if !self.memory_filter.is_empty() && ui.button("Clear").clicked() {
                self.memory_filter.clear();
            }
            ui.add_space(16.0);
            if ui.button(if self.show_manual_inject { "▼ Hide Injection" } else { "➕ Inject Custom Memory" }).clicked() {
                self.show_manual_inject = !self.show_manual_inject;
            }
            if ui
                .button("🧹 Clear Non-Archetypes")
                .on_hover_text("Preserves canonical archetypes while purging transient experience slots")
                .clicked()
            {
                network.memory_network.clear();
                self.status_message = Some("Memory cleared to canonical archetypes.".to_string());
            }
        });

        if self.show_manual_inject {
            ui.add_space(6.0);
            egui::Frame::NONE
                .fill(Color32::from_rgb(0x1a, 0x22, 0x2e))
                .stroke(Stroke::new(1.0, Color32::from_rgb(0x00, 0xaa, 0xff)))
                .corner_radius(4.0)
                .inner_margin(egui::Margin::symmetric(10, 8))
                .show(ui, |ui| {
                    ui.label(RichText::new("Inject Manual Memory Slot").strong().color(Color32::from_rgb(0x00, 0xee, 0xff)));
                    ui.horizontal(|ui| {
                        ui.label("Domain:");
                        egui::ComboBox::from_id_salt("inject_domain_combo")
                            .selected_text(&self.inject_domain)
                            .show_ui(ui, |ui| {
                                for d in &["Coder", "Researcher", "Cyber / Critic", "Planner", "Writer", "General"] {
                                    ui.selectable_value(&mut self.inject_domain, d.to_string(), *d);
                                }
                            });
                        ui.label("Label:");
                        ui.text_edit_singleline(&mut self.inject_label);
                        ui.label("Reward:");
                        ui.add(egui::Slider::new(&mut self.inject_reward, 0.1..=2.0).step_by(0.1));

                        if ui.button("Inject Slot").clicked() {
                            let dummy_key = extract_probe_features(&self.inject_label);
                            let dummy_val = vec![1.0; network.memory_network.value_dim];
                            network.memory_network.write(
                                dummy_key.as_slice().unwrap_or(&[]),
                                &dummy_val,
                                &self.inject_domain,
                                &self.inject_label,
                                self.inject_reward,
                            );
                            self.status_message = Some(format!("Injected custom memory slot: {}", self.inject_label));
                            self.show_manual_inject = false;
                        }
                    });
                });
        }

        ui.add_space(8.0);

        // Memory slots table
        let filter_lc = self.memory_filter.to_lowercase();
        let recent_attn = network.memory_network.last_attention_weights.clone();

        egui::Grid::new("memory_slots_table")
            .striped(true)
            .min_col_width(70.0)
            .show(ui, |ui| {
                ui.label(RichText::new("ID").strong());
                ui.label(RichText::new("Domain").strong());
                ui.label(RichText::new("Label / Experience Anchor").strong());
                ui.label(RichText::new("Accesses").strong());
                ui.label(RichText::new("Retention Score").strong());
                ui.label(RichText::new("Last Attention").strong());
                ui.label(RichText::new("Action").strong());
                ui.end_row();

                for (idx, slot) in network.memory_network.slots.iter().enumerate() {
                    if !filter_lc.is_empty()
                        && !slot.domain.to_lowercase().contains(&filter_lc)
                        && !slot.label.to_lowercase().contains(&filter_lc)
                    {
                        continue;
                    }

                    ui.label(format!("#{}", slot.id));

                    // Domain badge with specific colors
                    let dom_color = match slot.domain.as_str() {
                        "Coder" => Color32::from_rgb(0x00, 0xff, 0x88),
                        "Researcher" => Color32::from_rgb(0x00, 0xbb, 0xff),
                        "Cyber / Critic" => Color32::from_rgb(0xff, 0x55, 0x66),
                        "Planner" => Color32::from_rgb(0xdd, 0x66, 0xff),
                        "Writer" => Color32::from_rgb(0xff, 0xaa, 0x33),
                        _ => Color32::from_rgb(0xee, 0xdd, 0x44),
                    };
                    ui.label(RichText::new(&slot.domain).color(dom_color).strong());

                    ui.label(&slot.label);
                    ui.label(format!("{}", slot.access_count));

                    // Retention bar
                    let ret_norm = (slot.retention_score / 3.0).clamp(0.0, 1.0);
                    ui.horizontal(|ui| {
                        ui.add(egui::ProgressBar::new(ret_norm).desired_width(70.0).text(format!("{:.2}", slot.retention_score)));
                    });

                    // Attention weight
                    let attn = recent_attn.get(idx).copied().unwrap_or(0.0);
                    let attn_pct = attn * 100.0;
                    let attn_color = if attn > 0.15 {
                        Color32::from_rgb(0x00, 0xee, 0xff)
                    } else {
                        Color32::from_rgb(0x88, 0x88, 0x88)
                    };
                    ui.label(RichText::new(format!("{:.1}%", attn_pct)).color(attn_color));

                    if ui.small_button("Inspect").clicked() {
                        self.selected_memory_slot = Some(idx);
                    }
                    ui.end_row();
                }
            });

        // Detail inspector for selected memory slot
        if let Some(sel_idx) = self.selected_memory_slot {
            if let Some(slot) = network.memory_network.slots.get(sel_idx) {
                ui.add_space(10.0);
                egui::Frame::NONE
                    .fill(Color32::from_rgb(0x13, 0x19, 0x22))
                    .stroke(Stroke::new(1.0, Color32::from_rgb(0x00, 0xbb, 0xff)))
                    .corner_radius(6.0)
                    .inner_margin(egui::Margin::symmetric(12, 10))
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.label(RichText::new(format!("Memory Slot #{}: {} [{}]", slot.id, slot.label, slot.domain)).strong().color(Color32::from_rgb(0x00, 0xee, 0xff)));
                            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                if ui.button("Close").clicked() {
                                    self.selected_memory_slot = None;
                                }
                            });
                        });
                        ui.add_space(4.0);
                        ui.horizontal(|ui| {
                            ui.label(RichText::new("Key Vector (8D):").strong());
                            for k in &slot.key {
                                ui.label(RichText::new(format!("{:.2}", k)).color(Color32::from_rgb(0x88, 0xcc, 0xff)));
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.label(RichText::new("Value Vector (8D):").strong());
                            for v in &slot.value {
                                ui.label(RichText::new(format!("{:.2}", v)).color(Color32::from_rgb(0xff, 0xcc, 0x88)));
                            }
                        });
                    });
            }
        }
    }

    // =========================================================================
    // SUBTAB 4: SYNAPTIC LAYER INSPECTOR
    // =========================================================================

    fn render_synaptic_stats(&mut self, ui: &mut Ui, network: &ModelProfileNetwork) {
        let (param_count, layer_stats) = network.compute_synapse_stats();

        ui.horizontal(|ui| {
            ui.label(RichText::new("Deep Neural Synaptic Layer Distribution").size(14.0).strong());
            ui.add_space(12.0);
            ui.label(
                RichText::new(format!("Total Trainable Parameters: {}", param_count))
                    .size(11.0)
                    .color(Color32::from_rgb(0x00, 0xee, 0xff)),
            );
        });
        ui.label(
            RichText::new("Layer-by-layer statistical moments, weight dispersion, Frobenius L2 norms, and zero-weight sparsity.")
                .size(11.0)
                .color(Color32::from_rgb(0x88, 0xaa, 0xcc)),
        );
        ui.add_space(10.0);

        egui::Grid::new("synaptic_stats_grid")
            .striped(true)
            .min_col_width(75.0)
            .show(ui, |ui| {
                ui.label(RichText::new("Layer").strong());
                ui.label(RichText::new("Shape").strong());
                ui.label(RichText::new("Params").strong());
                ui.label(RichText::new("Mean (μ)").strong());
                ui.label(RichText::new("Std Dev (σ)").strong());
                ui.label(RichText::new("Min / Max").strong());
                ui.label(RichText::new("L2 Norm (‖W‖₂)").strong());
                ui.label(RichText::new("Sparsity %").strong());
                ui.end_row();

                for st in &layer_stats {
                    ui.label(RichText::new(&st.name).strong());
                    ui.label(format!("{} × {}", st.rows, st.cols));
                    ui.label(format!("{}", st.param_count));
                    ui.label(format!("{:.5}", st.mean));
                    ui.label(format!("{:.5}", st.std_dev));
                    ui.label(format!("[{:.3}, {:.3}]", st.min, st.max));
                    ui.label(format!("{:.4}", st.l2_norm));

                    let spar_color = if st.sparsity > 50.0 {
                        Color32::from_rgb(0xff, 0xaa, 0x33)
                    } else {
                        Color32::from_rgb(0x00, 0xff, 0x88)
                    };
                    ui.label(RichText::new(format!("{:.1}%", st.sparsity)).color(spar_color));
                    ui.end_row();
                }
            });

        ui.add_space(16.0);
        ui.separator();
        ui.add_space(8.0);

        // Architecture Topology Visual Breakdown
        ui.label(RichText::new("Topology Pipeline").size(13.0).strong());
        ui.horizontal(|ui| {
            ui.label(format!("Input Dim: {}", network.input_dim));
            ui.label("➔");
            for (idx, &h) in network.hidden_dims.iter().enumerate() {
                ui.label(format!("Hidden #{}: {} neurons", idx + 1, h));
                ui.label("➔");
            }
            ui.label(format!("Output Dim: {} profiles", network.output_dim));
        });
    }

    // =========================================================================
    // SUBTAB 5: BENCHMARK EVALUATION SUITE
    // =========================================================================

    fn render_benchmark_suite(&mut self, ui: &mut Ui, network: &ModelProfileNetwork) {
        ui.horizontal(|ui| {
            ui.label(RichText::new("Autonomous Multi-Domain Benchmark Suite").size(14.0).strong());
            ui.add_space(12.0);
            if let Some(t) = self.last_benchmark_time {
                let el = t.elapsed().as_secs();
                ui.label(RichText::new(format!("Last evaluated {}s ago", el)).size(11.0).color(Color32::from_rgb(0x88, 0xaa, 0xcc)));
            }
        });
        ui.label(
            RichText::new("Validates routing accuracy against canonical archetypes across all 6 specialized domains.")
                .size(11.0)
                .color(Color32::from_rgb(0x88, 0x88, 0x99)),
        );
        ui.add_space(8.0);

        ui.horizontal(|ui| {
            if ui
                .button(RichText::new("🎯 Execute Canonical Benchmark Now").color(Color32::from_rgb(0xff, 0xaa, 0x00)).strong())
                .clicked()
            {
                let metrics = network.evaluate_benchmark();
                self.benchmark_results = Some(metrics);
                self.last_benchmark_time = Some(Instant::now());
                self.status_message = Some("🎯 Benchmark evaluation complete.".to_string());
            }
        });
        ui.add_space(10.0);

        if let Some(bench) = &self.benchmark_results {
            ui.horizontal(|ui| {
                // Accuracy Gauge
                let acc_color = if bench.accuracy >= 80.0 {
                    Color32::from_rgb(0x00, 0xff, 0x88)
                } else if bench.accuracy >= 50.0 {
                    Color32::from_rgb(0xff, 0xcc, 0x00)
                } else {
                    Color32::from_rgb(0xff, 0x66, 0x66)
                };
                self.badge(ui, "Overall Accuracy", &format!("{:.1}%", bench.accuracy), acc_color);

                // Test Loss
                self.badge(
                    ui,
                    "Test Loss (MSE)",
                    &format!("{:.6}", bench.test_loss),
                    Color32::from_rgb(0x00, 0xcc, 0xff),
                );

                // Quantum Entropy
                self.badge(
                    ui,
                    "Avg Quantum Entropy",
                    &format!("{:.4} nats", bench.avg_entropy),
                    Color32::from_rgb(0xcc, 0x88, 0xff),
                );

                // Memory Confidence
                self.badge(
                    ui,
                    "Memory Confidence",
                    &format!("{:.1}%", bench.memory_retrieval_confidence * 100.0),
                    Color32::from_rgb(0xff, 0xaa, 0x33),
                );
            });

            ui.add_space(14.0);
            ui.label(RichText::new("Domain-by-Domain Accuracy Breakdown").strong());
            ui.add_space(6.0);

            egui::Grid::new("benchmark_domain_grid")
                .striped(true)
                .min_col_width(120.0)
                .show(ui, |ui| {
                    ui.label(RichText::new("Domain").strong());
                    ui.label(RichText::new("Test Accuracy %").strong());
                    ui.label(RichText::new("Visual Confidence Bar").strong());
                    ui.end_row();

                    for (dom, acc) in &bench.domain_accuracies {
                        ui.label(dom);
                        let bar_color = if *acc >= 80.0 {
                            Color32::from_rgb(0x00, 0xff, 0x88)
                        } else if *acc >= 50.0 {
                            Color32::from_rgb(0xff, 0xcc, 0x00)
                        } else {
                            Color32::from_rgb(0xff, 0x66, 0x66)
                        };
                        ui.label(RichText::new(format!("{:.1}%", acc)).color(bar_color).strong());
                        ui.horizontal(|ui| {
                            ui.add(egui::ProgressBar::new(acc / 100.0).desired_width(120.0));
                        });
                        ui.end_row();
                    }
                });
        } else {
            ui.scope(|ui| {
                let frame = egui::Frame::NONE
                    .fill(Color32::from_rgb(0x15, 0x1c, 0x24))
                    .corner_radius(6.0)
                    .inner_margin(egui::Margin::symmetric(16, 24));
                frame.show(ui, |ui| {
                    ui.vertical_centered(|ui| {
                        ui.label(RichText::new("No benchmark evaluation has been conducted yet.").size(13.0).color(Color32::from_rgb(0x88, 0xaa, 0xcc)));
                        ui.label(RichText::new("Click [🎯 Evaluate Benchmark] in the toolbar above to run testing.").size(11.0).color(Color32::from_rgb(0x66, 0x77, 0x88)));
                    });
                });
            });
        }
    }

    // =========================================================================
    // SUBTAB 6: LIVE PREDICTION PLAYGROUND
    // =========================================================================

    fn render_playground(&mut self, ui: &mut Ui, network: &mut ModelProfileNetwork, models: &[Model]) {
        ui.horizontal(|ui| {
            ui.label(RichText::new("Live Feature Extraction & Neural Routing Playground").size(14.0).strong());
            ui.add_space(12.0);
            ui.label(
                RichText::new("Interactive probe vector testing, memory soft attention retrieval, and quantum state distribution.")
                    .size(11.0)
                    .color(Color32::from_rgb(0x88, 0xaa, 0xcc)),
            );
        });
        ui.add_space(8.0);

        // Pre-fill quick buttons
        ui.horizontal_wrapped(|ui| {
            ui.label("Quick Presets:");
            if ui.small_button("💻 Coder Task").clicked() {
                self.test_prompt = "fn quicksort<T: Ord>(arr: &mut [T]) { /* recursive partition */ }".to_string();
                self.evaluate_test_prompt(network);
            }
            if ui.small_button("🛡 Cyber / Security").clicked() {
                self.test_prompt = "Audit cryptographic secret leakage, reentrancy attacks, and timing side-channels".to_string();
                self.evaluate_test_prompt(network);
            }
            if ui.small_button("🔬 Scientific Research").clicked() {
                self.test_prompt = "Formulate a literature review and hypothesis on biological ant swarm stigmergy".to_string();
                self.evaluate_test_prompt(network);
            }
            if ui.small_button("📋 DAG Planner").clicked() {
                self.test_prompt = "Design system architecture, decompose milestones into topological DAG nodes".to_string();
                self.evaluate_test_prompt(network);
            }
            if ui.small_button("✍ Technical Writer").clicked() {
                self.test_prompt = "Draft comprehensive user manual, installation guide, and executive summary".to_string();
                self.evaluate_test_prompt(network);
            }
        });

        ui.add_space(6.0);
        ui.horizontal(|ui| {
            ui.label("Test Prompt:");
            let resp = ui.text_edit_singleline(&mut self.test_prompt);
            if resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                self.evaluate_test_prompt(network);
            }
            if ui.button("🚀 Evaluate").clicked() {
                self.evaluate_test_prompt(network);
            }
        });

        if let Some(res) = &self.test_result {
            ui.add_space(10.0);
            egui::Frame::NONE
                .fill(Color32::from_rgb(0x11, 0x18, 0x22))
                .stroke(Stroke::new(1.0, Color32::from_rgb(0x00, 0xee, 0xff)))
                .corner_radius(6.0)
                .inner_margin(egui::Margin::symmetric(12, 10))
                .show(ui, |ui| {
                    // Header badges
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("Prediction Results").size(13.0).strong().color(Color32::from_rgb(0x00, 0xee, 0xff)));
                        ui.add_space(12.0);
                        self.badge(ui, "Inferred Domain", &res.best_domain, Color32::from_rgb(0x00, 0xff, 0x88));
                        self.badge(ui, "Target Slot", &format!("#{}", res.best_slot), Color32::from_rgb(0x00, 0xbb, 0xff));
                        self.badge(ui, "Confidence", &format!("{:.1}%", res.confidence * 100.0), Color32::from_rgb(0xff, 0xcc, 0x00));
                        self.badge(ui, "Von Neumann Entropy", &format!("{:.4} nats", res.quantum_entropy), Color32::from_rgb(0xdd, 0x88, 0xff));
                    });

                    ui.add_space(4.0);
                    ui.label(RichText::new(format!("Input Evaluated: \"{}\"", res.prompt)).size(11.0).italics().color(Color32::from_rgb(0x88, 0xaa, 0xbb)));
                    ui.add_space(6.0);

                    // Extracted 8D Feature Probe
                    ui.label(RichText::new("Extracted 8D Probe Features:").strong());
                    ui.horizontal_wrapped(|ui| {
                        let feature_names = [
                            "Length",
                            "Code Tokens",
                            "Security Terms",
                            "Research Words",
                            "Punctuation",
                            "Numeric Density",
                            "Command Words",
                            "Domain Bias",
                        ];
                        for (idx, &val) in res.probe_features.iter().enumerate() {
                            let name = feature_names.get(idx).unwrap_or(&"Feature");
                            ui.label(RichText::new(format!("{}: {:.2}", name, val)).size(11.0).color(Color32::from_rgb(0xaa, 0xcc, 0xee)));
                        }
                    });

                    ui.add_space(8.0);

                    // Associative Memory Soft-Attention Retrieval
                    ui.label(RichText::new("Associative Memory Soft-Attention Distribution:").strong());
                    ui.horizontal_wrapped(|ui| {
                        for (slot_id, domain, label, weight) in &res.memory_attention {
                            if *weight > 0.05 {
                                let pct = weight * 100.0;
                                ui.label(
                                    RichText::new(format!("Slot #{}: {} ({:.1}%) - {}", slot_id, domain, pct, label))
                                        .size(11.0)
                                        .color(Color32::from_rgb(0x00, 0xee, 0xcc)),
                                );
                            }
                        }
                    });

                    ui.add_space(8.0);

                    // Blended Memory Vector
                    ui.label(RichText::new("Retrieved Memory Vector (Blended Context):").strong());
                    ui.horizontal_wrapped(|ui| {
                        for (idx, &val) in res.retrieved_memory_vec.iter().enumerate() {
                            ui.label(RichText::new(format!("m[{}]: {:.3}", idx, val)).size(11.0).color(Color32::from_rgb(0xff, 0xbb, 0x66)));
                        }
                    });

                    ui.add_space(8.0);

                    // Quantum State Superposition
                    if !res.quantum_probs.is_empty() {
                        ui.label(RichText::new("Quantum Layer Superposition Probabilities:").strong());
                        ui.horizontal_wrapped(|ui| {
                            for (idx, &prob) in res.quantum_probs.iter().enumerate() {
                                ui.label(RichText::new(format!("|q_{}⟩: {:.1}%", idx, prob * 100.0)).size(11.0).color(Color32::from_rgb(0xcc, 0x88, 0xff)));
                            }
                        });
                        ui.add_space(8.0);
                    }

                    // Quantum State Layer & Output Profile Distribution
                    ui.label(RichText::new("Final Swarm Profile Softmax Distribution:").strong());
                    ui.horizontal(|ui| {
                        for (idx, &prob) in res.distribution.iter().enumerate() {
                            let is_best = idx == res.best_slot;
                            let color = if is_best {
                                Color32::from_rgb(0x00, 0xff, 0x88)
                            } else {
                                Color32::from_rgb(0x88, 0x99, 0xaa)
                            };
                            let slot_name = models
                                .get(idx)
                                .map(|m| m.name.as_str())
                                .unwrap_or("Slot");
                            ui.label(
                                RichText::new(format!("Slot #{}: {:.1}% ({})", idx, prob * 100.0, slot_name))
                                    .color(color)
                                    .strong(),
                            );
                        }
                    });
                });
        }
    }

    /// Evaluates test prompt through memory network and quantum neural pipeline.
    fn evaluate_test_prompt(&mut self, network: &mut ModelProfileNetwork) {
        if self.test_prompt.trim().is_empty() {
            return;
        }

        let probe = extract_probe_features(&self.test_prompt);
        let (slot, domain, conf, entropy, dist, retrieved_val, attn_weights) =
            network.query_with_memory(&self.test_prompt);

        let mut memory_attention = Vec::new();
        for (i, &w) in attn_weights.iter().enumerate() {
            if let Some(slot) = network.memory_network.slots.get(i) {
                memory_attention.push((slot.id, slot.domain.clone(), slot.label.clone(), w));
            }
        }
        memory_attention.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));

        let quantum_probs = if let Some(ql) = &network.quantum_layer {
            let (probs, _) = ql.transform(&probe);
            probs.to_vec()
        } else {
            vec![]
        };

        self.test_result = Some(TestPredictionResult {
            prompt: self.test_prompt.clone(),
            probe_features: probe.to_vec(),
            memory_attention,
            retrieved_memory_vec: retrieved_val,
            quantum_entropy: entropy,
            quantum_probs,
            best_slot: slot,
            best_domain: domain.to_string(),
            confidence: conf,
            distribution: dist,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_learning_panel_default_state() {
        let panel = LearningPanel::new();
        assert_eq!(panel.active_subtab, LearningSubtab::Overview);
        assert!(panel.benchmark_results.is_none());
        assert!(!panel.test_prompt.is_empty());
        assert!(panel.test_result.is_none());
    }

    #[test]
    fn test_learning_subtabs_count_and_labels() {
        let tabs = LearningSubtab::all();
        assert_eq!(tabs.len(), 6);
        for t in tabs {
            assert!(!t.label().is_empty());
        }
    }

    #[test]
    fn test_evaluate_test_prompt_flow() {
        let mut panel = LearningPanel::new();
        let mut net = ModelProfileNetwork::new(8, vec![16, 16], 4);
        panel.test_prompt = "fn quicksort() { let x = 1; }".to_string();
        panel.evaluate_test_prompt(&mut net);

        assert!(panel.test_result.is_some());
        let res = panel.test_result.unwrap();
        assert_eq!(res.best_domain, "Coder");
        assert_eq!(res.probe_features.len(), 8);
        assert!(!res.memory_attention.is_empty());
        assert!(res.confidence > 0.0 && res.confidence <= 1.0);
        assert!(res.quantum_entropy >= 0.0);
    }
}
