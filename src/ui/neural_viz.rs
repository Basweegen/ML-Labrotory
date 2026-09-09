use crate::neural::ModelProfileNetwork;
use egui::{Color32, Pos2, Rect, Sense, Stroke, Vec2};
use egui_plot::{Line, Plot, PlotPoints};
use ndarray::Array2;
use std::collections::VecDeque;

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct NeuralVizPanel {
    pub training_history: VecDeque<f32>,
    pub max_history_len: usize,
    pub show_weights: bool,
    pub show_architecture: bool,
    pub selected_layer: usize,
    pub weight_hovered: Option<(usize, usize, usize)>, // (layer, row, col)
    pub last_loss: Option<f32>,
}

impl Default for NeuralVizPanel {
    fn default() -> Self {
        Self {
            training_history: VecDeque::with_capacity(500),
            max_history_len: 500,
            show_weights: true,
            show_architecture: true,
            selected_layer: 0,
            weight_hovered: None,
            last_loss: None,
        }
    }
}

impl NeuralVizPanel {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_training_loss(&mut self, loss: f32) {
        self.training_history.push_back(loss);
        if self.training_history.len() > self.max_history_len {
            self.training_history.pop_front();
        }
    }

    pub fn show(&mut self, ui: &mut egui::Ui, network: &ModelProfileNetwork) {
        ui.horizontal(|ui| {
            ui.heading(egui::RichText::new("Neural Network Visualization").size(22.0).color(Color32::from_rgb(0x00, 0xaa, 0xff)));
            ui.add_space(16.0);
            ui.checkbox(&mut self.show_architecture, "Architecture");
            ui.checkbox(&mut self.show_weights, "Weight Heatmaps");
        });
        ui.add_space(8.0);
        ui.separator();
        ui.add_space(12.0);

        if self.show_architecture {
            self.render_architecture(ui, network);
            ui.add_space(16.0);
            ui.separator();
            ui.add_space(16.0);
        }

        if self.show_weights {
            self.render_weight_heatmaps(ui, network);
            ui.add_space(16.0);
            ui.separator();
            ui.add_space(16.0);
        }

        self.render_training_curves(ui);
        ui.add_space(16.0);
        ui.separator();
        ui.add_space(16.0);

        self.render_network_stats(ui, network);
    }

    fn render_architecture(&mut self, ui: &mut egui::Ui, network: &ModelProfileNetwork) {
        ui.label(egui::RichText::new("Network Architecture").size(18.0).color(Color32::from_rgb(0x00, 0xaa, 0xff)).strong());
        ui.add_space(12.0);

        let layer_dims = Self::get_layer_dims(network);
        let num_layers = layer_dims.len();

        if num_layers == 0 {
            ui.label("No layers to display");
            return;
        }

        let available_width = ui.available_width().max(600.0);
        let layer_spacing = available_width / (num_layers as f32 + 1.0);
        let max_neurons = *layer_dims.iter().max().unwrap_or(&1) as f32;
        let neuron_radius = (40.0 / max_neurons.sqrt()).min(20.0).max(6.0);
        let vertical_spacing = (neuron_radius * 2.0 + 8.0).min(60.0);

        let (response, painter) = ui.allocate_painter(
            Vec2::new(available_width, vertical_spacing * max_neurons + 100.0),
            Sense::hover(),
        );

        let rect = response.rect;
        let center_y = rect.center().y;

        // Convert each weight matrix ONCE (was: cloned+converted per connection,
        // i.e. rows*cols clones per layer per frame at 60fps).
        let w_mats: Vec<Array2<f32>> = network.weights.iter().map(|w| w.clone().into()).collect();
        // Draw connections first (behind neurons)
        for l in 0..num_layers - 1 {
            let x1 = rect.left() + layer_spacing * (l + 1) as f32;
            let x2 = rect.left() + layer_spacing * (l + 2) as f32;
            let n1 = layer_dims[l] as f32;
            let n2 = layer_dims[l + 1] as f32;

            let start_y = center_y - (n1 - 1.0) * vertical_spacing / 2.0;
            let end_y = center_y - (n2 - 1.0) * vertical_spacing / 2.0;

            for i in 0..layer_dims[l] {
                for j in 0..layer_dims[l + 1] {
                    let y1 = start_y + i as f32 * vertical_spacing;
                    let y2 = end_y + j as f32 * vertical_spacing;

                    // Get weight value for color intensity
                    let weight_val = if l < w_mats.len() {
                        let w_arr = &w_mats[l];
                        if j < w_arr.nrows() && i < w_arr.ncols() {
                            w_arr[[j, i]].abs().min(1.0)
                        } else {
                            0.0
                        }
                    } else {
                        0.0
                    };

                    let alpha = (weight_val * 180.0 + 40.0) as u8;
                    let color = if weight_val > 0.0 {
                        Color32::from_rgba_unmultiplied(0x00, 0xaa, 0xff, alpha)
                    } else {
                        Color32::from_rgba_unmultiplied(0xff, 0x66, 0x66, alpha)
                    };

                    painter.line_segment(
                        [Pos2::new(x1, y1), Pos2::new(x2, y2)],
                        Stroke::new(0.5, color),
                    );
                }
            }
        }

        // Draw neurons
        for (l, &dim) in layer_dims.iter().enumerate() {
            let x = rect.left() + layer_spacing * (l + 1) as f32;
            let n = dim as f32;
            let start_y = center_y - (n - 1.0) * vertical_spacing / 2.0;

            for i in 0..dim {
                let y = start_y + i as f32 * vertical_spacing;
                let pos = Pos2::new(x, y);

                let is_selected = self.selected_layer == l;
                let fill_color = if is_selected {
                    Color32::from_rgb(0x00, 0x77, 0xcc)
                } else if l == 0 {
                    Color32::from_rgb(0x00, 0xcc, 0x88)
                } else if l == num_layers - 1 {
                    Color32::from_rgb(0xff, 0xaa, 0x00)
                } else {
                    Color32::from_rgb(0xaa, 0x66, 0xff)
                };

                let stroke_color = if is_selected { Color32::WHITE } else { Color32::from_rgb(0x44, 0x44, 0x44) };
                let stroke_width = if is_selected { 2.0 } else { 1.0 };

                painter.circle_filled(pos, neuron_radius, fill_color);
                painter.circle_stroke(pos, neuron_radius, Stroke::new(stroke_width, stroke_color));

                // Neuron index label for small networks
                if dim <= 16 {
                    painter.text(
                        pos,
                        egui::Align2::CENTER_CENTER,
                        format!("{}", i),
                        egui::FontId::proportional(10.0),
                        Color32::WHITE,
                    );
                }
            }

            // Layer label
            let label_y = rect.bottom() - 20.0;
            let label = match l {
                0 => "Input",
                _ if l == num_layers - 1 => "Output",
                _ => &format!("Hidden {}", l),
            };
            painter.text(
                Pos2::new(x, label_y),
                egui::Align2::CENTER_TOP,
                label,
                egui::FontId::proportional(12.0),
                Color32::from_rgb(0xaa, 0xaa, 0xaa),
            );

            // Dimension label
            painter.text(
                Pos2::new(x, label_y + 20.0),
                egui::Align2::CENTER_TOP,
                format!("({})", dim),
                egui::FontId::proportional(10.0),
                Color32::from_rgb(0x88, 0x88, 0x88),
            );
        }

        // Click handling for layer selection
        if response.clicked() {
            if let Some(pos) = response.interact_pointer_pos() {
                for (l, &_dim) in layer_dims.iter().enumerate() {
                    let x = rect.left() + layer_spacing * (l + 1) as f32;
                    if (pos.x - x).abs() < neuron_radius * 2.0 {
                        // This would require mut self - we'll handle selection differently
                    }
                }
            }
        }

        // Layer selector combo box below
        ui.add_space(vertical_spacing * max_neurons + 60.0);
        ui.horizontal(|ui| {
            ui.label("Inspect Layer:");
            egui::ComboBox::from_id_salt("layer_selector")
                .selected_text(format!("Layer {} ({} neurons)", self.selected_layer, layer_dims[self.selected_layer]))
                .show_ui(ui, |ui| {
                    for (l, &dim) in layer_dims.iter().enumerate() {
                        let label = match l {
                            0 => format!("Layer {}: Input ({})", l, dim),
                            _ if l == num_layers - 1 => format!("Layer {}: Output ({})", l, dim),
                            _ => format!("Layer {}: Hidden ({})", l, dim),
                        };
                        ui.selectable_value(&mut self.selected_layer, l, label);
                    }
                });
        });
    }

    fn get_layer_dims(network: &ModelProfileNetwork) -> Vec<usize> {
        let mut dims = vec![network.input_dim];
        dims.extend(network.hidden_dims.iter().cloned());
        dims.push(network.output_dim);
        dims
    }

    fn render_weight_heatmaps(&mut self, ui: &mut egui::Ui, network: &ModelProfileNetwork) {
        ui.label(egui::RichText::new("Weight Matrices").size(18.0).color(Color32::from_rgb(0x00, 0xaa, 0xff)).strong());
        ui.add_space(8.0);

        let num_layers = network.weights.len();

        egui::ScrollArea::horizontal().show(ui, |ui| {
            ui.horizontal(|ui| {
                for (l, weight_data) in network.weights.iter().enumerate() {
                    let w_arr: ndarray::Array2<f32> = weight_data.clone().into();
                    let rows = w_arr.nrows();
                    let cols = w_arr.ncols();

                    let label = match l {
                        0 => format!("W1: {}×{} (Input→Hidden)", rows, cols),
                        _ if l == num_layers - 1 => format!("W{}: {}×{} (Hidden→Output)", l + 1, rows, cols),
                        _ => format!("W{}: {}×{} (Hidden→Hidden)", l + 1, rows, cols),
                    };

                    ui.vertical(|ui| {
                        ui.add_space(8.0);
                        ui.label(egui::RichText::new(&label).size(12.0).color(Color32::from_rgb(0xcc, 0xcc, 0xcc)));
                        self.render_heatmap(ui, &w_arr, &format!("heatmap_layer_{}", l));
                        ui.add_space(8.0);

                        // Stats
                        let (min_val, max_val, mean_val) = Self::compute_stats(&w_arr);
                        ui.label(egui::RichText::new(format!("Min: {:.4}  Max: {:.4}  Mean: {:.4}", min_val, max_val, mean_val))
                            .size(10.0).color(Color32::from_rgb(0x88, 0x88, 0x88)));
                    });
                    ui.add_space(16.0);
                }
            });
        });

        // Bias heatmaps
        ui.add_space(16.0);
        ui.label(egui::RichText::new("Bias Vectors").size(16.0).color(Color32::from_rgb(0xff, 0xaa, 0x00)).strong());
        ui.add_space(8.0);

        egui::ScrollArea::horizontal().show(ui, |ui| {
            ui.horizontal(|ui| {
                for (l, bias_data) in network.biases.iter().enumerate() {
                    let b_arr: ndarray::Array1<f32> = bias_data.clone().into();
                    let len = b_arr.len();

                    ui.vertical(|ui| {
                        ui.add_space(8.0);
                        ui.label(egui::RichText::new(format!("b{}: {} neurons", l + 1, len))
                            .size(12.0).color(Color32::from_rgb(0xcc, 0xcc, 0xcc)));
                        self.render_bias_heatmap(ui, &b_arr, &format!("bias_heatmap_{}", l));
                        ui.add_space(8.0);
                    });
                    ui.add_space(12.0);
                }
            });
        });
    }

    fn render_heatmap(&self, ui: &mut egui::Ui, matrix: &Array2<f32>, _id: &str) {
        let rows = matrix.nrows();
        let cols = matrix.ncols();

        if rows == 0 || cols == 0 {
            ui.label("Empty matrix");
            return;
        }

        // Limit display size for large matrices
        let display_rows = rows.min(64);
        let display_cols = cols.min(64);
        let cell_size = (200.0 / display_rows.max(display_cols) as f32).max(2.0).min(8.0);

        let (response, painter) = ui.allocate_painter(
            Vec2::new(display_cols as f32 * cell_size, display_rows as f32 * cell_size),
            Sense::hover(),
        );

        let rect = response.rect;

        // Find min/max for normalization
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        for val in matrix.iter() {
            min_val = min_val.min(*val);
            max_val = max_val.max(*val);
        }
        let range = (max_val - min_val).max(1e-6);

        for i in 0..display_rows {
            for j in 0..display_cols {
                let val = matrix[[i, j]];
                let normalized = (val - min_val) / range;
                let color = Self::value_to_color(normalized);

                let x = rect.left() + j as f32 * cell_size;
                let y = rect.top() + i as f32 * cell_size;
                let cell_rect = Rect::from_min_size(Pos2::new(x, y), Vec2::splat(cell_size));
                painter.rect_filled(cell_rect, 0.0, color);
            }
        }

        // Color bar legend
        let legend_height = 80.0;
        let legend_width = 20.0;
        let legend_x = rect.right() + 8.0;
        let legend_y = rect.top();

        for i in 0..=50 {
            let t = i as f32 / 50.0;
            let color = Self::value_to_color(t);
            let y = legend_y + (1.0 - t) * legend_height;
            painter.rect_filled(
                Rect::from_min_size(Pos2::new(legend_x, y), Vec2::new(legend_width, legend_height / 50.0)),
                0.0,
                color,
            );
        }
        painter.text(
            Pos2::new(legend_x + legend_width + 4.0, legend_y),
            egui::Align2::LEFT_TOP,
            format!("{:.3}", max_val),
            egui::FontId::proportional(10.0),
            Color32::from_rgb(0xaa, 0xaa, 0xaa),
        );
        painter.text(
            Pos2::new(legend_x + legend_width + 4.0, legend_y + legend_height),
            egui::Align2::LEFT_BOTTOM,
            format!("{:.3}", min_val),
            egui::FontId::proportional(10.0),
            Color32::from_rgb(0xaa, 0xaa, 0xaa),
        );

        // Hover tooltip
        if response.hovered() {
            if let Some(pos) = response.interact_pointer_pos() {
                let col = ((pos.x - rect.left()) / cell_size) as usize;
                let row = ((pos.y - rect.top()) / cell_size) as usize;
                if row < rows && col < cols {
                    let val = matrix[[row, col]];
                    response.on_hover_text(format!("W[{}][{}] = {:.4}", row, col, val));
                }
            }
        }
    }

    fn render_bias_heatmap(&self, ui: &mut egui::Ui, vector: &ndarray::Array1<f32>, _id: &str) {
        let len = vector.len();
        if len == 0 {
            ui.label("Empty bias vector");
            return;
        }

        let display_len = len.min(128);
        let cell_width = 8.0;
        let cell_height = 40.0;

        let (response, painter) = ui.allocate_painter(
            Vec2::new(display_len as f32 * cell_width, cell_height),
            Sense::hover(),
        );

        let rect = response.rect;

        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        for val in vector.iter() {
            min_val = min_val.min(*val);
            max_val = max_val.max(*val);
        }
        let range = (max_val - min_val).max(1e-6);

        for i in 0..display_len {
            let val = vector[i];
            let normalized = (val - min_val) / range;
            let color = Self::value_to_color(normalized);

            let x = rect.left() + i as f32 * cell_width;
            let y = rect.top();
            painter.rect_filled(
                Rect::from_min_size(Pos2::new(x, y), Vec2::new(cell_width, cell_height)),
                0.0,
                color,
            );
        }

        if response.hovered() {
            if let Some(pos) = response.interact_pointer_pos() {
                let idx = ((pos.x - rect.left()) / cell_width) as usize;
                if idx < len {
                    let val = vector[idx];
                    response.on_hover_text(format!("b[{}] = {:.4}", idx, val));
                }
            }
        }
    }

    fn value_to_color(normalized: f32) -> Color32 {
        // Blue-White-Red diverging colormap
        let t = normalized.clamp(0.0, 1.0);
        if t < 0.5 {
            // Blue to white
            let k = t * 2.0;
            Color32::from_rgb(
                (k * 255.0) as u8,
                (k * 255.0) as u8,
                255,
            )
        } else {
            // White to red
            let k = (t - 0.5) * 2.0;
            Color32::from_rgb(
                255,
                ((1.0 - k) * 255.0) as u8,
                ((1.0 - k) * 255.0) as u8,
            )
        }
    }

    fn compute_stats(matrix: &Array2<f32>) -> (f32, f32, f32) {
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        let mut sum = 0.0;
        let mut count = 0;
        for val in matrix.iter() {
            min_val = min_val.min(*val);
            max_val = max_val.max(*val);
            sum += *val;
            count += 1;
        }
        let mean = if count > 0 { sum / count as f32 } else { 0.0 };
        (min_val, max_val, mean)
    }

    fn render_training_curves(&self, ui: &mut egui::Ui) {
        ui.label(egui::RichText::new("Training Loss Curve").size(18.0).color(Color32::from_rgb(0x00, 0xaa, 0xff)).strong());
        ui.add_space(8.0);

        if self.training_history.is_empty() {
            ui.centered_and_justified(|ui| {
                ui.add_space(40.0);
                ui.label(egui::RichText::new("No training data yet. Train the network to see loss curves.")
                    .size(14.0).color(Color32::from_rgb(0x88, 0x88, 0x88)));
            });
            return;
        }

        let points: PlotPoints = self.training_history
            .iter()
            .enumerate()
            .map(|(i, &v)| [i as f64, v as f64])
            .collect();

        let line = Line::new("Training Loss", points)
            .color(Color32::from_rgb(0x00, 0xaa, 0xff))
            .width(2.0)
            .fill(0.0);

        let plot = Plot::new("training_loss_curve")
            .view_aspect(3.0)
            .height(200.0)
            .show_axes([true, true])
            .show_grid([true, true])
            .show_background(true)
            .y_axis_label("Loss")
            .x_axis_label("Step")
            .allow_zoom(true)
            .allow_drag(true)
            .allow_boxed_zoom(true);

        plot.show(ui, |plot_ui| {
            plot_ui.line(line);
        });

        // Summary stats
        ui.add_space(8.0);
        ui.horizontal(|ui| {
            let losses: Vec<f32> = self.training_history.iter().cloned().collect();
            let min_loss = losses.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let max_loss = losses.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let avg_loss = losses.iter().sum::<f32>() / losses.len() as f32;
            let last_loss = *losses.last().unwrap_or(&0.0);

            ui.label(egui::RichText::new(format!("Current: {:.6}", last_loss)).size(12.0).color(Color32::from_rgb(0x00, 0xaa, 0xff)));
            ui.add_space(16.0);
            ui.label(egui::RichText::new(format!("Min: {:.6}", min_loss)).size(12.0).color(Color32::from_rgb(0x00, 0xcc, 0x88)));
            ui.add_space(16.0);
            ui.label(egui::RichText::new(format!("Avg: {:.6}", avg_loss)).size(12.0).color(Color32::from_rgb(0xff, 0xaa, 0x00)));
            ui.add_space(16.0);
            ui.label(egui::RichText::new(format!("Max: {:.6}", max_loss)).size(12.0).color(Color32::from_rgb(0xff, 0x66, 0x66)));
            ui.add_space(16.0);
            ui.label(egui::RichText::new(format!("Steps: {}", losses.len())).size(12.0).color(Color32::from_rgb(0xaa, 0xaa, 0xaa)));
        });
    }

    fn render_network_stats(&self, ui: &mut egui::Ui, network: &ModelProfileNetwork) {
        ui.label(egui::RichText::new("Network Statistics").size(18.0).color(Color32::from_rgb(0x00, 0xaa, 0xff)).strong());
        ui.add_space(8.0);

        let total_params: usize = network.weights.iter().map(|w| w.data.len()).sum::<usize>()
            + network.biases.iter().map(|b| b.data.len()).sum::<usize>();

        let layer_dims = Self::get_layer_dims(network);

        egui::Grid::new("network_stats_grid")
            .num_columns(4)
            .spacing([24.0, 8.0])
            .striped(true)
            .show(ui, |ui| {
                self.stat_row(ui, "Input Dim", &network.input_dim.to_string());
                self.stat_row(ui, "Output Dim", &network.output_dim.to_string());
                self.stat_row(ui, "Hidden Layers", &network.hidden_dims.len().to_string());
                self.stat_row(ui, "Total Params", &format!("{:.1}K", total_params as f32 / 1000.0));
                ui.end_row();

                self.stat_row(ui, "Learning Rate", &format!("{:.4}", network.learning_rate));
                self.stat_row(ui, "Experience Buffer", &network.experience_buffer.len().to_string());
                self.stat_row(ui, "Performance History", &network.performance_history.len().to_string());
                self.stat_row(ui, "Avg Performance", &format!("{:.3}", network.avg_performance()));
                ui.end_row();

                for (i, &dim) in layer_dims.iter().enumerate() {
                    let label = match i {
                        0 => "Input Layer",
                        _ if i == layer_dims.len() - 1 => "Output Layer",
                        _ => &format!("Hidden Layer {}", i),
                    };
                    self.stat_row(ui, label, &dim.to_string());
                }
                ui.end_row();
            });

        ui.add_space(4.0);
        match self.last_loss {
            Some(l) => self.stat_row(
                ui,
                "Last train loss",
                &format!("{:.6} ({} experiences)", l, network.experience_buffer.len()),
            ),
            None => self.stat_row(
                ui,
                "Training",
                &format!(
                    "collecting ({} / 32 turns to first step)",
                    network.experience_buffer.len().min(32)
                ),
            ),
        }
    }

    fn stat_row(&self, ui: &mut egui::Ui, label: &str, value: &str) {
        ui.label(egui::RichText::new(label).size(12.0).color(Color32::from_rgb(0x88, 0x88, 0x88)));
        ui.label(egui::RichText::new(value).size(12.0).color(Color32::WHITE).strong());
    }
}