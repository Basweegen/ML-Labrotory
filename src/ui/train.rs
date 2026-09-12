//! Train tab: the self-learning center. Past chats already feed the
//! profiler network as experiences; here that learning is visible and
//! controllable: dataset size, exploration/exploitation knobs, on-demand
//! training, and reset.

use eframe::egui;
use std::sync::mpsc;

use crate::neural::{Experience, ModelProfileNetwork};
use crate::storage::{ChatSession, Storage};
use ndarray::Array1;
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
        storage: &Storage,
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

        // ---- skills by role ----
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Skills").size(15.0).strong());
        });
        if network.skill_stats.is_empty() {
            ui.label(
                egui::RichText::new("No finished turns yet - chat and each role earns its record here.")
                    .size(12.0)
                    .color(egui::Color32::from_rgb(0x88, 0x88, 0x88)),
            );
        } else {
            egui::Grid::new("skill_table")
                .num_columns(4)
                .spacing([16.0, 6.0])
                .striped(true)
                .show(ui, |ui| {
                    ui.strong("Role");
                    ui.strong("Turns");
                    ui.strong("Win");
                    ui.strong("Avg reward");
                    ui.end_row();
                    let mut roles: Vec<u8> = network.skill_stats.keys().copied().collect();
                    roles.sort_unstable();
                    for r in roles {
                        if let Some(s) = network.skill_stats.get(&r) {
                            ui.label(role_name(r));
                            ui.label(format!("{}", s.trials));
                            ui.label(format!("{:.0}%", s.win_rate() * 100.0));
                            ui.label(format!("{:+.2}", s.avg_reward()));
                            ui.end_row();
                        }
                    }
                });
        }
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
                .small_button("Learn from history")
                .on_hover_text("Replay saved chats as weak training signal (replies exist = success)")
                .clicked()
            {
                self.status = learn_from_history(network, neural_panel, storage);
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

/// Display name for a skill-table role index (mirrors ModelRole order).
fn role_name(idx: u8) -> &'static str {
    match idx {
        0 => "General",
        1 => "Coder",
        2 => "Researcher",
        3 => "Critic",
        4 => "Planner",
        5 => "Writer",
        _ => "Custom",
    }
}

/// Replay saved sessions as weak training signal: a stored assistant reply
/// counts as a 0.75 success with neutral latency. Role/model context is
/// approximated (General role, slot 0); live turns remain the strong signal.
/// Pure builder below is unit-tested; I/O + training live in the caller.
fn learn_from_history(
    network: &mut ModelProfileNetwork,
    neural_panel: &mut NeuralVizPanel,
    storage: &Storage,
) -> String {
    let mut sessions = match storage.load_sessions() {
        Ok(s) => s,
        Err(e) => return format!("History unreadable: {e:#}"),
    };
    sessions.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
    sessions.truncate(200);
    let mut used = 0;
    for s in &sessions {
        if let Some(exp) = experience_from_session(s) {
            let action = {
                let state: Array1<f32> = exp.state.clone().into();
                network.select_model_profile(&state)
            };
            let mut exp = exp;
            exp.action = action;
            network.add_experience(exp);
            network.update_performance(0.75);
            used += 1;
        }
    }
    if used < 4 {
        return format!("Only {used} usable past chats (need 4 with replies)");
    }
    let mut first = None;
    let mut last = 0.0;
    let mut ran = 0;
    for _ in 0..20 {
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
    format!(
        "Learned from {used} past chats (weak signal): {ran} batches, loss {:.4} \u{2192} {:.4}",
        first.unwrap_or(last),
        last
    )
}

/// Weak-signal experience from one saved session. None when no assistant
/// reply exists (nothing to learn).
fn experience_from_session(s: &ChatSession) -> Option<Experience> {
    let user = s.messages.iter().rev().find(|m| m.role == "user")?;
    let asst = s.messages.iter().rev().find(|m| m.role == "assistant")?;
    let hash = s
        .model
        .bytes()
        .fold(0u64, |a, b| a.wrapping_mul(31).wrapping_add(b as u64));
    let ctx = Array1::from_vec(vec![
        (user.content.len() as f32 / 50_000.0).min(1.0),
        (s.messages.len() as f32 / 20.0).min(1.0),
        1.0,
        30.0 / 120.0,
        (asst.content.len() as f32 / 50_000.0).min(1.0),
        0.0,
        0.0,
        ((hash % 100) as f32) / 100.0,
    ]);
    Some(Experience {
        state: ctx.clone().into(),
        action: 0,
        reward: 0.75,
        next_state: ctx.into(),
        done: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use uuid::Uuid;

    fn session_with_reply() -> ChatSession {
        use crate::storage::ChatMessage;
        ChatSession {
            id: Uuid::new_v4(),
            name: "t".to_string(),
            model: "m".to_string(),
            messages: vec![
                ChatMessage {
                    role: "user".to_string(),
                    content: "hi".to_string(),
                    timestamp: Utc::now(),
                },
                ChatMessage {
                    role: "assistant".to_string(),
                    content: "hello".to_string(),
                    timestamp: Utc::now(),
                },
            ],
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }

    #[test]
    fn replay_needs_a_reply() {
        let mut s = session_with_reply();
        let exp = experience_from_session(&s).unwrap();
        assert_eq!(exp.reward, 0.75);
        let state: Array1<f32> = exp.state.clone().into();
        assert_eq!(state.len(), 8);
        assert!(state.iter().all(|v| v.is_finite()));
        s.messages.pop();
        assert!(experience_from_session(&s).is_none());
    }
}
