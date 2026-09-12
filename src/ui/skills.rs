// Copyright 2026 Sean M. Stow. All rights reserved.
//! Autonomous Skills & Self-Learning Engine UI
//! Bio-inspired capability registry that reinforces neural weights and stigmergic
//! pheromone paths through multi-agent self-learning feedback loops.

use eframe::egui;
use std::sync::mpsc;
use uuid::Uuid;
use crate::storage::{Skill, Storage};
use crate::ui::app::AppMessage;

pub struct SkillsPanel {
    pub selected_domain: Option<usize>,
    pub search_query: String,
    pub is_creating: bool,
    pub new_name: String,
    pub new_desc: String,
    pub new_prompt: String,
    pub new_domain: usize,
    pub cached_skills: Vec<Skill>,
    pub last_refresh: Option<std::time::Instant>,
}

impl Default for SkillsPanel {
    fn default() -> Self {
        Self::new()
    }
}

impl SkillsPanel {
    pub fn new() -> Self {
        Self {
            selected_domain: None,
            search_query: String::new(),
            is_creating: false,
            new_name: String::new(),
            new_desc: String::new(),
            new_prompt: String::new(),
            new_domain: 0,
            cached_skills: Vec::new(),
            last_refresh: None,
        }
    }

    pub fn refresh_skills(&mut self, storage: &Option<Storage>) {
        if let Some(st) = storage.as_ref() {
            if let Ok(skills) = st.load_skills() {
                self.cached_skills = skills;
                self.last_refresh = Some(std::time::Instant::now());
            }
        }
    }

    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        storage: &Option<Storage>,
        tx: &mpsc::Sender<AppMessage>,
        focused_slot: usize,
    ) {
        // Refresh cache periodically or on initial load
        if self.cached_skills.is_empty()
            || self.last_refresh.map(|t| t.elapsed().as_secs() > 5).unwrap_or(true)
        {
            self.refresh_skills(storage);
        }

        ui.add_space(8.0);

        // Header and Stats
        ui.horizontal(|ui| {
            ui.heading("⚡ Autonomous Skills & Self-Learning Engine");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("➕ Create New Skill").clicked() {
                    self.is_creating = true;
                    self.new_name.clear();
                    self.new_desc.clear();
                    self.new_prompt.clear();
                    self.new_domain = 0;
                }
                if ui.button("🔄 Refresh").clicked() {
                    self.refresh_skills(storage);
                }
            });
        });

        ui.label(
            "Bio-inspired agent capability registry. Skills dynamically adapt neural weights and \
             deposit stigmergic pheromones across cohabitating model slots through positive feedback.",
        );

        ui.add_space(6.0);

        // Stats summary chips
        let total_skills = self.cached_skills.len();
        let total_runs: u64 = self.cached_skills.iter().map(|s| s.execution_count).sum();
        let avg_score = if total_skills > 0 {
            self.cached_skills.iter().map(|s| s.reinforcement_score).sum::<f32>() / total_skills as f32
        } else {
            0.0
        };

        ui.horizontal(|ui| {
            egui::Frame::NONE
                .fill(egui::Color32::from_rgb(0x13, 0x1a, 0x2b))
                .corner_radius(egui::CornerRadius::same(6))
                .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(0x1e, 0x29, 0x3b)))
                .inner_margin(egui::Margin::symmetric(10, 6))
                .show(ui, |ui| {
                    ui.label(format!("📦 Registered Skills: {}", total_skills));
                    ui.separator();
                    ui.label(format!("🎯 Total Executions: {}", total_runs));
                    ui.separator();
                    ui.label(format!("⭐ Mean Reinforcement: {:.2}", avg_score));
                });
        });

        ui.add_space(10.0);

        // Filters and Search Row
        ui.horizontal(|ui| {
            ui.label("Filter Domain:");
            if ui.selectable_label(self.selected_domain.is_none(), "All").clicked() {
                self.selected_domain = None;
            }
            let domains = [
                (3, "🛡 Cyber/Critic"),
                (1, "💻 Coder"),
                (2, "🔬 Researcher"),
                (4, "📋 Planner"),
                (5, "✍ Writer"),
                (0, "🌐 General"),
            ];
            for (idx, label) in domains {
                let active = self.selected_domain == Some(idx);
                if ui.selectable_label(active, label).clicked() {
                    self.selected_domain = if active { None } else { Some(idx) };
                }
            }

            ui.add_space(12.0);
            ui.label("🔍");
            ui.add(
                egui::TextEdit::singleline(&mut self.search_query)
                    .hint_text("Search skills...")
                    .desired_width(180.0),
            );
        });

        ui.add_space(10.0);
        ui.separator();
        ui.add_space(8.0);

        // Creation Modal Dialog
        if self.is_creating {
            egui::Window::new("➕ Create Autonomous Skill")
                .collapsible(false)
                .resizable(false)
                .min_width(450.0)
                .show(ui.ctx(), |ui| {
                    ui.label("Define a reusable agent capability. Skills are encrypted and saved to the post-quantum vault.");
                    ui.add_space(6.0);

                    ui.horizontal(|ui| {
                        ui.label("Skill Name:   ");
                        ui.add(egui::TextEdit::singleline(&mut self.new_name).hint_text("e.g. fuzz_tester, api_auditor"));
                    });

                    ui.horizontal(|ui| {
                        ui.label("Domain Niche: ");
                        egui::ComboBox::from_id_salt("new_skill_domain")
                            .selected_text(match self.new_domain {
                                1 => "💻 Coder",
                                2 => "🔬 Researcher",
                                3 => "🛡 Cyber / Critic",
                                4 => "📋 Planner",
                                5 => "✍ Writer",
                                _ => "🌐 General",
                            })
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut self.new_domain, 0, "🌐 General");
                                ui.selectable_value(&mut self.new_domain, 1, "💻 Coder");
                                ui.selectable_value(&mut self.new_domain, 2, "🔬 Researcher");
                                ui.selectable_value(&mut self.new_domain, 3, "🛡 Cyber / Critic");
                                ui.selectable_value(&mut self.new_domain, 4, "📋 Planner");
                                ui.selectable_value(&mut self.new_domain, 5, "✍ Writer");
                            });
                    });

                    ui.horizontal(|ui| {
                        ui.label("Description:  ");
                        ui.add(egui::TextEdit::singleline(&mut self.new_desc).hint_text("Brief description of capability"));
                    });

                    ui.label("Prompt Template (Instructional System Directive):");
                    ui.add(
                        egui::TextEdit::multiline(&mut self.new_prompt)
                            .hint_text("Enter the specialized prompt template that drives this agent skill...")
                            .desired_rows(6)
                            .desired_width(f32::INFINITY),
                    );

                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        let can_save = !self.new_name.trim().is_empty() && !self.new_prompt.trim().is_empty();
                        if ui.add_enabled(can_save, egui::Button::new("💾 Save Skill to Vault")).clicked() {
                            if let Some(st) = storage.as_ref() {
                                let skill = Skill {
                                    id: Uuid::new_v4(),
                                    name: self.new_name.trim().to_string(),
                                    description: self.new_desc.trim().to_string(),
                                    prompt_template: self.new_prompt.trim().to_string(),
                                    domain_idx: self.new_domain,
                                    reinforcement_score: 1.0,
                                    execution_count: 0,
                                    last_used: None,
                                    is_built_in: false,
                                };
                                if st.save_skill(&skill).is_ok() {
                                    self.refresh_skills(storage);
                                    self.is_creating = false;
                                }
                            }
                        }
                        if ui.button("Cancel").clicked() {
                            self.is_creating = false;
                        }
                    });
                });
        }

        // Skills Grid View
        let query = self.search_query.trim().to_lowercase();
        let filtered_skills: Vec<Skill> = self
            .cached_skills
            .iter()
            .filter(|s| {
                if let Some(d) = self.selected_domain {
                    if s.domain_idx != d {
                        return false;
                    }
                }
                if !query.is_empty() {
                    let matches_name = s.name.to_lowercase().contains(&query);
                    let matches_desc = s.description.to_lowercase().contains(&query);
                    return matches_name || matches_desc;
                }
                true
            })
            .cloned()
            .collect();

        if filtered_skills.is_empty() {
            ui.add_space(20.0);
            ui.vertical_centered(|ui| {
                ui.label("No skills match the current filter or search criteria.");
                if ui.button("➕ Create a Skill").clicked() {
                    self.is_creating = true;
                }
            });
            return;
        }

        let mut to_delete: Option<Uuid> = None;

        for skill in &filtered_skills {
            egui::Frame::NONE
                .fill(egui::Color32::from_rgb(0x11, 0x18, 0x27))
                .corner_radius(egui::CornerRadius::same(8))
                .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(0x1e, 0x29, 0x3b)))
                .inner_margin(egui::Margin::symmetric(14, 12))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        // Title
                        ui.label(
                            egui::RichText::new(format!("⚡ {}", skill.name))
                                .strong()
                                .size(16.0)
                                .color(egui::Color32::from_rgb(0x38, 0xbd, 0xf8)),
                        );

                        // Domain Badge
                        let (domain_label, badge_color) = match skill.domain_idx {
                            1 => ("💻 Coder", egui::Color32::from_rgb(0x34, 0xd3, 0x99)),
                            2 => ("🔬 Researcher", egui::Color32::from_rgb(0xa7, 0x8b, 0xfa)),
                            3 => ("🛡 Cyber / Critic", egui::Color32::from_rgb(0x38, 0xbd, 0xf8)),
                            4 => ("📋 Planner", egui::Color32::from_rgb(0xfb, 0xbf, 0x24)),
                            5 => ("✍ Writer", egui::Color32::from_rgb(0xf4, 0x72, 0xb6)),
                            _ => ("🌐 General", egui::Color32::from_rgb(0x94, 0xa3, 0xb8)),
                        };

                        egui::Frame::NONE
                            .fill(egui::Color32::from_rgba_premultiplied(
                                badge_color.r() / 4,
                                badge_color.g() / 4,
                                badge_color.b() / 4,
                                120,
                            ))
                            .corner_radius(egui::CornerRadius::same(4))
                            .stroke(egui::Stroke::new(1.0, badge_color))
                            .inner_margin(egui::Margin::symmetric(6, 2))
                            .show(ui, |ui| {
                                ui.label(egui::RichText::new(domain_label).size(11.0).color(badge_color));
                            });

                        if skill.is_built_in {
                            ui.label(egui::RichText::new("🔒 Built-in Seed").size(11.0).color(egui::Color32::GRAY));
                        }

                        // Right-aligned metrics
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.label(format!("🎯 {} runs", skill.execution_count));
                            ui.label(
                                egui::RichText::new(format!("⭐ {:.2}", skill.reinforcement_score))
                                    .strong()
                                    .color(egui::Color32::from_rgb(0xf5, 0x9e, 0x0b)),
                            );
                        });
                    });

                    ui.add_space(4.0);
                    ui.label(egui::RichText::new(&skill.description).italics());

                    // Collapsible prompt template
                    ui.collapsing("📜 View Prompt Template", |ui| {
                        egui::Frame::NONE
                            .fill(egui::Color32::from_rgb(0x07, 0x0a, 0x12))
                            .corner_radius(egui::CornerRadius::same(4))
                            .inner_margin(egui::Margin::symmetric(8, 6))
                            .show(ui, |ui| {
                                ui.label(
                                    egui::RichText::new(&skill.prompt_template)
                                        .monospace()
                                        .size(12.0)
                                        .color(egui::Color32::from_rgb(0x94, 0xa3, 0xb8)),
                                );
                            });
                    });

                    ui.add_space(6.0);
                    ui.horizontal(|ui| {
                        // Execute trigger
                        if ui
                            .button("⚡ Run Skill")
                            .on_hover_text("Load prompt template into active chat slot and run")
                            .clicked()
                        {
                            let _ = tx.send(AppMessage::RunSkill {
                                skill_name: skill.name.clone(),
                                target_slot: Some(focused_slot),
                            });
                        }

                        // Reinforcement triggers
                        if ui
                            .button("👍 Reinforce (+0.5)")
                            .on_hover_text("Reward skill and strengthen pheromone trail")
                            .clicked()
                        {
                            let _ = tx.send(AppMessage::ReinforceSkill {
                                skill_id: skill.id,
                                reward_delta: 0.5,
                            });
                        }

                        if ui
                            .button("👎 Penalize (-0.2)")
                            .on_hover_text("Penalize skill reinforcement score")
                            .clicked()
                        {
                            let _ = tx.send(AppMessage::ReinforceSkill {
                                skill_id: skill.id,
                                reward_delta: -0.2,
                            });
                        }

                        if !skill.is_built_in {
                            if ui
                                .button("🗑 Delete")
                                .on_hover_text("Delete this custom skill from vault")
                                .clicked()
                            {
                                to_delete = Some(skill.id);
                            }
                        }
                    });
                });

            ui.add_space(8.0);
        }

        if let Some(id) = to_delete {
            if let Some(st) = storage.as_ref() {
                let _ = st.delete_skill(id);
                self.refresh_skills(storage);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skills_panel_initial_state() {
        let panel = SkillsPanel::new();
        assert!(panel.cached_skills.is_empty());
        assert_eq!(panel.selected_domain, None);
        assert!(!panel.is_creating);
    }

    #[test]
    fn skills_filtering_and_search() {
        let mut panel = SkillsPanel::new();
        panel.cached_skills = vec![
            Skill {
                id: Uuid::new_v4(),
                name: "vulnerability_scan".to_string(),
                description: "Security scan".to_string(),
                prompt_template: "Prompt".to_string(),
                domain_idx: 3,
                reinforcement_score: 5.0,
                execution_count: 2,
                last_used: None,
                is_built_in: true,
            },
            Skill {
                id: Uuid::new_v4(),
                name: "code_optimizer".to_string(),
                description: "Refactor loops".to_string(),
                prompt_template: "Prompt".to_string(),
                domain_idx: 1,
                reinforcement_score: 4.0,
                execution_count: 1,
                last_used: None,
                is_built_in: true,
            },
        ];

        // Filter by domain 3 (Cyber)
        panel.selected_domain = Some(3);
        let cyber_matches: Vec<_> = panel.cached_skills.iter().filter(|s| panel.selected_domain == Some(s.domain_idx)).collect();
        assert_eq!(cyber_matches.len(), 1);
        assert_eq!(cyber_matches[0].name, "vulnerability_scan");

        // Filter by search query
        panel.selected_domain = None;
        panel.search_query = "loops".to_string();
        let query = panel.search_query.to_lowercase();
        let query_matches: Vec<_> = panel.cached_skills.iter().filter(|s| s.description.to_lowercase().contains(&query)).collect();
        assert_eq!(query_matches.len(), 1);
        assert_eq!(query_matches[0].name, "code_optimizer");
    }
}
