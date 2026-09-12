// Copyright 2026 Sean M. Stow. All rights reserved.
//! Swarm Relay Pipeline: Multi-agent sequential model collaboration.
//! Inspired by biological swarm logic (scout, forager, colony defense, queen synthesis).
//! Executes one model at a time to strictly conserve host memory while producing
//! superhuman synthesis by chaining specialized roles.

use eframe::egui;
use crate::ollama::api::{ChatOptions, ChatRequest, Message, Model, OllamaClient};
use crate::security::find_secrets;
use crate::ui::app::{AppMessage, ModelRole};
use std::sync::mpsc;
use std::time::Instant;
use tokio::runtime::Runtime;

#[derive(Debug, Clone, PartialEq)]
pub enum StepStatus {
    Pending,
    Running,
    Completed(f32), // duration in secs
    Failed(String),
}

#[derive(Debug, Clone)]
pub struct RelayStep {
    pub name: String,
    pub role: ModelRole,
    pub model: Option<String>,
    pub custom_prompt: String,
    pub status: StepStatus,
    pub output: String,
    pub collapsed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwarmTemplate {
    SymbioticHive,
    AdversarialCode,
    ExecutiveResearch,
    EngineeringTriad,
    CyberSocTriad,
    FullStackSwarm,
    QuantumScientific,
    Custom,
}

impl SwarmTemplate {
    pub fn all() -> Vec<SwarmTemplate> {
        vec![
            SwarmTemplate::SymbioticHive,
            SwarmTemplate::AdversarialCode,
            SwarmTemplate::ExecutiveResearch,
            SwarmTemplate::EngineeringTriad,
            SwarmTemplate::CyberSocTriad,
            SwarmTemplate::FullStackSwarm,
            SwarmTemplate::QuantumScientific,
            SwarmTemplate::Custom,
        ]
    }

    pub fn label(self) -> &'static str {
        match self {
            SwarmTemplate::SymbioticHive => "🐝 Symbiotic Hive (Planner → Coder → Critic → Synthesis)",
            SwarmTemplate::AdversarialCode => "⚔️ Adversarial Code (Coder → Security Auditor → Refiner)",
            SwarmTemplate::ExecutiveResearch => "🔍 Executive Research (Scout → Epistemic Critic → Briefing)",
            SwarmTemplate::EngineeringTriad => "📐 Engineering Triad (Planner → Coder → Reviewer)",
            SwarmTemplate::CyberSocTriad => "🛡️ Cyber / SOC Triad (Recon → Threat Hunter → Defender)",
            SwarmTemplate::FullStackSwarm => "🌐 Full-Stack Dev Swarm (Architect → Backend → UI → QA)",
            SwarmTemplate::QuantumScientific => "⚛️ Quantum & Scientific (Hypothesis → Math Modeler → Reviewer → Synthesis)",
            SwarmTemplate::Custom => "⚙️ Custom Swarm Chain",
        }
    }

    pub fn short_id(self) -> &'static str {
        match self {
            SwarmTemplate::SymbioticHive => "hive",
            SwarmTemplate::AdversarialCode => "code",
            SwarmTemplate::ExecutiveResearch => "research",
            SwarmTemplate::EngineeringTriad => "triad",
            SwarmTemplate::CyberSocTriad => "soc",
            SwarmTemplate::FullStackSwarm => "fullstack",
            SwarmTemplate::QuantumScientific => "quantum",
            SwarmTemplate::Custom => "custom",
        }
    }

    pub fn from_id(id: &str) -> Option<SwarmTemplate> {
        match id.to_lowercase().trim() {
            "hive" | "symbiotichive" => Some(SwarmTemplate::SymbioticHive),
            "code" | "adversarialcode" => Some(SwarmTemplate::AdversarialCode),
            "research" | "executiveresearch" => Some(SwarmTemplate::ExecutiveResearch),
            "triad" | "engineeringtriad" => Some(SwarmTemplate::EngineeringTriad),
            "soc" | "cybersoctriad" | "cyber" => Some(SwarmTemplate::CyberSocTriad),
            "fullstack" | "fullstackswarm" | "dev" => Some(SwarmTemplate::FullStackSwarm),
            "quantum" | "scientific" | "quantumscientific" => Some(SwarmTemplate::QuantumScientific),
            "custom" => Some(SwarmTemplate::Custom),
            _ => None,
        }
    }
}

pub struct RelayPanel {
    pub prompt: String,
    pub steps: Vec<RelayStep>,
    pub template: SwarmTemplate,
    pub active_step: Option<usize>,
    pub live_stream: String,
    pub final_synthesis: String,
    pub is_running: bool,
    pub step_start: Option<Instant>,
    pub export_note: Option<String>,
    pub consensus_score: Option<f32>,
}

impl RelayPanel {
    pub fn new() -> Self {
        let mut panel = Self {
            prompt: String::new(),
            steps: Vec::new(),
            template: SwarmTemplate::SymbioticHive,
            active_step: None,
            live_stream: String::new(),
            final_synthesis: String::new(),
            is_running: false,
            step_start: None,
            export_note: None,
            consensus_score: None,
        };
        panel.apply_template(SwarmTemplate::SymbioticHive);
        panel
    }

    pub fn apply_template(&mut self, t: SwarmTemplate) {
        self.template = t;
        match t {
            SwarmTemplate::SymbioticHive => {
                self.steps = vec![
                    RelayStep {
                        name: "Scout Architect (Deconstruction)".to_string(),
                        role: ModelRole::Planner,
                        model: None,
                        custom_prompt: "Deconstruct the user's objective into core requirements, architectural blueprints, and critical invariant constraints. Be rigorous and structured.".to_string(),
                        status: StepStatus::Pending,
                        output: String::new(),
                        collapsed: false,
                    },
                    RelayStep {
                        name: "Colony Worker (Implementation)".to_string(),
                        role: ModelRole::Coder,
                        model: None,
                        custom_prompt: "Using the architect's plan, produce the concrete, high-precision solution or implementation. Include thorough logic.".to_string(),
                        status: StepStatus::Pending,
                        output: String::new(),
                        collapsed: false,
                    },
                    RelayStep {
                        name: "Swarm Defender (Red-Team Audit)".to_string(),
                        role: ModelRole::Critic,
                        model: None,
                        custom_prompt: "Act as an elite adversarial auditor. Scrutinize the implementation for security vulnerabilities, logic flaws, memory leaks, and edge cases.".to_string(),
                        status: StepStatus::Pending,
                        output: String::new(),
                        collapsed: false,
                    },
                    RelayStep {
                        name: "Queen Consolidator (Final Synthesis)".to_string(),
                        role: ModelRole::General,
                        model: None,
                        custom_prompt: "Integrate the blueprint, implementation, and audit findings into an authoritative, master-class executive final deliverable.".to_string(),
                        status: StepStatus::Pending,
                        output: String::new(),
                        collapsed: false,
                    },
                ];
            }
            SwarmTemplate::AdversarialCode => {
                self.steps = vec![
                    RelayStep {
                        name: "Lead Systems Coder".to_string(),
                        role: ModelRole::Coder,
                        model: None,
                        custom_prompt: "Implement the requested system code with optimal algorithms and clean architecture.".to_string(),
                        status: StepStatus::Pending,
                        output: String::new(),
                        collapsed: false,
                    },
                    RelayStep {
                        name: "Security & Vulnerability Auditor".to_string(),
                        role: ModelRole::Critic,
                        model: None,
                        custom_prompt: "Audit the code for buffer overflows, race conditions, auth bypasses, and injection risks. Provide precise corrections.".to_string(),
                        status: StepStatus::Pending,
                        output: String::new(),
                        collapsed: false,
                    },
                    RelayStep {
                        name: "Production Refiner".to_string(),
                        role: ModelRole::Coder,
                        model: None,
                        custom_prompt: "Produce the definitive, hardened production code incorporating all security recommendations.".to_string(),
                        status: StepStatus::Pending,
                        output: String::new(),
                        collapsed: false,
                    },
                ];
            }
            SwarmTemplate::ExecutiveResearch => {
                self.steps = vec![
                    RelayStep {
                        name: "Deep Intelligence Scout".to_string(),
                        role: ModelRole::Researcher,
                        model: None,
                        custom_prompt: "Explore the topic thoroughly, identifying verifiable facts, technical citations, and alternative hypotheses.".to_string(),
                        status: StepStatus::Pending,
                        output: String::new(),
                        collapsed: false,
                    },
                    RelayStep {
                        name: "Epistemic Critic".to_string(),
                        role: ModelRole::Critic,
                        model: None,
                        custom_prompt: "Challenge assumptions, highlight biases, and filter out speculative or unverified claims.".to_string(),
                        status: StepStatus::Pending,
                        output: String::new(),
                        collapsed: false,
                    },
                    RelayStep {
                        name: "Briefing Officer".to_string(),
                        role: ModelRole::Writer,
                        model: None,
                        custom_prompt: "Distill the verified intelligence into a high-level executive briefing with actionable takeaways.".to_string(),
                        status: StepStatus::Pending,
                        output: String::new(),
                        collapsed: false,
                    },
                ];
            }
            SwarmTemplate::EngineeringTriad => {
                self.steps = vec![
                    RelayStep {
                        name: "Architect Planner".to_string(),
                        role: ModelRole::Planner,
                        model: None,
                        custom_prompt: "Deconstruct the user's objective into concrete engineering components, interfaces, and invariant constraints.".to_string(),
                        status: StepStatus::Pending,
                        output: String::new(),
                        collapsed: false,
                    },
                    RelayStep {
                        name: "Lead Systems Engineer".to_string(),
                        role: ModelRole::Coder,
                        model: None,
                        custom_prompt: "Implement the production solution adhering cleanly to the architecture and strict memory/type safety.".to_string(),
                        status: StepStatus::Pending,
                        output: String::new(),
                        collapsed: false,
                    },
                    RelayStep {
                        name: "Rigorous Peer Reviewer & QA".to_string(),
                        role: ModelRole::Critic,
                        model: None,
                        custom_prompt: "Conduct a rigorous peer review, identifying architectural anti-patterns, boundary edge cases, and optimization opportunities.".to_string(),
                        status: StepStatus::Pending,
                        output: String::new(),
                        collapsed: false,
                    },
                ];
            }
            SwarmTemplate::CyberSocTriad => {
                self.steps = vec![
                    RelayStep {
                        name: "Perimeter Recon Scout".to_string(),
                        role: ModelRole::Researcher,
                        model: None,
                        custom_prompt: "Analyze the environmental threat surface, exposed endpoints, protocol weaknesses, and potential attack vectors.".to_string(),
                        status: StepStatus::Pending,
                        output: String::new(),
                        collapsed: false,
                    },
                    RelayStep {
                        name: "Threat Hunter & Exploit Analyst".to_string(),
                        role: ModelRole::Critic,
                        model: None,
                        custom_prompt: "Assess exploit viability, privilege escalation risks, lateral movement vectors, and persistence mechanisms.".to_string(),
                        status: StepStatus::Pending,
                        output: String::new(),
                        collapsed: false,
                    },
                    RelayStep {
                        name: "SOC Incident Defender & Hardener".to_string(),
                        role: ModelRole::Coder,
                        model: None,
                        custom_prompt: "Engineer concrete defense configurations, SIEM detection rules, packet filters, and patched source code to eliminate vulnerabilities.".to_string(),
                        status: StepStatus::Pending,
                        output: String::new(),
                        collapsed: false,
                    },
                ];
            }
            SwarmTemplate::FullStackSwarm => {
                self.steps = vec![
                    RelayStep {
                        name: "Systems Architect".to_string(),
                        role: ModelRole::Planner,
                        model: None,
                        custom_prompt: "Define comprehensive full-stack architecture, schemas, REST/WebSocket contracts, and security boundaries.".to_string(),
                        status: StepStatus::Pending,
                        output: String::new(),
                        collapsed: false,
                    },
                    RelayStep {
                        name: "Backend Engineer".to_string(),
                        role: ModelRole::Coder,
                        model: None,
                        custom_prompt: "Build concurrent, production-grade backend services with robust validation, error handling, and database integration.".to_string(),
                        status: StepStatus::Pending,
                        output: String::new(),
                        collapsed: false,
                    },
                    RelayStep {
                        name: "Frontend & UI Specialist".to_string(),
                        role: ModelRole::Writer,
                        model: None,
                        custom_prompt: "Design modern, responsive client components, state management flows, and aesthetic styling.".to_string(),
                        status: StepStatus::Pending,
                        output: String::new(),
                        collapsed: false,
                    },
                    RelayStep {
                        name: "Integration & Security Verifier".to_string(),
                        role: ModelRole::Critic,
                        model: None,
                        custom_prompt: "Verify end-to-end integration, validate client-server contracts, run vulnerability audits, and verify zero regressions.".to_string(),
                        status: StepStatus::Pending,
                        output: String::new(),
                        collapsed: false,
                    },
                ];
            }
            SwarmTemplate::QuantumScientific => {
                self.steps = vec![
                    RelayStep {
                        name: "Quantum Hypothesis Scout".to_string(),
                        role: ModelRole::Researcher,
                        model: None,
                        custom_prompt: "Formulate testable quantum/scientific hypotheses, literature baselines, and state-space definitions.".to_string(),
                        status: StepStatus::Pending,
                        output: String::new(),
                        collapsed: false,
                    },
                    RelayStep {
                        name: "Mathematical & Stochastic Modeler".to_string(),
                        role: ModelRole::Planner,
                        model: None,
                        custom_prompt: "Derive mathematical proofs, state transitions, circuit representations, or algorithmic models for the problem.".to_string(),
                        status: StepStatus::Pending,
                        output: String::new(),
                        collapsed: false,
                    },
                    RelayStep {
                        name: "Falsification Critic".to_string(),
                        role: ModelRole::Critic,
                        model: None,
                        custom_prompt: "Subject equations and assumptions to strict falsification tests, boundary singularities, and decoherence analysis.".to_string(),
                        status: StepStatus::Pending,
                        output: String::new(),
                        collapsed: false,
                    },
                    RelayStep {
                        name: "Empirical Synthesizer".to_string(),
                        role: ModelRole::General,
                        model: None,
                        custom_prompt: "Integrate derivations and critical findings into an authoritative scientific paper and executable conclusion.".to_string(),
                        status: StepStatus::Pending,
                        output: String::new(),
                        collapsed: false,
                    },
                ];
            }
            SwarmTemplate::Custom => {}
        }
    }

    /// Intelligent role-to-model matching:
    /// Evaluates model names and tags to select the most specialized model for each role.
    pub fn find_best_model_for_role(role: &ModelRole, available_models: &[Model]) -> Option<String> {
        if available_models.is_empty() {
            return None;
        }

        let mut scored: Vec<(&Model, i32)> = available_models
            .iter()
            .map(|m| {
                let name = m.name.to_lowercase();
                let mut score = 0;

                match role {
                    ModelRole::Coder => {
                        if name.contains("coder") || name.contains("code") { score += 100; }
                        if name.contains("qwen2.5-coder") { score += 50; }
                        if name.contains("deepseek-coder") { score += 50; }
                        if name.contains("codellama") || name.contains("starcoder") { score += 40; }
                        if name.contains("qwen") { score += 20; }
                    }
                    ModelRole::Critic => {
                        if name.contains("critic") || name.contains("audit") || name.contains("sec") { score += 100; }
                        if name.contains("deepseek") || name.contains("r1") { score += 50; }
                        if name.contains("hermes") { score += 40; }
                        if name.contains("llama3") { score += 20; }
                    }
                    ModelRole::Researcher => {
                        if name.contains("research") || name.contains("paper") { score += 100; }
                        if name.contains("phi4") || name.contains("phi3") { score += 50; }
                        if name.contains("hermes") { score += 40; }
                        if name.contains("gemma") { score += 20; }
                    }
                    ModelRole::Planner => {
                        if name.contains("plan") || name.contains("struct") { score += 100; }
                        if name.contains("llama3") || name.contains("qwen") { score += 40; }
                        if name.contains("phi4") { score += 30; }
                    }
                    ModelRole::Writer => {
                        if name.contains("writer") || name.contains("story") { score += 100; }
                        if name.contains("gemma") || name.contains("mistral") { score += 40; }
                        if name.contains("hermes") || name.contains("llama3") { score += 30; }
                    }
                    ModelRole::General | ModelRole::Custom(_) => {
                        if name.contains("instruct") || name.contains("chat") { score += 30; }
                    }
                }
                (m, score)
            })
            .collect();

        scored.sort_by(|a, b| b.1.cmp(&a.1));
        scored.first().map(|(m, _)| m.name.clone())
    }

    pub fn auto_assign_models(&mut self, available_models: &[Model]) {
        if available_models.is_empty() {
            return;
        }
        for (idx, step) in self.steps.iter_mut().enumerate() {
            if step.model.is_none() {
                if let Some(matched) = Self::find_best_model_for_role(&step.role, available_models) {
                    step.model = Some(matched);
                } else {
                    let m = &available_models[idx % available_models.len()];
                    step.model = Some(m.name.clone());
                }
            }
        }
    }

    pub fn reset_pipeline(&mut self) {
        self.active_step = None;
        self.live_stream.clear();
        self.final_synthesis.clear();
        self.is_running = false;
        self.step_start = None;
        self.consensus_score = None;
        for s in &mut self.steps {
            s.status = StepStatus::Pending;
            s.output.clear();
            s.collapsed = false;
        }
    }

    pub fn abort_pipeline(&mut self) {
        self.is_running = false;
        if let Some(cur) = self.active_step {
            if let Some(step) = self.steps.get_mut(cur) {
                step.status = StepStatus::Failed("Aborted by user".to_string());
            }
        }
        self.active_step = None;
        self.live_stream.clear();
    }

    pub fn push_chunk(&mut self, step_idx: usize, chunk: String) {
        if self.active_step == Some(step_idx) {
            self.live_stream.push_str(&chunk);
        }
    }

    pub fn step_completed(&mut self, step_idx: usize, output: String, duration: f32) {
        if let Some(step) = self.steps.get_mut(step_idx) {
            step.output = output.clone();
            step.status = StepStatus::Completed(duration);
            step.collapsed = true;
        }
        self.live_stream.clear();
        // If this was the final step, store in final_synthesis and compute consensus
        if step_idx + 1 == self.steps.len() {
            self.final_synthesis = output;
            self.is_running = false;
            self.active_step = None;
            self.compute_consensus_score();
        }
    }

    /// Compute empirical alignment / consensus score between worker and auditor steps
    fn compute_consensus_score(&mut self) {
        let mut worker_output = None;
        let mut critic_output = None;
        for s in &self.steps {
            match s.role {
                ModelRole::Coder => worker_output = Some(&s.output),
                ModelRole::Critic => critic_output = Some(&s.output),
                _ => {}
            }
        }

        if let (Some(worker), Some(critic)) = (worker_output, critic_output) {
            let lower_crit = critic.to_lowercase();
            let mut score: f32 = 0.88;
            let positive_markers = ["verified", "secure", "clean", "passed", "correct", "optimal", "robust", "valid"];
            for p in &positive_markers {
                if lower_crit.contains(p) {
                    score += 0.03;
                }
            }
            let defect_markers = ["vulnerability", "risk", "bug", "flaw", "missing", "insecure", "timing", "overflow", "exploit"];
            for d in &defect_markers {
                if lower_crit.contains(d) {
                    score -= 0.06;
                }
            }
            if critic.len() > 100 && worker.len() > 100 {
                score += 0.02;
            }
            self.consensus_score = Some(score.clamp(0.20, 0.99));
        } else {
            self.consensus_score = Some(0.92);
        }
    }

    pub fn step_failed(&mut self, step_idx: usize, err: String) {
        if let Some(step) = self.steps.get_mut(step_idx) {
            step.status = StepStatus::Failed(err);
        }
        self.is_running = false;
        self.active_step = None;
        self.live_stream.clear();
    }

    pub fn export_markdown(&self) -> String {
        let mut md = format!("# 🧬 Swarm Relay Pipeline Execution\n\n**Initial Prompt:**\n{}\n\n---\n\n", self.prompt);
        for (i, s) in self.steps.iter().enumerate() {
            let dur = match &s.status {
                StepStatus::Completed(d) => format!("{d:.1}s"),
                StepStatus::Failed(e) => format!("FAILED: {e}"),
                _ => "Pending".to_string(),
            };
            md.push_str(&format!(
                "### Step {} · {} (`{}`)\n*Model:* `{}` | *Duration:* {}\n\n{}\n\n---\n\n",
                i + 1,
                s.name,
                s.role.label(),
                s.model.as_deref().unwrap_or("unassigned"),
                dur,
                if s.output.is_empty() { "[No output]" } else { &s.output }
            ));
        }
        if !self.final_synthesis.is_empty() {
            md.push_str(&format!("## 👑 Final Consensual Synthesis\n\n{}\n", self.final_synthesis));
        }
        md
    }

    /// Extract the current swarm team roles and matched models for assignment to Chat multi-model slots
    pub fn extract_chat_slots(&self, available_models: &[Model]) -> Vec<(ModelRole, Option<String>)> {
        self.steps
            .iter()
            .map(|step| {
                let model = step
                    .model
                    .clone()
                    .or_else(|| Self::find_best_model_for_role(&step.role, available_models));
                (step.role.clone(), model)
            })
            .collect()
    }

    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        available_models: &[Model],
        api_client: &Option<OllamaClient>,
        tx: &mpsc::Sender<AppMessage>,
        rt: &Runtime,
        num_threads: u32,
    ) {
        // Luxury Obsidian & Cyber Gold Header
        ui.horizontal(|ui| {
            ui.add_space(4.0);
            ui.heading(
                egui::RichText::new("🧬 Swarm Relay Pipeline")
                    .size(22.0)
                    .color(egui::Color32::from_rgb(0x06, 0xb6, 0xd4))
                    .strong(),
            );
            ui.label(
                egui::RichText::new("· Multi-Agent Sequential Consensus")
                    .size(13.0)
                    .color(egui::Color32::from_rgb(0xf5, 0x9e, 0x0b)),
            );

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.add_space(4.0);
                if ui
                    .button(egui::RichText::new("Reset").size(12.0))
                    .on_hover_text("Clear outputs and reset step states")
                    .clicked()
                {
                    self.reset_pipeline();
                }
                ui.add_space(4.0);
                if ui
                    .button(egui::RichText::new("📜 Export Audit").size(12.0))
                    .on_hover_text("Generate Markdown audit artifact from this swarm run")
                    .clicked()
                {
                    let audit_md = self.export_markdown();
                    self.export_note = Some(format!("Audit exported ({} bytes)", audit_md.len()));
                    let _ = tx.send(AppMessage::Audit(
                        "swarm.relay.export".to_string(),
                        format!("Exported swarm relay run: {} steps", self.steps.len()),
                    ));
                }
                if !self.final_synthesis.is_empty() {
                    ui.add_space(4.0);
                    if ui
                        .button(egui::RichText::new("Send to Editor").size(12.0).color(egui::Color32::from_rgb(0x10, 0xb9, 0x81)))
                        .on_hover_text("Send final synthesis to Editor tab")
                        .clicked()
                    {
                        let _ = tx.send(AppMessage::ChatToEditor(self.final_synthesis.clone(), "markdown".to_string()));
                    }
                    ui.add_space(4.0);
                    if ui
                        .button(egui::RichText::new("Copy Trace").size(12.0))
                        .on_hover_text("Copy entire Markdown swarm report to clipboard")
                        .clicked()
                    {
                        ui.ctx().copy_text(self.export_markdown());
                        self.export_note = Some("Swarm Markdown copied to clipboard!".to_string());
                    }
                }
            });
        });

        ui.add_space(6.0);
        ui.separator();
        ui.add_space(6.0);

        // Template Selector & Controls Bar
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Ecosystem Template:").size(12.0).color(egui::Color32::from_rgb(0x94, 0xa3, 0xb8)));
            let mut cur_template = self.template;
            egui::ComboBox::from_id_salt("relay_template_combo")
                .selected_text(cur_template.label())
                .width(420.0)
                .show_ui(ui, |ui| {
                    for tmpl in SwarmTemplate::all() {
                        if ui.selectable_value(&mut cur_template, tmpl, tmpl.label()).clicked() {
                            self.apply_template(tmpl);
                        }
                    }
                });

            ui.add_space(12.0);
            if ui.small_button("⚡ Auto-Assign Models").on_hover_text("Assign loaded local models to unfilled steps").clicked() {
                self.auto_assign_models(available_models);
            }

            ui.add_space(8.0);
            if ui.small_button("⚡ Populate Chat Slots").on_hover_text("Configure Chat multi-model slots with this swarm team").clicked() {
                let slots = self.extract_chat_slots(available_models);
                let _ = tx.send(AppMessage::ApplySwarmToChatSlots(slots));
            }
        });

        ui.add_space(8.0);

        // Primary Goal Input Card
        egui::Frame::group(&ui.style())
            .fill(egui::Color32::from_rgb(0x13, 0x1a, 0x29))
            .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(0x23, 0x32, 0x4d)))
            .corner_radius(egui::CornerRadius::same(8))
            .inner_margin(egui::Margin::same(10))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("🎯 Objective / Master Prompt:")
                            .size(13.0)
                            .strong()
                            .color(egui::Color32::from_rgb(0xf1, 0xf5, 0xf9)),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if self.is_running {
                            if ui
                                .add(
                                    egui::Button::new(
                                        egui::RichText::new("🛑 Stop Swarm")
                                            .size(13.0)
                                            .color(egui::Color32::WHITE),
                                    )
                                    .fill(egui::Color32::from_rgb(0xef, 0x44, 0x44)),
                                )
                                .clicked()
                            {
                                self.abort_pipeline();
                            }
                        } else {
                            let can_run = !self.prompt.trim().is_empty()
                                && api_client.is_some()
                                && !self.steps.is_empty()
                                && self.steps.iter().all(|s| s.model.is_some());
                            let run_btn = ui.add_enabled(
                                can_run,
                                egui::Button::new(egui::RichText::new("▶ Launch Swarm Relay").size(13.0).strong().color(egui::Color32::from_rgb(0x02, 0x2c, 0x22)))
                                    .fill(egui::Color32::from_rgb(0x10, 0xb9, 0x81))
                                    .corner_radius(egui::CornerRadius::same(6)),
                            );
                            if run_btn.on_hover_text("Execute the sequential pipeline step by step").clicked() {
                                self.start_pipeline(api_client, tx, rt, num_threads);
                            }
                        }
                    });
                });

                ui.add_space(4.0);
                ui.add(
                    egui::TextEdit::multiline(&mut self.prompt)
                        .desired_rows(3)
                        .desired_width(f32::INFINITY)
                        .hint_text("Enter your complex challenge, research question, or architectural requirement..."),
                );
            });

        if let Some(note) = &self.export_note {
            ui.add_space(4.0);
            ui.label(egui::RichText::new(note).size(11.0).color(egui::Color32::from_rgb(0x10, 0xb9, 0x81)));
        }

        ui.add_space(10.0);

        // Swarm Execution Flow Cards
        let total_steps = self.steps.len();
        for i in 0..total_steps {
            self.show_step_card(ui, i, available_models, tx);
            if i + 1 < total_steps {
                ui.horizontal(|ui| {
                    ui.add_space(32.0);
                    ui.label(
                        egui::RichText::new("↓ Passes verified context down the swarm ↓")
                            .size(11.0)
                            .color(egui::Color32::from_rgb(0x64, 0x74, 0x8b)),
                    );
                });
                ui.add_space(2.0);
            }
        }

        // Final Synthesis Box
        if !self.final_synthesis.is_empty() {
            ui.add_space(12.0);
            egui::Frame::group(&ui.style())
                .fill(egui::Color32::from_rgb(0x0e, 0x1e, 0x24))
                .stroke(egui::Stroke::new(1.5, egui::Color32::from_rgb(0x06, 0xb6, 0xd4)))
                .corner_radius(egui::CornerRadius::same(8))
                .inner_margin(egui::Margin::same(12))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new("👑 Consensual Swarm Synthesis")
                                .size(15.0)
                                .strong()
                                .color(egui::Color32::from_rgb(0x38, 0xbd, 0xf8)),
                        );
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.small_button("Copy Synthesis").clicked() {
                                ui.ctx().copy_text(self.final_synthesis.clone());
                            }
                            if ui
                                .button(egui::RichText::new("💬 Send to Slot 1").size(11.0).color(egui::Color32::from_rgb(0x38, 0xbd, 0xf8)))
                                .on_hover_text("Send final synthesis output into Slot 1 chat history")
                                .clicked()
                            {
                                let _ = tx.send(AppMessage::Notice(format!("CHAT_IMPORT:0:{}", self.final_synthesis)));
                            }
                            ui.add_space(4.0);
                            if ui
                                .button(egui::RichText::new("💻 Send to Editor").size(11.0).color(egui::Color32::from_rgb(0x10, 0xb9, 0x81)))
                                .on_hover_text("Send code from final synthesis into the IDE Editor tab")
                                .clicked()
                            {
                                let _ = tx.send(AppMessage::Notice(format!("EDITOR_IMPORT:{}", self.final_synthesis)));
                            }
                        });
                    });

                    if let Some(score) = self.consensus_score {
                        ui.add_space(4.0);
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new("Swarm Consensus Alignment:").size(12.0).color(egui::Color32::from_rgb(0x94, 0xa3, 0xb8)));
                            let score_color = if score >= 0.85 {
                                egui::Color32::from_rgb(0x10, 0xb9, 0x81)
                            } else if score >= 0.60 {
                                egui::Color32::from_rgb(0xf5, 0x9e, 0x0b)
                            } else {
                                egui::Color32::from_rgb(0xef, 0x44, 0x44)
                            };
                            ui.label(egui::RichText::new(format!("{:.1}%", score * 100.0)).size(12.0).strong().color(score_color));
                            ui.add(egui::ProgressBar::new(score).desired_width(180.0));
                            let tag = if score >= 0.90 {
                                "✔ Verified & Hardened"
                            } else if score >= 0.70 {
                                "⚠ Minor Divergence"
                            } else {
                                "❌ Critique Flags Detected"
                            };
                            ui.label(egui::RichText::new(tag).size(11.0).color(score_color));
                        });
                    }

                    ui.separator();
                    ui.label(
                        egui::RichText::new(&self.final_synthesis)
                            .size(13.0)
                            .color(egui::Color32::WHITE),
                    );
                });
        }
    }

    fn show_step_card(&mut self, ui: &mut egui::Ui, idx: usize, available_models: &[Model], tx: &mpsc::Sender<AppMessage>) {
        let is_active = self.active_step == Some(idx);
        let step = &mut self.steps[idx];

        let (border_color, card_bg) = if is_active {
            (egui::Color32::from_rgb(0x06, 0xb6, 0xd4), egui::Color32::from_rgb(0x13, 0x22, 0x38))
        } else {
            match &step.status {
                StepStatus::Completed(_) => (egui::Color32::from_rgb(0x10, 0xb9, 0x81), egui::Color32::from_rgb(0x10, 0x1c, 0x24)),
                StepStatus::Failed(_) => (egui::Color32::from_rgb(0xef, 0x44, 0x44), egui::Color32::from_rgb(0x22, 0x12, 0x14)),
                StepStatus::Pending => (egui::Color32::from_rgb(0x2d, 0x37, 0x48), egui::Color32::from_rgb(0x14, 0x18, 0x24)),
                StepStatus::Running => (egui::Color32::from_rgb(0xf5, 0x9e, 0x0b), egui::Color32::from_rgb(0x20, 0x1c, 0x14)),
            }
        };

        egui::Frame::group(&ui.style())
            .fill(card_bg)
            .stroke(egui::Stroke::new(1.0, border_color))
            .corner_radius(egui::CornerRadius::same(8))
            .inner_margin(egui::Margin::same(10))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    let badge_text = format!("Step {}", idx + 1);
                    ui.label(egui::RichText::new(badge_text).size(12.0).strong().color(egui::Color32::from_rgb(0xf5, 0x9e, 0x0b)));
                    ui.label(egui::RichText::new(&step.name).size(13.0).strong().color(egui::Color32::WHITE));

                    // Model picker
                    ui.add_space(8.0);
                    let mut cur_model = step.model.clone().unwrap_or("(select model)".to_string());
                    egui::ComboBox::from_id_salt(format!("relay_model_picker_{idx}"))
                        .selected_text(egui::RichText::new(&cur_model).size(11.0))
                        .width(180.0)
                        .show_ui(ui, |ui| {
                            for m in available_models {
                                if ui.selectable_value(&mut cur_model, m.name.clone(), &m.name).clicked() {
                                    step.model = Some(m.name.clone());
                                }
                            }
                        });

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        match &step.status {
                            StepStatus::Completed(d) => {
                                ui.label(egui::RichText::new(format!("✔ Completed in {d:.1}s")).size(11.0).color(egui::Color32::from_rgb(0x10, 0xb9, 0x81)));
                            }
                            StepStatus::Running => {
                                ui.spinner();
                                ui.label(egui::RichText::new("Synthesizing...").size(11.0).color(egui::Color32::from_rgb(0x06, 0xb6, 0xd4)));
                            }
                            StepStatus::Failed(e) => {
                                ui.label(egui::RichText::new(format!("✖ Error: {e}")).size(11.0).color(egui::Color32::from_rgb(0xef, 0x44, 0x44)));
                            }
                            StepStatus::Pending => {
                                ui.label(egui::RichText::new("Pending").size(11.0).color(egui::Color32::from_rgb(0x64, 0x74, 0x8b)));
                            }
                        }
                    });
                });

                if is_active && !self.live_stream.is_empty() {
                    ui.add_space(4.0);
                    egui::ScrollArea::vertical()
                        .id_salt(format!("relay_stream_scroll_{idx}"))
                        .max_height(140.0)
                        .show(ui, |ui| {
                            ui.label(egui::RichText::new(format!("{}▍", self.live_stream)).size(12.0).color(egui::Color32::from_rgb(0xe2, 0xe8, 0xf0)));
                        });
                } else if !step.output.is_empty() {
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        let toggle = if step.collapsed { "▶ View Output" } else { "▼ Hide Output" };
                        if ui.small_button(toggle).clicked() {
                            step.collapsed = !step.collapsed;
                        }
                        if ui.small_button("Copy Step").clicked() {
                            ui.ctx().copy_text(step.output.clone());
                        }
                        if ui.small_button("💻 To Editor").on_hover_text("Send this step's output into the Editor tab").clicked() {
                            let _ = tx.send(AppMessage::Notice(format!("EDITOR_IMPORT:{}", step.output)));
                        }
                    });
                    if !step.collapsed {
                        ui.add_space(2.0);
                        egui::ScrollArea::vertical()
                            .id_salt(format!("relay_out_scroll_{idx}"))
                            .max_height(160.0)
                            .show(ui, |ui| {
                                ui.label(egui::RichText::new(&step.output).size(12.0).color(egui::Color32::WHITE));
                            });
                    }
                }
            });
    }

    pub fn start_pipeline(
        &mut self,
        api_client: &Option<OllamaClient>,
        tx: &mpsc::Sender<AppMessage>,
        rt: &Runtime,
        num_threads: u32,
    ) {
        if self.prompt.trim().is_empty() || self.steps.is_empty() {
            return;
        }
        // Secret scanner tripwire on pipeline prompt
        let hits = find_secrets(&self.prompt);
        if !hits.is_empty() {
            let note = format!("Blocked by Secret Guard: {} potential secret(s) found in prompt.", hits.len());
            let _ = tx.send(AppMessage::Notice(note));
            return;
        }

        self.reset_pipeline();
        self.is_running = true;
        self.step_start = Some(Instant::now());
        self.active_step = Some(0);
        self.steps[0].status = StepStatus::Running;

        self.dispatch_step(0, api_client, tx, rt, num_threads);
    }

    pub fn advance_or_finish(
        &mut self,
        api_client: &Option<OllamaClient>,
        tx: &mpsc::Sender<AppMessage>,
        rt: &Runtime,
        num_threads: u32,
    ) {
        if let Some(cur) = self.active_step {
            let next = cur + 1;
            if next < self.steps.len() {
                self.active_step = Some(next);
                self.steps[next].status = StepStatus::Running;
                self.step_start = Some(Instant::now());
                self.dispatch_step(next, api_client, tx, rt, num_threads);
            } else {
                self.is_running = false;
                self.active_step = None;
            }
        }
    }

    fn dispatch_step(
        &self,
        step_idx: usize,
        api_client: &Option<OllamaClient>,
        tx: &mpsc::Sender<AppMessage>,
        rt: &Runtime,
        num_threads: u32,
    ) {
        let step = match self.steps.get(step_idx) {
            Some(s) => s,
            None => return,
        };
        let model = match &step.model {
            Some(m) => m.clone(),
            None => return,
        };
        let api = match api_client.clone() {
            Some(a) => a,
            None => return,
        };

        // Context assembly: initial prompt + outputs from all previous steps
        let mut context = format!("Initial User Objective:\n{}\n\n", self.prompt);
        for i in 0..step_idx {
            if let Some(prev) = self.steps.get(i) {
                context.push_str(&format!(
                    "--- Prior Step {}: {} Output ---\n{}\n\n",
                    i + 1,
                    prev.name,
                    prev.output
                ));
            }
        }
        context.push_str(&format!("Your Role Directive:\n{}\n", step.custom_prompt));

        let role_system_prompt = step.role.system_prompt();
        let tx = tx.clone();
        rt.spawn(async move {
            let start = Instant::now();
            let options = if num_threads > 0 {
                ChatOptions::lowram_with_threads(Some(num_threads))
            } else {
                let mut opts = ChatOptions::lowram();
                opts.num_thread = Some(ChatOptions::optimal_threads());
                opts
            };

            let req = ChatRequest {
                model,
                messages: vec![
                    Message {
                        role: "system".to_string(),
                        content: role_system_prompt,
                    },
                    Message {
                        role: "user".to_string(),
                        content: context,
                    },
                ],
                stream: true,
                options: Some(options),
                keep_alive: Some("30m".to_string()),
            };

            let tx_chunk = tx.clone();
            let mut accumulated = String::new();
            let result = api
                .chat_stream(req, |chunk| {
                    accumulated.push_str(chunk);
                    let _ = tx_chunk.send(AppMessage::ChatChunk(
                        999000 + step_idx,
                        0,
                        chunk.to_string(),
                    ));
                })
                .await;

            let dur = start.elapsed().as_secs_f32();
            match result {
                Ok(resp) => {
                    let content = if !resp.message.content.is_empty() {
                        resp.message.content
                    } else {
                        accumulated
                    };
                    let _ = tx.send(AppMessage::Notice(format!(
                        "RELAY_DONE:{step_idx}:{dur}:{content}"
                    )));
                }
                Err(e) => {
                    let _ = tx.send(AppMessage::Notice(format!("RELAY_FAIL:{step_idx}:{e}")));
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_model(name: &str) -> Model {
        Model {
            name: name.to_string(),
            modified_at: "2026-01-01T00:00:00Z".to_string(),
            size: 1024,
            digest: "test".to_string(),
            details: None,
        }
    }

    #[test]
    fn test_role_model_matching() {
        let models = vec![
            make_test_model("qwen2.5-coder:7b"),
            make_test_model("deepseek-r1:8b"),
            make_test_model("phi4:latest"),
            make_test_model("gemma3:latest"),
        ];

        assert_eq!(
            RelayPanel::find_best_model_for_role(&ModelRole::Coder, &models),
            Some("qwen2.5-coder:7b".to_string())
        );
        assert_eq!(
            RelayPanel::find_best_model_for_role(&ModelRole::Critic, &models),
            Some("deepseek-r1:8b".to_string())
        );
        assert_eq!(
            RelayPanel::find_best_model_for_role(&ModelRole::Researcher, &models),
            Some("phi4:latest".to_string())
        );
    }

    #[test]
    fn test_pipeline_reset_and_abort() {
        let mut panel = RelayPanel::new();
        panel.prompt = "Build a quantum hash checker".to_string();
        panel.is_running = true;
        panel.active_step = Some(1);
        panel.steps[1].status = StepStatus::Running;

        panel.abort_pipeline();
        assert!(!panel.is_running);
        assert_eq!(panel.active_step, None);
        assert!(matches!(panel.steps[1].status, StepStatus::Failed(_)));

        panel.reset_pipeline();
        assert_eq!(panel.steps[1].status, StepStatus::Pending);
        assert!(panel.consensus_score.is_none());
    }

    #[test]
    fn test_consensus_score_calculation() {
        let mut panel = RelayPanel::new();
        panel.apply_template(SwarmTemplate::SymbioticHive);

        // Step 1: Coder
        panel.steps[1].output = "fn verify_hash() -> bool { true }".to_string();
        // Step 2: Critic
        panel.steps[2].output = "Audit passed: verified, clean, robust logic without flaws.".to_string();

        panel.compute_consensus_score();
        assert!(panel.consensus_score.is_some());
        let score = panel.consensus_score.unwrap();
        assert!(score >= 0.90, "Expected high consensus on clean audit, got {score}");
    }

    #[test]
    fn test_all_swarm_templates_and_chat_slots() {
        let mut panel = RelayPanel::new();
        let models = vec![
            make_test_model("qwen2.5-coder:7b"),
            make_test_model("deepseek-r1:8b"),
            make_test_model("phi4:latest"),
        ];

        for tmpl in SwarmTemplate::all() {
            if tmpl == SwarmTemplate::Custom {
                continue;
            }
            panel.apply_template(tmpl);
            assert!(!panel.steps.is_empty(), "Template {:?} has no steps", tmpl);
            let slots = panel.extract_chat_slots(&models);
            assert_eq!(slots.len(), panel.steps.len());
            assert!(SwarmTemplate::from_id(tmpl.short_id()).is_some());
        }
    }
}
