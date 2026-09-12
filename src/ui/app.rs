// Copyright 2026 Sean M. Stow. All rights reserved.
use anyhow::Result;
use chrono::Utc;
use eframe::egui;
use std::collections::HashMap;
use std::sync::mpsc;
use std::time::Instant;
use tokio::runtime::Runtime;
use uuid::Uuid;
use ndarray::Array1;
use serde::{Deserialize, Serialize};

use crate::neural::ModelProfileNetwork;
use crate::ollama::api::{ChatResponse, Model, OllamaClient};
use crate::ollama::cli::OllamaCli;
use crate::resources::{self, ESTIMATED_MODEL_BYTES, ResourceGuard, ResourceReport};
use crate::voice::VoiceEngine;
use crate::storage::{AppSettings, ChatSession, SlotConfig, Storage, Theme};
use crate::ui::chat::ChatPanel;
use crate::ui::editor::EditorPanel;
use crate::ui::history::HistoryPanel;
use crate::ui::models::ModelsPanel;
use crate::ui::neural_viz::NeuralVizPanel;
use crate::ui::relay::RelayPanel;
use crate::ui::settings::SettingsPanel;
use crate::ui::skills::SkillsPanel;
use crate::ui::tools_panel::ToolsPanel;
use crate::ui::learning::LearningPanel;

/// Messages sent from background tasks / panels to the app.
pub enum AppMessage {
    ChatResponse(usize, u64, Result<ChatResponse>),
    EditorSuggestion(usize, Result<ChatResponse>),
    /// Code handed from a chat bubble to the Editor tab (code, lang tag).
    ChatToEditor(String, String),
    EditorToChat(String),
    ModelsLoaded(Result<Vec<Model>>),
    OllamaVersion(String),
    RefreshModels,
    ModelSelected(String),
    ModelPulled(Result<String>),
    ModelDeleted(Result<String>),
    NewChat,
    LoadSession(ChatSession),
    VoiceToggled(bool),
    VoiceListen(usize),
    VoiceState { enabled: bool, note: String },
    VoiceInput(usize, String),
    ChatChunk(usize, u64, String),
    EditorChunk(usize, String),
    Audit(String, String),
    Broadcast(String),
    PullProgress(String),
    SpeakText(String),
    StopSpeak,
    StreamHandle(usize, tokio::task::JoinHandle<()>),
    StopStream(usize),
    StopAll,
    Notice(String),
    LaunchSwarmTask(String),
    SkillsCommand(usize, crate::commands::SkillsCommand),
    ShowAudit(usize),
    SetThreads(u32),
    ShowStatus(usize),
    RunSkill { skill_name: String, target_slot: Option<usize> },
    ReinforceSkill { skill_id: Uuid, reward_delta: f32 },
    TerminalRun(String),
    TerminalFinished(crate::workspace::CommandResult),
    WorkspaceMkdir(String),
    WorkspaceTouch(String),
    WorkspaceRm(String),
    WorkspaceMv { src: String, dst: String },
    WorkspaceLs(Option<String>),
    WorkspaceProject(crate::commands::ProjectCommand),
    ToolsCommand(usize, crate::commands::ToolsCommand),
    GuardrailCommand(usize, crate::commands::GuardrailCommand),
    RunTool { id: String, args: Option<String> },
    ApplySwarmToChatSlots(Vec<(ModelRole, Option<String>)>),
    LaunchSwarmPreset {
        template: crate::ui::relay::SwarmTemplate,
        prompt: Option<String>,
    },
    LaunchSwarmDag {
        preset: Option<crate::swarm::DagPreset>,
        prompt: Option<String>,
    },
    ShowBlackboard,
    AbortSwarm,
    #[allow(dead_code)]
    SetGuardrailTier(crate::guardrails::GuardrailTier),
    #[allow(dead_code)]
    DeployApp { template: crate::workspace::AppTemplateType, name: String },
    #[allow(dead_code)]
    DeployMultiFile(String),
}

/// Role assigned to a model slot. Prepended as a system prompt to every chat.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelRole {
    General,
    Coder,
    Researcher,
    Critic,
    Planner,
    Writer,
    Custom(String),
}

impl ModelRole {
    /// Fixed roles shown in the picker (Custom is handled separately).
    pub fn all_fixed() -> Vec<ModelRole> {
        vec![
            ModelRole::General,
            ModelRole::Coder,
            ModelRole::Researcher,
            ModelRole::Critic,
            ModelRole::Planner,
            ModelRole::Writer,
        ]
    }

    pub fn label(&self) -> String {
        match self {
            ModelRole::General => "General".to_string(),
            ModelRole::Coder => "Coder".to_string(),
            ModelRole::Researcher => "Researcher".to_string(),
            ModelRole::Critic => "Critic".to_string(),
            ModelRole::Planner => "Planner".to_string(),
            ModelRole::Writer => "Writer".to_string(),
            ModelRole::Custom(s) if s.trim().is_empty() => "Custom".to_string(),
            ModelRole::Custom(s) => format!("Custom: {}", s),
        }
    }

    pub fn system_prompt(&self) -> String {
        match self {
            ModelRole::General => "You are a helpful AI assistant.".to_string(),
            ModelRole::Coder => "You are an expert software engineer. Answer with correct, idiomatic code and brief explanations.".to_string(),
            ModelRole::Researcher => "You are a careful research assistant. Reason step by step and distinguish fact from speculation.".to_string(),
            ModelRole::Critic => "You are a rigorous critic. Point out flaws, risks, and better alternatives.".to_string(),
            ModelRole::Planner => "You are a planning assistant. Break goals into concrete, ordered steps.".to_string(),
            ModelRole::Writer => "You are a skilled writing assistant. Produce clear, well-structured prose.".to_string(),
            ModelRole::Custom(s) => s.clone(),
        }
    }
}

/// Shared identity + memory + slot role + guardrail directive, sent as the system prompt.
fn compose_system_prompt(
    persona: &str,
    memory: &str,
    slot: &ModelSlot,
    guardrail_tier: crate::guardrails::GuardrailTier,
) -> String {
    let mut parts: Vec<String> = Vec::new();
    let directive = guardrail_tier.system_prompt_directive();
    if !directive.trim().is_empty() {
        parts.push(directive.trim().to_string());
    }
    if !persona.trim().is_empty() {
        parts.push(persona.trim().to_string());
    }
    if !memory.trim().is_empty() {
        parts.push(format!("Remembered facts:\n{}", memory.trim()));
    }
    parts.push(slot.role_prompt());
    parts.join("\n\n")
}

/// One runnable model slot: a model assignment + a role + its own chat history.
pub struct ModelSlot {
    pub id: usize,
    pub model: Option<String>,
    pub role: ModelRole,
    pub custom_role: String,
    pub chat: ChatPanel,
    pub session_id: Uuid,
    pub session_created: chrono::DateTime<chrono::Utc>,
    pub last_saved_revision: u64,
}

impl ModelSlot {
    pub fn new(id: usize, role: ModelRole) -> Self {
        Self {
            id,
            model: None,
            role,
            custom_role: String::new(),
            chat: ChatPanel::new(),
            session_id: Uuid::new_v4(),
            session_created: Utc::now(),
            last_saved_revision: 0,
        }
    }

    /// Effective system prompt for this slot.
    pub fn role_prompt(&self) -> String {
        match &self.role {
            ModelRole::Custom(_) if !self.custom_role.trim().is_empty() => {
                self.custom_role.clone()
            }
            _ => self.role.system_prompt(),
        }
    }
}

pub fn apply_luxury_visuals(ctx: &egui::Context, theme: &Theme) {
    match theme {
        Theme::Light => {
            ctx.set_visuals(egui::Visuals::light());
        }
        _ => {
            let mut visuals = egui::Visuals::dark();
            visuals.dark_mode = true;
            visuals.panel_fill = egui::Color32::from_rgb(0x0a, 0x0e, 0x17);
            visuals.window_fill = egui::Color32::from_rgb(0x0e, 0x14, 0x22);
            visuals.extreme_bg_color = egui::Color32::from_rgb(0x1a, 0x25, 0x3d);
            visuals.faint_bg_color = egui::Color32::from_rgb(0x13, 0x1a, 0x2b);
            visuals.code_bg_color = egui::Color32::from_rgb(0x07, 0x0a, 0x12);
            visuals.override_text_color = Some(egui::Color32::from_rgb(0xf1, 0xf5, 0xf9));

            visuals.widgets.noninteractive.bg_fill = egui::Color32::from_rgb(0x11, 0x18, 0x27);
            visuals.widgets.noninteractive.bg_stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(0x1e, 0x29, 0x3b));
            visuals.widgets.noninteractive.corner_radius = egui::CornerRadius::same(6);

            visuals.widgets.inactive.bg_fill = egui::Color32::from_rgb(0x16, 0x1f, 0x33);
            visuals.widgets.inactive.bg_stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(0x24, 0x32, 0x4f));
            visuals.widgets.inactive.corner_radius = egui::CornerRadius::same(6);
            visuals.widgets.inactive.fg_stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(0xe2, 0xe8, 0xf0));

            visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(0x1f, 0x2c, 0x47);
            visuals.widgets.hovered.bg_stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(0x06, 0xb6, 0xd4));
            visuals.widgets.hovered.corner_radius = egui::CornerRadius::same(6);
            visuals.widgets.hovered.fg_stroke = egui::Stroke::new(1.0, egui::Color32::WHITE);

            visuals.widgets.active.bg_fill = egui::Color32::from_rgb(0x28, 0x38, 0x5a);
            visuals.widgets.active.bg_stroke = egui::Stroke::new(1.5, egui::Color32::from_rgb(0xf5, 0x9e, 0x0b));
            visuals.widgets.active.corner_radius = egui::CornerRadius::same(6);
            visuals.widgets.active.fg_stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(0xf5, 0x9e, 0x0b));

            visuals.widgets.open.bg_fill = egui::Color32::from_rgb(0x13, 0x1a, 0x2b);
            visuals.widgets.open.bg_stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(0x06, 0xb6, 0xd4));
            visuals.widgets.open.corner_radius = egui::CornerRadius::same(6);

            visuals.selection.bg_fill = egui::Color32::from_rgb(0x02, 0x84, 0xc7);
            visuals.selection.stroke = egui::Stroke::new(1.0, egui::Color32::WHITE);

            ctx.set_visuals(visuals);
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Tab {
    Chat,
    Relay,
    Compare,
    Editor,
    Models,
    History,
    Neural,
    Learning,
    Skills,
    Tools,
    Settings,
}

impl Tab {
    pub fn label(self) -> &'static str {
        match self {
            Tab::Chat => "Chat",
            Tab::Relay => "Swarm Relay",
            Tab::Compare => "Compare",
            Tab::Editor => "Editor",
            Tab::Models => "Models",
            Tab::History => "History",
            Tab::Neural => "Neural",
            Tab::Learning => "Learning",
            Tab::Skills => "Skills",
            Tab::Tools => "Tools",
            Tab::Settings => "Settings",
        }
    }

    pub fn icon(self) -> &'static str {
        match self {
            Tab::Chat => "💬",
            Tab::Relay => "🧬",
            Tab::Compare => "⚖",
            Tab::Editor => "📝",
            Tab::Models => "🤖",
            Tab::History => "📜",
            Tab::Neural => "🧠",
            Tab::Learning => "📈",
            Tab::Skills => "⚡",
            Tab::Tools => "🛠",
            Tab::Settings => "⚙",
        }
    }

    pub fn from_label(label: &str) -> Option<Tab> {
        match label.trim() {
            "Chat" => Some(Tab::Chat),
            "Swarm Relay" | "Relay" => Some(Tab::Relay),
            "Compare" => Some(Tab::Compare),
            "Editor" => Some(Tab::Editor),
            "Models" => Some(Tab::Models),
            "History" => Some(Tab::History),
            "Neural" => Some(Tab::Neural),
            "Learning" | "Learning Algorithm" => Some(Tab::Learning),
            "Skills" => Some(Tab::Skills),
            "Tools" => Some(Tab::Tools),
            "Settings" => Some(Tab::Settings),
            _ => None,
        }
    }

    pub fn is_closable(self) -> bool {
        !matches!(self, Tab::Chat | Tab::Settings)
    }

    pub fn all() -> Vec<Tab> {
        vec![
            Tab::Chat,
            Tab::Relay,
            Tab::Compare,
            Tab::Editor,
            Tab::Models,
            Tab::History,
            Tab::Neural,
            Tab::Learning,
            Tab::Skills,
            Tab::Tools,
            Tab::Settings,
        ]
    }
}

pub struct AiDashboardApp {
    rt: Runtime,
    tx: mpsc::Sender<AppMessage>,
    rx: mpsc::Receiver<AppMessage>,
    api_client: Option<OllamaClient>,
    cli_client: Option<OllamaCli>,
    voice: Option<VoiceEngine>,
    inflight: HashMap<usize, tokio::task::JoinHandle<()>>,
    last_ollama_url: String,
    last_allow_remote: bool,
    linked: bool,
    ollama_version: Option<String>,
    frame_ms: f32,
    clear_chats_armed: bool,
    last_theme: Theme,
    models: Vec<Model>,
    models_loading: bool,
    status: String,
    slots: Vec<ModelSlot>,
    show_model_slots: bool,
    pub split_chat_view: bool,
    pub dual_run_mode: bool,
    next_slot_id: usize,
    focused_slot: usize,
    tab: Tab,
    editor: EditorPanel,
    relay: RelayPanel,
    models_panel: ModelsPanel,
    history: HistoryPanel,
    skills_panel: SkillsPanel,
    tools_panel: ToolsPanel,
    settings_panel: SettingsPanel,
    learning_panel: LearningPanel,
    settings: AppSettings,
    storage: Option<Storage>,
    neural_panel: NeuralVizPanel,
    network: ModelProfileNetwork,
    mem: resources::MemoryStats,
    mem_checked: Instant,
    report: ResourceReport,
}

impl AiDashboardApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let storage = Storage::new().ok();
        let mut settings = storage
            .as_ref()
            .and_then(|s| s.load_settings().ok())
            .unwrap_or_default();
        if !settings.visible_tabs.iter().any(|v| v == "Learning") {
            settings.visible_tabs.push("Learning".to_string());
        }
        apply_luxury_visuals(&cc.egui_ctx, &settings.theme);
        let rt = Runtime::new().expect("tokio runtime");
        let (tx, rx) = mpsc::channel();
        let api_client =
            OllamaClient::new(settings.ollama_url.clone(), settings.allow_remote).ok();
        let cli_client = OllamaCli::new().ok();
        let mem = resources::system_memory();
        let report = ResourceGuard::evaluate(&mem, &[]);
        let mut app = Self {
            rt,
            tx,
            rx,
            api_client,
            cli_client,
            voice: None,
            inflight: HashMap::new(),
            last_ollama_url: settings.ollama_url.clone(),
            last_allow_remote: settings.allow_remote,
            linked: false,
            ollama_version: None,
            frame_ms: 0.0,
            clear_chats_armed: false,
            last_theme: settings.theme.clone(),
            models: Vec::new(),
            models_loading: false,
            status: "Ready".to_string(),
            slots: Self::restore_slots(&settings),
            show_model_slots: true,
            split_chat_view: settings.chat_split_view_default,
            dual_run_mode: true,
            next_slot_id: settings.slot_layout.len().max(2),
            focused_slot: 0,
            tab: Tab::from_label(&settings.default_tab).unwrap_or(Tab::Chat),
            editor: EditorPanel::new(),
            relay: RelayPanel::new(),
            models_panel: ModelsPanel::new(),
            history: HistoryPanel::new(),
            skills_panel: SkillsPanel::new(),
            tools_panel: ToolsPanel::new(),
            settings_panel: SettingsPanel::new(),
            learning_panel: LearningPanel::new(),
            settings,
            storage,
            neural_panel: NeuralVizPanel::new(),
            network: Self::load_network(),
            mem,
            mem_checked: Instant::now(),
            report,
        };
        if let Some(ref st) = app.storage {
            if let Ok(Some(saved_dag)) = st.load_swarm_dag() {
                app.relay.dag = saved_dag;
            }
            if let Ok(Some(saved_bb)) = st.load_blackboard() {
                app.relay.blackboard = saved_bb;
            }
        }
        app.refresh_models();
        app
    }

    // ---------- resource management ----------

    fn known_sizes(&self) -> HashMap<String, u64> {
        self.models
            .iter()
            .map(|m| (m.name.clone(), m.size))
            .collect()
    }

    /// Sizes of models assigned to slots, optionally skipping one slot.
    fn current_sizes(&self, skip: Option<usize>) -> Vec<u64> {
        let known = self.known_sizes();
        self.slots
            .iter()
            .enumerate()
            .filter(|(i, s)| Some(*i) != skip && s.model.is_some())
            .map(|(_, s)| {
                ResourceGuard::size_for_model(s.model.as_deref().unwrap_or(""), &known)
            })
            .collect()
    }

    fn refresh_resources(&mut self) {
        if self.mem_checked.elapsed().as_secs() >= 1 {
            self.mem = resources::system_memory();
            self.mem_checked = Instant::now();
        }
        let sizes = self.current_sizes(None);
        self.report = ResourceGuard::evaluate(&self.mem, &sizes);
    }

    /// Assign a model to a slot. Ensures model is assigned so local inference
    /// is never hard-blocked, while providing clear defensive warnings if RAM is constrained.
    fn try_assign_model(&mut self, idx: usize, name: String) {
        if self.slots.get(idx).is_none() {
            return;
        }
        let known = self.known_sizes();
        let new_size = ResourceGuard::size_for_model(&name, &known);
        let others = self.current_sizes(Some(idx));
        let fit_res = ResourceGuard::can_fit(&self.mem, &others, new_size);
        if let Some(slot) = self.slots.get_mut(idx) {
            slot.model = Some(name.clone());
        }
        match fit_res {
            Ok(()) => {
                self.status = format!("Slot {} now runs {}", idx + 1, name);
                self.audit("model.assign", format!("slot {} -> {}", idx + 1, name));
            }
            Err(e) => {
                self.status = format!("Slot {} runs {} (Warning: {})", idx + 1, name, e);
                self.audit("model.assign_warn", format!("slot {} -> {} ({})", idx + 1, name, e));
            }
        }
        if let Some(api) = self.api_client.clone() {
            let warm_name = name;
            self.rt.spawn(async move {
                let _ = api.warm_model(&warm_name, Some("30m")).await;
            });
        }
        let sizes = self.current_sizes(None);
        self.report = ResourceGuard::evaluate(&self.mem, &sizes);
    }

    /// Add a new empty slot for multi-model workflows. Capped at 16 slots.
    fn try_add_slot(&mut self) {
        if self.slots.len() >= 16 {
            self.status = "Maximum slot count (16) reached".to_string();
            return;
        }
        let id = self.next_slot_id;
        self.next_slot_id += 1;
        self.slots.push(ModelSlot::new(id, ModelRole::General));
        self.focused_slot = self.slots.len() - 1;
        self.status = format!("Slot {} added", self.slots.len());
        self.audit("slot.add", format!("slot {}", self.slots.len()));
        let sizes = self.current_sizes(None);
        self.report = ResourceGuard::evaluate(&self.mem, &sizes);
    }

    /// Assign the first listed model to every slot that has none.
    /// Skips slots already set; reports how many changed.
    fn fill_empty_slots(&mut self) {
        let first = self.models.first().map(|m| m.name.clone());
        match first {
            Some(name) => {
                let mut filled = 0usize;
                for slot in &mut self.slots {
                    if slot.model.is_none() {
                        slot.model = Some(name.clone());
                        filled += 1;
                    }
                }
                self.status = if filled == 0 {
                    "No empty slots".to_string()
                } else {
                    format!("Filled {} empty slot{} with {}", filled, if filled == 1 { "" } else { "s" }, name)
                };
                self.audit("slot.fill", format!("{} slots -> {}", filled, name));
            }
            None => {
                self.status = "No models listed — refresh first".to_string();
            }
        }
    }

    /// Duplicate the focused slot (model, role, history) behind the same
    /// budget gate as a fresh slot. The clone gets its own session id so
    /// future saves never collide with the original.
    fn try_clone_slot(&mut self) {
        let f = self.focused_slot.min(self.slots.len().saturating_sub(1));
        let sizes = self.current_sizes(None);
        match ResourceGuard::can_fit(&self.mem, &sizes, ESTIMATED_MODEL_BYTES) {
            Ok(()) => {
                let id = self.next_slot_id;
                self.next_slot_id += 1;
                let src = &self.slots[f];
                let mut slot = ModelSlot::new(id, src.role.clone());
                slot.model = src.model.clone();
                slot.custom_role = src.custom_role.clone();
                slot.chat = src.chat.carry_messages();
                self.slots.push(slot);
                self.focused_slot = self.slots.len() - 1;
                self.status = format!("Slot {} cloned", self.slots.len());
                self.audit("slot.clone", format!("slot {} from {}", self.slots.len(), f + 1));
            }
            Err(e) => {
                self.status = format!("Cannot clone slot: {}", e);
            }
        }
    }

    // ---------- chat persistence ----------

    /// Write slots whose chat changed since the last save. Empty chats are
    /// skipped so cleared/new slots don't leave junk rows.
    fn autosave_dirty_slots(&mut self) {
        for idx in 0..self.slots.len() {
            let dirty =
                self.slots[idx].chat.revision() != self.slots[idx].last_saved_revision;
            if !dirty || self.slots[idx].chat.messages().is_empty() {
                continue;
            }
            let session = ChatSession {
                id: self.slots[idx].session_id,
                name: Self::session_name(idx, &self.slots[idx]),
                model: self.slots[idx]
                    .model
                    .clone()
                    .unwrap_or_else(|| "(no model)".to_string()),
                messages: self.slots[idx].chat.messages().to_vec(),
                created_at: self.slots[idx].session_created,
                updated_at: Utc::now(),
            };
            if let Some(st) = self.storage.as_ref() {
                if st.save_session(&session).is_ok() {
                    self.slots[idx].last_saved_revision =
                        self.slots[idx].chat.revision();
                    self.history.mark_dirty();
                }
            }
        }
    }

    /// Session title from the first user message (first 8 words, 48 chars max).
    fn session_name(idx: usize, slot: &ModelSlot) -> String {
        let first_user = slot
            .chat
            .messages()
            .iter()
            .find(|m| m.role == "user")
            .map(|m| m.content.clone())
            .unwrap_or_default();
        let title: String = first_user
            .split_whitespace()
            .take(8)
            .collect::<Vec<_>>()
            .join(" ");
        if title.is_empty() {
            return format!("Slot {} chat", idx + 1);
        }
        let mut end = title.len().min(48);
        while !title.is_char_boundary(end) {
            end -= 1;
        }
        title[..end].to_string()
    }

    // ---------- profiler network training ----------

    fn network_path() -> std::path::PathBuf {
        dirs::data_dir()
            .unwrap_or_else(|| std::path::PathBuf::from("."))
            .join("ai-dashboard")
            .join("network.bin")
    }

    /// Load the persisted profiler net; fall back to fresh on any mismatch.
    fn load_network() -> ModelProfileNetwork {
        let path = Self::network_path().to_string_lossy().to_string();
        if let Ok(net) = ModelProfileNetwork::load(&path) {
            if net.input_dim == 8 && net.hidden_dims == vec![16, 16] && net.output_dim == 4 {
                return net;
            }
        }
        ModelProfileNetwork::new(8, vec![16, 16], 4)
    }

    /// True while anything animates (stream, spinner, pull, relay, coder): deserves a fast tick.
    fn animating(&self) -> bool {
        self.models_loading
            || self.models_panel.is_busy()
            || self.slots.iter().any(|s| s.chat.is_streaming())
            || self.relay.is_running
            || self.editor.coder_is_streaming
    }

    /// Drop the trained profiler net and start fresh (weights + loss curve).
    fn reset_network(&mut self) {
        self.network = ModelProfileNetwork::new(8, vec![16, 16], 4);
        let path = Self::network_path().to_string_lossy().to_string();
        let saved = self.network.save(&path).is_ok();
        self.neural_panel.training_history.clear();
        self.neural_panel.last_loss = None;
        self.status = if saved {
            "Profiler network reset".to_string()
        } else {
            "Profiler network reset (save failed)".to_string()
        };
        self.audit("neural.reset", "fresh net".to_string());
    }

    /// Feed a completed chat turn into the profiler network as training signal.
    /// Context is an 8-dim heuristic feature vector (sizes, latency, outcome).
    fn observe_chat(
        &mut self,
        idx: usize,
        ok: bool,
        elapsed: f32,
        prompt_chars: usize,
        resp_chars: usize,
        hist_len: usize,
        role_idx: usize,
        model_name: &str,
    ) {
        let hash = model_name
            .bytes()
            .fold(0u64, |a, b| a.wrapping_mul(31).wrapping_add(b as u64));
        let ctx = Array1::from_vec(vec![
            (prompt_chars as f32 / 50_000.0).min(1.0),
            (hist_len as f32 / 20.0).min(1.0),
            if ok { 1.0 } else { 0.0 },
            (elapsed / 120.0).min(1.0),
            (resp_chars as f32 / 50_000.0).min(1.0),
            (idx as f32 / 8.0).min(1.0),
            role_idx as f32 / 6.0,
            ((hash % 100) as f32) / 100.0,
        ]);
        let action = self.network.select_model_profile(&ctx);
        let reward = if ok {
            1.0 + (1.0 - (elapsed / 120.0).min(1.0)) * 0.5
        } else {
            -0.5
        };
        self.network.add_experience(crate::neural::Experience {
            state: ctx.clone().into(),
            action,
            reward,
            next_state: ctx.clone().into(),
            done: true,
        });
        self.network.update_performance(reward);
        let (domain_idx, domain_label) = match role_idx {
            1 => (1, "Coder"),
            2 => (2, "Researcher"),
            3 => (3, "Cyber / Critic"),
            4 => (4, "Planner"),
            5 => (5, "Writer"),
            _ => (0, "General"),
        };
        self.network.swarm_pheromones.deposit(domain_idx, idx, reward);

        if ok && reward > 0.5 {
            let val = vec![
                if ok { 1.0 } else { 0.0 },
                (elapsed / 60.0).min(1.0),
                (resp_chars as f32 / 10_000.0).min(1.0),
                role_idx as f32 / 6.0,
                idx as f32 / 8.0,
                reward.min(2.0),
                0.8,
                1.0,
            ];
            let label = format!("{}: slot #{} ({} chars, {:.1}s)", domain_label, idx, resp_chars, elapsed);
            self.network.memory_network.write(ctx.as_slice().unwrap_or(&[]), &val, domain_label, &label, reward);
        }

        if self.network.optimizer_config.auto_train && self.network.experience_buffer.len() >= 4 {
            if let Ok(loss) = self.network.train_step() {
                if loss.is_finite() && loss > 0.0 {
                    if let Some(ql) = &mut self.network.quantum_layer {
                        let grad = vec![loss * 0.01; ql.num_qubits];
                        ql.update_phases(self.network.learning_rate, &grad);
                    }
                    self.neural_panel.add_training_loss(loss);
                    self.neural_panel.last_loss = Some(loss);
                    let path = Self::network_path().to_string_lossy().to_string();
                    let _ = self.network.save(&path);
                }
            }
        }
    }

    /// Append a security-activity event (best effort; never fails the action).
    fn audit(&self, kind: &str, detail: String) {
        if let Some(st) = self.storage.as_ref() {
            let _ = st.log_audit(kind, &detail);
        }
    }

    /// Abort any in-flight stream for a slot (Stop / resend / remove).
    fn abort_slot(&mut self, idx: usize) {
        if let Some(h) = self.inflight.remove(&idx) {
            h.abort();
        }
    }

    /// Rebuild slots from persisted layout (roles always; models are
    /// re-attached later once the model list arrives so the RAM guard
    /// can vet each one).
    fn restore_slots(settings: &AppSettings) -> Vec<ModelSlot> {
        if settings.slot_layout.is_empty() {
            return vec![
                ModelSlot::new(0, ModelRole::General),
                ModelSlot::new(1, ModelRole::Coder),
            ];
        }
        settings
            .slot_layout
            .iter()
            .take(32)
            .enumerate()
            .map(|(i, c)| {
                let mut s =
                    ModelSlot::new(i, Self::role_from_layout(&c.role, &c.custom_role));
                s.custom_role = c.custom_role.clone();
                s
            })
            .collect()
    }

    fn role_from_layout(role: &str, custom: &str) -> ModelRole {
        match role {
            "General" => ModelRole::General,
            "Coder" => ModelRole::Coder,
            "Researcher" => ModelRole::Researcher,
            "Critic" => ModelRole::Critic,
            "Planner" => ModelRole::Planner,
            "Writer" => ModelRole::Writer,
            r if r.starts_with("Custom") => ModelRole::Custom(custom.to_string()),
            _ => ModelRole::General,
        }
    }

    /// Persist slot model/role assignments when they change.
    fn sync_slot_layout(&mut self) {
        let changed = self.slots.len() != self.settings.slot_layout.len()
            || self.slots.iter().zip(self.settings.slot_layout.iter()).any(|(s, cfg)| {
                s.model != cfg.model || s.role.label() != cfg.role || s.custom_role != cfg.custom_role
            });

        if changed {
            let layout: Vec<SlotConfig> = self
                .slots
                .iter()
                .map(|s| SlotConfig {
                    model: s.model.clone(),
                    role: s.role.label(),
                    custom_role: s.custom_role.clone(),
                })
                .collect();
            self.settings.slot_layout = layout;
            if let Some(st) = self.storage.as_ref() {
                let _ = st.save_settings(&self.settings);
            }
        }
    }

    /// Side-by-side latest assistant reply per slot. Read-only: broadcast
    /// once in Chat, compare here.
    fn show_compare(&self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.add_space(4.0);
            ui.heading(
                egui::RichText::new("\u{2696} Compare slot answers")
                    .size(22.0)
                    .color(egui::Color32::from_rgb(0x00, 0xaa, 0xff)),
            );
        });
        ui.add_space(8.0);
        ui.separator();
        ui.add_space(8.0);
        ui.label(
            egui::RichText::new("Latest assistant reply per slot — broadcast once, read side by side.")
                .size(12.0)
                .color(egui::Color32::from_rgb(0x88, 0x88, 0x88)),
        );
        ui.add_space(8.0);
        let answered: Vec<(usize, &ModelSlot)> = self
            .slots
            .iter()
            .enumerate()
            .filter(|(_, s)| s.chat.messages().iter().any(|m| m.role == "assistant"))
            .collect();
        if answered.is_empty() {
            ui.label(
                egui::RichText::new("No answers yet — ask something in Chat (Ask all hits every slot).")
                    .size(13.0)
                    .color(egui::Color32::from_rgb(0x99, 0x99, 0x99)),
            );
            return;
        }
        egui::ScrollArea::horizontal()
            .id_salt("compare_scroll")
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    for (i, slot) in answered {
                        if let Some(last) = slot
                            .chat
                            .messages()
                            .iter()
                            .rev()
                            .find(|m| m.role == "assistant")
                        {
                            egui::Frame::group(&ui.style())
                                .fill(egui::Color32::from_rgb(0x1e, 0x1e, 0x1e))
                                .corner_radius(egui::CornerRadius::same(8))
                                .inner_margin(egui::Margin::same(8))
                                .show(ui, |ui| {
                                    ui.set_min_width(280.0);
                                    ui.set_max_width(340.0);
                                    ui.label(
                                        egui::RichText::new(format!(
                                            "Slot {} \u{00B7} {} [{}]",
                                            i + 1,
                                            slot.model
                                                .clone()
                                                .unwrap_or("(empty)".to_string()),
                                            slot.role.label()
                                        ))
                                        .size(13.0)
                                        .strong(),
                                    );
                                    ui.separator();
                                    let mut text = last.content.chars().take(4000).collect::<String>();
                                    if last.content.chars().count() > 4000 {
                                        text.push_str("\n…[truncated for compare]");
                                    }
                                    egui::ScrollArea::vertical()
                                        .id_salt(format!("compare_card_{i}"))
                                        .max_height(300.0)
                                        .show(ui, |ui| {
                                            ui.label(
                                                egui::RichText::new(text)
                                                    .size(13.0)
                                                    .color(egui::Color32::WHITE),
                                            );
                                        });
                                    ui.add_space(4.0);
                                    ui.label(
                                        egui::RichText::new(last.timestamp.format("%H:%M").to_string())
                                            .size(11.0)
                                            .color(egui::Color32::from_rgb(0xaa, 0xaa, 0xaa)),
                                    );
                                });
                        }
                    }
                });
            });
    }

    // ---------- model loading ----------

    fn refresh_models(&mut self) {
        self.models_loading = true;
        let tx = self.tx.clone();
        let api = self.api_client.clone();
        let cli = self.cli_client.clone();
        self.rt.spawn(async move {
            let version = if let Some(a) = api.clone() {
                a.version().await.ok()
            } else {
                None
            };
            let result = if let Some(api) = api {
                api.list_models().await
            } else if let Some(cli) = cli {
                cli.list_models().await.map(|cms| {
                    cms.into_iter()
                        .map(|c| Model {
                            name: c.name,
                            modified_at: c.modified,
                            size: c.size.parse().unwrap_or(0),
                            digest: c.digest,
                            details: None,
                        })
                        .collect()
                })
            } else {
                Err(anyhow::anyhow!("No Ollama service available"))
            };
            if let Some(v) = version {
                let _ = tx.send(AppMessage::OllamaVersion(v));
            }
            let _ = tx.send(AppMessage::ModelsLoaded(result));
        });
    }

    fn poll_messages(&mut self) {
        while let Ok(msg) = self.rx.try_recv() {
            match msg {
                AppMessage::Broadcast(prompt) => {
                    let trimmed = prompt.trim();
                    if trimmed.starts_with('/') {
                        if let Ok(Some(cmd)) = crate::commands::SlashCommand::parse(trimmed) {
                            match cmd {
                                crate::commands::SlashCommand::Help => {
                                    let f0 = self.focused_slot.min(self.slots.len().saturating_sub(1));
                                    if let Some(slot) = self.slots.get_mut(f0) {
                                        slot.chat.push_system_note(crate::commands::SlashCommand::help_manual());
                                    }
                                }
                                crate::commands::SlashCommand::Clear => {
                                    for slot in &mut self.slots {
                                        slot.chat.clear_chat();
                                        slot.chat.push_system_note("⚡ **Chat cleared from viewport.**");
                                    }
                                    self.status = "All active chats cleared".to_string();
                                }
                                crate::commands::SlashCommand::New => {
                                    for slot in &mut self.slots {
                                        slot.chat.clear_chat();
                                    }
                                    self.status = "Fresh session started across all slots".to_string();
                                }
                                crate::commands::SlashCommand::Model(name) => {
                                    let _ = self.tx.send(crate::ui::app::AppMessage::ModelSelected(name.clone()));
                                }
                                crate::commands::SlashCommand::Swarm(cmd) => match cmd {
                                    crate::commands::SwarmCommand::Dispatch(task) => {
                                        let _ = self.tx.send(crate::ui::app::AppMessage::LaunchSwarmTask(task));
                                    }
                                    crate::commands::SwarmCommand::Preset { template, prompt } => {
                                        if let Some(tmpl) = crate::ui::relay::SwarmTemplate::from_id(&template) {
                                            let _ = self.tx.send(crate::ui::app::AppMessage::LaunchSwarmPreset {
                                                template: tmpl,
                                                prompt,
                                            });
                                        }
                                    }
                                    crate::commands::SwarmCommand::ListPresets => {}
                                    crate::commands::SwarmCommand::Dag { preset, prompt } => {
                                        let dag_preset = preset.as_deref().and_then(crate::swarm::DagPreset::from_id);
                                        let _ = self.tx.send(crate::ui::app::AppMessage::LaunchSwarmDag {
                                            preset: dag_preset,
                                            prompt,
                                        });
                                    }
                                    crate::commands::SwarmCommand::Blackboard => {
                                        let _ = self.tx.send(crate::ui::app::AppMessage::ShowBlackboard);
                                    }
                                    crate::commands::SwarmCommand::Abort => {
                                        let _ = self.tx.send(crate::ui::app::AppMessage::AbortSwarm);
                                    }
                                }
                                crate::commands::SlashCommand::Skills(sub) => {
                                    let f0 = self.focused_slot.min(self.slots.len().saturating_sub(1));
                                    let _ = self.tx.send(crate::ui::app::AppMessage::SkillsCommand(f0, sub));
                                }
                                crate::commands::SlashCommand::Audit => {
                                    let f0 = self.focused_slot.min(self.slots.len().saturating_sub(1));
                                    let _ = self.tx.send(crate::ui::app::AppMessage::ShowAudit(f0));
                                }
                                crate::commands::SlashCommand::Threads(n) => {
                                    let _ = self.tx.send(crate::ui::app::AppMessage::SetThreads(n));
                                }
                                crate::commands::SlashCommand::Status => {
                                    let f0 = self.focused_slot.min(self.slots.len().saturating_sub(1));
                                    let _ = self.tx.send(crate::ui::app::AppMessage::ShowStatus(f0));
                                }
                                crate::commands::SlashCommand::Exec(cmd_line) => {
                                    let _ = self.tx.send(crate::ui::app::AppMessage::TerminalRun(cmd_line));
                                }
                                crate::commands::SlashCommand::Mkdir(dir_path) => {
                                    let _ = self.tx.send(crate::ui::app::AppMessage::WorkspaceMkdir(dir_path));
                                }
                                crate::commands::SlashCommand::Touch(file_path) => {
                                    let _ = self.tx.send(crate::ui::app::AppMessage::WorkspaceTouch(file_path));
                                }
                                crate::commands::SlashCommand::Rm(target) => {
                                    let _ = self.tx.send(crate::ui::app::AppMessage::WorkspaceRm(target));
                                }
                                crate::commands::SlashCommand::Mv { src, dst } => {
                                    let _ = self.tx.send(crate::ui::app::AppMessage::WorkspaceMv { src, dst });
                                }
                                crate::commands::SlashCommand::Ls(path_opt) => {
                                    let _ = self.tx.send(crate::ui::app::AppMessage::WorkspaceLs(path_opt));
                                }
                                crate::commands::SlashCommand::Project(proj) => {
                                    let _ = self.tx.send(crate::ui::app::AppMessage::WorkspaceProject(proj));
                                }
                                crate::commands::SlashCommand::Tools(sub) => {
                                    let f0 = self.focused_slot.min(self.slots.len().saturating_sub(1));
                                    let _ = self.tx.send(crate::ui::app::AppMessage::ToolsCommand(f0, sub));
                                }
                                crate::commands::SlashCommand::Guardrail(sub) => {
                                    let f0 = self.focused_slot.min(self.slots.len().saturating_sub(1));
                                    let _ = self.tx.send(crate::ui::app::AppMessage::GuardrailCommand(f0, sub));
                                }
                            }
                            continue;
                        }
                    }

                    let f0 = self.focused_slot.min(self.slots.len().saturating_sub(1));
                    if let Some(note) = self
                        .slots
                        .get_mut(f0)
                        .and_then(|s| s.chat.broadcast_check(&prompt).err())
                    {
                        if let Some(slot) = self.slots.get_mut(f0) {
                            slot.chat.push_system_note(note);
                        }
                        self.status =
                            "Ask all blocked: possible secret — resend to override".to_string();
                        self.audit("broadcast.blocked", "secret guard".to_string());
                        continue;
                    }
                    let models = self.models.clone();
                    let api = self.api_client.clone();
                    let persona = self.settings.persona.clone();
                    let memory = self.settings.memory.clone();
                    let mut sent = 0usize;
                    for idx in 0..self.slots.len() {
                        let model = self.slots[idx].model.clone();
                        let role_prompt = {
                            let s = &self.slots[idx];
                            compose_system_prompt(&persona, &memory, s, self.settings.guardrail_tier)
                        };
                        if let Some(slot) = self.slots.get_mut(idx) {
                            slot.chat.num_threads = self.settings.num_threads;
                            if slot.chat.send_prompt(
                                prompt.clone(),
                                &models,
                                &model,
                                &role_prompt,
                                idx,
                                &api,
                                &self.tx,
                                &self.rt,
                            ) {
                                sent += 1;
                            }
                        }
                    }
                    let skipped = self.slots.len().saturating_sub(sent);
                    self.status = if sent == 0 {
                        "Ask all: no slot answered (need models, idle chats)".to_string()
                    } else if skipped > 0 {
                        format!(
                            "Asked {} slot{} ({} skipped: no model or busy)",
                            sent,
                            if sent == 1 { "" } else { "s" },
                            skipped
                        )
                    } else {
                        format!("Asked {} slot{}", sent, if sent == 1 { "" } else { "s" })
                    };
                    if sent > 0 {
                        self.audit("broadcast.sent", format!("{} slots", sent));
                    }
                }
                AppMessage::OllamaVersion(v) => {
                    self.ollama_version = Some(v);
                }
                AppMessage::ChatChunk(idx, seq, piece) => {
                    if idx >= 999000 {
                        self.relay.push_chunk(idx - 999000, piece);
                    } else if let Some(slot) = self.slots.get_mut(idx) {
                        slot.chat.push_chunk(seq, &piece);
                    }
                }
                AppMessage::ChatResponse(idx, seq, res) => {
                    // Retire the handle even when stale: a stopped or
                    // superseded request must not leave a finished task behind.
                    self.inflight.remove(&idx);
                    if self.slots.get(idx).map(|s| s.chat.stream_seq()) != Some(seq) {
                        continue; // stale: stopped or superseded by a newer send
                    }
                    if idx < self.slots.len() {
                        let (ok, elapsed, resp_chars);
                        let prompt_chars: usize;
                        let hist_len: usize;
                        let role_idx: usize;
                        let model_name: String;
                        {
                            let slot = &mut self.slots[idx];
                            prompt_chars = slot
                                .chat
                                .messages()
                                .iter()
                                .rev()
                                .find(|m| m.role == "user")
                                .map(|m| m.content.len())
                                .unwrap_or(0);
                            hist_len = slot.chat.messages().len();
                            role_idx = match slot.role {
                                ModelRole::General => 0,
                                ModelRole::Coder => 1,
                                ModelRole::Researcher => 2,
                                ModelRole::Critic => 3,
                                ModelRole::Planner => 4,
                                ModelRole::Writer => 5,
                                ModelRole::Custom(_) => 6,
                            };
                            model_name = slot.model.clone().unwrap_or_default();
                            let out = slot.chat.handle_response(res);
                            ok = out.0;
                            elapsed = out.1;
                            resp_chars = out.2;
                        }
                        self.observe_chat(
                            idx,
                            ok,
                            elapsed,
                            prompt_chars,
                            resp_chars,
                            hist_len,
                            role_idx,
                            &model_name,
                        );
                        if self.settings.voice_enabled {
                            if let Some(eng) = self.voice.clone() {
                                if let Some(last) = self.slots[idx]
                                    .chat
                                    .messages()
                                    .iter()
                                    .rev()
                                    .find(|m| m.role == "assistant")
                                {
                                    let text: String =
                                        last.content.chars().take(1000).collect();
                                    self.rt.spawn(async move {
                                        let _ = eng.speak(&text).await;
                                    });
                                }
                            }
                        }
                    }
                }
                AppMessage::EditorChunk(id, piece) => {
                    self.editor.push_chunk(id, &piece);
                }
                AppMessage::EditorSuggestion(id, res) => {
                    self.editor.handle_ai_suggestion(id, res);
                }
                AppMessage::EditorToChat(code) => {
                    let f = self.focused_slot.min(self.slots.len().saturating_sub(1));
                    if let Some(slot) = self.slots.get_mut(f) {
                        slot.chat.append_input(&code);
                    }
                    self.tab = Tab::Chat;
                    self.status = format!("Editor code sent to slot {}", f + 1);
                    self.audit("editor.to_chat", format!("slot {}", f + 1));
                }
                AppMessage::ChatToEditor(code, lang) => {
                    self.editor.set_code_from_chat(code, lang);
                    self.tab = Tab::Editor;
                    self.status = "Code moved to Editor — pick a file and Save".to_string();
                }
                AppMessage::ModelsLoaded(res) => {
                    self.models_loading = false;
                    match res {
                        Ok(m) => {
                            self.linked = true;
                            self.status = format!("{} models loaded", m.len());
                            self.models = m;
                            self.relay.auto_assign_models(&self.models);
                            // First: re-attach models saved in the slot layout
                            // (each vetted by the RAM guard); then auto-fill
                            // any still-empty slots with the smallest models
                            // that fit, so Chat works immediately (Send
                            // enables as soon as a slot has a model).
                            for i in 0..self.slots.len() {
                                if self.slots[i].model.is_none() {
                                    if let Some(want) = self
                                        .settings
                                        .slot_layout
                                        .get(i)
                                        .and_then(|c| c.model.clone())
                                    {
                                        self.try_assign_model(i, want);
                                    }
                                }
                            }
                            if self.slots.iter().any(|s| s.model.is_none()) {
                                let mut by_size = self.models.clone();
                                by_size.sort_by_key(|m| m.size);
                                for i in 0..self.slots.len() {
                                    if self.slots[i].model.is_some() {
                                        continue;
                                    }
                                    for cand in &by_size {
                                        let known = self.known_sizes();
                                        let others = self.current_sizes(Some(i));
                                        let size = ResourceGuard::size_for_model(
                                            &cand.name,
                                            &known,
                                        );
                                        if ResourceGuard::can_fit(&self.mem, &others, size).is_ok() {
                                            self.try_assign_model(i, cand.name.clone());
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            self.linked = false;
                            self.status = format!("Model refresh failed: {}", e);
                        }
                    }
                }
                AppMessage::ModelSelected(name) => {
                    let f = self.focused_slot.min(self.slots.len().saturating_sub(1));
                    self.try_assign_model(f, name);
                    self.tab = Tab::Chat;
                }
                AppMessage::ModelPulled(res) => {
                    match res {
                        Ok(n) => {
                            self.status = format!("Pulled {}", n);
                            self.audit("model.pull", format!("ok {}", n));
                            self.models_panel.note_pull_finished(true);
                        }
                        Err(e) => {
                            self.status = format!("Pull failed: {}", e);
                            self.audit("model.pull_failed", format!("{}", e));
                            self.models_panel.note_pull_finished(false);
                        }
                    }
                    self.refresh_models();
                }
                AppMessage::RefreshModels => {
                    self.refresh_models();
                }
                AppMessage::ModelDeleted(res) => {
                    match res {
                        Ok(n) => {
                            self.status = format!("Deleted {}", n);
                            self.audit("model.delete", format!("{}", n));
                            for slot in &mut self.slots {
                                if slot.model.as_deref() == Some(&n) {
                                    slot.model = None;
                                }
                            }
                        }
                        Err(e) => {
                            self.status = format!("Delete failed: {}", e);
                            self.audit("model.delete_failed", format!("{}", e));
                        }
                    }
                    self.models_panel.note_transfer_finished();
                    self.refresh_models();
                }
                AppMessage::PullProgress(line) => {
                    self.models_panel.push_progress(line);
                }
                AppMessage::StreamHandle(idx, h) => {
                    if let Some(old) = self.inflight.insert(idx, h) {
                        old.abort();
                    }
                }
                AppMessage::StopStream(idx) => {
                    self.abort_slot(idx);
                    if let Some(slot) = self.slots.get_mut(idx) {
                        slot.chat.stop_stream();
                    }
                    self.status = format!("Slot {} stopped", idx + 1);
                    self.audit("chat.stop", format!("slot {}", idx + 1));
                }
                AppMessage::StopAll => {
                    for i in 0..self.slots.len() {
                        self.abort_slot(i);
                        if let Some(slot) = self.slots.get_mut(i) {
                            slot.chat.stop_stream();
                        }
                    }
                    self.status = "Stopped all slots".to_string();
                    self.audit("chat.stop_all", "all slots".to_string());
                }
                AppMessage::Notice(s) => {
                    if let Some(rest) = s.strip_prefix("RELAY_DONE:") {
                        let parts: Vec<&str> = rest.splitn(3, ':').collect();
                        if parts.len() >= 3 {
                            let step_idx: usize = parts[0].parse().unwrap_or(0);
                            let dur: f32 = parts[1].parse().unwrap_or(0.0);
                            let content = parts[2].to_string();
                            self.relay.step_completed(step_idx, content, dur);
                            if let Some(step) = self.relay.steps.get(step_idx) {
                                let domain_idx = match step.role {
                                    ModelRole::Planner => 4,
                                    ModelRole::Coder => 1,
                                    ModelRole::Critic => 3,
                                    ModelRole::Researcher => 2,
                                    ModelRole::Writer => 5,
                                    _ => 0,
                                };
                                let reward = 1.0 + (1.0 - (dur / 120.0).min(1.0)) * 0.5;
                                self.network.swarm_pheromones.deposit(domain_idx, step_idx.min(7), reward);
                            }
                            self.relay.advance_or_finish(&self.api_client, &self.tx, &self.rt, self.settings.num_threads);
                            self.status = format!("Swarm step {} completed in {:.1}s", step_idx + 1, dur);
                            self.audit("relay.step_done", format!("step {} ({:.1}s)", step_idx + 1, dur));
                        }
                    } else if let Some(rest) = s.strip_prefix("RELAY_FAIL:") {
                        let parts: Vec<&str> = rest.splitn(2, ':').collect();
                        let step_idx: usize = parts.first().and_then(|x| x.parse().ok()).unwrap_or(0);
                        let err = parts.get(1).unwrap_or(&"Unknown error").to_string();
                        self.relay.step_failed(step_idx, err.clone());
                        self.status = format!("Swarm step {} failed: {}", step_idx + 1, err);
                        self.audit("relay.step_failed", format!("step {}: {}", step_idx + 1, err));
                    } else if let Some(rest) = s.strip_prefix("CHAT_IMPORT:") {
                        let parts: Vec<&str> = rest.splitn(2, ':').collect();
                        let target_slot: usize = parts.first().and_then(|x| x.parse().ok()).unwrap_or(0);
                        let content = parts.get(1).unwrap_or(&"").to_string();
                        if let Some(slot) = self.slots.get_mut(target_slot) {
                            slot.chat.push_assistant_message(format!("**[Swarm Relay Consensus Output]**\n\n{}", content));
                            self.tab = Tab::Chat;
                            self.focused_slot = target_slot;
                            self.status = format!("Imported Swarm output into Slot {}", target_slot + 1);
                            self.audit("relay.import_chat", format!("slot {}", target_slot + 1));
                        }
                    } else if let Some(rest) = s.strip_prefix("EDITOR_IMPORT:") {
                        self.editor.load_imported_code(rest);
                        self.tab = Tab::Editor;
                        self.status = "Imported Swarm code into Editor IDE".to_string();
                        self.audit("relay.import_editor", "editor".to_string());
                    } else if let Some(rest) = s.strip_prefix("DAG_CHUNK:") {
                        let parts: Vec<&str> = rest.splitn(2, ':').collect();
                        if parts.len() == 2 {
                            if let Ok(node_id) = parts[0].parse::<usize>() {
                                self.relay.push_dag_chunk(node_id, parts[1].to_string());
                            }
                        }
                    } else if let Some(rest) = s.strip_prefix("DAG_DONE:") {
                        let parts: Vec<&str> = rest.splitn(3, ':').collect();
                        if parts.len() >= 3 {
                            let node_id: usize = parts[0].parse().unwrap_or(0);
                            let dur: f32 = parts[1].parse().unwrap_or(0.0);
                            let content = parts[2].to_string();

                            // Phase 3: Autonomous Physical Tool Execution Check
                            let (auto_tool, tool_id_opt) = self
                                .relay
                                .dag
                                .find_node(node_id)
                                .map(|n| (n.auto_exec_tool, n.tool_id.clone()))
                                .unwrap_or((false, None));

                            // Populate node output for tool examination
                            if let Some(node) = self.relay.dag.find_node_mut(node_id) {
                                node.output = content.clone();
                            }

                            let mut tool_failed = false;
                            if auto_tool && tool_id_opt.is_some() {
                                let ws_root = self.editor.workspace.root_path.to_str();
                                match self.relay.execute_node_tool(
                                    node_id,
                                    &self.tools_panel.registry,
                                    self.settings.guardrail_tier,
                                    ws_root,
                                ) {
                                    Ok(res) => {
                                        if !res.success {
                                            tool_failed = true;
                                            let err_msg = format!(
                                                "Physical tool '{}' failed (exit code {}):\nSTDERR:\n{}\nSTDOUT:\n{}",
                                                res.tool_id,
                                                res.exit_code,
                                                if res.stderr.is_empty() { "[empty]" } else { &res.stderr },
                                                if res.stdout.is_empty() { "[empty]" } else { &res.stdout },
                                            );
                                            let will_retry = self.relay.dag_node_failed(node_id, err_msg.clone(), &self.models);
                                            if let Some(st) = self.storage.as_ref() {
                                                let _ = st.save_swarm_dag(&self.relay.dag);
                                                let _ = st.save_blackboard(&self.relay.blackboard);
                                            }
                                            if will_retry {
                                                self.status = format!(
                                                    "Node #{} tool verification failed (exit {}) -> auto-retrying with compiler diagnostics",
                                                    node_id, res.exit_code
                                                );
                                                self.audit(
                                                    "swarm.dag_tool_retry",
                                                    format!("node {} tool {}: exit {}", node_id, res.tool_id, res.exit_code),
                                                );
                                                self.relay.dispatch_ready_dag_nodes(
                                                    &self.api_client,
                                                    &self.tx,
                                                    &self.rt,
                                                    self.settings.num_threads,
                                                );
                                            } else {
                                                self.status = format!(
                                                    "Node #{} tool verification failed (max retries reached): {}",
                                                    node_id, res.tool_id
                                                );
                                                self.audit(
                                                    "swarm.dag_tool_failed",
                                                    format!("node {} tool {}", node_id, res.tool_id),
                                                );
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        self.status = format!("Tool execution error on node {}: {}", node_id, e);
                                        self.audit("swarm.dag_tool_error", format!("node {}: {}", node_id, e));
                                    }
                                }
                            }

                            if !tool_failed {
                                // Complete node and deposit artifact into stigmergic blackboard
                                self.relay.dag_node_completed(node_id, content, dur);

                                // Stigmergic pheromone deposit and memory network write
                                if let Some(node) = self.relay.dag.find_node(node_id) {
                                    let (domain_idx, domain_label) = match node.role {
                                        ModelRole::Planner => (4, "Planner"),
                                        ModelRole::Coder => (1, "Coder"),
                                        ModelRole::Critic => (3, "Cyber / Critic"),
                                        ModelRole::Researcher => (2, "Researcher"),
                                        ModelRole::Writer => (5, "Writer"),
                                        _ => (0, "General"),
                                    };
                                    let reward = 1.0 + (1.0 - (dur / 120.0).min(1.0)) * 0.5;
                                    self.network.swarm_pheromones.deposit(domain_idx, node_id.min(7), reward);

                                    // Formulate probe and memory vector from completed task
                                    let probe = crate::neural::extract_probe_features(&node.directive);
                                    let val = vec![
                                        1.0,
                                        (dur / 60.0).min(1.0),
                                        (node.output.len() as f32 / 10_000.0).min(1.0),
                                        domain_idx as f32 / 6.0,
                                        (node_id as f32 / 8.0).min(1.0),
                                        reward.min(2.0),
                                        1.0,
                                        0.9,
                                    ];
                                    let label = format!("{}: DAG Node #{} '{}'", domain_label, node_id, node.name);
                                    self.network.memory_network.write(probe.as_slice().unwrap_or(&[]), &val, domain_label, &label, reward);

                                    // Add to experience buffer for neural learning
                                    self.network.add_experience(crate::neural::Experience {
                                        state: probe.clone().into(),
                                        action: domain_idx.min(self.network.output_dim.saturating_sub(1)),
                                        reward,
                                        next_state: probe.into(),
                                        done: true,
                                    });

                                    // Auto-train if enabled
                                    if self.network.optimizer_config.auto_train && self.network.experience_buffer.len() >= 4 {
                                        if let Ok(loss) = self.network.train_step() {
                                            if loss.is_finite() && loss > 0.0 {
                                                self.neural_panel.add_training_loss(loss);
                                                self.neural_panel.last_loss = Some(loss);
                                                let path = Self::network_path().to_string_lossy().to_string();
                                                let _ = self.network.save(&path);
                                            }
                                        }
                                    }
                                }

                                // Persist DAG and Blackboard encrypted at rest in local Sled vault
                                if let Some(st) = self.storage.as_ref() {
                                    let _ = st.save_swarm_dag(&self.relay.dag);
                                    let _ = st.save_blackboard(&self.relay.blackboard);
                                }

                                // Advance DAG execution by dispatching newly ready nodes
                                self.relay.dispatch_ready_dag_nodes(
                                    &self.api_client,
                                    &self.tx,
                                    &self.rt,
                                    self.settings.num_threads,
                                );

                                let node_name = self
                                    .relay
                                    .dag
                                    .find_node(node_id)
                                    .map(|n| n.name.clone())
                                    .unwrap_or_else(|| format!("Node {}", node_id));

                                if self.relay.dag.is_finished() {
                                    self.status = format!(
                                        "Swarm DAG '{}' successfully completed all nodes!",
                                        self.relay.dag.preset.label()
                                    );
                                    self.audit(
                                        "swarm.dag_finished",
                                        self.relay.dag.preset.label().to_string(),
                                    );
                                } else {
                                    self.status = format!("Swarm DAG node '{}' completed in {:.1}s", node_name, dur);
                                    self.audit("swarm.dag_node_done", format!("node {} ({:.1}s)", node_id, dur));
                                }
                            }
                        }
                    } else if let Some(rest) = s.strip_prefix("DAG_FAIL:") {
                        let parts: Vec<&str> = rest.splitn(2, ':').collect();
                        let node_id: usize = parts.first().and_then(|x| x.parse().ok()).unwrap_or(0);
                        let err = parts.get(1).unwrap_or(&"Unknown error").to_string();
                        let will_retry = self.relay.dag_node_failed(node_id, err.clone(), &self.models);
                        if let Some(st) = self.storage.as_ref() {
                            let _ = st.save_swarm_dag(&self.relay.dag);
                        }
                        if will_retry {
                            self.status = format!("Swarm DAG node {} failed (recovering via auto-retry): {}", node_id, err);
                            self.audit("swarm.dag_node_retry", format!("node {}: {}", node_id, err));
                            // Automatically re-dispatch ready retry nodes
                            self.relay.dispatch_ready_dag_nodes(
                                &self.api_client,
                                &self.tx,
                                &self.rt,
                                self.settings.num_threads,
                            );
                        } else {
                            self.status = format!("Swarm DAG node {} failed (max retries exceeded): {}", node_id, err);
                            self.audit("swarm.dag_node_failed", format!("node {}: {}", node_id, err));
                        }
                    } else if let Some(rest) = s.strip_prefix("DAG_RETRY:") {
                        if let Ok(node_id) = rest.parse::<usize>() {
                            self.relay.dag.reset_node_for_retry(node_id, None);
                            self.relay.dag.is_running = true;
                            if let Some(st) = self.storage.as_ref() {
                                let _ = st.save_swarm_dag(&self.relay.dag);
                            }
                            self.relay.dispatch_ready_dag_nodes(
                                &self.api_client,
                                &self.tx,
                                &self.rt,
                                self.settings.num_threads,
                            );
                            self.status = format!("Manually retrying Swarm DAG node {}", node_id);
                            self.audit("swarm.dag_node_manual_retry", format!("node {}", node_id));
                        }
                    } else if let Some(rest) = s.strip_prefix("DAG_RUN_TOOL:") {
                        if let Ok(node_id) = rest.parse::<usize>() {
                            let ws_root = self.editor.workspace.root_path.to_str();
                            match self.relay.execute_node_tool(
                                node_id,
                                &self.tools_panel.registry,
                                self.settings.guardrail_tier,
                                ws_root,
                            ) {
                                Ok(res) => {
                                    self.status = format!(
                                        "Tool '{}' executed on node {}: exit code {}",
                                        res.tool_id, node_id, res.exit_code
                                    );
                                    self.audit(
                                        "swarm.dag_run_tool",
                                        format!("node {}: {} (exit {})", node_id, res.tool_id, res.exit_code),
                                    );
                                    if let Some(st) = self.storage.as_ref() {
                                        let _ = st.save_swarm_dag(&self.relay.dag);
                                        let _ = st.save_blackboard(&self.relay.blackboard);
                                    }
                                }
                                Err(err) => {
                                    self.status = format!("Tool execution failed for node {}: {}", node_id, err);
                                    self.audit("swarm.dag_run_tool_error", format!("node {}: {}", node_id, err));
                                }
                            }
                        }
                    } else {
                        self.status = s;
                    }
                }
                AppMessage::Audit(kind, detail) => {
                    self.audit(&kind, detail);
                }
                AppMessage::NewChat => {
                    let f = self.focused_slot.min(self.slots.len().saturating_sub(1));
                    self.audit("chat.new", format!("slot {}", f + 1));
                    self.abort_slot(f);
                    if let Some(slot) = self.slots.get_mut(f) {
                        slot.chat.clear_chat();
                        slot.session_id = Uuid::new_v4();
                        slot.session_created = Utc::now();
                        self.status = format!("Slot {} started a new chat", f + 1);
                    }
                }
                AppMessage::LoadSession(s) => {
                    let f = self.focused_slot.min(self.slots.len().saturating_sub(1));
                    self.abort_slot(f);
                    if let Some(slot) = self.slots.get_mut(f) {
                        slot.chat.load_messages(s.messages.clone());
                        slot.session_id = s.id;
                        slot.session_created = s.created_at;
                        slot.last_saved_revision = slot.chat.revision();
                        self.status = format!("Loaded '{}' into slot {}", s.name, f + 1);
                        self.audit("chat.load", format!("'{}' -> slot {}", s.name, f + 1));
                    }
                    self.tab = Tab::Chat;
                }
                AppMessage::VoiceToggled(req) => {
                    if req {
                        match VoiceEngine::new(
                            self.settings.tts_voice.clone(),
                            self.settings.stt_model.clone(),
                        ) {
                            Ok(eng) if eng.is_available() => {
                                self.voice = Some(eng);
                                self.settings.voice_enabled = true;
                                if let Some(st) = self.storage.as_ref() {
                                    let _ = st.save_settings(&self.settings);
                                }
                                self.status =
                                    "Voice on: replies will be read aloud".to_string();
                                self.audit("voice.on", String::new());
                                let _ = self.tx.send(AppMessage::VoiceState {
                                    enabled: true,
                                    note: String::new(),
                                });
                            }
                            Ok(_) | Err(_) => {
                                self.status =
                                    "Voice unavailable: install piper + whisper-cli"
                                        .to_string();
                                let _ = self.tx.send(AppMessage::VoiceState {
                                    enabled: false,
                                    note: "missing piper/whisper binaries".to_string(),
                                });
                            }
                        }
                    } else {
                        self.voice = None;
                        self.settings.voice_enabled = false;
                        if let Some(st) = self.storage.as_ref() {
                            let _ = st.save_settings(&self.settings);
                        }
                        self.status = "Voice off".to_string();
                        self.audit("voice.off", String::new());
                        let _ = self.tx.send(AppMessage::VoiceState {
                            enabled: false,
                            note: String::new(),
                        });
                    }
                }
                AppMessage::VoiceListen(idx) => {
                    if let Some(eng) = self.voice.clone() {
                        let tx = self.tx.clone();
                        self.status = "Listening (5s)...".to_string();
                        self.rt.spawn(async move {
                            let result = eng.listen().await.unwrap_or_default();
                            let _ = tx.send(AppMessage::VoiceInput(idx, result));
                        });
                    } else {
                        self.status = "Voice is off — toggle Voice ON first".to_string();
                    }
                }
                AppMessage::VoiceState { enabled, note } => {
                    for slot in &mut self.slots {
                        slot.chat.set_voice_enabled(enabled);
                    }
                    if !note.is_empty() {
                        self.status = format!("Voice off ({})", note);
                    }
                }
                AppMessage::VoiceInput(idx, text) => {
                    if text.trim().is_empty() {
                        self.status = "Heard nothing".to_string();
                    } else if let Some(slot) = self.slots.get_mut(idx) {
                        slot.chat.append_input(&text);
                        self.status = "Dictated into chat input".to_string();
                    }
                }
                AppMessage::StopSpeak => {
                    if let Some(eng) = self.voice.clone() {
                        self.rt.spawn(async move {
                            eng.stop().await;
                        });
                        self.status = "Voice stopped".to_string();
                    } else {
                        self.status = "Voice unavailable — toggle Voice ON first".to_string();
                    }
                }
                AppMessage::SpeakText(text) => {
                    if let Some(eng) = self.voice.clone() {
                        let text: String = text.chars().take(1000).collect();
                        self.status = "Reading message aloud…".to_string();
                        self.rt.spawn(async move {
                            let _ = eng.speak(&text).await;
                        });
                    } else {
                        self.status = "Voice unavailable — toggle Voice ON first".to_string();
                    }
                }
                AppMessage::LaunchSwarmTask(task) => {
                    self.relay.prompt = task;
                    self.tab = Tab::Relay;
                    self.status = "Swarm Relay loaded with task".to_string();
                }
                AppMessage::ApplySwarmToChatSlots(roles) => {
                    if !roles.is_empty() {
                        self.slots.clear();
                        for (idx, (role, model)) in roles.into_iter().enumerate() {
                            let mut slot = ModelSlot::new(idx, role);
                            slot.model = model;
                            self.slots.push(slot);
                        }
                        self.focused_slot = 0;
                        self.split_chat_view = self.slots.len() > 1;
                        self.tab = Tab::Chat;
                        self.status = format!("Loaded {} swarm roles into Chat slots", self.slots.len());
                        self.audit("swarm.chat_slots_applied", format!("{} slots", self.slots.len()));
                    }
                }
                AppMessage::LaunchSwarmPreset { template, prompt } => {
                    self.relay.apply_template(template);
                    if let Some(p) = prompt {
                        self.relay.prompt = p;
                    }
                    self.tab = Tab::Relay;
                    self.status = format!("Loaded Swarm preset: {}", template.label());
                    self.audit("swarm.preset_launched", template.short_id().to_string());
                }
                AppMessage::LaunchSwarmDag { preset, prompt } => {
                    self.tab = Tab::Relay;
                    self.relay.swarm_mode = crate::ui::relay::SwarmMode::Dag;
                    if let Some(p) = preset {
                        self.relay.dag.apply_preset(p);
                    }
                    // Auto-assign models to nodes that don't have one
                    for node in &mut self.relay.dag.nodes {
                        if node.model.is_none() {
                            node.model = crate::ui::relay::RelayPanel::find_best_model_for_role(&node.role, &self.models);
                        }
                    }
                    if let Some(p) = prompt {
                        self.relay.dag.objective = p;
                        if self.relay.dag.nodes.iter().all(|n| n.model.is_some()) {
                            self.relay.start_dag(&self.api_client, &self.tx, &self.rt, self.settings.num_threads);
                            self.status = format!("Launched Swarm DAG: {}", self.relay.dag.preset.label());
                        } else {
                            self.status = format!("Configured Swarm DAG: {} (Assign remaining models to run)", self.relay.dag.preset.label());
                        }
                    } else {
                        self.status = format!("Loaded Swarm DAG preset: {}", self.relay.dag.preset.label());
                    }
                    self.audit("swarm.dag_launched", self.relay.dag.preset.label().to_string());
                }
                AppMessage::ShowBlackboard => {
                    self.tab = Tab::Relay;
                    self.relay.swarm_mode = crate::ui::relay::SwarmMode::Dag;
                    self.relay.show_blackboard_drawer = true;
                    self.status = "Swarm Stigmergic Blackboard opened".to_string();
                    self.audit("swarm.blackboard_opened", "blackboard drawer".to_string());
                }
                AppMessage::AbortSwarm => {
                    self.relay.abort_dag();
                    self.relay.abort_pipeline();
                    self.status = "Swarm execution aborted".to_string();
                    self.audit("swarm.abort", "all nodes and steps aborted".to_string());
                }
                AppMessage::SetThreads(n) => {
                    self.settings.num_threads = n;
                    if let Some(st) = self.storage.as_ref() {
                        let _ = st.save_settings(&self.settings);
                    }
                    for slot in &mut self.slots {
                        slot.chat.num_threads = n;
                    }
                    self.status = format!("Inference threads set to {}", n);
                }
                AppMessage::ShowAudit(slot_idx) => {
                    if let Some(st) = self.storage.as_ref() {
                        if let Ok(entries) = st.load_audit() {
                            let mut msg = String::from("### 🛡 **Recent Security & Cryptographic Audit Log**\n\n| Timestamp | Event Kind | Detail |\n|---|---|---|\n");
                            if entries.is_empty() {
                                msg.push_str("| - | *Clean* | No security incidents or blocked secrets |\n");
                            } else {
                                for e in entries.iter().take(15) {
                                    msg.push_str(&format!("| `{}` | `{}` | {} |\n", e.ts.format("%H:%M:%S"), e.kind, e.detail));
                                }
                            }
                            if let Some(slot) = self.slots.get_mut(slot_idx) {
                                slot.chat.push_system_note(&msg);
                            }
                        }
                    }
                }
                AppMessage::ShowStatus(slot_idx) => {
                    let mem = crate::resources::system_memory();
                    let v = self.ollama_version.clone().unwrap_or_else(|| "offline".to_string());
                    let msg = format!(
                        "### ⚡ **ML Laboratory System Status**\n\n\
                         • **Ollama Engine:** `{}`\n\
                         • **System Memory:** Used `{:.2} GB` / Total `{:.2} GB`\n\
                         • **Active CPU Inference Threads:** `{}` (optimal: `{}`)\n\
                         • **Post-Quantum Storage Vault:** `AES-256-GCM (Active)`\n\
                         • **Active Model Slots:** `{}` slots configured\n",
                        v,
                        mem.used_bytes() as f64 / (1024.0 * 1024.0 * 1024.0),
                        mem.total_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                        self.settings.num_threads,
                        crate::ollama::api::ChatOptions::optimal_threads(),
                        self.slots.len()
                    );
                    if let Some(slot) = self.slots.get_mut(slot_idx) {
                        slot.chat.push_system_note(msg);
                    }
                }
                AppMessage::SkillsCommand(slot_idx, sub) => {
                    match sub {
                        crate::commands::SkillsCommand::List => {
                            if let Some(st) = self.storage.as_ref() {
                                if let Ok(skills) = st.load_skills() {
                                    let mut msg = String::from("### ⚡ **Autonomous Skills Registry**\n\n");
                                    for s in skills {
                                        let domain_name = match s.domain_idx {
                                            1 => "Coder",
                                            2 => "Researcher",
                                            3 => "Cyber/Critic",
                                            4 => "Planner",
                                            5 => "Writer",
                                            _ => "General",
                                        };
                                        msg.push_str(&format!("• **`{}`** `[{}]` — Score: `{:.2}` (Runs: {})\n  _{}_\n\n",
                                            s.name, domain_name, s.reinforcement_score, s.execution_count, s.description));
                                    }
                                    msg.push_str("To execute: `/skills run <name>` or visit the **⚡ Skills** tab.");
                                    if let Some(slot) = self.slots.get_mut(slot_idx) {
                                        slot.chat.push_system_note(&msg);
                                    }
                                }
                            }
                        }
                        crate::commands::SkillsCommand::Run(name) => {
                            if let Some(st) = self.storage.as_ref() {
                                if let Ok(skills) = st.load_skills() {
                                    if let Some(skill) = skills.iter().find(|s| s.name.eq_ignore_ascii_case(&name)) {
                                        let _ = st.reinforce_skill(skill.id, 0.1);
                                        if let Some(slot) = self.slots.get_mut(slot_idx) {
                                            slot.chat.append_input(&skill.prompt_template);
                                            slot.chat.push_system_note(&format!("⚡ **Loaded skill `{}`.** Provide your context or hit Send.", skill.name));
                                        }
                                    } else if let Some(slot) = self.slots.get_mut(slot_idx) {
                                        slot.chat.push_system_note(&format!("⚠ Skill `{}` not found. Type `/skills list` for registered skills.", name));
                                    }
                                }
                            }
                        }
                        crate::commands::SkillsCommand::Add { name, description } => {
                            if let Some(st) = self.storage.as_ref() {
                                let skill = crate::storage::Skill {
                                    id: Uuid::new_v4(),
                                    name: name.clone(),
                                    description,
                                    prompt_template: format!("You are an expert specialist performing {}:", name),
                                    domain_idx: 0,
                                    reinforcement_score: 1.0,
                                    execution_count: 0,
                                    last_used: None,
                                    is_built_in: false,
                                };
                                if st.save_skill(&skill).is_ok() {
                                    if let Some(slot) = self.slots.get_mut(slot_idx) {
                                        slot.chat.push_system_note(&format!("⚡ **Skill `{}` successfully created and encrypted in vault.**", name));
                                    }
                                }
                            }
                        }
                    }
                }
                AppMessage::RunSkill { skill_name, target_slot } => {
                    let target = target_slot.unwrap_or(self.focused_slot).min(self.slots.len().saturating_sub(1));
                    if let Some(st) = self.storage.as_ref() {
                        if let Ok(skills) = st.load_skills() {
                            if let Some(skill) = skills.iter().find(|s| s.name.eq_ignore_ascii_case(&skill_name)) {
                                let _ = st.reinforce_skill(skill.id, 0.2);
                                self.tab = Tab::Chat;
                                self.focused_slot = target;
                                if let Some(slot) = self.slots.get_mut(target) {
                                    slot.chat.append_input(&skill.prompt_template);
                                    slot.chat.push_system_note(&format!("⚡ **Activated autonomous skill `{}`.**", skill.name));
                                }
                            }
                        }
                    }
                }
                AppMessage::ReinforceSkill { skill_id, reward_delta } => {
                    if let Some(st) = self.storage.as_ref() {
                        if let Ok(Some(skill)) = st.reinforce_skill(skill_id, reward_delta) {
                            self.network.swarm_pheromones.deposit(skill.domain_idx, self.focused_slot, reward_delta);
                            self.status = format!("Skill '{}' reinforced ({:+.2})", skill.name, reward_delta);
                        }
                    }
                }
                AppMessage::ToolsCommand(slot_idx, sub) => {
                    match sub {
                        crate::commands::ToolsCommand::List => {
                            let mut msg = String::from("### 🛠 **External Tools Registry (BYOT)**\n\n");
                            for t in &self.tools_panel.registry.tools {
                                let (sens, _) = match t.sensitivity {
                                    crate::tools::SensitivityLevel::Low => ("1-Click Run", ()),
                                    crate::tools::SensitivityLevel::High => ("Confirm Req", ()),
                                };
                                msg.push_str(&format!(
                                    "• **`{}`** `[{}]` `[{}]` Min: `{}`\n  `{} {}`\n  _{}_\n\n",
                                    t.id,
                                    t.execution_type.short_badge(),
                                    sens,
                                    t.min_guardrail.short_label(),
                                    t.command,
                                    t.args_template,
                                    t.description
                                ));
                            }
                            msg.push_str("To run: `/tools run <id> [args]` or switch to the **🛠 Tools** tab.");
                            if let Some(slot) = self.slots.get_mut(slot_idx) {
                                slot.chat.push_system_note(&msg);
                            }
                        }
                        crate::commands::ToolsCommand::Run { id, args } => {
                            let _ = self.tx.send(AppMessage::RunTool { id, args });
                        }
                        crate::commands::ToolsCommand::Add { id, command, args_template, detached, high_sensitivity } => {
                            let tool = crate::tools::UserTool {
                                id: id.clone(),
                                name: id.clone(),
                                command,
                                args_template,
                                execution_type: if detached { crate::tools::ToolExecutionType::Detached } else { crate::tools::ToolExecutionType::TerminalDock },
                                sensitivity: if high_sensitivity { crate::tools::SensitivityLevel::High } else { crate::tools::SensitivityLevel::Low },
                                min_guardrail: crate::guardrails::GuardrailTier::Heavy,
                                description: "User established tool".to_string(),
                            };
                            self.tools_panel.registry.add_or_update(tool);
                            self.tools_panel.save(&self.storage);
                            if let Some(slot) = self.slots.get_mut(slot_idx) {
                                slot.chat.push_system_note(&format!("🛠 **Tool `{}` registered and encrypted into vault.**", id));
                            }
                        }
                        crate::commands::ToolsCommand::Rm(id) => {
                            if self.tools_panel.registry.remove(&id) {
                                self.tools_panel.save(&self.storage);
                                if let Some(slot) = self.slots.get_mut(slot_idx) {
                                    slot.chat.push_system_note(&format!("🛠 **Tool `{}` removed from registry.**", id));
                                }
                            } else if let Some(slot) = self.slots.get_mut(slot_idx) {
                                slot.chat.push_system_note(&format!("⚠ Tool `{}` not found in registry.", id));
                            }
                        }
                    }
                }
                AppMessage::GuardrailCommand(slot_idx, sub) => {
                    match sub {
                        crate::commands::GuardrailCommand::Status => {
                            let msg = format!(
                                "### 🛡 **Active Safety Guardrails**\n\n\
                                 • **Tier:** `{}`\n\
                                 • **Description:** {}\n\
                                 • **Unrestricted Waiver Accepted:** `{}`\n\n\
                                 Use `/guardrail [heavy | medium | none]` or the **🛠 Tools** tab to modify.",
                                self.settings.guardrail_tier.label(),
                                self.settings.guardrail_tier.description(),
                                if self.settings.unrestricted_waiver_accepted { "YES" } else { "NO" }
                            );
                            if let Some(slot) = self.slots.get_mut(slot_idx) {
                                slot.chat.push_system_note(&msg);
                            }
                        }
                        crate::commands::GuardrailCommand::Set(tier) => {
                            if tier == crate::guardrails::GuardrailTier::None && !self.settings.unrestricted_waiver_accepted {
                                self.tab = Tab::Tools;
                                self.tools_panel.show_waiver_modal = true;
                                if let Some(slot) = self.slots.get_mut(slot_idx) {
                                    slot.chat.push_system_note(
                                        "⚠ **Unrestricted Guardrails require typed waiver acceptance.**\n\
                                         Opening legal authorization modal in the **🛠 Tools** tab."
                                    );
                                }
                            } else {
                                self.settings.guardrail_tier = tier;
                                if let Some(st) = self.storage.as_ref() {
                                    let _ = st.save_settings(&self.settings);
                                }
                                if let Some(slot) = self.slots.get_mut(slot_idx) {
                                    slot.chat.push_system_note(&format!("🛡 **Safety guardrail tier set to `{}`.**", tier.label()));
                                }
                                self.status = format!("Guardrails set to {}", tier.short_label());
                            }
                        }
                    }
                }
                AppMessage::RunTool { id, args } => {
                    let tool_opt = self.tools_panel.registry.get(&id).cloned();
                    let f = self.focused_slot.min(self.slots.len().saturating_sub(1));
                    if let Some(tool) = tool_opt {
                        let ws_root = self.editor.workspace.root_path.to_str();
                        let active_file = if self.editor.file_path.is_empty() { None } else { Some(self.editor.file_path.as_str()) };
                        let cmd_line = tool.format_command_line(active_file, ws_root, args.as_deref());

                        // Validate against active guardrails
                        let ws_path = Some(self.editor.workspace.root_path.as_path());
                        if let Err(violation) = self.settings.guardrail_tier.validate_command(&cmd_line, ws_path) {
                            if let Some(slot) = self.slots.get_mut(f) {
                                slot.chat.push_system_note(&format!("❌ **Guardrail Block:**\n{}", violation));
                            }
                            if let Some(st) = self.storage.as_ref() {
                                let _ = st.log_audit("guardrail.blocked", &format!("tool {}: {}", id, violation));
                            }
                            self.status = "Command blocked by active guardrail tier".to_string();
                        } else {
                            if let Some(st) = self.storage.as_ref() {
                                let _ = st.log_audit("tool.exec", &format!("{}: {}", id, cmd_line));
                            }
                            match tool.execution_type {
                                crate::tools::ToolExecutionType::Detached => {
                                    match self.tools_panel.registry.spawn_detached(&id, active_file, ws_root, args.as_deref()) {
                                        Ok(pid) => {
                                            if let Some(slot) = self.slots.get_mut(f) {
                                                slot.chat.push_system_note(&format!("🛠 **Spawned `{}`** (PID: `{}`) in detached process.", tool.name, pid));
                                            }
                                            self.status = format!("Tool '{}' spawned (PID: {})", tool.name, pid);
                                        }
                                        Err(e) => {
                                            if let Some(slot) = self.slots.get_mut(f) {
                                                slot.chat.push_system_note(&format!("❌ **Tool Error:** {}", e));
                                            }
                                        }
                                    }
                                }
                                crate::tools::ToolExecutionType::TerminalDock => {
                                    if let Some(slot) = self.slots.get_mut(f) {
                                        slot.chat.push_system_note(&format!("🛠 **Dispatched `{}` to Terminal Dock:**\n`{}`", tool.name, cmd_line));
                                    }
                                    let _ = self.tx.send(AppMessage::TerminalRun(cmd_line));
                                }
                            }
                        }
                    } else if let Some(slot) = self.slots.get_mut(f) {
                        slot.chat.push_system_note(&format!("⚠ Tool `{}` not found in registry. Type `/tools list` to check registered tools.", id));
                    }
                }
                AppMessage::SetGuardrailTier(tier) => {
                    self.settings.guardrail_tier = tier;
                    if let Some(st) = self.storage.as_ref() {
                        let _ = st.save_settings(&self.settings);
                    }
                }
                AppMessage::TerminalRun(cmd_line) => {
                    self.editor.run_terminal_command(&cmd_line, &self.tx, &self.rt);
                    self.tab = Tab::Editor;
                }
                AppMessage::TerminalFinished(result) => {
                    if let Some(st) = self.storage.as_ref() {
                        let status_str = if result.success { "success" } else { "failed" };
                        let _ = st.log_audit("terminal.exec", &format!("{}: {}", status_str, result.cmd));
                    }
                    self.editor.on_terminal_finished(result);
                }
                AppMessage::WorkspaceMkdir(dir_path) => {
                    match self.editor.workspace.create_dir(std::path::Path::new(&dir_path)) {
                        Ok(p) => {
                            self.status = format!("Created folder: {}", p.display());
                            if let Some(st) = self.storage.as_ref() {
                                let _ = st.log_audit("workspace.mkdir", &p.display().to_string());
                            }
                        }
                        Err(e) => {
                            self.status = format!("Failed to create folder: {}", e);
                        }
                    }
                }
                AppMessage::WorkspaceTouch(file_path) => {
                    match self.editor.workspace.create_file(std::path::Path::new(&file_path), "") {
                        Ok(p) => {
                            self.status = format!("Created file: {}", p.display());
                            self.editor.load_file_from_path(&p);
                            if let Some(st) = self.storage.as_ref() {
                                let _ = st.log_audit("workspace.touch", &p.display().to_string());
                            }
                        }
                        Err(e) => {
                            self.status = format!("Failed to create file: {}", e);
                        }
                    }
                }
                AppMessage::WorkspaceRm(target) => {
                    match self.editor.workspace.delete_entry(std::path::Path::new(&target)) {
                        Ok(()) => {
                            self.status = format!("Deleted: {}", target);
                            if let Some(st) = self.storage.as_ref() {
                                let _ = st.log_audit("workspace.rm", &target);
                            }
                        }
                        Err(e) => {
                            self.status = format!("Delete failed: {}", e);
                        }
                    }
                }
                AppMessage::WorkspaceMv { src, dst } => {
                    match self.editor.workspace.move_entry(std::path::Path::new(&src), std::path::Path::new(&dst)) {
                        Ok(d) => {
                            self.status = format!("Moved {} -> {}", src, d.display());
                            if let Some(st) = self.storage.as_ref() {
                                let _ = st.log_audit("workspace.mv", &format!("{} -> {}", src, dst));
                            }
                        }
                        Err(e) => {
                            self.status = format!("Move failed: {}", e);
                        }
                    }
                }
                AppMessage::WorkspaceLs(path_opt) => {
                    let p = path_opt.unwrap_or_else(|| self.editor.workspace.root_path.display().to_string());
                    self.status = format!("Listed contents of {}", p);
                    let _ = self.editor.workspace.refresh_tree();
                }
                AppMessage::WorkspaceProject(proj) => match proj {
                    crate::commands::ProjectCommand::Scaffold { template, name } => {
                        let tmpl = match template.to_lowercase().as_str() {
                            "python" | "py" | "swarm" => crate::workspace::AppTemplateType::PythonAiSwarm,
                            "web" | "js" | "html" => crate::workspace::AppTemplateType::ModernWeb,
                            "cyber" | "quantum" | "vault" => crate::workspace::AppTemplateType::QuantumCyberSecurity,
                            _ => crate::workspace::AppTemplateType::RustHighPerformance,
                        };
                        match self.editor.scaffold_app(tmpl, &name) {
                            Ok(rep) => {
                                self.status = format!("Scaffolded {} in {}", rep.app_name, rep.target_dir.display());
                                if let Some(st) = self.storage.as_ref() {
                                    let _ = st.log_audit("project.scaffold", &format!("{}: {}", rep.app_name, rep.target_dir.display()));
                                }
                            }
                            Err(e) => {
                                self.status = format!("Scaffold failed: {}", e);
                            }
                        }
                    }
                    crate::commands::ProjectCommand::Build => {
                        self.editor.run_build(&self.tx, &self.rt);
                    }
                    crate::commands::ProjectCommand::Test => {
                        self.editor.run_test(&self.tx, &self.rt);
                    }
                    crate::commands::ProjectCommand::Run => {
                        self.editor.run_start(&self.tx, &self.rt);
                    }
                    crate::commands::ProjectCommand::Setup => {
                        self.editor.run_setup(&self.tx, &self.rt);
                    }
                    crate::commands::ProjectCommand::Verify => {
                        self.editor.run_verify(&self.tx, &self.rt);
                    }
                    crate::commands::ProjectCommand::Clean => {
                        self.editor.run_clean(&self.tx, &self.rt);
                    }
                    crate::commands::ProjectCommand::GenerateScripts => {
                        match self.editor.generate_automation_scripts() {
                            Ok(n) => {
                                self.status = format!("Generated {} automation scripts in workspace", n);
                            }
                            Err(e) => {
                                self.status = format!("Script generation failed: {}", e);
                            }
                        }
                    }
                },
                AppMessage::DeployApp { template, name } => {
                    match self.editor.scaffold_app(template, &name) {
                        Ok(rep) => {
                            self.status = format!("App deployed to {}", rep.target_dir.display());
                            if let Some(st) = self.storage.as_ref() {
                                let _ = st.log_audit("app.deployed", &rep.target_dir.display().to_string());
                            }
                        }
                        Err(e) => {
                            self.status = format!("App deployment failed: {}", e);
                        }
                    }
                }
                AppMessage::DeployMultiFile(text) => {
                    match self.editor.deploy_multi_file_from_text(&text) {
                        Ok(rep) => {
                            self.status = format!("Deployed {} files to {}", rep.created_files.len(), rep.target_dir.display());
                            if let Some(st) = self.storage.as_ref() {
                                let _ = st.log_audit("app.multi_file_deployed", &format!("{} files in {}", rep.created_files.len(), rep.target_dir.display()));
                            }
                        }
                        Err(e) => {
                            self.status = format!("Multi-file deploy failed: {}", e);
                        }
                    }
                }
            }
        }
    }

    // ---------- UI ----------

    /// Set the Settings font size and persist it (shared by the A+/A-
    /// buttons and the Ctrl+= / Ctrl+- / Ctrl+0 shortcuts).
    fn set_zoom(&mut self, size: f32) {
        self.settings.font_size = size.clamp(10.0, 32.0);
        if let Some(st) = self.storage.as_ref() {
            let _ = st.save_settings(&self.settings);
        }
    }

    fn bump_zoom(&mut self, delta: f32) {
        let next = self.settings.font_size + delta;
        self.set_zoom(next);
    }

    fn show_top_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.heading(
                egui::RichText::new("ML Laboratory")
                    .size(20.0)
                    .color(egui::Color32::from_rgb(0x00, 0xaa, 0xff)),
            );
            ui.separator();
            let committed = self
                .report
                .system_used_bytes
                .saturating_add(self.report.models_used_bytes);
            ui.label(
                egui::RichText::new(format!(
                    "RAM {}/{} (15% = {} reserved)",
                    resources::format_bytes(committed),
                    resources::format_bytes(self.report.budget_bytes),
                    resources::format_bytes(self.report.reserve_bytes),
                ))
                .size(12.0),
            );
            ui.add(
                egui::ProgressBar::new(self.report.usage_fraction)
                    .desired_width(120.0)
                    .show_percentage(),
            );
            if self.report.over_budget {
                ui.label(
                    egui::RichText::new("OVER BUDGET")
                        .size(12.0)
                        .color(egui::Color32::from_rgb(0xff, 0x55, 0x55)),
                );
            } else {
                ui.label(
                    egui::RichText::new(format!(
                        "{} free · ~{} more fit",
                        resources::format_bytes(self.report.free_for_models_bytes),
                        self.report.max_models_fit,
                    ))
                    .size(12.0)
                    .color(egui::Color32::from_rgb(0x88, 0xcc, 0x88)),
                );
            }
        });
        ui.horizontal(|ui| {
            ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                let (dot, txt) = if self.linked {
                    (
                        egui::Color32::from_rgb(0x00, 0xcc, 0x88),
                        format!("\u{25CF} linked ({})", self.models.len()),
                    )
                } else {
                    (
                        egui::Color32::from_rgb(0xcc, 0x66, 0x66),
                        "\u{25CF} down".to_string(),
                    )
                };
                ui.label(egui::RichText::new(txt).size(12.0).color(dot));
                ui.separator();
                ui.label(
                    egui::RichText::new(self.status.clone())
                        .size(12.0)
                        .color(egui::Color32::from_rgb(0xaa, 0xaa, 0xaa)),
                );
                ui.separator();
                let (gr_r, gr_g, gr_b) = self.settings.guardrail_tier.badge_rgb();
                let gr_badge = if self.settings.guardrail_tier == crate::guardrails::GuardrailTier::None {
                    "⚠ [UNRESTRICTED - AUTHORIZED USE ONLY]"
                } else {
                    match self.settings.guardrail_tier {
                        crate::guardrails::GuardrailTier::Heavy => "[🛡 Guardrail: Heavy]",
                        crate::guardrails::GuardrailTier::Medium => "[🛡 Guardrail: Medium]",
                        crate::guardrails::GuardrailTier::None => "[⚠ UNRESTRICTED]",
                    }
                };
                if ui.small_button(
                    egui::RichText::new(gr_badge)
                        .size(11.0)
                        .strong()
                        .color(egui::Color32::from_rgb(gr_r, gr_g, gr_b))
                ).on_hover_text("Click to configure Safety Guardrails and External Tools in the Tools tab").clicked() {
                    self.tab = Tab::Tools;
                }
            });
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .small_button("A+")
                    .on_hover_text("Zoom in (Ctrl+=, saved to Settings → Font size)")
                    .clicked()
                {
                    self.bump_zoom(1.0);
                }
                ui.label(
                    egui::RichText::new(format!(
                        "{:.0}% \u{00B7} {:.1}ms",
                        (self.settings.font_size / 14.0).clamp(0.5, 1.75) * 100.0,
                        self.frame_ms,
                    ))
                    .size(12.0)
                    .color(egui::Color32::from_rgb(0x88, 0x88, 0x88)),
                );
                if ui
                    .small_button("A−")
                    .on_hover_text("Zoom out (Ctrl+-, saved to Settings → Font size)")
                    .clicked()
                {
                    self.bump_zoom(-1.0);
                }
            });
        });
    }

    fn show_tabs(&mut self, ui: &mut egui::Ui) {
        ui.add_space(8.0);
        let mut ordered_tabs = Vec::new();
        for label in &self.settings.tab_order {
            if let Some(tab) = Tab::from_label(label) {
                if !ordered_tabs.contains(&tab) {
                    ordered_tabs.push(tab);
                }
            }
        }
        for tab in Tab::all() {
            if !ordered_tabs.contains(&tab) {
                ordered_tabs.push(tab);
            }
        }

        for tab in ordered_tabs {
            // Visibility check: Chat and Settings are permanently visible anchors
            let is_visible = !tab.is_closable()
                || self.settings.visible_tabs.iter().any(|v| v == tab.label() || (v == "Relay" && tab == Tab::Relay));
            if !is_visible {
                continue;
            }

            let selected = self.tab == tab;
            let resp = ui.selectable_label(
                selected,
                egui::RichText::new(format!("{} {}", tab.icon(), tab.label())).size(14.0),
            );
            if resp.clicked() {
                self.tab = tab;
            }
            ui.add_space(4.0);
        }
        ui.add_space(16.0);
        ui.separator();
        ui.add_space(8.0);
        ui.label(
            egui::RichText::new(format!("{} model slot(s)", self.slots.len()))
                .size(12.0)
                .color(egui::Color32::from_rgb(0x88, 0x88, 0x88)),
        );
        for (i, slot) in self.slots.iter().enumerate() {
            let name = slot.model.clone().unwrap_or("(empty)".to_string());
            ui.label(
                egui::RichText::new(format!("{}. {} [{}]", i + 1, name, slot.role.label()))
                    .size(12.0),
            );
        }
    }

    fn show_chat(&mut self, ui: &mut egui::Ui) {
        let models = self.models.clone();
        let known = self.known_sizes();

        // Snapshot for read-only rendering; actions applied after the strip.
        let snapshot: Vec<(usize, Option<String>, ModelRole, String)> = self
            .slots
            .iter()
            .map(|s| {
                (
                    s.id,
                    s.model.clone(),
                    s.role.clone(),
                    s.custom_role.clone(),
                )
            })
            .collect();

        let mut pending_assign: Option<(usize, String)> = None;
        let mut pending_unassign: Option<usize> = None;
        let mut pending_focus: Option<usize> = None;
        let mut pending_remove: Option<usize> = None;
        let mut pending_role: Option<(usize, ModelRole)> = None;
        let mut pending_custom: Option<(usize, String)> = None;
        let mut add_pressed = false;

        ui.horizontal(|ui| {
            ui.add_space(4.0);
            if ui
                .small_button("\u{23F9} Stop all")
                .on_hover_text("Stop generation on every slot")
                .clicked()
            {
                let _ = self.tx.send(AppMessage::StopAll);
            }
            ui.add_space(8.0);
            if ui
                .small_button("Clone slot")
                .on_hover_text("Duplicate the focused slot with its history")
                .clicked()
            {
                self.try_clone_slot();
            }
            ui.add_space(8.0);
            if ui
                .small_button("Fill empty")
                .on_hover_text("Assign the first listed model to every empty slot")
                .clicked()
            {
                self.fill_empty_slots();
            }
            ui.add_space(8.0);
            if self.clear_chats_armed {
                if ui.small_button("Confirm clear").clicked() {
                    let mut n = 0usize;
                    for i in 0..self.slots.len() {
                        self.abort_slot(i);
                        if let Some(slot) = self.slots.get_mut(i) {
                            n += slot.chat.messages().len();
                            slot.chat.clear_chat();
                        }
                    }
                    self.status = format!("Cleared {} messages", n);
                    self.audit("chat.clear_all", format!("{} messages", n));
                    self.clear_chats_armed = false;
                }
            } else if ui
                .small_button("Clear chats")
                .on_hover_text("Empty every slot (history keeps saved sessions)")
                .clicked()
            {
                self.clear_chats_armed = true;
            }
            ui.add_space(8.0);
            if ui
                .small_button("Export all .md")
                .on_hover_text("Export every slot transcript to one markdown file")
                .clicked()
            {
                let mut md = String::new();
                for (i, slot) in self.slots.iter().enumerate() {
                    md.push_str(&format!(
                        "# Slot {} \u{00B7} {} [{}]\n\n{}",
                        i + 1,
                        slot.model.clone().unwrap_or("(empty)".to_string()),
                        slot.role.label(),
                        crate::ui::chat::ChatPanel::slot_markdown(&slot.model, slot.chat.messages())
                    ));
                }
                self.status = match crate::ui::history::HistoryPanel::export_path("all-slots") {
                    Some(path) => {
                        if let Some(parent) = path.parent() {
                            let _ = std::fs::create_dir_all(parent);
                        }
                        match std::fs::write(&path, md) {
                            Ok(()) => format!("Saved {}", path.display()),
                            Err(e) => format!("Export failed: {e}"),
                        }
                    }
                    None => "Export failed: bad export name.".to_string(),
                };
            }
            ui.add_space(8.0);
            let toggle_label = if self.show_model_slots {
                "⊟ Collapse Slots"
            } else {
                "⊞ Multi-Model Slots"
            };
            if ui
                .small_button(toggle_label)
                .on_hover_text("Toggle the multi-model slot bar (hide to maximize chat area, show to adjust slots)")
                .clicked()
            {
                self.show_model_slots = !self.show_model_slots;
            }
        });
        ui.add_space(4.0);
        if self.show_model_slots {
            egui::ScrollArea::horizontal()
                .id_salt("slot_cards")
                .show(ui, |ui| {
            ui.horizontal(|ui| {
            for (i, (id, model, role, custom)) in snapshot.iter().enumerate() {
                let is_focused = i == self.focused_slot;
                egui::Frame::group(&ui.style())
                    .fill(if is_focused {
                        egui::Color32::from_rgb(0x0a, 0x2a, 0x44)
                    } else {
                        egui::Color32::from_rgb(0x1e, 0x1e, 0x1e)
                    })
                    .stroke(egui::Stroke::new(
                        1.0,
                        if is_focused {
                            egui::Color32::from_rgb(0x00, 0xaa, 0xff)
                        } else {
                            egui::Color32::from_rgb(0x44, 0x44, 0x44)
                        },
                    ))
                    .corner_radius(egui::CornerRadius::same(8))
                    .inner_margin(egui::Margin::same(8))
                    .show(ui, |ui| {
                        ui.set_min_width(230.0);
                        ui.set_max_width(300.0);
                        let stale = model
                            .as_ref()
                            .is_some_and(|m| !known.contains_key(m));
                        ui.horizontal(|ui| {
                            ui.label(
                                egui::RichText::new(format!(
                                    "Slot {} · {}",
                                    i + 1,
                                    role.label()
                                ))
                                .size(13.0)
                                .strong(),
                            );
                            if is_focused {
                                ui.label(
                                    egui::RichText::new("●")
                                        .size(10.0)
                                        .color(egui::Color32::from_rgb(0x00, 0xcc, 0x88)),
                                );
                            }
                            if stale {
                                ui.label(
                                    egui::RichText::new("gone from Ollama?")
                                        .size(11.0)
                                        .color(egui::Color32::from_rgb(0xff, 0x66, 0x66)),
                                );
                            }
                        });
                        ui.add_space(4.0);

                        // Model picker
                        let current = model.clone().unwrap_or("(none)".to_string());
                        egui::ComboBox::from_id_salt(format!("slot_model_{}", id))
                            .selected_text(current)
                            .width(210.0)
                            .show_ui(ui, |ui| {
                                if ui
                                    .selectable_label(model.is_none(), "(none)")
                                    .clicked()
                                {
                                    pending_unassign = Some(i);
                                }
                                for m in &models {
                                    let selected =
                                        model.as_deref() == Some(&m.name);
                                    // Would this model fit if assigned here?
                                    let others: Vec<u64> = snapshot
                                        .iter()
                                        .enumerate()
                                        .filter(|(j, (_, mm, _, _))| {
                                            *j != i && mm.is_some()
                                        })
                                        .map(|(_, (_, mm, _, _))| {
                                            ResourceGuard::size_for_model(
                                                mm.as_deref().unwrap_or(""),
                                                &known,
                                            )
                                        })
                                        .collect();
                                    let fits = ResourceGuard::can_fit(
                                        &self.mem,
                                        &others,
                                        ResourceGuard::size_for_model(&m.name, &known),
                                    )
                                    .is_ok();
                                    let mark = if selected {
                                        ""
                                    } else if fits {
                                        " ✓"
                                    } else {
                                        " ⚠ over budget"
                                    };
                                    let label = format!(
                                        "{} ({}){}",
                                        m.name,
                                        resources::format_bytes(m.size),
                                        mark
                                    );
                                    if ui.selectable_label(selected, label).clicked() {
                                        pending_assign =
                                            Some((i, m.name.clone()));
                                    }
                                }
                            });
                        ui.add_space(4.0);

                        // Role picker
                        egui::ComboBox::from_id_salt(format!("slot_role_{}", id))
                            .selected_text(role.label())
                            .width(210.0)
                            .show_ui(ui, |ui| {
                                for r in ModelRole::all_fixed() {
                                    if ui
                                        .selectable_label(*role == r, r.label())
                                        .clicked()
                                    {
                                        pending_role = Some((i, r));
                                    }
                                }
                                let is_custom = matches!(role, ModelRole::Custom(_));
                                if ui.selectable_label(is_custom, "Custom…").clicked() {
                                    pending_role = Some((
                                        i,
                                        ModelRole::Custom(custom.clone()),
                                    ));
                                }
                            });
                        if matches!(role, ModelRole::Custom(_)) {
                            let mut c = custom.clone();
                            if ui
                                .add(
                                    egui::TextEdit::singleline(&mut c)
                                        .desired_width(210.0)
                                        .hint_text("e.g. SQL tutor, sarcastic pirate…"),
                                )
                                .changed()
                            {
                                pending_custom = Some((i, c));
                            }
                        }
                        ui.add_space(4.0);

                        ui.horizontal(|ui| {
                            if !is_focused && ui.small_button("Focus").clicked() {
                                pending_focus = Some(i);
                            }
                            if snapshot.len() > 1 && ui.small_button("✕").clicked()
                            {
                                pending_remove = Some(i);
                            }
                        });
                    });
            }

            // "+" card to add another model slot (resource-gated).
            egui::Frame::group(&ui.style())
                .fill(egui::Color32::from_rgb(0x18, 0x28, 0x18))
                .stroke(egui::Stroke::new(
                    1.0,
                    egui::Color32::from_rgb(0x00, 0xaa, 0x55),
                ))
                .corner_radius(egui::CornerRadius::same(8))
                .inner_margin(egui::Margin::same(8))
                .show(ui, |ui| {
                    ui.set_min_width(120.0);
                    ui.vertical_centered(|ui| {
                        ui.add_space(16.0);
                        let btn = ui.add(
                            egui::Button::new(
                                egui::RichText::new("+ Add model").size(15.0),
                            )
                            .fill(egui::Color32::from_rgb(0x00, 0x77, 0x44))
                            .corner_radius(egui::CornerRadius::same(8)),
                        );
                        if btn
                            .on_hover_text(
                                "Add another model slot with its own role. Allowed only while RAM headroom (15% always reserved) permits.",
                            )
                            .clicked()
                        {
                            add_pressed = true;
                        }
                        ui.add_space(4.0);
                        ui.label(
                            egui::RichText::new(format!(
                                "~{} more fit",
                                self.report.max_models_fit
                            ))
                            .size(11.0)
                            .color(egui::Color32::from_rgb(0x88, 0x88, 0x88)),
                        );
                        ui.add_space(16.0);
                    });
                });
            });
        });
        }

        // Apply pending actions.
        if let Some((i, name)) = pending_assign {
            self.try_assign_model(i, name);
            self.focused_slot = i.min(self.slots.len().saturating_sub(1));
        }
        if let Some(i) = pending_unassign {
            if let Some(slot) = self.slots.get_mut(i) {
                slot.model = None;
                self.status = format!("Slot {} cleared", i + 1);
                self.audit("model.unassign", format!("slot {}", i + 1));
            }
        }
        if let Some((i, r)) = pending_role {
            if let Some(slot) = self.slots.get_mut(i) {
                slot.role = r;
            }
        }
        if let Some((i, c)) = pending_custom {
            if let Some(slot) = self.slots.get_mut(i) {
                slot.custom_role = c.clone();
                slot.role = ModelRole::Custom(c);
            }
        }
        if let Some(i) = pending_focus {
            self.focused_slot = i.min(self.slots.len().saturating_sub(1));
        }
        if let Some(i) = pending_remove {
            if self.slots.len() > 1 && i < self.slots.len() {
                self.abort_slot(i);
                self.inflight.remove(&i);
                self.slots.remove(i);
                self.focused_slot = self.focused_slot.min(self.slots.len() - 1);
                self.status = format!("Slot {} removed", i + 1);
                self.audit("slot.remove", format!("slot {}", i + 1));
            }
        }
        if add_pressed {
            self.try_add_slot();
        }

        ui.add_space(8.0);
        ui.separator();
        ui.add_space(8.0);

        let active_slots: Vec<usize> = self
            .slots
            .iter()
            .enumerate()
            .filter(|(_, s)| s.model.is_some())
            .map(|(i, _)| i)
            .collect();

        if active_slots.len() >= 2 {
            ui.horizontal(|ui| {
                ui.add_space(4.0);
                let active_summary: Vec<String> = active_slots
                    .iter()
                    .map(|&idx| {
                        format!(
                            "Slot {}: {}",
                            idx + 1,
                            self.slots[idx].model.as_deref().unwrap_or("none")
                        )
                    })
                    .collect();
                ui.label(
                    egui::RichText::new(format!("⚡ Multi-Model: {}", active_summary.join("  |  ")))
                        .size(13.0)
                        .strong()
                        .color(egui::Color32::from_rgb(0x00, 0xee, 0xff)),
                );
                ui.add_space(8.0);
                ui.checkbox(&mut self.dual_run_mode, "⚡ Run Both on Enter (Dual Mode)")
                    .on_hover_text("When enabled, typing a prompt and pressing Enter runs all active models simultaneously.");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.add_space(4.0);
                    if ui.selectable_label(self.split_chat_view, "⊞ Split View").clicked() {
                        self.split_chat_view = true;
                    }
                    if ui.selectable_label(!self.split_chat_view, "⬚ Single View").clicked() {
                        self.split_chat_view = false;
                    }
                });
            });
            ui.add_space(4.0);
            ui.separator();
            ui.add_space(4.0);
        }

        if active_slots.len() >= 2 && self.split_chat_view {
            let f = self.focused_slot.min(self.slots.len().saturating_sub(1));
            let input_reserve = 180.0 * ui.ctx().zoom_factor();
            let list_h = (ui.available_height() - input_reserve).max(100.0);
            let num_cols = active_slots.len();

            ui.columns(num_cols, |cols| {
                for (col_idx, &slot_idx) in active_slots.iter().enumerate() {
                    let col = &mut cols[col_idx];
                    let is_focused = slot_idx == f;
                    let slot = &mut self.slots[slot_idx];
                    let slot_m = slot.model.clone().unwrap_or_default();
                    let slot_r = slot.role.label();

                    col.horizontal(|ui| {
                        ui.add_space(2.0);
                        let title = format!("Slot {} · {} [{}]", slot_idx + 1, slot_m, slot_r);
                        let text_color = if is_focused {
                            egui::Color32::from_rgb(0x00, 0xee, 0xff)
                        } else {
                            egui::Color32::from_rgb(0xdd, 0xdd, 0xdd)
                        };
                        ui.label(egui::RichText::new(title).size(13.0).strong().color(text_color));
                        if slot.chat.is_streaming() {
                            ui.spinner();
                            ui.label(egui::RichText::new("Streaming...").size(11.0).color(egui::Color32::YELLOW));
                        }
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.small_button("Clear").on_hover_text("Clear this slot").clicked() {
                                slot.chat.clear_chat();
                            }
                        });
                    });
                    col.add_space(2.0);
                    col.separator();
                    slot.chat.show_message_list(col, &self.tx, list_h);
                }
            });

            ui.separator();
            ui.add_space(4.0);

            // Unified bottom input dock
            let response = ui.add(
                egui::TextEdit::multiline(&mut self.slots[f].chat.input)
                    .id_salt("chat_unified_multiline_input")
                    .desired_rows(3)
                    .desired_width(f32::INFINITY)
                    .hint_text(if self.dual_run_mode {
                        "Type message for ALL active models... (Enter to send to both, Shift+Enter for newline)"
                    } else {
                        "Type message for focused slot... (Enter to send, Shift+Enter for newline)"
                    })
                    .font(egui::TextStyle::Body),
            );

            let input_text = self.slots[f].chat.input.trim().to_string();
            let domain_info = if !input_text.is_empty() && input_text.len() >= 6 {
                Some(crate::neural::classify_prompt_domain(&input_text))
            } else {
                None
            };

            ui.add_space(4.0);
            ui.horizontal(|ui| {
                let voice_btn = ui.add(
                    egui::Button::new(if self.slots[f].chat.voice_enabled() { "Voice ON" } else { "Voice OFF" })
                        .corner_radius(egui::CornerRadius::same(6)),
                );
                if voice_btn.clicked() {
                    let cur = self.slots[f].chat.voice_enabled();
                    let _ = self.tx.send(crate::ui::app::AppMessage::VoiceToggled(!cur));
                }
                let mic_btn = ui.add(
                    egui::Button::new("Dictate").corner_radius(egui::CornerRadius::same(6)),
                );
                if mic_btn.clicked() {
                    let _ = self.tx.send(crate::ui::app::AppMessage::VoiceListen(f));
                }
                if let Some((domain_idx, domain_name)) = domain_info {
                    let badge = match domain_idx {
                        1 => "💻 Coder",
                        2 => "🔬 Researcher",
                        3 => "🛡️ Cyber / Critic",
                        4 => "📋 Planner",
                        5 => "✍️ Writer",
                        _ => "🌐 General",
                    };
                    ui.label(
                        egui::RichText::new(format!("🐝 {badge} ({domain_name})"))
                            .size(11.0)
                            .color(egui::Color32::from_rgb(0x00, 0xee, 0xff)),
                    );
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let any_streaming = active_slots.iter().any(|&s| self.slots[s].chat.is_streaming());
                    let can_send = !input_text.is_empty() && !any_streaming && self.api_client.is_some();
                    let label = if any_streaming {
                        "Working..."
                    } else if self.dual_run_mode {
                        "⚡ Send to Both (Enter)"
                    } else {
                        "Send to Focused (Enter)"
                    };
                    let fill = if can_send {
                        egui::Color32::from_rgb(0x00, 0x66, 0xcc)
                    } else {
                        egui::Color32::from_rgb(0x1a, 0x3a, 0x66)
                    };
                    let send_btn = ui.add_enabled(
                        can_send,
                        egui::Button::new(egui::RichText::new(label).size(13.0).color(egui::Color32::WHITE).strong())
                            .fill(fill)
                            .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(0x44, 0xaa, 0xff)))
                            .corner_radius(egui::CornerRadius::same(6))
                            .min_size(egui::vec2(150.0, 34.0)),
                    ).on_hover_text("Send prompt (Enter to send, Shift+Enter for newline)");
                    if send_btn.clicked() {
                        if self.dual_run_mode {
                            if let Some(prompt) = self.slots[f].chat.take_broadcast() {
                                let _ = self.tx.send(crate::ui::app::AppMessage::Broadcast(prompt));
                            }
                        } else {
                            let slot_model = self.slots[f].model.clone();
                            let role_prompt = compose_system_prompt(&self.settings.persona, &self.settings.memory, &self.slots[f], self.settings.guardrail_tier);
                            self.slots[f].chat.send_message(&self.models, &slot_model, &role_prompt, f, &self.api_client, &self.tx, &self.rt);
                        }
                    }

                    // Individual slot triggers
                    for &slot_idx in active_slots.iter().rev() {
                        let btn_lbl = format!("Slot {} Only", slot_idx + 1);
                        let btn = ui.add_enabled(
                            can_send,
                            egui::Button::new(egui::RichText::new(btn_lbl).size(11.0))
                                .corner_radius(egui::CornerRadius::same(6)),
                        );
                        if btn.clicked() {
                            let prompt = self.slots[f].chat.take_broadcast().unwrap_or_default();
                            if !prompt.is_empty() {
                                let slot_model = self.slots[slot_idx].model.clone();
                                let role_prompt = compose_system_prompt(&self.settings.persona, &self.settings.memory, &self.slots[slot_idx], self.settings.guardrail_tier);
                                self.slots[slot_idx].chat.send_prompt(prompt, &self.models, &slot_model, &role_prompt, slot_idx, &self.api_client, &self.tx, &self.rt);
                            }
                        }
                    }

                    if any_streaming {
                        ui.add_space(4.0);
                        let stop_btn = ui.add(
                            egui::Button::new(egui::RichText::new("Stop All").size(13.0).strong())
                                .fill(egui::Color32::from_rgb(0xaa, 0x33, 0x33))
                                .corner_radius(egui::CornerRadius::same(6)),
                        );
                        if stop_btn.clicked() {
                            let _ = self.tx.send(crate::ui::app::AppMessage::StopAll);
                        }
                    }
                });
                // Enter sends
                let send_triggered = ui.input(|i| {
                    i.key_pressed(egui::Key::Enter) && !i.modifiers.shift && !i.modifiers.ctrl
                }) && response.has_focus()
                    && !self.slots[f].chat.input.trim().is_empty()
                    && !active_slots.iter().any(|&s| self.slots[s].chat.is_streaming());
                if send_triggered {
                    if self.dual_run_mode {
                        if let Some(prompt) = self.slots[f].chat.take_broadcast() {
                            let _ = self.tx.send(crate::ui::app::AppMessage::Broadcast(prompt));
                        }
                    } else {
                        let slot_model = self.slots[f].model.clone();
                        let role_prompt = compose_system_prompt(&self.settings.persona, &self.settings.memory, &self.slots[f], self.settings.guardrail_tier);
                        self.slots[f].chat.send_message(&self.models, &slot_model, &role_prompt, f, &self.api_client, &self.tx, &self.rt);
                    }
                }
                let broadcast_triggered = ui.input(|i| {
                    i.key_pressed(egui::Key::Enter) && i.modifiers.ctrl
                }) && response.has_focus()
                    && !self.slots[f].chat.input.trim().is_empty()
                    && !active_slots.iter().any(|&s| self.slots[s].chat.is_streaming());
                if broadcast_triggered {
                    if let Some(prompt) = self.slots[f].chat.take_broadcast() {
                        let _ = self.tx.send(crate::ui::app::AppMessage::Broadcast(prompt));
                    }
                }
            });
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.add_space(4.0);
                ui.label(
                    egui::RichText::new(format!(
                        "⚡ Split View: {} models active | Ollama: {} | Press Enter to send to all active models concurrently",
                        active_slots.len(),
                        if self.api_client.is_some() { "linked" } else { "NOT linked" }
                    ))
                    .size(11.0)
                    .color(egui::Color32::from_rgb(0x88, 0x88, 0x88)),
                );
            });
        } else {
            // Focused slot chat. One shared identity for every model:
            // persona (who I am) + memory (what I remember) + slot role (job).
            let f = self.focused_slot.min(self.slots.len().saturating_sub(1));
            let role_prompt = compose_system_prompt(
                &self.settings.persona,
                &self.settings.memory,
                &self.slots[f],
                self.settings.guardrail_tier,
            );
            let slot_model = self.slots[f].model.clone();
            let history_depth = self.settings.history_depth.max(1) as usize;
            let slot_role = self.slots[f].role.label();
            let mut chat_hdr_assign: Option<String> = None;
            let mut chat_hdr_unassign = false;
            ui.horizontal(|ui| {
                ui.add_space(4.0);
                ui.label(
                    egui::RichText::new(format!("Slot {} ·", f + 1))
                        .size(14.0)
                        .strong()
                        .color(egui::Color32::from_rgb(0x00, 0xaa, 0xff)),
                );
                let current_txt = slot_model.clone().unwrap_or("(select model)".to_string());
                egui::ComboBox::from_id_salt(format!("active_chat_hdr_model_{}", f))
                    .selected_text(current_txt)
                    .width(220.0)
                    .show_ui(ui, |ui| {
                        if ui.selectable_label(slot_model.is_none(), "(none)").clicked() {
                            chat_hdr_unassign = true;
                        }
                        for m in &models {
                            let is_sel = slot_model.as_deref() == Some(&m.name);
                            let lbl = format!("{} ({})", m.name, crate::resources::format_bytes(m.size));
                            if ui.selectable_label(is_sel, lbl).clicked() {
                                chat_hdr_assign = Some(m.name.clone());
                            }
                        }
                    });
                ui.label(
                    egui::RichText::new("as")
                        .size(13.0)
                        .color(egui::Color32::from_rgb(0x88, 0x88, 0x88)),
                );
                ui.label(
                    egui::RichText::new(&slot_role)
                        .size(13.0)
                        .strong()
                        .color(egui::Color32::from_rgb(0x00, 0xcc, 0x88)),
                );
                if !self.show_model_slots {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.add_space(4.0);
                        if ui
                            .small_button(format!("⊞ Model Slots ({})", self.slots.len()))
                            .on_hover_text("Show multi-model slots to add, remove, or configure")
                            .clicked()
                        {
                            self.show_model_slots = true;
                        }
                    });
                }
            });
            if let Some(m_name) = chat_hdr_assign {
                self.try_assign_model(f, m_name);
            }
            if chat_hdr_unassign {
                if let Some(slot) = self.slots.get_mut(f) {
                    slot.model = None;
                    self.status = format!("Slot {} cleared", f + 1);
                }
            }
            let slot_model = self.slots[f].model.clone();
            ui.add_space(4.0);
            if let Some(slot) = self.slots.get_mut(f) {
                slot.chat.history_depth = history_depth;
                slot.chat.num_threads = self.settings.num_threads;
                let dual_active = self.dual_run_mode && active_slots.len() >= 2;
                slot.chat.show(
                    ui,
                    &models,
                    &slot_model,
                    &role_prompt,
                    &self.api_client,
                    f,
                    &self.tx,
                    &self.rt,
                    dual_active,
                );
            }
        }
    }
}

impl eframe::App for AiDashboardApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let frame_start = std::time::Instant::now();
        // Apply the Settings-tab font size as a global zoom factor so the UI
        // fits small screens (e.g. MacBook Air) and large monitors alike.
        // 14pt == 100%. Takes effect from the next frame.
        let zoom = (self.settings.font_size / 14.0).clamp(0.5, 1.75);
        if (ui.ctx().zoom_factor() - zoom).abs() > 0.001 {
            ui.ctx().set_zoom_factor(zoom);
        }
        // Keyboard zoom: Ctrl+= / Ctrl+- / Ctrl+0 mirror the A+/A- buttons.
        if ui.ctx().input(|i| i.modifiers.ctrl) {
            if ui.ctx().input(|i| i.key_pressed(egui::Key::Equals)) {
                self.bump_zoom(1.0);
            } else if ui.ctx().input(|i| i.key_pressed(egui::Key::Minus)) {
                self.bump_zoom(-1.0);
            } else if ui.ctx().input(|i| i.key_pressed(egui::Key::Num0)) {
                self.set_zoom(14.0);
            }
        }
        // Slot focus: Alt+1/2/3 or F1/F2/F3 (Alt/F-keys never clash with typing).
        // F-keys exist because synthetic key events don't always produce digits.
        let mut focus_req: Option<usize> = None;
        if ui.ctx().input(|i| i.modifiers.alt) {
            for (key, idx) in [
                (egui::Key::Num1, 0),
                (egui::Key::Num2, 1),
                (egui::Key::Num3, 2),
            ] {
                if ui.ctx().input(|i| i.key_pressed(key)) {
                    focus_req = Some(idx);
                }
            }
        }
        for (key, idx) in [(egui::Key::F1, 0), (egui::Key::F2, 1), (egui::Key::F3, 2)] {
            if ui.ctx().input(|i| i.key_pressed(key)) {
                focus_req = Some(idx);
            }
        }
        if let Some(idx) = focus_req {
            if idx < self.slots.len() {
                self.focused_slot = idx;
                self.status = format!("Slot {} focused", idx + 1);
            }
        }
        // F4 cycles focus through the slots (Shift+F4 goes backwards).
        if ui.ctx().input(|i| i.key_pressed(egui::Key::F4)) && !self.slots.is_empty() {
            let back = ui.ctx().input(|i| i.modifiers.shift);
            let n = self.slots.len();
            self.focused_slot = if back {
                (self.focused_slot + n - 1) % n
            } else {
                (self.focused_slot + 1) % n
            };
            self.status = format!("Slot {} focused", self.focused_slot + 1);
        }
        // Apply the Settings-tab theme choice (Dark/Light); System falls back to luxury dark.
        if self.settings.theme != self.last_theme {
            self.last_theme = self.settings.theme.clone();
            apply_luxury_visuals(ui.ctx(), &self.last_theme);
        }
        self.poll_messages();

        // Re-create the API client if the URL or the remote flag changed.
        if self.settings.ollama_url != self.last_ollama_url
            || self.settings.allow_remote != self.last_allow_remote
        {
            self.last_ollama_url = self.settings.ollama_url.clone();
            self.last_allow_remote = self.settings.allow_remote;
            self.api_client = OllamaClient::new(
                self.settings.ollama_url.clone(),
                self.settings.allow_remote,
            )
            .ok();
            self.refresh_models();
        }

        self.refresh_resources();

        egui::Panel::top("top_bar").show(ui, |ui| {
            self.show_top_bar(ui);
        });

        egui::Panel::left("side_tabs")
            .default_size(160.0)
            .resizable(true)
            .show(ui, |ui| {
                self.show_tabs(ui);
            });

        egui::CentralPanel::default().show(ui, |ui| {
            // Chat owns its message scroll area + a pinned input row, so it
            // must NOT live inside the outer scroll area — otherwise the
            // input row (and Send button) scrolls out of view on small
            // windows and looks "missing".
            if self.tab == Tab::Chat {
                self.show_chat(ui);
                return;
            }
            egui::ScrollArea::vertical().show(ui, |ui| match self.tab {
                Tab::Chat => self.show_chat(ui),
                Tab::Relay => {
                    self.relay.show(
                        ui,
                        &self.models,
                        &self.api_client,
                        &self.tx,
                        &self.rt,
                        self.settings.num_threads,
                        &self.tools_panel.registry,
                    );
                }
                Tab::Models => {
                    self.models_panel.show(
                        ui,
                        &mut self.models,
                        &self.ollama_version,
                        &self.api_client,
                        &self.cli_client,
                        &self.tx,
                        &self.rt,
                    );
                    if self.models_loading {
                        ui.horizontal(|ui| {
                            ui.spinner();
                            ui.label("Loading models…");
                        });
                    }
                }
                Tab::Editor => {
                    let f = self.focused_slot.min(self.slots.len().saturating_sub(1));
                    let model = self.slots.get(f).and_then(|s| s.model.clone());
                    let models = self.models.clone();
                    let editor_system = {
                        let mut parts: Vec<String> = Vec::new();
                        if !self.settings.persona.trim().is_empty() {
                            parts.push(self.settings.persona.trim().to_string());
                        }
                        if !self.settings.memory.trim().is_empty() {
                            parts.push(format!(
                                "Remembered facts:\n{}",
                                self.settings.memory.trim()
                            ));
                        }
                        parts.push(
                            "You are an expert software engineer. Answer with correct, idiomatic code and brief explanations."
                                .to_string(),
                        );
                        parts.join("\n\n")
                    };
                    self.editor.show(
                        ui,
                        &models,
                        &model,
                        &editor_system,
                        &self.api_client,
                        &self.tx,
                        &self.rt,
                        self.settings.num_threads,
                    );
                }
                Tab::History => {
                    if let Some(st) = self.storage.as_ref() {
                        self.history.show(ui, st, &self.tx);
                    } else {
                        ui.label("Storage unavailable.");
                    }
                }
                Tab::Neural => {
                    ui.horizontal(|ui| {
                        ui.add_space(4.0);
                        if ui
                            .small_button("Reset network")
                            .on_hover_text("Discard training and start a fresh profiler net")
                            .clicked()
                        {
                            self.reset_network();
                        }
                    });
                    ui.add_space(4.0);
                    self.neural_panel.show(ui, &mut self.network);
                }
                Tab::Learning => {
                    self.learning_panel.show(ui, &mut self.network, &self.models, &self.tx);
                }
                Tab::Skills => {
                    self.skills_panel.show(ui, &self.storage, &self.tx, self.focused_slot);
                }
                Tab::Tools => {
                    let ws_root = self.editor.workspace.root_path.to_str();
                    let focused = self.focused_slot.min(self.slots.len().saturating_sub(1));
                    let mut settings = self.settings.clone();
                    let storage = self.storage.take();
                    self.tools_panel.show(ui, &storage, &mut settings, &self.tx, focused, ws_root);
                    self.storage = storage;
                    self.settings = settings;
                }
                Tab::Compare => {
                    self.show_compare(ui);
                }
                Tab::Settings => {
                    if self.storage.is_some() {
                        let mut settings = self.settings.clone();
                        // Take storage out briefly to satisfy the borrow checker.
                        let storage = self.storage.take();
                        if let Some(st) = storage.as_ref() {
                            self.settings_panel.show(ui, &mut settings, st, &self.tx);
                        }
                        self.storage = storage;
                        self.settings = settings;
                    } else {
                        ui.label("Storage unavailable -- settings cannot be saved.");
                    }
                }
            });
        });

        self.autosave_dirty_slots();
        self.sync_slot_layout();

        // Keep the meter/spinners live, but don't burn CPU when idle:
        // 16ms (60 FPS) tick while something animates or streams tokens; 1000ms when idle.
        let tick = if self.animating() { 16 } else { 1000 };
        ui.ctx().request_repaint_after(std::time::Duration::from_millis(tick));
        let ms = frame_start.elapsed().as_secs_f32() * 1000.0;
        self.frame_ms = if self.frame_ms <= 0.0 { ms } else { self.frame_ms * 0.9 + ms * 0.1 };
    }
}
