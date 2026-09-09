use anyhow::Result;
use chrono::Utc;
use eframe::egui;
use std::collections::HashMap;
use std::sync::mpsc;
use std::time::Instant;
use tokio::runtime::Runtime;
use uuid::Uuid;
use ndarray::Array1;

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
use crate::ui::settings::SettingsPanel;

/// Messages sent from background tasks / panels to the app.
pub enum AppMessage {
    ChatResponse(usize, u64, Result<ChatResponse>),
    EditorSuggestion(usize, Result<ChatResponse>),
    /// Code handed from a chat bubble to the Editor tab (code, lang tag).
    ChatToEditor(String, String),
    ModelsLoaded(Result<Vec<Model>>),
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
    Broadcast(String),
    PullProgress(String),
    StreamHandle(usize, tokio::task::JoinHandle<()>),
    StopStream(usize),
}

/// Role assigned to a model slot. Prepended as a system prompt to every chat.
#[derive(Debug, Clone, PartialEq)]
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

/// Shared identity + memory + slot role, sent as the system prompt.
fn compose_system_prompt(persona: &str, memory: &str, slot: &ModelSlot) -> String {
    let mut parts: Vec<String> = Vec::new();
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

#[derive(Debug, Clone, Copy, PartialEq)]
enum Tab {
    Chat,
    Models,
    Editor,
    History,
    Neural,
    Settings,
}

impl Tab {
    fn label(self) -> &'static str {
        match self {
            Tab::Chat => "Chat",
            Tab::Models => "Models",
            Tab::Editor => "Editor",
            Tab::History => "History",
            Tab::Neural => "Neural",
            Tab::Settings => "Settings",
        }
    }

    fn icon(self) -> &'static str {
        match self {
            Tab::Chat => "💬",
            Tab::Models => "🤖",
            Tab::Editor => "📝",
            Tab::History => "📜",
            Tab::Neural => "🧠",
            Tab::Settings => "⚙",
        }
    }

    fn all() -> Vec<Tab> {
        vec![
            Tab::Chat,
            Tab::Models,
            Tab::Editor,
            Tab::History,
            Tab::Neural,
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
    last_theme: Theme,
    models: Vec<Model>,
    models_loading: bool,
    status: String,
    slots: Vec<ModelSlot>,
    next_slot_id: usize,
    focused_slot: usize,
    tab: Tab,
    editor: EditorPanel,
    models_panel: ModelsPanel,
    history: HistoryPanel,
    settings_panel: SettingsPanel,
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
        cc.egui_ctx.set_visuals(egui::Visuals::dark());
        let rt = Runtime::new().expect("tokio runtime");
        let (tx, rx) = mpsc::channel();
        let storage = Storage::new().ok();
        let settings = storage
            .as_ref()
            .and_then(|s| s.load_settings().ok())
            .unwrap_or_default();
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
            last_theme: settings.theme.clone(),
            models: Vec::new(),
            models_loading: false,
            status: "Ready".to_string(),
            slots: Self::restore_slots(&settings),
            next_slot_id: settings.slot_layout.len().max(2),
            focused_slot: 0,
            tab: Tab::Chat,
            editor: EditorPanel::new(),
            models_panel: ModelsPanel::new(),
            history: HistoryPanel::new(),
            settings_panel: SettingsPanel::new(),
            settings,
            storage,
            neural_panel: NeuralVizPanel::new(),
            network: Self::load_network(),
            mem,
            mem_checked: Instant::now(),
            report,
        };
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

    /// Try to assign a model to a slot. Blocked when it would break the reserve.
    fn try_assign_model(&mut self, idx: usize, name: String) {
        if self.slots.get(idx).is_none() {
            return;
        }
        let known = self.known_sizes();
        let new_size = ResourceGuard::size_for_model(&name, &known);
        let others = self.current_sizes(Some(idx));
        match ResourceGuard::can_fit(&self.mem, &others, new_size) {
            Ok(()) => {
                if let Some(slot) = self.slots.get_mut(idx) {
                    slot.model = Some(name.clone());
                }
                self.status = format!("Slot {} now runs {}", idx + 1, name);
                let sizes = self.current_sizes(None);
                self.report = ResourceGuard::evaluate(&self.mem, &sizes);
            }
            Err(e) => {
                self.status = format!("Blocked: {}", e);
            }
        }
    }

    /// Try to add a new empty slot. Blocked when even an average model would
    /// exceed the budget — the "+" is gated by real headroom, so a weak box
    /// stays at 1 slot while a big server can grow toward dozens or more.
    fn try_add_slot(&mut self) {
        let sizes = self.current_sizes(None);
        match ResourceGuard::can_fit(&self.mem, &sizes, ESTIMATED_MODEL_BYTES) {
            Ok(()) => {
                let id = self.next_slot_id;
                self.next_slot_id += 1;
                self.slots.push(ModelSlot::new(id, ModelRole::General));
                self.focused_slot = self.slots.len() - 1;
                self.status = format!("Slot {} added", self.slots.len());
            }
            Err(e) => {
                self.status = format!("Cannot add slot: {}", e);
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
            next_state: ctx.into(),
            done: true,
        });
        self.network.update_performance(reward);
        if self.network.experience_buffer.len() >= 4 {
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
        let layout: Vec<SlotConfig> = self
            .slots
            .iter()
            .map(|s| SlotConfig {
                model: s.model.clone(),
                role: s.role.label(),
                custom_role: s.custom_role.clone(),
            })
            .collect();
        if layout != self.settings.slot_layout {
            self.settings.slot_layout = layout;
            if let Some(st) = self.storage.as_ref() {
                let _ = st.save_settings(&self.settings);
            }
        }
    }

    // ---------- model loading ----------

    fn refresh_models(&mut self) {
        self.models_loading = true;
        let tx = self.tx.clone();
        let api = self.api_client.clone();
        let cli = self.cli_client.clone();
        self.rt.spawn(async move {
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
            let _ = tx.send(AppMessage::ModelsLoaded(result));
        });
    }

    fn poll_messages(&mut self) {
        while let Ok(msg) = self.rx.try_recv() {
            match msg {
                AppMessage::Broadcast(prompt) => {
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
                            compose_system_prompt(&persona, &memory, s)
                        };
                        if let Some(slot) = self.slots.get_mut(idx) {
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
                    self.status = if sent == 0 {
                        "Ask all: no slot answered (need models, idle chats)".to_string()
                    } else {
                        format!("Asked {} slot{}", sent, if sent == 1 { "" } else { "s" })
                    };
                }
                AppMessage::ChatChunk(idx, seq, piece) => {
                    if let Some(slot) = self.slots.get_mut(idx) {
                        slot.chat.push_chunk(seq, &piece);
                    }
                }
                AppMessage::ChatResponse(idx, seq, res) => {
                    if self.slots.get(idx).map(|s| s.chat.stream_seq()) != Some(seq) {
                        continue; // stale: stopped or superseded by a newer send
                    }
                    self.inflight.remove(&idx);
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
                AppMessage::ChatToEditor(code, lang) => {
                    self.editor.set_code_from_chat(code, lang);
                    self.tab = Tab::Editor;
                    self.status = "Code moved to Editor — pick a file and Save".to_string();
                }
                AppMessage::ModelsLoaded(res) => {
                    self.models_loading = false;
                    match res {
                        Ok(m) => {
                            self.status = format!("{} models loaded", m.len());
                            self.models = m;
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
                            self.status = format!("Model refresh failed: {}", e);
                        }
                    }
                }
                AppMessage::ModelSelected(name) => {
                    let f = self.focused_slot.min(self.slots.len().saturating_sub(1));
                    self.try_assign_model(f, name);
                }
                AppMessage::ModelPulled(res) => {
                    match res {
                        Ok(n) => {
                            self.status = format!("Pulled {}", n);
                            self.models_panel.note_pull_finished(true);
                        }
                        Err(e) => {
                            self.status = format!("Pull failed: {}", e);
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
                            for slot in &mut self.slots {
                                if slot.model.as_deref() == Some(&n) {
                                    slot.model = None;
                                }
                            }
                        }
                        Err(e) => self.status = format!("Delete failed: {}", e),
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
                }
                AppMessage::NewChat => {
                    let f = self.focused_slot.min(self.slots.len().saturating_sub(1));
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
            }
        }
    }

    // ---------- UI ----------

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
            ui.label(
                egui::RichText::new(self.status.clone())
                    .size(12.0)
                    .color(egui::Color32::from_rgb(0xaa, 0xaa, 0xaa)),
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .small_button("A+")
                    .on_hover_text("Zoom in (saved to Settings → Font size)")
                    .clicked()
                {
                    self.settings.font_size = (self.settings.font_size + 1.0).clamp(10.0, 32.0);
                    if let Some(st) = self.storage.as_ref() {
                        let _ = st.save_settings(&self.settings);
                    }
                }
                ui.label(
                    egui::RichText::new(format!(
                        "{:.0}%",
                        (self.settings.font_size / 14.0).clamp(0.5, 1.75) * 100.0
                    ))
                    .size(12.0)
                    .color(egui::Color32::from_rgb(0x88, 0x88, 0x88)),
                );
                if ui
                    .small_button("A−")
                    .on_hover_text("Zoom out (saved to Settings → Font size)")
                    .clicked()
                {
                    self.settings.font_size = (self.settings.font_size - 1.0).clamp(10.0, 32.0);
                    if let Some(st) = self.storage.as_ref() {
                        let _ = st.save_settings(&self.settings);
                    }
                }
            });
        });
    }

    fn show_tabs(&mut self, ui: &mut egui::Ui) {
        ui.add_space(8.0);
        for tab in Tab::all() {
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

        // Apply pending actions.
        if let Some((i, name)) = pending_assign {
            self.try_assign_model(i, name);
            self.focused_slot = i.min(self.slots.len().saturating_sub(1));
        }
        if let Some(i) = pending_unassign {
            if let Some(slot) = self.slots.get_mut(i) {
                slot.model = None;
                self.status = format!("Slot {} cleared", i + 1);
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
            }
        }
        if add_pressed {
            self.try_add_slot();
        }

        ui.add_space(8.0);
        ui.separator();
        ui.add_space(8.0);

        // Focused slot chat. One shared identity for every model:
        // persona (who I am) + memory (what I remember) + slot role (job).
        let f = self.focused_slot.min(self.slots.len().saturating_sub(1));
        let role_prompt = compose_system_prompt(
            &self.settings.persona,
            &self.settings.memory,
            &self.slots[f],
        );
        let slot_model = self.slots[f].model.clone();
        let slot_role = self.slots[f].role.label();
        ui.horizontal(|ui| {
            ui.label(
                egui::RichText::new(format!(
                    "Chatting with {} as {}",
                    slot_model.clone().unwrap_or("(no model — pick one above)".to_string()),
                    slot_role,
                ))
                .size(14.0)
                .color(egui::Color32::from_rgb(0x00, 0xaa, 0xff)),
            );
        });
        ui.add_space(4.0);
        if let Some(slot) = self.slots.get_mut(f) {
            slot.chat.show(
                ui,
                &models,
                &slot_model,
                &role_prompt,
                &self.api_client,
                f,
                &self.tx,
                &self.rt,
            );
        }
    }
}

impl eframe::App for AiDashboardApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        // Apply the Settings-tab font size as a global zoom factor so the UI
        // fits small screens (e.g. MacBook Air) and large monitors alike.
        // 14pt == 100%. Takes effect from the next frame.
        let zoom = (self.settings.font_size / 14.0).clamp(0.5, 1.75);
        if (ui.ctx().zoom_factor() - zoom).abs() > 0.001 {
            ui.ctx().set_zoom_factor(zoom);
        }
        // Apply the Settings-tab theme choice (Dark/Light); System falls back to dark.
        if self.settings.theme != self.last_theme {
            self.last_theme = self.settings.theme.clone();
            ui.ctx().set_visuals(match self.last_theme {
                Theme::Dark | Theme::System => egui::Visuals::dark(),
                Theme::Light => egui::Visuals::light(),
            });
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
                Tab::Models => {
                    self.models_panel.show(
                        ui,
                        &mut self.models,
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
                    self.neural_panel.show(ui, &self.network);
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

        // Keep the resource meter and spinners live.
        ui.ctx().request_repaint_after(std::time::Duration::from_millis(500));
    }
}
