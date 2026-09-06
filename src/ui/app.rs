use anyhow::Result;
use eframe::egui;
use std::collections::HashMap;
use std::sync::mpsc;
use std::time::Instant;
use tokio::runtime::Runtime;

use crate::neural::ModelProfileNetwork;
use crate::ollama::api::{ChatResponse, Model, OllamaClient};
use crate::ollama::cli::OllamaCli;
use crate::resources::{self, ESTIMATED_MODEL_BYTES, ResourceGuard, ResourceReport};
use crate::storage::{AppSettings, Storage, Theme};
use crate::ui::chat::ChatPanel;
use crate::ui::editor::EditorPanel;
use crate::ui::history::HistoryPanel;
use crate::ui::models::ModelsPanel;
use crate::ui::neural_viz::NeuralVizPanel;
use crate::ui::settings::SettingsPanel;

/// Messages sent from background tasks / panels to the app.
pub enum AppMessage {
    ChatResponse(usize, Result<ChatResponse>),
    EditorSuggestion(usize, Result<ChatResponse>),
    ModelsLoaded(Result<Vec<Model>>),
    RefreshModels,
    ModelSelected(String),
    ModelPulled(Result<String>),
    ModelDeleted(Result<String>),
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

/// One runnable model slot: a model assignment + a role + its own chat history.
pub struct ModelSlot {
    pub id: usize,
    pub model: Option<String>,
    pub role: ModelRole,
    pub custom_role: String,
    pub chat: ChatPanel,
}

impl ModelSlot {
    pub fn new(id: usize, role: ModelRole) -> Self {
        Self {
            id,
            model: None,
            role,
            custom_role: String::new(),
            chat: ChatPanel::new(),
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
    last_ollama_url: String,
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
        let api_client = OllamaClient::new(settings.ollama_url.clone()).ok();
        let cli_client = OllamaCli::new().ok();
        let mem = resources::system_memory();
        let report = ResourceGuard::evaluate(&mem, &[]);
        let mut app = Self {
            rt,
            tx,
            rx,
            api_client,
            cli_client,
            last_ollama_url: settings.ollama_url.clone(),
            last_theme: settings.theme.clone(),
            models: Vec::new(),
            models_loading: false,
            status: "Ready".to_string(),
            slots: vec![
                ModelSlot::new(0, ModelRole::General),
                ModelSlot::new(1, ModelRole::Coder),
            ],
            next_slot_id: 2,
            focused_slot: 0,
            tab: Tab::Chat,
            editor: EditorPanel::new(),
            models_panel: ModelsPanel::new(),
            history: HistoryPanel::new(),
            settings_panel: SettingsPanel::new(),
            settings,
            storage,
            neural_panel: NeuralVizPanel::new(),
            network: ModelProfileNetwork::new(8, vec![16, 16], 4),
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
                eprintln!("AUTOFILL slot={} model={}", idx + 1, name);
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
                AppMessage::ChatResponse(idx, res) => {
                    if let Some(slot) = self.slots.get_mut(idx) {
                        slot.chat.handle_response(res);
                    }
                }
                AppMessage::EditorSuggestion(id, res) => {
                    self.editor.handle_ai_suggestion(id, res);
                }
                AppMessage::ModelsLoaded(res) => {
                    self.models_loading = false;
                    match res {
                        Ok(m) => {
                            self.status = format!("{} models loaded", m.len());
                            self.models = m;
                            // Auto-fill empty slots with the smallest models
                            // that fit, so Chat works immediately (Send
                            // enables as soon as a slot has a model).
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
                        Ok(n) => self.status = format!("Pulled {}", n),
                        Err(e) => self.status = format!("Pull failed: {}", e),
                    }
                    self.models_panel.note_transfer_finished();
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
                    .desired_width(150.0)
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
                        "{} free · ~{} more models fit",
                        resources::format_bytes(self.report.free_for_models_bytes),
                        self.report.max_models_fit,
                    ))
                    .size(12.0)
                    .color(egui::Color32::from_rgb(0x88, 0xcc, 0x88)),
                );
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(
                    egui::RichText::new(self.status.clone())
                        .size(12.0)
                        .color(egui::Color32::from_rgb(0xaa, 0xaa, 0xaa)),
                );
                ui.separator();
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

        ui.horizontal_wrapped(|ui| {
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

        // Focused slot chat.
        let f = self.focused_slot.min(self.slots.len().saturating_sub(1));
        let role_prompt = self.slots[f].role_prompt();
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

        // Re-create the API client if the URL changed in settings.
        if self.settings.ollama_url != self.last_ollama_url {
            self.last_ollama_url = self.settings.ollama_url.clone();
            self.api_client = OllamaClient::new(self.settings.ollama_url.clone()).ok();
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
                    self.editor.show(
                        ui,
                        &models,
                        &model,
                        &self.api_client,
                        &self.tx,
                        &self.rt,
                    );
                }
                Tab::History => {
                    if let Some(st) = self.storage.as_ref() {
                        self.history.show(ui, st);
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

        // Keep the resource meter and spinners live.
        ui.ctx().request_repaint_after(std::time::Duration::from_millis(500));
    }
}
