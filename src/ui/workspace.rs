//! Files tab: browse, create, delete, rename, read, write and edit
//! project files under the configured root. Everything mutating is
//! audit-logged; deletes of whole directory trees need a confirm click.

use eframe::egui;
use std::sync::mpsc;

use crate::storage::{AppSettings, Storage};
use crate::ui::app::AppMessage;
use crate::workspace::{join_rel, DirEntry, Workspace};

pub struct WorkspacePanel {
    root_input: String,
    cur: String,
    entries: Vec<DirEntry>,
    loaded: bool,
    selected: Option<String>,
    buffer: String,
    dirty: bool,
    new_name: String,
    rename_to: String,
    delete_armed: Option<String>,
    status: String,
}

impl WorkspacePanel {
    pub fn new() -> Self {
        Self {
            root_input: String::new(),
            cur: String::new(),
            entries: Vec::new(),
            loaded: false,
            selected: None,
            buffer: String::new(),
            dirty: false,
            new_name: String::new(),
            rename_to: String::new(),
            delete_armed: None,
            status: String::new(),
        }
    }

    fn audit(&self, tx: &mpsc::Sender<AppMessage>, kind: &str, detail: String) {
        let _ = tx.send(AppMessage::Audit(kind.to_string(), detail));
    }

    fn refresh(&mut self, ws: &Workspace) {
        match ws.list(&self.cur) {
            Ok(e) => {
                self.entries = e;
                self.loaded = true;
            }
            Err(e) => {
                self.status = format!("List failed: {e:#}");
                self.entries.clear();
                self.loaded = true;
            }
        }
    }

    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        settings: &mut AppSettings,
        storage: &Storage,
        tx: &mpsc::Sender<AppMessage>,
    ) {
        ui.horizontal(|ui| {
            ui.add_space(4.0);
            ui.heading(
                egui::RichText::new("🗂 Project files")
                    .size(22.0)
                    .color(egui::Color32::from_rgb(0x00, 0xaa, 0xff)),
            );
        });
        ui.add_space(4.0);

        // ---- project root ----
        if self.root_input.is_empty() && !settings.workspace_root.is_empty() {
            self.root_input = settings.workspace_root.clone();
        }
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Project root:").size(13.0));
            ui.add(
                egui::TextEdit::singleline(&mut self.root_input)
                    .desired_width(360.0)
                    .font(egui::TextStyle::Monospace)
                    .hint_text("/home/you/projects/my-app"),
            );
            if ui.small_button("Set").clicked() {
                let root = self.root_input.trim().to_string();
                match Workspace::new(&root) {
                    Ok(_) => {
                        settings.workspace_root = root.clone();
                        let saved = storage.save_settings(settings).is_ok();
                        self.cur.clear();
                        self.selected = None;
                        self.buffer.clear();
                        self.dirty = false;
                        self.loaded = false;
                        self.status = if saved {
                            format!("Root set: {root}")
                        } else {
                            "Root set (settings save failed)".to_string()
                        };
                        self.audit(tx, "workspace.root", root);
                    }
                    Err(e) => {
                        self.status = format!("Not a directory: {e:#}");
                    }
                }
            }
        });
        ui.add_space(4.0);
        ui.separator();
        ui.add_space(4.0);

        let ws = match Workspace::new(&settings.workspace_root) {
            Ok(w) => w,
            Err(_) => {
                ui.label(
                    egui::RichText::new(
                        "Pick a project folder above. Everything below stays inside it: \
                         browse, create, edit, rename and delete files - the dashboard \
                         can scaffold whole projects here.",
                    )
                    .size(13.0)
                    .color(egui::Color32::from_rgb(0x99, 0x99, 0x99)),
                );
                if !self.status.is_empty() {
                    ui.add_space(4.0);
                    ui.label(
                        egui::RichText::new(self.status.clone())
                            .size(12.0)
                            .color(egui::Color32::from_rgb(0xcc, 0x66, 0x66)),
                    );
                }
                return;
            }
        };
        if !self.loaded {
            self.refresh(&ws);
        }

        // ---- breadcrumb ----
        ui.horizontal(|ui| {
            if ui.small_button("\u{2190} Up").clicked() && !self.cur.is_empty() {
                if let Some(pos) = self.cur.rfind('/') {
                    self.cur.truncate(pos);
                } else {
                    self.cur.clear();
                }
                self.loaded = false;
            }
            let here = if self.cur.is_empty() {
                ws.root().display().to_string()
            } else {
                format!("{}/{}", ws.root().display(), self.cur)
            };
            ui.label(egui::RichText::new(here).size(13.0).monospace());
        });
        ui.add_space(4.0);

        // ---- entries ----
        egui::ScrollArea::vertical()
            .id_salt("ws_entries")
            .max_height(180.0)
            .show(ui, |ui| {
                if self.entries.is_empty() {
                    ui.label(
                        egui::RichText::new("(empty folder)")
                            .size(12.0)
                            .color(egui::Color32::from_rgb(0x88, 0x88, 0x88)),
                    );
                }
                for e in self.entries.clone() {
                    let label = if e.is_dir {
                        format!("📁 {}/", e.name)
                    } else {
                        format!("📄 {}  ({} B)", e.name, e.size)
                    };
                    let sel = self.selected.as_deref() == Some(join_rel(&self.cur, &e.name).as_str())
                        || self.selected.is_none() && false;
                    if ui.selectable_label(sel, label).clicked() {
                        let rel = join_rel(&self.cur, &e.name);
                        if e.is_dir {
                            self.cur = rel;
                            self.loaded = false;
                        } else if self.dirty {
                            self.status = "Unsaved edits - Save or Revert first".to_string();
                        } else {
                            match ws.read(&rel) {
                                Ok(text) => {
                                    self.selected = Some(rel.clone());
                                    self.buffer = text;
                                    self.dirty = false;
                                    self.rename_to.clear();
                                    self.delete_armed = None;
                                    self.status = format!("Opened {rel}");
                                }
                                Err(err) => self.status = format!("Open failed: {err:#}"),
                            }
                        }
                    }
                }
            });
        ui.add_space(4.0);
        ui.separator();
        ui.add_space(4.0);

        // ---- editor ----
        if let Some(rel) = self.selected.clone() {
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new(format!("Editing {rel}"))
                        .size(13.0)
                        .monospace()
                        .color(egui::Color32::from_rgb(0x00, 0xaa, 0xff)),
                );
                if self.dirty {
                    ui.label(
                        egui::RichText::new("\u{25CF} unsaved")
                            .size(12.0)
                            .color(egui::Color32::from_rgb(0xcc, 0xaa, 0x44)),
                    );
                }
            });
            if ui
                .add(
                    egui::TextEdit::multiline(&mut self.buffer)
                        .desired_rows(10)
                        .desired_width(f32::INFINITY)
                        .font(egui::TextStyle::Monospace),
                )
                .changed()
            {
                self.dirty = true;
            }
            ui.horizontal(|ui| {
                if ui.small_button("Save").clicked() {
                    match ws.write(&rel, &self.buffer) {
                        Ok(()) => {
                            self.dirty = false;
                            self.status = format!("Saved {rel}");
                            self.audit(tx, "workspace.write", rel.clone());
                            self.loaded = false;
                        }
                        Err(e) => self.status = format!("Save failed: {e:#}"),
                    }
                }
                if ui.small_button("Revert").clicked() {
                    match ws.read(&rel) {
                        Ok(text) => {
                            self.buffer = text;
                            self.dirty = false;
                            self.status = "Reverted".to_string();
                        }
                        Err(e) => self.status = format!("Revert failed: {e:#}"),
                    }
                }
                ui.label(
                    egui::RichText::new(format!("{} chars", self.buffer.chars().count()))
                        .size(11.0)
                        .color(egui::Color32::from_rgb(0x88, 0x88, 0x88)),
                );
            });
            ui.add_space(4.0);
            ui.separator();
            ui.add_space(4.0);
        }

        // ---- create ----
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("New:").size(13.0));
            ui.add(
                egui::TextEdit::singleline(&mut self.new_name)
                    .desired_width(220.0)
                    .font(egui::TextStyle::Monospace)
                    .hint_text("name.py or sub/dir"),
            );
            if ui.small_button("File").clicked() && !self.new_name.trim().is_empty() {
                let rel = join_rel(&self.cur, self.new_name.trim());
                match ws.write(&rel, "") {
                    Ok(()) => {
                        self.status = format!("Created {rel}");
                        self.audit(tx, "workspace.create", rel);
                        self.new_name.clear();
                        self.loaded = false;
                    }
                    Err(e) => self.status = format!("Create failed: {e:#}"),
                }
            }
            if ui.small_button("Folder").clicked() && !self.new_name.trim().is_empty() {
                let rel = join_rel(&self.cur, self.new_name.trim());
                match ws.create_dir(&rel) {
                    Ok(()) => {
                        self.status = format!("Created {rel}/");
                        self.audit(tx, "workspace.mkdir", rel);
                        self.new_name.clear();
                        self.loaded = false;
                    }
                    Err(e) => self.status = format!("Create failed: {e:#}"),
                }
            }
        });
        ui.add_space(4.0);

        // ---- rename / delete selected ----
        if let Some(rel) = self.selected.clone() {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("Rename to:").size(13.0));
                ui.add(
                    egui::TextEdit::singleline(&mut self.rename_to)
                        .desired_width(220.0)
                        .font(egui::TextStyle::Monospace)
                        .hint_text("new/name.py"),
                );
                if ui.small_button("Rename").clicked() && !self.rename_to.trim().is_empty() {
                    let to = if self.rename_to.contains('/') {
                        self.rename_to.trim().to_string()
                    } else {
                        join_rel(&self.cur, self.rename_to.trim())
                    };
                    match ws.rename(&rel, &to) {
                        Ok(()) => {
                            self.status = format!("Renamed to {to}");
                            self.audit(tx, "workspace.rename", format!("{rel} -> {to}"));
                            self.selected = Some(to);
                            self.rename_to.clear();
                            self.loaded = false;
                        }
                        Err(e) => self.status = format!("Rename failed: {e:#}"),
                    }
                }
            });
            ui.horizontal(|ui| {
                if self.delete_armed.as_deref() == Some(&rel) {
                    if ui.small_button("Confirm delete (recursive for folders)").clicked() {
                        match ws.delete(&rel) {
                            Ok(()) => {
                                self.status = format!("Deleted {rel}");
                                self.audit(tx, "workspace.delete", rel.clone());
                                self.selected = None;
                                self.buffer.clear();
                                self.dirty = false;
                                self.delete_armed = None;
                                self.loaded = false;
                            }
                            Err(e) => self.status = format!("Delete failed: {e:#}"),
                        }
                    }
                } else if ui.small_button("Delete").clicked() {
                    self.delete_armed = Some(rel);
                }
            });
            ui.add_space(4.0);
        }

        if !self.status.is_empty() {
            ui.label(
                egui::RichText::new(self.status.clone())
                    .size(12.0)
                    .color(egui::Color32::from_rgb(0x99, 0x99, 0x99)),
            );
        }
    }
}
