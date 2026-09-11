mod ollama;
mod storage;
mod ui;
mod voice;
mod workspace;
mod neural;
mod resources;
mod security;

use eframe::egui;
use ui::app::AiDashboardApp;

/// First thing at boot: a panic hook that appends every crash to
/// crashes.log next to the database. A GUI that "just closes" with no
/// evidence is undebuggable; this leaves a timestamped line per death.
fn install_crash_log() {
    let path = dirs::data_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join("ai-dashboard")
        .join("crashes.log");
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let line = format!(
            "{} panic: {}\n",
            chrono::Local::now().format("%Y-%m-%d %H:%M:%S"),
            info
        );
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
        {
            use std::io::Write;
            let _ = f.write_all(line.as_bytes());
        }
        prev(info);
    }));
}

fn main() -> eframe::Result<()> {
    install_crash_log();
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([960.0, 600.0])
            .with_min_inner_size([640.0, 480.0]),
        ..Default::default()
    };

    eframe::run_native(
        "ML Laboratory",
        options,
        Box::new(|cc| {
            Ok(Box::new(AiDashboardApp::new(cc)) as Box<dyn eframe::App>)
        }),
    )
}
