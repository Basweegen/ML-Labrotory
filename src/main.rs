// Copyright 2026 Sean M. Stow. All rights reserved.
mod ollama;
mod storage;
mod ui;
mod voice;
mod neural;
mod resources;
mod security;
pub mod commands;
pub mod workspace;
pub mod guardrails;
pub mod tools;
pub mod swarm;

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
    let args: Vec<String> = std::env::args().skip(1).collect();
    if !args.is_empty() {
        let cli_args: Vec<String> = if args[0].eq_ignore_ascii_case("cli") {
            args[1..].to_vec()
        } else {
            args
        };
        if !cli_args.is_empty() {
            match commands::SlashCommand::parse_cli_args(&cli_args) {
                Ok(Some(cmd)) => {
                    let rt = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .expect("Failed to initialize CLI runtime");
                    if let Err(e) = rt.block_on(commands::run_headless_cli(cmd)) {
                        eprintln!("[ML Laboratory CLI Error] {}", e);
                        std::process::exit(1);
                    }
                    std::process::exit(0);
                }
                Ok(None) => {}
                Err(err_msg) => {
                    eprintln!("{}", err_msg);
                    std::process::exit(1);
                }
            }
        }
    }

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
