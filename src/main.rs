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

use eframe::egui;
use ui::app::AiDashboardApp;

fn main() -> eframe::Result<()> {
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
