mod ollama;
mod storage;
mod ui;
mod voice;
mod neural;
mod resources;

use eframe::egui;
use ui::app::AiDashboardApp;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1150.0, 750.0])
            .with_min_inner_size([760.0, 540.0]),
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
