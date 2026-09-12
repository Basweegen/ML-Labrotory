//! Stack-chan style reactive face for the chat UI (pure egui, no assets).
//!
//! Two call sites (the "two"):
//!   1. big face above the focused-slot chat (reactive: thinking/talking/happy)
//!   2. mini face on each slot card (streaming vs idle at a glance)

use eframe::egui;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaceMood {
    Idle,
    Thinking,
    Talking,
    Happy,
    Sad,
    Sleepy,
}

/// Pick a mood from cheap chat state. Callers already have these values;
/// no ChatPanel import here so slot cards and the focused header share it.
///
/// - streaming with no text yet = Thinking, with text = Talking
/// - fresh system note (blocks, errors) = Sad
/// - fresh assistant reply = Happy
/// - nothing for 5+ min = Sleepy
pub fn mood_for(
    streaming: bool,
    stream_len: usize,
    last_role: Option<&str>,
    secs_since_last: Option<i64>,
    secs_since_assistant: Option<i64>,
) -> FaceMood {
    if streaming && stream_len == 0 {
        return FaceMood::Thinking;
    }
    if streaming {
        return FaceMood::Talking;
    }
    let fresh = |s: Option<i64>| s.map(|v| v >= 0 && v < 8).unwrap_or(false);
    if last_role == Some("system") && fresh(secs_since_last) {
        return FaceMood::Sad;
    }
    if fresh(secs_since_assistant) {
        return FaceMood::Happy;
    }
    if secs_since_last.map(|v| v > 300).unwrap_or(false) {
        return FaceMood::Sleepy;
    }
    FaceMood::Idle
}

/// Draw the face in a `size x size` square. Animates via ctx time:
/// blink every ~3.4s, eyes follow the pointer, mouth moves when talking.
pub fn show_face(ui: &mut egui::Ui, size: f32, mood: FaceMood) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(size, size), egui::Sense::hover());
    let p = ui.painter_at(rect);
    let t = ui.ctx().input(|i| i.time);

    // Face plate.
    let plate = egui::Color32::from_rgb(0xE8, 0xEC, 0xF4);
    let stroke_col = match mood {
        FaceMood::Idle => egui::Color32::from_rgb(0x00, 0xAA, 0xFF),
        FaceMood::Thinking => egui::Color32::from_rgb(0xBB, 0x88, 0xFF),
        FaceMood::Talking => egui::Color32::from_rgb(0x00, 0xCC, 0x88),
        FaceMood::Happy => egui::Color32::from_rgb(0xFF, 0xCC, 0x33),
        FaceMood::Sad => egui::Color32::from_rgb(0xFF, 0x66, 0x66),
        FaceMood::Sleepy => egui::Color32::from_rgb(0x88, 0x88, 0xAA),
    };
    p.rect(
        rect,
        egui::CornerRadius::same((size * 0.22) as u8),
        plate,
        egui::Stroke::new(2.0, stroke_col),
        egui::StrokeKind::Inside,
    );

    // Eye look offset: follow pointer, clamped; Thinking looks up.
    let mut look = ui
        .ctx()
        .pointer_latest_pos()
        .map(|pos| pos - rect.center())
        .unwrap_or_default();
    if mood == FaceMood::Thinking {
        look = egui::vec2(look.x.clamp(-3.0, 3.0), -4.0);
    } else {
        let max = (size * 0.05).clamp(2.0, 5.0);
        look = egui::vec2(look.x.clamp(-max, max), look.y.clamp(-max, max));
    }

    // Blink: shut every ~3.4s for 0.12s (happy has ^^ eyes, sleepy stays shut).
    let phase = t % 3.4;
    let blinking =
        mood == FaceMood::Sleepy || (mood != FaceMood::Happy && phase < 0.12);

    let ink = egui::Color32::from_rgb(0x1A, 0x1A, 0x22);
    let cx = rect.center().x;
    let cy = rect.center().y - size * 0.06;
    let ex = size * 0.20;
    let ew = size * 0.13;
    let eh_full = size * 0.17;

    for side in [-1.0, 1.0] {
        let c = egui::pos2(cx + side * ex + look.x, cy + look.y);
        if mood == FaceMood::Happy {
            // ^^ eyes: two short arcs (polyline peaks).
            let r = ew * 0.9;
            let pts = (0..=8)
                .map(|k| {
                    let a = std::f64::consts::PI * (k as f32 / 8.0) as f64;
                    egui::pos2(
                        c.x - r + 2.0 * r * (k as f32 / 8.0),
                        c.y - (a.sin() as f32) * r * 0.9,
                    )
                })
                .collect::<Vec<_>>();
            p.add(egui::Shape::line(
                pts,
                egui::Stroke::new((size * 0.035).clamp(1.5, 3.0), ink),
            ));
        } else if blinking {
            // Shut eye: horizontal line.
            p.line_segment(
                [egui::pos2(c.x - ew * 0.8, c.y), egui::pos2(c.x + ew * 0.8, c.y)],
                egui::Stroke::new((size * 0.035).clamp(1.5, 3.0), ink),
            );
        } else {
            p.add(egui::Shape::ellipse_filled(c, egui::vec2(ew, eh_full), ink));
            // Catchlight.
            p.circle(
                egui::pos2(c.x - ew * 0.25, cy + look.y - eh_full * 0.3),
                (size * 0.022).clamp(1.0, 2.6),
                egui::Color32::WHITE,
                egui::Stroke::NONE,
            );
        }
    }

    // Blush (skip on tiny minis so they stay crisp).
    if size >= 36.0 {
        let blush = egui::Color32::from_rgb(0xFF, 0xB3, 0xC1);
        for side in [-1.0, 1.0] {
            p.add(egui::Shape::ellipse_filled(
                egui::pos2(cx + side * size * 0.30, cy + size * 0.16),
                egui::vec2(size * 0.055, size * 0.035),
                blush,
            ));
        }
    }

    // Mouth.
    let mc = egui::pos2(rect.center().x, rect.center().y + size * 0.22);
    let mstroke = egui::Stroke::new((size * 0.032).clamp(1.2, 2.6), ink);
    match mood {
        FaceMood::Talking => {
            // Chatter: height oscillates with time.
            let h = size * 0.055 + size * 0.045 * (0.5 + 0.5 * (t * 11.0).sin()) as f32;
            let w = size * 0.075;
            p.add(egui::Shape::ellipse_filled(mc, egui::vec2(w, h.max(2.0)), ink));
        }
        FaceMood::Happy => {
            // Open smile (D shape): filled half-disc.
            let r = size * 0.11;
            let pts = (0..=12)
                .map(|k| {
                    let a = std::f64::consts::PI * (k as f32 / 12.0) as f64;
                    egui::pos2(
                        mc.x - r + 2.0 * r * (k as f32 / 12.0),
                        mc.y - r * 0.35 + (a.sin() as f32) * r,
                    )
                })
                .collect::<Vec<_>>();
            p.add(egui::Shape::convex_polygon(
                pts,
                ink,
                egui::Stroke::NONE,
            ));
        }
        FaceMood::Thinking => {
            // Small "o".
            p.circle(mc, (size * 0.035).clamp(1.2, 3.2), ink, egui::Stroke::NONE);
        }
        FaceMood::Idle => {
            // Small calm smile.
            let w = size * 0.07;
            p.add(egui::Shape::line(
                vec![
                    egui::pos2(mc.x - w, mc.y - w * 0.35),
                    egui::pos2(mc.x, mc.y + w * 0.25),
                    egui::pos2(mc.x + w, mc.y - w * 0.35),
                ],
                mstroke,
            ));
        }
        FaceMood::Sad => {
            // Frown.
            let w = size * 0.07;
            p.add(egui::Shape::line(
                vec![
                    egui::pos2(mc.x - w, mc.y + w * 0.45),
                    egui::pos2(mc.x, mc.y - w * 0.15),
                    egui::pos2(mc.x + w, mc.y + w * 0.45),
                ],
                mstroke,
            ));
        }
        FaceMood::Sleepy => {
            // Tiny flat mouth + "z".
            let w = size * 0.05;
            p.line_segment(
                [egui::pos2(mc.x - w, mc.y), egui::pos2(mc.x + w, mc.y)],
                mstroke,
            );
            if size >= 44.0 {
                p.text(
                    egui::pos2(rect.right() - size * 0.10, rect.top() + size * 0.16),
                    egui::Align2::CENTER_CENTER,
                    "z",
                    egui::FontId::proportional(size * 0.22),
                    egui::Color32::from_rgb(0x88, 0x88, 0xAA),
                );
            }
        }
    }

    // Keep animating while visible: talking/thinking need every frame,
    // idle only needs a wakeup for the next blink.
    if mood == FaceMood::Talking || mood == FaceMood::Thinking {
        ui.ctx().request_repaint();
    } else {
        ui.ctx().request_repaint_after(std::time::Duration::from_millis(400));
    }
}
