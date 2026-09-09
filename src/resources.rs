use std::collections::HashMap;

/// Fraction of total system RAM always reserved for the OS.
/// Model allocation never exceeds (1 - reserve) of total RAM.
pub const SYSTEM_RESERVE_RATIO: f64 = 0.15;

/// Fallback size estimate for a model whose size is unknown (e.g. not yet pulled).
pub const ESTIMATED_MODEL_BYTES: u64 = 4 * 1024 * 1024 * 1024;

#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub total_bytes: u64,
    pub available_bytes: u64,
}

impl MemoryStats {
    pub fn used_bytes(&self) -> u64 {
        self.total_bytes.saturating_sub(self.available_bytes)
    }

    /// Budget available for AI models: total minus the 15% system reserve.
    pub fn model_budget_bytes(&self) -> u64 {
        ((self.total_bytes as f64) * (1.0 - SYSTEM_RESERVE_RATIO)) as u64
    }

    pub fn reserve_bytes(&self) -> u64 {
        ((self.total_bytes as f64) * SYSTEM_RESERVE_RATIO) as u64
    }
}

/// Read system memory. Linux: parse /proc/meminfo. Elsewhere: 8 GiB fallback.
pub fn system_memory() -> MemoryStats {
    #[cfg(target_os = "linux")]
    {
        if let Ok(text) = std::fs::read_to_string("/proc/meminfo") {
            let mut kb_total = 0u64;
            let mut kb_avail = 0u64;
            for line in text.lines() {
                let mut parts = line.split_whitespace();
                let key = parts.next().unwrap_or("");
                let val: u64 = parts.next().and_then(|v| v.parse().ok()).unwrap_or(0);
                match key {
                    "MemTotal:" => kb_total = val,
                    "MemAvailable:" => kb_avail = val,
                    _ => {}
                }
            }
            if kb_total > 0 {
                return MemoryStats {
                    total_bytes: kb_total * 1024,
                    available_bytes: kb_avail * 1024,
                };
            }
        }
    }
    MemoryStats {
        total_bytes: 8 * 1024 * 1024 * 1024,
        available_bytes: 4 * 1024 * 1024 * 1024,
    }
}

#[derive(Debug, Clone)]
pub struct ResourceReport {
    pub reserve_bytes: u64,
    pub budget_bytes: u64,
    pub system_used_bytes: u64,
    pub models_used_bytes: u64,
    pub free_for_models_bytes: u64,
    pub usage_fraction: f32,
    pub over_budget: bool,
    pub max_models_fit: usize,
}

pub struct ResourceGuard;

impl ResourceGuard {
    /// Evaluate current allocation against the budget.
    pub fn evaluate(stats: &MemoryStats, assigned_sizes: &[u64]) -> ResourceReport {
        let budget = stats.model_budget_bytes();
        let system_used = stats.used_bytes();
        let models_used: u64 = assigned_sizes.iter().sum();
        let committed = system_used.saturating_add(models_used);
        let over_budget = committed > budget;
        let free = budget.saturating_sub(committed);
        let usage_fraction = if budget == 0 {
            1.0
        } else {
            (committed as f32 / budget as f32).clamp(0.0, 1.0)
        };
        let avg: u64 = if assigned_sizes.is_empty() {
            ESTIMATED_MODEL_BYTES
        } else {
            (models_used / assigned_sizes.len() as u64).max(1)
        };
        let max_fit = (free / avg) as usize;
        ResourceReport {
            reserve_bytes: stats.reserve_bytes(),
            budget_bytes: budget,
            system_used_bytes: system_used,
            models_used_bytes: models_used,
            free_for_models_bytes: free,
            usage_fraction,
            over_budget,
            max_models_fit: max_fit,
        }
    }

    /// Can a new model of `new_size` be added given current assignments?
    pub fn can_fit(
        stats: &MemoryStats,
        assigned_sizes: &[u64],
        new_size: u64,
    ) -> Result<(), String> {
        let mut with_new = assigned_sizes.to_vec();
        with_new.push(new_size);
        let report = Self::evaluate(stats, &with_new);
        if report.over_budget {
            Err(format!(
                "Not enough RAM: this model plus system use exceeds the budget (15% of RAM reserved for OS). Free for models: {}.",
                format_bytes(report.free_for_models_bytes),
            ))
        } else {
            Ok(())
        }
    }

    /// Resolve a model's size from the Ollama list; fall back to estimate.
    pub fn size_for_model(name: &str, known: &HashMap<String, u64>) -> u64 {
        match known.get(name).copied().unwrap_or(ESTIMATED_MODEL_BYTES) {
            // Never treat a model as free: unknown or zero sizes estimate high.
            0 => ESTIMATED_MODEL_BYTES,
            n => n,
        }
    }
}

pub fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit = 0;
    while size >= 1024.0 && unit < UNITS.len() - 1 {
        size /= 1024.0;
        unit += 1;
    }
    format!("{:.1} {}", size, UNITS[unit])
}

/// Hardware tier for this box, from free-for-models budget. Shown in Models tab.
pub fn hardware_tier(free_for_models_bytes: u64) -> &'static str {
    const GB: u64 = 1024 * 1024 * 1024;
    if free_for_models_bytes < 2 * GB {
        "LOW: stick to ~1B models"
    } else if free_for_models_bytes < 4 * GB {
        "MEDIUM: 1-3B models fit"
    } else if free_for_models_bytes < 8 * GB {
        "GOOD: 3-4B models fit"
    } else {
        "HIGH: 4B+ models fit"
    }
}

/// Per-model fit badge for the Models list. Unknown/zero sizes estimate high.
pub fn fit_label(model_size: u64, free_for_models_bytes: u64) -> (&'static str, (u8, u8, u8)) {
    let size = if model_size == 0 { ESTIMATED_MODEL_BYTES } else { model_size };
    if size <= free_for_models_bytes {
        ("Fits", (0x88, 0xcc, 0x88))
    } else if size <= free_for_models_bytes.saturating_add(free_for_models_bytes / 2) {
        ("Tight", (0xcc, 0xaa, 0x44))
    } else {
        ("Skip", (0xcc, 0x66, 0x66))
    }
}
