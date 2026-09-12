// Copyright 2026 Sean M. Stow. All rights reserved.
use anyhow::Result;
use ndarray::{Array1, Array2};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

// Serializable wrappers for ndarray types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableArray2 {
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

impl From<Array2<f32>> for SerializableArray2 {
    fn from(arr: Array2<f32>) -> Self {
        let (rows, cols) = arr.dim();
        Self {
            data: arr.into_raw_vec_and_offset().0,
            rows,
            cols,
        }
    }
}

impl From<SerializableArray2> for Array2<f32> {
    fn from(s: SerializableArray2) -> Self {
        Array2::from_shape_vec((s.rows, s.cols), s.data)
            .unwrap_or_else(|_| Array2::zeros((s.rows, s.cols)))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableArray1 {
    pub data: Vec<f32>,
}

impl From<Array1<f32>> for SerializableArray1 {
    fn from(arr: Array1<f32>) -> Self {
        Self {
            data: arr.into_raw_vec_and_offset().0,
        }
    }
}

impl From<SerializableArray1> for Array1<f32> {
    fn from(s: SerializableArray1) -> Self {
        Array1::from_vec(s.data)
    }
}

/// Bio-Inspired Swarm Pheromone Stigmergy Matrix.
/// Models dynamic cohabitation and ecological niche specialization of multi-agent LLM slots.
/// Implements pheromone decay (stigmergic evaporation) and reinforcement deposits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmPheromoneMatrix {
    /// Number of task categories (e.g. 6: General, Coder, Researcher, Cyber/Critic, Planner, Writer)
    pub num_domains: usize,
    /// Number of cohabitating model slots (e.g. 8)
    pub num_slots: usize,
    /// Pheromone trail intensity matrix tau_{d, s} (domains x slots)
    pub pheromones: Vec<Vec<f32>>,
    /// Evaporation coefficient rho in (0, 1)
    pub evaporation_rate: f32,
    /// Pheromone sensitivity exponent alpha
    pub alpha: f32,
    /// Neural heuristic sensitivity exponent beta
    pub beta: f32,
}

impl Default for SwarmPheromoneMatrix {
    fn default() -> Self {
        Self::new(6, 8)
    }
}

impl SwarmPheromoneMatrix {
    pub fn new(num_domains: usize, num_slots: usize) -> Self {
        let d = num_domains.max(1);
        let s = num_slots.max(1);
        let pheromones = vec![vec![1.0; s]; d];
        Self {
            num_domains: d,
            num_slots: s,
            pheromones,
            evaporation_rate: 0.05,
            alpha: 1.0,
            beta: 1.5,
        }
    }

    /// Bio-inspired stigmergic evaporation: trails decay toward baseline 1.0 over time.
    pub fn evaporate(&mut self) {
        let rho = self.evaporation_rate.clamp(0.001, 0.5);
        for row in &mut self.pheromones {
            for trail in row {
                *trail = ((1.0 - rho) * *trail + rho * 1.0).clamp(0.1, 20.0);
            }
        }
    }

    /// Reinforcement deposit: successful completions strengthen the domain-slot trail.
    pub fn deposit(&mut self, domain: usize, slot: usize, reward: f32) {
        let d = domain % self.num_domains;
        let s = slot % self.num_slots;
        if reward > 0.0 {
            let deposit = (reward * 0.5).clamp(0.0, 5.0);
            self.pheromones[d][s] = (self.pheromones[d][s] + deposit).min(20.0);
        } else if reward < 0.0 {
            let penalty = (-reward * 0.2).clamp(0.0, 0.5);
            self.pheromones[d][s] = (self.pheromones[d][s] - penalty).max(0.1);
        }
    }

    /// Fuse stigmergic pheromone trail with neural heuristic desirability.
    /// Returns (best_slot, probability_distribution).
    pub fn fuse_decision(&self, domain: usize, neural_desirability: &[f32]) -> (usize, Vec<f32>) {
        let d = domain % self.num_domains;
        let n_slots = self.num_slots;
        let mut scores = Vec::with_capacity(n_slots);

        for s in 0..n_slots {
            let tau = if s < self.pheromones[d].len() { self.pheromones[d][s] } else { 1.0 };
            let eta = if s < neural_desirability.len() { neural_desirability[s].max(1e-4) } else { 1.0 / n_slots as f32 };
            let score = tau.powf(self.alpha) * eta.powf(self.beta);
            scores.push(if score.is_finite() && score > 0.0 { score } else { 1e-4 });
        }

        let total: f32 = scores.iter().sum();
        let probs: Vec<f32> = if total > 1e-6 {
            scores.into_iter().map(|v| v / total).collect()
        } else {
            vec![1.0 / n_slots as f32; n_slots]
        };

        let best_slot = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        (best_slot, probs)
    }
}

/// Parameterized Quantum-Inspired Superposition Layer.
/// Maps normalized classical input vectors into quantum state superposition amplitudes:
/// |psi_k> = cos(theta_k/2)|0> + e^(i*phi_k)*sin(theta_k/2)|1>
/// Computes quantum interference, measurement probabilities (Born's rule), and Von Neumann entropy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumStateLayer {
    pub num_qubits: usize,
    pub rotation_thetas: Vec<f32>,
    pub phase_phis: Vec<f32>,
    pub entanglement_couplings: Vec<f32>,
}

impl Default for QuantumStateLayer {
    fn default() -> Self {
        Self::new(8)
    }
}

impl QuantumStateLayer {
    pub fn new(num_qubits: usize) -> Self {
        let n = num_qubits.max(1);
        let mut rng = rand::thread_rng();
        let rotation_thetas: Vec<f32> = (0..n).map(|_| rng.gen_range(0.0..std::f32::consts::PI)).collect();
        let phase_phis: Vec<f32> = (0..n).map(|_| rng.gen_range(0.0..std::f32::consts::TAU)).collect();
        let entanglement_couplings: Vec<f32> = (0..n).map(|_| rng.gen_range(-0.5..0.5)).collect();
        Self {
            num_qubits: n,
            rotation_thetas,
            phase_phis,
            entanglement_couplings,
        }
    }

    /// Transform classical feature vector into quantum probability distribution & entropy.
    /// Maps each normalized feature x_k into state |psi_k> = cos(theta_k/2)|0> + e^(i phi_k) sin(theta_k/2)|1>.
    /// Applies nearest-neighbor quantum entanglement phase coupling.
    /// Computes Born's rule measurement probabilities P_k = |alpha_k|^2 + |beta_k'|^2.
    /// Calculates Von Neumann entropy S = - sum(p_k * ln(p_k)).
    pub fn transform(&self, input: &Array1<f32>) -> (Array1<f32>, f32) {
        let n = self.num_qubits;
        let mut probs = Vec::with_capacity(n);

        for k in 0..n {
            let x_k = if k < input.len() { input[k].clamp(0.0, 1.0) } else { 0.5 };
            let theta_param = if k < self.rotation_thetas.len() { self.rotation_thetas[k] } else { 0.0 };
            let phi_param = if k < self.phase_phis.len() { self.phase_phis[k] } else { 0.0 };
            let j_coupling = if k < self.entanglement_couplings.len() { self.entanglement_couplings[k] } else { 0.0 };

            // Angle of rotation around Y axis: theta = pi * x + param
            let theta = std::f32::consts::PI * x_k + theta_param;
            let alpha = (theta * 0.5).cos();
            let beta = (theta * 0.5).sin();

            // Next neighbor amplitude for entanglement coupling
            let next_k = (k + 1) % n;
            let next_x = if next_k < input.len() { input[next_k].clamp(0.0, 1.0) } else { 0.5 };
            let alpha_next = ((std::f32::consts::PI * next_x) * 0.5).cos();

            // Entangled phase rotation
            let phase = phi_param + j_coupling * alpha_next;
            let beta_entangled = beta * phase.cos();

            // Born's rule amplitude magnitude squared
            let p_raw = alpha * alpha + beta_entangled * beta_entangled;
            probs.push(if p_raw.is_finite() && p_raw > 0.0 { p_raw } else { 1e-4 });
        }

        // Normalize state vector probabilities: sum(P) = 1.0
        let total: f32 = probs.iter().sum();
        let norm_probs: Vec<f32> = if total > 1e-6 {
            probs.into_iter().map(|v| v / total).collect()
        } else {
            vec![1.0 / n as f32; n]
        };

        // Von Neumann entropy: S = - sum(p_k * ln(p_k))
        let entropy: f32 = norm_probs.iter().map(|&p| {
            if p > 1e-12 { -p * p.ln() } else { 0.0 }
        }).sum();

        (Array1::from_vec(norm_probs), entropy)
    }

    /// Parameterized quantum phase update using loss gradient feedback
    pub fn update_phases(&mut self, lr: f32, gradient: &[f32]) {
        for (i, &grad) in gradient.iter().enumerate() {
            if i < self.rotation_thetas.len() && grad.is_finite() {
                self.rotation_thetas[i] = (self.rotation_thetas[i] - lr * grad).clamp(0.0, std::f32::consts::PI);
            }
            if i < self.phase_phis.len() && grad.is_finite() {
                self.phase_phis[i] = (self.phase_phis[i] - lr * grad * 0.5) % std::f32::consts::TAU;
            }
        }
    }
}

/// Individual memory slot in the Associative Memory Network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySlot {
    pub id: usize,
    pub key: Vec<f32>,
    pub value: Vec<f32>,
    pub domain: String,
    pub label: String,
    pub access_count: u32,
    pub retention_score: f32, // 0.0 to 3.0; strengthened on attention, decays over time
    pub last_accessed_timestamp: u64,
}

/// Bio-inspired Associative Memory Network with soft-attention content addressing.
/// Implements key-value storage, least-recently-used / lowest-retention slot replacement,
/// and memory consolidation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssociativeMemoryNetwork {
    pub capacity: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub slots: Vec<MemorySlot>,
    pub temperature: f32,
    pub decay_rate: f32,
    pub total_reads: u64,
    pub total_writes: u64,
    pub last_attention_weights: Vec<f32>,
    pub last_query_domain: Option<String>,
}

impl Default for AssociativeMemoryNetwork {
    fn default() -> Self {
        Self::new(32, 8, 8)
    }
}

impl AssociativeMemoryNetwork {
    pub fn new(capacity: usize, key_dim: usize, value_dim: usize) -> Self {
        let cap = capacity.max(8);
        let mut net = Self {
            capacity: cap,
            key_dim,
            value_dim,
            slots: Vec::with_capacity(cap),
            temperature: 0.5,
            decay_rate: 0.05,
            total_reads: 0,
            total_writes: 0,
            last_attention_weights: Vec::new(),
            last_query_domain: None,
        };
        net.seed_canonical_archetypes();
        net
    }

    /// Seeds canonical task archetypes representing the 6 primary AI ecosystem domains
    pub fn seed_canonical_archetypes(&mut self) {
        let archetypes = [
            (
                "General",
                "Broad conversational, multi-domain problem solving & query dispatch",
                vec![0.2, 0.1, 0.5, 0.1, 0.2, 0.0, 0.0, 0.1],
                vec![0.7, 0.1, 0.1, 0.0, 0.0, 0.1, 0.0, 0.0],
            ),
            (
                "Coder",
                "High-performance systems programming, memory safety & compiler optimization",
                vec![0.6, 0.3, 0.8, 0.2, 0.7, 0.1, 0.16, 0.42],
                vec![0.1, 0.8, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0],
            ),
            (
                "Researcher",
                "Deep technical analysis, literature synthesis & factual verification",
                vec![0.8, 0.5, 0.9, 0.4, 0.85, 0.2, 0.33, 0.78],
                vec![0.1, 0.0, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0],
            ),
            (
                "Cyber/Critic",
                "Vulnerability inspection, post-quantum crypto analysis & threat auditing",
                vec![0.5, 0.2, 0.7, 0.1, 0.5, 0.3, 0.50, 0.91],
                vec![0.0, 0.0, 0.1, 0.8, 0.1, 0.0, 0.0, 0.0],
            ),
            (
                "Planner",
                "Swarm orchestration, DAG task decomposition & dependency resolution",
                vec![0.4, 0.2, 0.6, 0.15, 0.4, 0.4, 0.66, 0.25],
                vec![0.1, 0.0, 0.0, 0.0, 0.8, 0.1, 0.0, 0.0],
            ),
            (
                "Writer",
                "Architectural documentation, security advisory drafting & reporting",
                vec![0.7, 0.4, 0.8, 0.3, 0.8, 0.5, 0.83, 0.55],
                vec![0.1, 0.0, 0.0, 0.0, 0.1, 0.8, 0.0, 0.0],
            ),
        ];

        for (domain, label, key, value) in archetypes {
            let mut k = key;
            k.resize(self.key_dim, 0.0);
            let mut v = value;
            v.resize(self.value_dim, 0.0);
            self.slots.push(MemorySlot {
                id: self.slots.len(),
                key: k,
                value: v,
                domain: domain.to_string(),
                label: label.to_string(),
                access_count: 1,
                retention_score: 1.0,
                last_accessed_timestamp: 0,
            });
        }
    }

    /// Content-addressable soft-attention read head.
    /// Computes dot-product similarity against memory keys scaled by temperature and sqrt(d),
    /// applies Softmax to generate attention distribution, and retrieves weighted memory vector.
    pub fn read(&mut self, query: &[f32]) -> (Vec<f32>, Vec<f32>) {
        self.total_reads += 1;
        if self.slots.is_empty() {
            return (vec![0.0; self.value_dim], vec![]);
        }

        let temp = self.temperature.clamp(0.05, 5.0);
        let scale = (self.key_dim.max(1) as f32).sqrt() * temp;

        let mut raw_scores = Vec::with_capacity(self.slots.len());
        for slot in &self.slots {
            let mut dot = 0.0;
            let len = query.len().min(slot.key.len());
            for i in 0..len {
                dot += query[i] * slot.key[i];
            }
            raw_scores.push(dot / scale);
        }

        let max_score = raw_scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_scores: Vec<f32> = raw_scores.iter().map(|&s| (s - max_score).exp()).collect();
        let sum_exp: f32 = exp_scores.iter().sum();

        let attn_weights: Vec<f32> = if sum_exp > 1e-6 {
            exp_scores.iter().map(|&e| e / sum_exp).collect()
        } else {
            vec![1.0 / self.slots.len() as f32; self.slots.len()]
        };

        // Compute retrieved value as linear combination of values weighted by attention
        let mut retrieved = vec![0.0; self.value_dim];
        for (i, slot) in self.slots.iter_mut().enumerate() {
            let w = attn_weights[i];
            if w > 0.05 {
                slot.access_count = slot.access_count.saturating_add(1);
                slot.retention_score = (slot.retention_score + 0.1 * w).min(3.0);
            }
            for (j, &val) in slot.value.iter().enumerate().take(self.value_dim) {
                retrieved[j] += w * val;
            }
        }

        self.last_attention_weights = attn_weights.clone();
        (retrieved, attn_weights)
    }

    /// Writes or updates an experience vector in the memory network.
    /// If at capacity, overwrites the least-retained memory slot.
    pub fn write(&mut self, key: &[f32], value: &[f32], domain: &str, label: &str, reward: f32) {
        self.total_writes += 1;
        let mut padded_key = key.to_vec();
        padded_key.resize(self.key_dim, 0.0);
        let mut padded_val = value.to_vec();
        padded_val.resize(self.value_dim, 0.0);

        // Check if an existing slot matches closely (cosine similarity > 0.96)
        let mut closest_idx = None;
        let mut max_sim = 0.0;
        for (idx, slot) in self.slots.iter().enumerate() {
            let mut dot = 0.0;
            let mut norm_q = 0.0;
            let mut norm_k = 0.0;
            for i in 0..self.key_dim {
                let q_i = padded_key[i];
                let k_i = slot.key[i];
                dot += q_i * k_i;
                norm_q += q_i * q_i;
                norm_k += k_i * k_i;
            }
            let denom = (norm_q.sqrt() * norm_k.sqrt()).max(1e-6);
            let sim = dot / denom;
            if sim > max_sim {
                max_sim = sim;
                closest_idx = Some(idx);
            }
        }

        if max_sim > 0.96 {
            if let Some(idx) = closest_idx {
                let slot = &mut self.slots[idx];
                for j in 0..self.value_dim {
                    slot.value[j] = 0.7 * slot.value[j] + 0.3 * padded_val[j];
                }
                slot.retention_score = (slot.retention_score + 0.2 * reward.max(0.1)).min(3.0);
                slot.access_count = slot.access_count.saturating_add(1);
                return;
            }
        }

        // If at capacity, find slot with lowest retention score (excluding canonical archetypes 0..6)
        if self.slots.len() >= self.capacity {
            let min_slot_idx = self
                .slots
                .iter()
                .enumerate()
                .skip(6.min(self.slots.len()))
                .min_by(|a, b| a.1.retention_score.partial_cmp(&b.1.retention_score).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(self.slots.len().saturating_sub(1));

            if min_slot_idx < self.slots.len() {
                self.slots[min_slot_idx] = MemorySlot {
                    id: min_slot_idx,
                    key: padded_key,
                    value: padded_val,
                    domain: domain.to_string(),
                    label: label.to_string(),
                    access_count: 1,
                    retention_score: 1.0 + reward.max(0.0) * 0.5,
                    last_accessed_timestamp: 0,
                };
                return;
            }
        }

        let new_id = self.slots.len();
        self.slots.push(MemorySlot {
            id: new_id,
            key: padded_key,
            value: padded_val,
            domain: domain.to_string(),
            label: label.to_string(),
            access_count: 1,
            retention_score: 1.0 + reward.max(0.0) * 0.5,
            last_accessed_timestamp: 0,
        });
    }

    /// Memory consolidation: applies temporal decay to retention scores while preserving archetypes.
    pub fn consolidate(&mut self) {
        let decay = self.decay_rate.clamp(0.001, 0.5);
        for (i, slot) in self.slots.iter_mut().enumerate() {
            let min_floor = if i < 6 { 0.5 } else { 0.05 };
            slot.retention_score = (slot.retention_score * (1.0 - decay)).max(min_floor);
        }
    }

    /// Resets all memory slots back to canonical archetypes.
    pub fn clear(&mut self) {
        self.slots.clear();
        self.total_reads = 0;
        self.total_writes = 0;
        self.last_attention_weights.clear();
        self.last_query_domain = None;
        self.seed_canonical_archetypes();
    }
}

/// Available Machine Learning Optimizers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizerType {
    Adam,
    Momentum,
    RmsProp,
    Sgd,
}

impl OptimizerType {
    pub fn label(self) -> &'static str {
        match self {
            OptimizerType::Adam => "Adam (Adaptive Moments)",
            OptimizerType::Momentum => "SGD with Momentum",
            OptimizerType::RmsProp => "RMSProp",
            OptimizerType::Sgd => "Standard SGD",
        }
    }

    pub fn all() -> [OptimizerType; 4] {
        [
            OptimizerType::Adam,
            OptimizerType::Momentum,
            OptimizerType::RmsProp,
            OptimizerType::Sgd,
        ]
    }
}

/// Optimizer Configuration & Hyperparameter Setup.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerConfig {
    pub optimizer_type: OptimizerType,
    pub learning_rate: f32,
    pub momentum: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub weight_decay: f32,
    pub batch_size: usize,
    pub auto_train: bool,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            optimizer_type: OptimizerType::Adam,
            learning_rate: 0.005,
            momentum: 0.9,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0001,
            batch_size: 16,
            auto_train: true,
        }
    }
}

/// Synaptic weight statistics for an individual layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerStats {
    pub layer_index: usize,
    pub name: String,
    pub rows: usize,
    pub cols: usize,
    pub param_count: usize,
    pub mean: f32,
    pub std_dev: f32,
    pub min: f32,
    pub max: f32,
    pub l2_norm: f32,
    pub sparsity: f32,
}

/// Benchmark evaluation metrics over canonical domain tasks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkMetrics {
    pub test_loss: f32,
    pub accuracy: f32,
    pub avg_entropy: f32,
    pub memory_retrieval_confidence: f32,
    pub domain_accuracies: Vec<(String, f32)>,
}

/// Bounded replay buffer size to prevent memory bloat and slash serialization overhead.
pub const MAX_EXPERIENCE_BUFFER: usize = 512;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelProfileNetwork {
    pub id: Uuid,
    pub name: String,
    pub input_dim: usize,
    pub hidden_dims: Vec<usize>,
    pub output_dim: usize,
    pub weights: Vec<SerializableArray2>,
    pub biases: Vec<SerializableArray1>,
    pub learning_rate: f32,
    pub experience_buffer: Vec<Experience>,
    pub performance_history: Vec<f32>,
    #[serde(default)]
    pub swarm_pheromones: SwarmPheromoneMatrix,
    #[serde(default)]
    pub quantum_layer: Option<QuantumStateLayer>,
    #[serde(default)]
    pub memory_network: AssociativeMemoryNetwork,
    #[serde(default)]
    pub optimizer_config: OptimizerConfig,
    #[serde(default)]
    pub opt_m_weights: Vec<SerializableArray2>,
    #[serde(default)]
    pub opt_v_weights: Vec<SerializableArray2>,
    #[serde(default)]
    pub opt_m_biases: Vec<SerializableArray1>,
    #[serde(default)]
    pub opt_v_biases: Vec<SerializableArray1>,
    #[serde(default)]
    pub opt_step: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experience {
    pub state: SerializableArray1,
    pub action: usize,
    pub reward: f32,
    pub next_state: SerializableArray1,
    pub done: bool,
}

impl ModelProfileNetwork {
    pub fn new(input_dim: usize, hidden_dims: Vec<usize>, output_dim: usize) -> Self {
        let mut weights: Vec<SerializableArray2> = Vec::new();
        let mut biases: Vec<SerializableArray1> = Vec::new();
        let mut prev_dim = input_dim;
        
        let mut rng = rand::thread_rng();
        
        for &hidden_dim in &hidden_dims {
            let w = Array2::from_shape_fn((hidden_dim, prev_dim), |_| {
                rng.gen_range(-0.1..0.1)
            });
            let b = Array1::zeros(hidden_dim);
            weights.push(w.into());
            biases.push(b.into());
            prev_dim = hidden_dim;
        }
        
        let w = Array2::from_shape_fn((output_dim, prev_dim), |_| {
            rng.gen_range(-0.1..0.1)
        });
        let b = Array1::zeros(output_dim);
        weights.push(w.into());
        biases.push(b.into());
        
        let mut opt_m_weights = Vec::new();
        let mut opt_v_weights = Vec::new();
        for w in &weights {
            opt_m_weights.push(Array2::zeros((w.rows, w.cols)).into());
            opt_v_weights.push(Array2::zeros((w.rows, w.cols)).into());
        }
        let mut opt_m_biases = Vec::new();
        let mut opt_v_biases = Vec::new();
        for b in &biases {
            opt_m_biases.push(Array1::zeros(b.data.len()).into());
            opt_v_biases.push(Array1::zeros(b.data.len()).into());
        }

        Self {
            id: Uuid::new_v4(),
            name: "ModelProfileNN".to_string(),
            input_dim,
            hidden_dims,
            output_dim,
            weights,
            biases,
            learning_rate: 0.005,
            experience_buffer: Vec::new(),
            performance_history: Vec::new(),
            swarm_pheromones: SwarmPheromoneMatrix::new(6, output_dim.max(8)),
            quantum_layer: Some(QuantumStateLayer::new(input_dim)),
            memory_network: AssociativeMemoryNetwork::new(32, input_dim, input_dim),
            optimizer_config: OptimizerConfig::default(),
            opt_m_weights,
            opt_v_weights,
            opt_m_biases,
            opt_v_biases,
            opt_step: 0,
        }
    }
    
    fn weights_as_arrays(&self) -> Vec<Array2<f32>> {
        self.weights.iter().cloned().map(|w| w.into()).collect()
    }
    
    fn biases_as_arrays(&self) -> Vec<Array1<f32>> {
        self.biases.iter().cloned().map(|b| b.into()).collect()
    }
    
    fn update_weights(&mut self, weights: Vec<Array2<f32>>, biases: Vec<Array1<f32>>) {
        self.weights = weights.into_iter().map(|w| w.into()).collect();
        self.biases = biases.into_iter().map(|b| b.into()).collect();
    }

    pub fn forward(&self, input: &Array1<f32>) -> Array1<f32> {
        let weights = self.weights_as_arrays();
        let biases = self.biases_as_arrays();
        if weights.is_empty() || self.output_dim == 0 {
            return Array1::zeros(self.output_dim.max(1));
        }
        // Sanitize input: wrong dim or non-finite -> zeros (prevents dot panic / NaN spread)
        let mut x = if input.len() != self.input_dim {
            Array1::zeros(self.input_dim)
        } else {
            input.mapv(|v| if v.is_finite() { v.clamp(-1e3, 1e3) } else { 0.0 })
        };

        for i in 0..weights.len() {
            // Shape guard: weight rows must match bias len, cols must match x len
            if weights[i].ncols() != x.len() || weights[i].nrows() != biases[i].len() {
                return Array1::from_elem(self.output_dim.max(1), 1.0 / self.output_dim.max(1) as f32);
            }
            x = weights[i].dot(&x) + &biases[i];
            x.mapv_inplace(|v| if v.is_finite() { v } else { 0.0 });
            if i < weights.len() - 1 {
                x.mapv_inplace(|v| v.max(0.0));
            }
        }

        if x.is_empty() {
            return Array1::from_elem(self.output_dim.max(1), 1.0 / self.output_dim.max(1) as f32);
        }
        let max_val = x.fold(f32::NEG_INFINITY, |a, &b| {
            if b.is_finite() { a.max(b) } else { a }
        });
        let max_val = if max_val.is_finite() { max_val } else { 0.0 };
        let exp_x = x.mapv(|v| (v.clamp(-50.0, 50.0) - max_val).exp());
        let sum = exp_x.sum();
        if !sum.is_finite() || sum <= 1e-12 {
            return Array1::from_elem(x.len(), 1.0 / x.len() as f32);
        }
        exp_x / sum
    }

    /// Forward pass that also returns per-layer activations (for correct gradient updates).
    fn forward_with_activations(&self, input: &Array1<f32>) -> (Array1<f32>, Vec<Array1<f32>>) {
        let weights = self.weights_as_arrays();
        let biases = self.biases_as_arrays();
        let mut x = if input.len() != self.input_dim {
            Array1::zeros(self.input_dim)
        } else {
            input.mapv(|v| if v.is_finite() { v.clamp(-1e3, 1e3) } else { 0.0 })
        };
        let mut acts = vec![x.clone()];
        for i in 0..weights.len() {
            if weights[i].ncols() != x.len() || weights[i].nrows() != biases[i].len() {
                break;
            }
            x = weights[i].dot(&x) + &biases[i];
            x.mapv_inplace(|v| if v.is_finite() { v } else { 0.0 });
            if i < weights.len() - 1 {
                x.mapv_inplace(|v| v.max(0.0));
            }
            acts.push(x.clone());
        }
        // softmax on final
        let out = if x.is_empty() {
            Array1::zeros(self.output_dim.max(1))
        } else {
            let m = x.fold(f32::NEG_INFINITY, |a, &b| if b.is_finite() { a.max(b) } else { a });
            let m = if m.is_finite() { m } else { 0.0 };
            let e = x.mapv(|v| (v.clamp(-50.0, 50.0) - m).exp());
            let s = e.sum();
            if !s.is_finite() || s <= 1e-12 {
                Array1::from_elem(x.len(), 1.0 / x.len() as f32)
            } else {
                e / s
            }
        };
        (out, acts)
    }

    pub fn select_model_profile(&self, context: &Array1<f32>) -> usize {
        let output = self.forward(context);
        output.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    pub fn add_experience(&mut self, experience: Experience) {
        // Clamp action to valid range at ingest so train_step can never index OOB
        let mut exp = experience;
        if self.output_dim > 0 {
            exp.action = exp.action.min(self.output_dim - 1);
        }
        exp.reward = if exp.reward.is_finite() { exp.reward.clamp(-10.0, 10.0) } else { 0.0 };
        self.experience_buffer.push(exp);

        if self.experience_buffer.len() > MAX_EXPERIENCE_BUFFER {
            let overflow = self.experience_buffer.len() - MAX_EXPERIENCE_BUFFER;
            self.experience_buffer.drain(0..overflow);
        }
    }

    pub fn train_step(&mut self) -> Result<f32> {
        if self.experience_buffer.len() < 4 {
            return Ok(0.0);
        }

        let batch_size = self.optimizer_config.batch_size.clamp(4, 64).min(self.experience_buffer.len());
        let mut rng = rand::thread_rng();
        let mut batch = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            let i = rng.gen_range(0..self.experience_buffer.len());
            batch.push(self.experience_buffer[i].clone());
        }

        let mut weights = self.weights_as_arrays();
        let mut biases = self.biases_as_arrays();
        let num_layers = weights.len();
        if num_layers == 0 {
            return Ok(0.0);
        }

        let mut grad_weights: Vec<Array2<f32>> = weights
            .iter()
            .map(|w| Array2::zeros(w.dim()))
            .collect();
        let mut grad_biases: Vec<Array1<f32>> = biases
            .iter()
            .map(|b| Array1::zeros(b.len()))
            .collect();

        let mut total_loss = 0.0;
        let mut valid_samples = 0;

        for exp in &batch {
            let state: Array1<f32> = exp.state.clone().into();
            let (prediction, acts) = self.forward_with_activations(&state);
            if prediction.len() != self.output_dim || acts.len() != num_layers + 1 {
                continue;
            }

            let action = exp.action.min(self.output_dim.saturating_sub(1));
            let mut target = prediction.clone();
            target[action] = exp.reward.clamp(-10.0, 10.0);

            let diff = &prediction - &target;
            let loss: f32 = diff.mapv(|v| if v.is_finite() { v * v } else { 0.0 }).sum();
            total_loss += if loss.is_finite() { loss } else { 0.0 };
            valid_samples += 1;

            // Output gradient
            let n_out = self.output_dim.max(1) as f32;
            let mut delta: Array1<f32> = diff.mapv(|v| {
                if v.is_finite() { (2.0 * v / n_out).clamp(-2.0, 2.0) } else { 0.0 }
            });

            // Backpropagate through layers in reverse
            for l in (0..num_layers).rev() {
                let a_prev = &acts[l];
                let (rows, cols) = weights[l].dim();

                for r in 0..rows.min(delta.len()) {
                    let d = delta[r];
                    for c in 0..cols.min(a_prev.len()) {
                        let a = if a_prev[c].is_finite() { a_prev[c] } else { 0.0 };
                        grad_weights[l][[r, c]] += d * a;
                    }
                    if r < grad_biases[l].len() {
                        grad_biases[l][r] += d;
                    }
                }

                if l > 0 {
                    let mut delta_prev = Array1::zeros(cols);
                    for c in 0..cols {
                        let mut sum = 0.0;
                        for r in 0..rows.min(delta.len()) {
                            sum += weights[l][[r, c]] * delta[r];
                        }
                        let relu_deriv = if a_prev[c] > 0.0 { 1.0 } else { 0.0 };
                        delta_prev[c] = (sum * relu_deriv).clamp(-2.0, 2.0);
                    }
                    delta = delta_prev;
                }
            }

            // Write rewarded experience to Associative Memory Network
            if exp.reward > 0.5 {
                let domain_idx = exp.action % 6;
                let domain_tag = match domain_idx {
                    1 => "Coder",
                    2 => "Researcher",
                    3 => "Cyber/Critic",
                    4 => "Planner",
                    5 => "Writer",
                    _ => "General",
                };
                self.memory_network.write(
                    &state.to_vec(),
                    &prediction.to_vec(),
                    domain_tag,
                    "rewarded_experience",
                    exp.reward,
                );
            }
        }

        if valid_samples == 0 {
            return Ok(0.0);
        }

        let scale = 1.0 / valid_samples as f32;
        self.opt_step += 1;
        let opt_cfg = self.optimizer_config.clone();
        let lr = self.learning_rate.clamp(1e-6, 0.5);
        let lambda = opt_cfg.weight_decay.clamp(0.0, 0.05);

        // Ensure optimizer momentum/velocity buffers match weights
        if self.opt_m_weights.len() != num_layers {
            self.opt_m_weights = weights.iter().map(|w| Array2::zeros(w.dim()).into()).collect();
            self.opt_v_weights = weights.iter().map(|w| Array2::zeros(w.dim()).into()).collect();
            self.opt_m_biases = biases.iter().map(|b| Array1::zeros(b.len()).into()).collect();
            self.opt_v_biases = biases.iter().map(|b| Array1::zeros(b.len()).into()).collect();
        }

        let mut opt_m_w: Vec<Array2<f32>> = self.opt_m_weights.iter().map(|w| w.clone().into()).collect();
        let mut opt_v_w: Vec<Array2<f32>> = self.opt_v_weights.iter().map(|w| w.clone().into()).collect();
        let mut opt_m_b: Vec<Array1<f32>> = self.opt_m_biases.iter().map(|b| b.clone().into()).collect();
        let mut opt_v_b: Vec<Array1<f32>> = self.opt_v_biases.iter().map(|b| b.clone().into()).collect();

        // Apply optimizer update step
        for l in 0..num_layers {
            let (rows, cols) = weights[l].dim();

            for r in 0..rows {
                for c in 0..cols {
                    let mut g = grad_weights[l][[r, c]] * scale + lambda * weights[l][[r, c]];
                    if !g.is_finite() { g = 0.0; }
                    g = g.clamp(-5.0, 5.0);

                    let delta_w = match opt_cfg.optimizer_type {
                        OptimizerType::Adam => {
                            let b1 = opt_cfg.beta1;
                            let b2 = opt_cfg.beta2;
                            opt_m_w[l][[r, c]] = b1 * opt_m_w[l][[r, c]] + (1.0 - b1) * g;
                            opt_v_w[l][[r, c]] = b2 * opt_v_w[l][[r, c]] + (1.0 - b2) * g * g;
                            let step = self.opt_step as f32;
                            let m_hat = opt_m_w[l][[r, c]] / (1.0 - b1.powf(step)).max(1e-7);
                            let v_hat = opt_v_w[l][[r, c]] / (1.0 - b2.powf(step)).max(1e-7);
                            lr * m_hat / (v_hat.sqrt() + opt_cfg.epsilon)
                        }
                        OptimizerType::Momentum => {
                            let mu = opt_cfg.momentum;
                            opt_m_w[l][[r, c]] = mu * opt_m_w[l][[r, c]] + lr * g;
                            opt_m_w[l][[r, c]]
                        }
                        OptimizerType::RmsProp => {
                            let gamma = 0.9;
                            opt_v_w[l][[r, c]] = gamma * opt_v_w[l][[r, c]] + (1.0 - gamma) * g * g;
                            lr * g / (opt_v_w[l][[r, c]].sqrt() + opt_cfg.epsilon)
                        }
                        OptimizerType::Sgd => lr * g,
                    };

                    if delta_w.is_finite() {
                        weights[l][[r, c]] -= delta_w;
                    }
                }
            }

            for r in 0..biases[l].len() {
                let mut g = grad_biases[l][r] * scale;
                if !g.is_finite() { g = 0.0; }
                g = g.clamp(-5.0, 5.0);

                let delta_b = match opt_cfg.optimizer_type {
                    OptimizerType::Adam => {
                        let b1 = opt_cfg.beta1;
                        let b2 = opt_cfg.beta2;
                        opt_m_b[l][r] = b1 * opt_m_b[l][r] + (1.0 - b1) * g;
                        opt_v_b[l][r] = b2 * opt_v_b[l][r] + (1.0 - b2) * g * g;
                        let step = self.opt_step as f32;
                        let m_hat = opt_m_b[l][r] / (1.0 - b1.powf(step)).max(1e-7);
                        let v_hat = opt_v_b[l][r] / (1.0 - b2.powf(step)).max(1e-7);
                        lr * m_hat / (v_hat.sqrt() + opt_cfg.epsilon)
                    }
                    OptimizerType::Momentum => {
                        let mu = opt_cfg.momentum;
                        opt_m_b[l][r] = mu * opt_m_b[l][r] + lr * g;
                        opt_m_b[l][r]
                    }
                    OptimizerType::RmsProp => {
                        let gamma = 0.9;
                        opt_v_b[l][r] = gamma * opt_v_b[l][r] + (1.0 - gamma) * g * g;
                        lr * g / (opt_v_b[l][r].sqrt() + opt_cfg.epsilon)
                    }
                    OptimizerType::Sgd => lr * g,
                };

                if delta_b.is_finite() {
                    biases[l][r] -= delta_b;
                }
            }
        }

        self.update_weights(weights, biases);
        self.opt_m_weights = opt_m_w.into_iter().map(|w| w.into()).collect();
        self.opt_v_weights = opt_v_w.into_iter().map(|w| w.into()).collect();
        self.opt_m_biases = opt_m_b.into_iter().map(|b| b.into()).collect();
        self.opt_v_biases = opt_v_b.into_iter().map(|b| b.into()).collect();

        // Memory network consolidation
        self.memory_network.consolidate();

        Ok(total_loss / valid_samples as f32)
    }

    pub fn update_performance(&mut self, reward: f32) {
        self.performance_history.push(if reward.is_finite() { reward } else { 0.0 });
        if self.performance_history.len() > 1000 {
            let overflow = self.performance_history.len() - 1000;
            self.performance_history.drain(0..overflow);
        }
    }

    pub fn avg_performance(&self) -> f32 {
        let vals: Vec<f32> = self.performance_history.iter().copied().filter(|v| v.is_finite()).collect();
        if vals.is_empty() {
            0.0
        } else {
            vals.iter().sum::<f32>() / vals.len() as f32
        }
    }

    /// Forward pass through the quantum superposition layer and neural classifier.
    /// Returns (fused_probabilities, von_neumann_entropy).
    pub fn forward_quantum(&self, input: &Array1<f32>) -> (Array1<f32>, f32) {
        let (processed_input, entropy) = if let Some(ql) = &self.quantum_layer {
            ql.transform(input)
        } else {
            (input.clone(), 0.0)
        };
        let out = self.forward(&processed_input);
        (out, entropy)
    }

    /// Bio-inspired swarm auto-router:
    /// Evaluates text prompt through probe features, quantum superposition amplitudes,
    /// and stigmergic pheromone trails.
    /// Returns (best_slot, domain_name, confidence, entropy, distribution).
    pub fn recommend_swarm_slot(&self, text: &str) -> (usize, &'static str, f32, f32, Vec<f32>) {
        let (domain_idx, domain_name) = classify_prompt_domain(text);
        let probe = extract_probe_features(text);
        let (neural_dist, entropy) = self.forward_quantum(&probe);
        let (best_slot, fused_probs) = self.swarm_pheromones.fuse_decision(domain_idx, &neural_dist.to_vec());
        let confidence = if best_slot < fused_probs.len() { fused_probs[best_slot] } else { 0.0 };
        (best_slot, domain_name, confidence, entropy, fused_probs)
    }

    /// Evaluates prompt with full Memory Network soft-attention, Quantum layer, and Swarm Pheromones.
    /// Returns (best_slot, domain_name, confidence, entropy, distribution, retrieved_memory, attention_weights).
    pub fn query_with_memory(
        &mut self,
        text: &str,
    ) -> (usize, &'static str, f32, f32, Vec<f32>, Vec<f32>, Vec<f32>) {
        let (domain_idx, domain_name) = classify_prompt_domain(text);
        let probe = extract_probe_features(text);
        let (mem_val, mem_attn) = self.memory_network.read(&probe.to_vec());
        self.memory_network.last_query_domain = Some(domain_name.to_string());

        // Modulate probe features with retrieved memory vector
        let mut modulated_probe = probe.clone();
        for i in 0..modulated_probe.len().min(mem_val.len()) {
            modulated_probe[i] = 0.8 * modulated_probe[i] + 0.2 * mem_val[i];
        }

        let (neural_dist, entropy) = self.forward_quantum(&modulated_probe);
        let (best_slot, fused_probs) = self.swarm_pheromones.fuse_decision(domain_idx, &neural_dist.to_vec());
        let confidence = if best_slot < fused_probs.len() { fused_probs[best_slot] } else { 0.0 };

        (best_slot, domain_name, confidence, entropy, fused_probs, mem_val, mem_attn)
    }

    /// Computes synaptic weight statistics (mean, std dev, min, max, L2 norm, sparsity) across all layers.
    pub fn compute_synapse_stats(&self) -> (usize, Vec<LayerStats>) {
        let weights = self.weights_as_arrays();
        let mut total_params = 0;
        let mut layer_stats = Vec::with_capacity(weights.len());

        for (idx, w) in weights.iter().enumerate() {
            let (rows, cols) = w.dim();
            let count = rows * cols;
            total_params += count;

            let mut sum = 0.0;
            let mut sum_sq = 0.0;
            let mut min_val = f32::INFINITY;
            let mut max_val = f32::NEG_INFINITY;
            let mut zero_count = 0;

            for &val in w.iter() {
                if val.is_finite() {
                    sum += val;
                    sum_sq += val * val;
                    if val < min_val { min_val = val; }
                    if val > max_val { max_val = val; }
                    if val.abs() < 1e-4 { zero_count += 1; }
                }
            }

            let mean = if count > 0 { sum / count as f32 } else { 0.0 };
            let variance = if count > 1 {
                ((sum_sq - sum * sum / count as f32) / (count - 1) as f32).max(0.0)
            } else {
                0.0
            };
            let std_dev = variance.sqrt();
            let l2_norm = sum_sq.sqrt();
            let sparsity = if count > 0 { zero_count as f32 / count as f32 } else { 0.0 };

            let name = if idx == 0 {
                format!("Input [{}] -> Hidden 0 [{}]", cols, rows)
            } else if idx + 1 == weights.len() {
                format!("Hidden {} [{}] -> Output [{}]", idx - 1, cols, rows)
            } else {
                format!("Hidden {} [{}] -> Hidden {} [{}]", idx - 1, cols, idx, rows)
            };

            layer_stats.push(LayerStats {
                layer_index: idx,
                name,
                rows,
                cols,
                param_count: count,
                mean,
                std_dev,
                min: if min_val.is_finite() { min_val } else { 0.0 },
                max: if max_val.is_finite() { max_val } else { 0.0 },
                l2_norm,
                sparsity,
            });
        }

        (total_params, layer_stats)
    }

    /// Evaluates the model profile and memory network against canonical benchmark probes.
    /// Computes test loss, prediction accuracy, average entropy, and memory confidence.
    pub fn evaluate_benchmark(&self) -> BenchmarkMetrics {
        let samples = generate_synthetic_benchmark_dataset();
        let mut total_loss = 0.0;
        let mut correct = 0;
        let mut total_entropy = 0.0;
        let mut total_mem_conf = 0.0;
        let mut domain_counts: HashMap<String, (usize, usize)> = HashMap::new();

        for s in &samples {
            let probe = extract_probe_features(s.prompt);
            let (pred, entropy) = self.forward_quantum(&probe);
            total_entropy += entropy;

            let mut dot_sum: f32 = 0.0;
            for slot in &self.memory_network.slots {
                let dot: f32 = probe.iter().zip(slot.key.iter()).map(|(&a, &b)| a * b).sum();
                dot_sum = dot_sum.max(dot.abs());
            }
            total_mem_conf += dot_sum.min(1.0);

            let chosen_action = pred
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);

            let is_correct = chosen_action == s.target_slot;
            if is_correct {
                correct += 1;
            }

            let entry = domain_counts.entry(s.domain_name.to_string()).or_insert((0, 0));
            entry.0 += 1;
            if is_correct {
                entry.1 += 1;
            }

            let mut target = pred.clone();
            target[s.target_slot.min(self.output_dim.saturating_sub(1))] = s.expected_reward;
            let diff = &pred - &target;
            total_loss += diff.mapv(|v| v * v).sum();
        }

        let n = samples.len().max(1) as f32;
        let mut domain_accuracies = Vec::new();
        for (dom, (tot, corr)) in domain_counts {
            let acc = if tot > 0 { corr as f32 / tot as f32 } else { 0.0 };
            domain_accuracies.push((dom, acc));
        }
        domain_accuracies.sort_by(|a, b| a.0.cmp(&b.0));

        BenchmarkMetrics {
            test_loss: total_loss / n,
            accuracy: correct as f32 / n,
            avg_entropy: total_entropy / n,
            memory_retrieval_confidence: total_mem_conf / n,
            domain_accuracies,
        }
    }

    /// Run a synthetic multi-domain training epoch:
    /// Samples canonical task archetypes, propagates through quantum layer,
    /// updates synaptic weights via gradient descent, deposits reinforcement pheromones,
    /// and applies stigmergic evaporation.
    /// Returns (training_loss, avg_quantum_entropy).
    pub fn train_synthetic_epoch(&mut self) -> Result<(f32, f32)> {
        let samples = generate_synthetic_benchmark_dataset();
        let mut total_entropy = 0.0;

        for s in &samples {
            let probe = extract_probe_features(s.prompt);
            let (quantum_input, entropy) = if let Some(ql) = &self.quantum_layer {
                ql.transform(&probe)
            } else {
                (probe.clone(), 0.5)
            };
            total_entropy += entropy;

            let exp = Experience {
                state: quantum_input.clone().into(),
                action: s.target_slot.min(self.output_dim.saturating_sub(1)),
                reward: s.expected_reward,
                next_state: quantum_input.into(),
                done: true,
            };
            self.add_experience(exp);

            // Reinforce memory network with canonical archetype
            self.memory_network.write(
                &probe.to_vec(),
                &s.prompt.as_bytes().iter().take(self.memory_network.value_dim).map(|&b| b as f32 / 255.0).collect::<Vec<_>>(),
                s.domain_name,
                s.target_role,
                s.expected_reward,
            );

            // Reinforce swarm pheromones for the correct domain/slot pair
            self.swarm_pheromones.deposit(s.domain_idx, s.target_slot, s.expected_reward);
        }

        // Run gradient descent
        let mut total_loss = 0.0;
        if self.experience_buffer.len() >= 4 {
            let loss = self.train_step()?;
            total_loss = loss;
            self.performance_history.push(total_loss);
        }

        if let Some(ql) = &mut self.quantum_layer {
            let grad = vec![total_loss * 0.01; ql.num_qubits];
            ql.update_phases(self.learning_rate, &grad);
        }

        // Natural bio-inspired pheromone evaporation
        self.swarm_pheromones.evaporate();

        let avg_entropy = total_entropy / samples.len().max(1) as f32;
        Ok((total_loss, avg_entropy))
    }

    pub fn save(&self, path: &str) -> Result<()> {
        let raw = bincode::serialize(self)?;
        let key_path = std::path::Path::new(path).with_file_name("vault.key");
        let vault = crate::storage::StorageVault::load_or_create(&key_path)?;
        let data = vault.encrypt(&raw)?;
        let tmp_path = format!("{}.tmp", path);
        std::fs::write(&tmp_path, &data)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&tmp_path, std::fs::Permissions::from_mode(0o600));
        }
        std::fs::rename(&tmp_path, path)?;
        Ok(())
    }

    pub fn load(path: &str) -> Result<Self> {
        let raw = std::fs::read(path)?;
        let key_path = std::path::Path::new(path).with_file_name("vault.key");
        let vault = crate::storage::StorageVault::load_or_create(&key_path)?;
        let data = vault.decrypt(&raw)?;
        let mut network: Self = match bincode::deserialize(&data) {
            Ok(net) => net,
            Err(_) => {
                // If legacy schema without quantum/swarm, create a fresh initialized net
                let net = Self::new(8, vec![16, 16], 4);
                let _ = net.save(path);
                return Ok(net);
            }
        };
        network.sanitize();
        // Structural validation: layer count must be hidden+1, dims sane
        let expect_layers = network.hidden_dims.len() + 1;
        if network.weights.len() != expect_layers || network.biases.len() != expect_layers {
            anyhow::bail!("corrupt network db: layer count mismatch");
        }
        if network.input_dim == 0 || network.input_dim > 100000 || network.output_dim == 0 || network.output_dim > 100000 {
            anyhow::bail!("corrupt network db: bad dims");
        }
        let mut prev = network.input_dim;
        for (k, w) in network.weights.iter().enumerate() {
            let expect_rows = if k < network.hidden_dims.len() { network.hidden_dims[k] } else { network.output_dim };
            if w.rows != expect_rows || w.cols != prev {
                anyhow::bail!("corrupt network db: weight shape mismatch at layer {k}");
            }
            prev = expect_rows;
        }
        Ok(network)
    }

    /// Replace non-finite weights/biases with 0 so one corrupt save can't NaN-poison the net.
    fn sanitize(&mut self) {
        for w in &mut self.weights {
            for v in w.data.iter_mut() {
                if !v.is_finite() { *v = 0.0; }
            }
        }
        for b in &mut self.biases {
            for v in b.data.iter_mut() {
                if !v.is_finite() { *v = 0.0; }
            }
        }
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            self.learning_rate = 0.01;
        }
        if let Some(ql) = &mut self.quantum_layer {
            for v in &mut ql.rotation_thetas {
                if !v.is_finite() { *v = 0.0; }
            }
            for v in &mut ql.phase_phis {
                if !v.is_finite() { *v = 0.0; }
            }
            for v in &mut ql.entanglement_couplings {
                if !v.is_finite() { *v = 0.0; }
            }
        }
        for row in &mut self.swarm_pheromones.pheromones {
            for trail in row {
                if !trail.is_finite() || *trail <= 0.0 { *trail = 1.0; }
            }
        }
        if self.experience_buffer.len() > MAX_EXPERIENCE_BUFFER {
            let overflow = self.experience_buffer.len() - MAX_EXPERIENCE_BUFFER;
            self.experience_buffer.drain(0..overflow);
        }
    }
}

/// Canonical 8-dimensional feature extractor for prompts & context:
/// [0] Character length ratio
/// [1] Word count ratio
/// [2] Markdown code block indicator (1.0 or 0.0)
/// [3] Question mark density
/// [4] Line count ratio
/// [5] Digit density
/// [6] Uppercase density
/// [7] Programming syntax / symbol density
pub fn extract_probe_features(text: &str) -> Array1<f32> {
    let chars = text.chars().count().max(1) as f32;
    let words = text.split_whitespace().count() as f32;
    let lines = text.lines().count() as f32;
    let qmarks = text.chars().filter(|&c| c == '?').count() as f32;
    let digits = text.chars().filter(|c| c.is_ascii_digit()).count() as f32;
    let upper = text
        .chars()
        .filter(|c| c.is_alphabetic() && c.is_uppercase())
        .count() as f32;
    let codey = text
        .chars()
        .filter(|c| matches!(c, '{' | '}' | ';' | '=' | '(' | ')' | '`' | '<' | '>'))
        .count() as f32;
    let fences = if text.contains("```") { 1.0 } else { 0.0 };
    Array1::from_vec(vec![
        (chars / 2000.0).min(1.0),
        (words / 300.0).min(1.0),
        fences,
        (qmarks / 3.0).min(1.0),
        (lines / 40.0).min(1.0),
        (digits / chars * 5.0).min(1.0),
        (upper / chars * 3.0).min(1.0),
        (codey / 20.0).min(1.0),
    ])
}

/// Classify prompt into one of 6 ecological domain niches:
/// 0: General, 1: Coder, 2: Researcher, 3: Cyber / Critic, 4: Planner, 5: Writer
pub fn classify_prompt_domain(text: &str) -> (usize, &'static str) {
    let lower = text.to_lowercase();
    if lower.contains("audit") || lower.contains("vulnerab") || lower.contains("secur")
        || lower.contains("secret") || lower.contains("encrypt") || lower.contains("timing")
        || lower.contains("quantum") || lower.contains("qubit") || lower.contains("shannon")
    {
        (3, "Cyber / Critic")
    } else if lower.contains("fn ") || lower.contains("def ") || lower.contains("impl ")
        || lower.contains("struct ") || lower.contains("class ") || lower.contains("```")
        || lower.contains("bug") || lower.contains("refactor") || lower.contains("compile")
    {
        (1, "Coder")
    } else if lower.contains("why") || lower.contains("explain") || lower.contains("how does")
        || lower.contains("paper") || lower.contains("research") || lower.contains("theory")
        || lower.contains("analyze")
    {
        (2, "Researcher")
    } else if lower.contains("plan") || lower.contains("steps") || lower.contains("roadmap")
        || lower.contains("schedule") || lower.contains("architecture") || lower.contains("design")
    {
        (4, "Planner")
    } else if lower.contains("write") || lower.contains("story") || lower.contains("draft")
        || lower.contains("essay") || lower.contains("summary") || lower.contains("walkthrough")
    {
        (5, "Writer")
    } else {
        (0, "General")
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SyntheticTaskSample {
    pub domain_idx: usize,
    pub domain_name: &'static str,
    pub prompt: &'static str,
    pub target_slot: usize,
    pub target_role: &'static str,
    pub expected_reward: f32,
}

pub fn generate_synthetic_benchmark_dataset() -> Vec<SyntheticTaskSample> {
    vec![
        SyntheticTaskSample {
            domain_idx: 1, // Coder
            domain_name: "Coder",
            prompt: "fn quicksort<T: Ord>(arr: &mut [T]) { /* implement partitioning */ }",
            target_slot: 1,
            target_role: "Coder",
            expected_reward: 1.5,
        },
        SyntheticTaskSample {
            domain_idx: 3, // Cyber / Critic
            domain_name: "Cyber / Critic",
            prompt: "Audit this loopback client for memory safety, side-channel timing leaks, and Shannon entropy threshold breaches.",
            target_slot: 0,
            target_role: "Critic",
            expected_reward: 1.8,
        },
        SyntheticTaskSample {
            domain_idx: 2, // Researcher
            domain_name: "Researcher",
            prompt: "Synthesize empirical benchmarks on quantum annealing vs gate-based quantum phase estimation for discrete optimization.",
            target_slot: 2,
            target_role: "Researcher",
            expected_reward: 1.6,
        },
        SyntheticTaskSample {
            domain_idx: 4, // Planner
            domain_name: "Planner",
            prompt: "Decompose this distributed multi-agent swarm into parallel DAG task stages with fault tolerance.",
            target_slot: 3,
            target_role: "Planner",
            expected_reward: 1.4,
        },
        SyntheticTaskSample {
            domain_idx: 5, // Writer
            domain_name: "Writer",
            prompt: "Draft an executive summary and architectural walkthrough of the bio-inspired swarm stigmergy engine.",
            target_slot: 0,
            target_role: "Writer",
            expected_reward: 1.5,
        },
        SyntheticTaskSample {
            domain_idx: 0, // General
            domain_name: "General",
            prompt: "What are the core operating parameters of this laboratory environment?",
            target_slot: 0,
            target_role: "General",
            expected_reward: 1.2,
        },
    ]
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct AdaptiveModelSelector {
    pub network: ModelProfileNetwork,
    pub profiles: Vec<ModelProfileInfo>,
    pub context_encoder: ContextEncoder,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct ModelProfileInfo {
    pub id: Uuid,
    pub name: String,
    pub model_name: String,
    pub system_prompt: String,
    pub temperature: f32,
    pub top_p: f32,
    pub skills: Vec<SkillInfo>,
    pub success_rate: f32,
    pub avg_response_time: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct SkillInfo {
    pub name: String,
    pub proficiency: f32,
    pub category: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct ContextEncoder {
    pub vocab_size: usize,
    pub embedding_dim: usize,
    pub embeddings: SerializableArray2,
}

#[allow(dead_code)]
impl ContextEncoder {
    pub fn new(vocab_size: usize, embedding_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let embeddings = Array2::from_shape_fn((vocab_size, embedding_dim), |_| {
            rng.gen_range(-0.1..0.1)
        });
        
        Self {
            vocab_size,
            embedding_dim,
            embeddings: embeddings.into(),
        }
    }

    pub fn encode(&self, text: &str) -> Array1<f32> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let embeddings: Array2<f32> = self.embeddings.clone().into();
        let mut vec = Array1::zeros(self.embedding_dim);
        
        for word in &words {
            let hash = Self::hash_word(word) % self.vocab_size;
            vec += &embeddings.row(hash);
        }
        
        if !words.is_empty() {
            vec /= words.len() as f32;
        }
        
        vec
    }

    fn hash_word(word: &str) -> usize {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        word.hash(&mut hasher);
        hasher.finish() as usize
    }
}

#[allow(dead_code)]
impl AdaptiveModelSelector {
    pub fn new(num_profiles: usize, context_dim: usize) -> Self {
        Self {
            network: ModelProfileNetwork::new(context_dim, vec![128, 64], num_profiles),
            profiles: Vec::new(),
            context_encoder: ContextEncoder::new(10000, context_dim),
        }
    }

    pub fn add_profile(&mut self, profile: ModelProfileInfo) {
        self.profiles.push(profile);
    }

    pub fn select_best_profile(&self, user_input: &str, task_type: &str) -> Option<&ModelProfileInfo> {
        let context = self.build_context(user_input, task_type);
        let profile_idx = self.network.select_model_profile(&context);
        
        self.profiles.get(profile_idx)
    }

    fn build_context(&self, user_input: &str, task_type: &str) -> Array1<f32> {
        let text_embedding = self.context_encoder.encode(user_input);
        let task_embedding = self.context_encoder.encode(task_type);
        
        let mut context = Array1::zeros(self.network.input_dim);
        let half = self.network.input_dim / 2;
        
        for i in 0..half.min(text_embedding.len()) {
            context[i] = text_embedding[i];
        }
        for i in 0..(self.network.input_dim - half).min(task_embedding.len()) {
            context[half + i] = task_embedding[i];
        }
        
        context
    }

    pub fn record_interaction(&mut self, user_input: &str, task_type: &str, profile_idx: usize, success: bool, response_time: f32) {
        let context = self.build_context(user_input, task_type);
        let reward = if success { 1.0 } else { -0.5 };
        
        let time_reward = (1.0 - (response_time / 30.0).min(1.0)) * 0.5;
        let total_reward = reward + time_reward;
        
        let experience = Experience {
            state: context.clone().into(),
            action: profile_idx,
            reward: total_reward,
            next_state: context.into(),
            done: true,
        };
        
        self.network.add_experience(experience);
        self.network.update_performance(total_reward);
        
        if let Some(profile) = self.profiles.get_mut(profile_idx) {
            profile.success_rate = 0.9 * profile.success_rate + 0.1 * if success { 1.0 } else { 0.0 };
            profile.avg_response_time = 0.9 * profile.avg_response_time + 0.1 * response_time;
        }
        
        if self.network.experience_buffer.len() % 50 == 0 {
            let _ = self.network.train_step();
        }
    }

    pub fn learn_skill(&mut self, profile_idx: usize, skill_name: String, skill_category: String) {
        if let Some(profile) = self.profiles.get_mut(profile_idx) {
            if !profile.skills.iter().any(|s| s.name == skill_name) {
                profile.skills.push(SkillInfo {
                    name: skill_name.clone(),
                    proficiency: 0.1,
                    category: skill_category,
                });
            } else if let Some(skill) = profile.skills.iter_mut().find(|s| s.name == skill_name) {
                skill.proficiency = (skill.proficiency + 0.1).min(1.0);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_is_distribution() {
        let net = ModelProfileNetwork::new(4, vec![8], 3);
        let out = net.forward(&Array1::zeros(4));
        assert_eq!(out.len(), 3);
        let s: f32 = out.iter().sum();
        assert!((s - 1.0).abs() < 1e-4, "softmax sums to 1, got {s}");
        assert!(out.iter().all(|&v| v > 0.0 && v < 1.0));
        assert!(net.select_model_profile(&Array1::zeros(4)) < 3);
    }

    #[test]
    fn experience_clamped_and_train_runs() {
        let mut net = ModelProfileNetwork::new(4, vec![8], 3);
        assert_eq!(net.train_step().unwrap(), 0.0);
        for _ in 0..8 {
            net.add_experience(Experience {
                state: Array1::zeros(4).into(),
                action: 99,
                reward: f32::INFINITY,
                next_state: Array1::zeros(4).into(),
                done: true,
            });
        }
        assert_eq!(net.experience_buffer.len(), 8);
        assert!(net.experience_buffer.iter().all(|e| e.action == 2));
        assert!(net.experience_buffer.iter().all(|e| e.reward == 0.0));
        let loss = net.train_step().unwrap();
        assert!(loss.is_finite() && loss >= 0.0);
    }

    #[test]
    fn quantum_layer_normalization_and_entropy() {
        let ql = QuantumStateLayer::new(8);
        let input = Array1::from_vec(vec![0.1, 0.9, 0.5, 0.0, 0.8, 0.2, 0.4, 0.7]);
        let (probs, entropy) = ql.transform(&input);
        assert_eq!(probs.len(), 8);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "quantum probabilities must sum to 1, got {sum}");
        assert!(entropy >= 0.0 && entropy <= 3.0, "entropy must be non-negative: {entropy}");
    }

    #[test]
    fn swarm_pheromone_stigmergy_and_fusion() {
        let mut swarm = SwarmPheromoneMatrix::new(6, 4);
        assert_eq!(swarm.pheromones[1][1], 1.0);
        swarm.deposit(1, 1, 2.0); // Reward coder domain slot 1
        assert!(swarm.pheromones[1][1] > 1.0);

        swarm.evaporate();
        assert!(swarm.pheromones[1][1] > 1.0 && swarm.pheromones[1][1] < 2.0);

        let neural_desirability = [0.25, 0.25, 0.25, 0.25];
        let (best_slot, probs) = swarm.fuse_decision(1, &neural_desirability);
        assert_eq!(best_slot, 1, "fused decision should choose the reinforced slot");
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
    }

    #[test]
    fn synthetic_benchmark_training_epoch() {
        let mut net = ModelProfileNetwork::new(8, vec![16, 16], 4);
        let res = net.train_synthetic_epoch();
        assert!(res.is_ok());
        let (loss, entropy) = res.unwrap();
        assert!(loss >= 0.0 && loss.is_finite());
        assert!(entropy >= 0.0 && entropy.is_finite());
        assert!(!net.experience_buffer.is_empty());
    }

    #[test]
    fn recommend_swarm_slot_matches_domains() {
        let net = ModelProfileNetwork::new(8, vec![16, 16], 4);
        let (slot, domain, conf, entropy, dist) = net.recommend_swarm_slot("fn quicksort() { let x = 1; }");
        assert_eq!(domain, "Coder");
        assert!(conf > 0.0 && conf <= 1.0);
        assert!(entropy >= 0.0);
        assert_eq!(dist.len(), net.swarm_pheromones.num_slots);
        assert!(slot < net.swarm_pheromones.num_slots);

        let (_, cyber_domain, _, _, _) = net.recommend_swarm_slot("Audit security vulnerabilities, secret leakage, and timing attacks");
        assert_eq!(cyber_domain, "Cyber / Critic");
    }

    #[test]
    fn neural_encrypted_at_rest_roundtrip() {
        let dir = std::env::temp_dir().join(format!("aidash-neural-enc-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("neural.bin").to_string_lossy().to_string();

        let mut net = ModelProfileNetwork::new(8, vec![16, 16], 4);
        net.swarm_pheromones.deposit(1, 2, 5.0);
        net.save(&path).expect("save neural network");

        // Inspect raw disk bytes to ensure AES-256-GCM envelope magic b"A256"
        let raw = std::fs::read(&path).expect("read raw bytes");
        assert_eq!(&raw[0..4], crate::storage::ENVELOPE_MAGIC, "saved network must be AES-256 encrypted");

        // Load network back and verify weights & pheromones
        let loaded = ModelProfileNetwork::load(&path).expect("load neural network");
        assert_eq!(loaded.swarm_pheromones.pheromones[1][2], net.swarm_pheromones.pheromones[1][2]);
        assert_eq!(loaded.input_dim, 8);
        assert_eq!(loaded.output_dim, 4);

        let input = Array1::from_vec(vec![0.5; 8]);
        let out = loaded.forward(&input);
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_associative_memory_network_read_write_and_decay() {
        let mut mem = AssociativeMemoryNetwork::new(16, 8, 8);
        assert_eq!(mem.slots.len(), 6, "pre-seeded with 6 domain archetypes");

        let probe = vec![0.1, 0.9, 0.2, 0.1, 0.1, 0.3, 0.1, 0.2];
        let (read_vec, weights) = mem.read(&probe);
        assert_eq!(read_vec.len(), 8);
        assert_eq!(weights.len(), 6);
        let w_sum: f32 = weights.iter().sum();
        assert!((w_sum - 1.0).abs() < 1e-4, "attention weights must sum to 1.0, got {w_sum}");

        // Highest weight should correspond to Coder (slot 1)
        assert!(weights[1] > weights[0]);

        // Write new memory slot
        let key = vec![0.1, 0.1, 0.9, 0.2, 0.1, 0.1, 0.1, 0.8];
        let val = vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        mem.write(&key, &val, "Cyber / Critic", "Audit secret leakage", 1.0);
        assert_eq!(mem.slots.len(), 7);

        // Decay retention via consolidation
        mem.consolidate();
        assert!(mem.slots[0].retention_score <= 3.0);
    }

    #[test]
    fn test_adam_optimizer_multi_layer_backprop() {
        let mut net = ModelProfileNetwork::new(8, vec![16, 16], 4);
        net.optimizer_config.optimizer_type = OptimizerType::Adam;
        net.optimizer_config.learning_rate = 0.05;
        net.optimizer_config.weight_decay = 0.0001;

        let target_state = Array1::from_vec(vec![0.5; 8]);
        for _ in 0..16 {
            net.add_experience(Experience {
                state: target_state.clone().into(),
                action: 2,
                reward: 1.0,
                next_state: target_state.clone().into(),
                done: true,
            });
        }

        let first_loss = net.train_step().unwrap();
        assert!(first_loss.is_finite() && first_loss >= 0.0);

        // Train multiple steps
        let mut final_loss = first_loss;
        for _ in 0..10 {
            for _ in 0..8 {
                net.add_experience(Experience {
                    state: target_state.clone().into(),
                    action: 2,
                    reward: 1.0,
                    next_state: target_state.clone().into(),
                    done: true,
                });
            }
            final_loss = net.train_step().unwrap();
        }
        assert!(final_loss.is_finite());

        // Check synapse stats
        let (param_count, layer_stats) = net.compute_synapse_stats();
        assert!(param_count > 0);
        assert_eq!(layer_stats.len(), 3); // 2 hidden + 1 output
        for st in &layer_stats {
            assert!(st.l2_norm >= 0.0);
            assert!(st.sparsity >= 0.0 && st.sparsity <= 100.0);
        }

        // Check benchmark evaluation
        let bench = net.evaluate_benchmark();
        assert!(bench.test_loss.is_finite());
        assert!(bench.accuracy >= 0.0 && bench.accuracy <= 100.0);
        assert!(bench.memory_retrieval_confidence >= 0.0);
    }
}


