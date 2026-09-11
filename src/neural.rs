use anyhow::Result;
use ndarray::{Array1, Array2};
use rand::Rng;
use serde::{Deserialize, Serialize};
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
    pub epsilon: f32,
    pub experience_buffer: Vec<Experience>,
    pub performance_history: Vec<f32>,
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
        let mut weights = Vec::new();
        let mut biases = Vec::new();
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
        
        Self {
            id: Uuid::new_v4(),
            name: "ModelProfileNN".to_string(),
            input_dim,
            hidden_dims,
            output_dim,
            weights,
            biases,
            learning_rate: 0.01,
            epsilon: 0.1,
            experience_buffer: Vec::new(),
            performance_history: Vec::new(),
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

    /// Shared forward pass: softmax output plus per-layer pre-activations
    /// (z) and post-activations (a) for backprop. posts[0] is the sanitized
    /// input; posts[l+1] is layer l's output (ReLU for hidden, logits last).
    fn forward_cache(
        &self,
        input: &Array1<f32>,
    ) -> (Array1<f32>, Vec<Array1<f32>>, Vec<Array1<f32>>) {
        let weights = self.weights_as_arrays();
        let biases = self.biases_as_arrays();
        let uniform =
            Array1::from_elem(self.output_dim.max(1), 1.0 / self.output_dim.max(1) as f32);
        let mut x = if input.len() != self.input_dim {
            Array1::zeros(self.input_dim)
        } else {
            input.mapv(|v| if v.is_finite() { v.clamp(-1e3, 1e3) } else { 0.0 })
        };
        let mut pres = Vec::with_capacity(weights.len());
        let mut posts = Vec::with_capacity(weights.len() + 1);
        posts.push(x.clone());
        for i in 0..weights.len() {
            // Shape guard: weight rows must match bias len, cols must match x len
            if weights[i].ncols() != x.len() || weights[i].nrows() != biases[i].len() {
                return (uniform, pres, posts);
            }
            let z = weights[i].dot(&x) + &biases[i];
            let z = z.mapv(|v| if v.is_finite() { v } else { 0.0 });
            pres.push(z.clone());
            x = if i < weights.len() - 1 {
                z.mapv(|v| v.max(0.0))
            } else {
                z
            };
            posts.push(x.clone());
        }
        if x.is_empty() {
            return (uniform, pres, posts);
        }
        (Self::softmax(&x), pres, posts)
    }

    fn softmax(logits: &Array1<f32>) -> Array1<f32> {
        let max_val =
            logits.fold(f32::NEG_INFINITY, |a, &b| if b.is_finite() { a.max(b) } else { a });
        let max_val = if max_val.is_finite() { max_val } else { 0.0 };
        let exp_x = logits.mapv(|v| (v.clamp(-50.0, 50.0) - max_val).exp());
        let sum = exp_x.sum();
        if !sum.is_finite() || sum <= 1e-12 {
            Array1::from_elem(logits.len(), 1.0 / logits.len().max(1) as f32)
        } else {
            exp_x / sum
        }
    }

    pub fn forward(&self, input: &Array1<f32>) -> Array1<f32> {
        let weights = self.weights_as_arrays();
        if weights.is_empty() || self.output_dim == 0 {
            return Array1::zeros(self.output_dim.max(1));
        }
        self.forward_cache(input).0
    }

    /// Map a raw reward (-0.5 fail .. ~1.5 fast success) onto a target
    /// probability in [0,1] so it lives on the softmax output's scale.
    fn reward_target(reward: f32) -> f32 {
        ((reward + 0.5) / 2.0).clamp(0.0, 1.0)
    }

    pub fn select_model_profile(&self, context: &Array1<f32>) -> usize {
        if self.output_dim == 0 {
            return 0;
        }
        let eps = if self.epsilon.is_finite() {
            self.epsilon.clamp(0.0, 0.5)
        } else {
            0.1
        };
        let mut rng = rand::thread_rng();
        if rng.gen_range(0.0..1.0) < eps {
            return rng.gen_range(0..self.output_dim);
        }
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

        if self.experience_buffer.len() > 10000 {
            let overflow = self.experience_buffer.len() - 10000;
            self.experience_buffer.drain(0..overflow);
        }
    }

    /// Full backprop through every layer (ReLU hidden, softmax output,
    /// exact MSE-through-softmax gradient). Previously only the output
    /// layer learned and hidden layers stayed frozen at their init.
    pub fn train_step(&mut self) -> Result<f32> {
        // Small-batch friendly: sample with replacement so training starts
        // after a handful of turns instead of waiting for 32.
        if self.experience_buffer.len() < 4 {
            return Ok(0.0);
        }

        let mut rng = rand::thread_rng();
        let n = self.experience_buffer.len().min(32);
        let batch: Vec<_> = (0..n)
            .map(|_| {
                let i = rng.gen_range(0..self.experience_buffer.len());
                self.experience_buffer[i].clone()
            })
            .collect();

        let mut weights = self.weights_as_arrays();
        let mut biases = self.biases_as_arrays();
        let mut total_loss = 0.0;
        let lr = if self.learning_rate.is_finite() {
            self.learning_rate.clamp(1e-6, 0.1)
        } else {
            0.01
        };
        let out_dim = self.output_dim.max(1);

        for exp in &batch {
            let state: Array1<f32> = exp.state.clone().into();
            let (prediction, pres, posts) = self.forward_cache(&state);
            if prediction.len() != self.output_dim
                || pres.len() != weights.len()
                || posts.len() != weights.len() + 1
            {
                continue; // corrupt entry / shape mismatch -> skip, don't poison weights
            }
            let action = exp.action.min(self.output_dim.saturating_sub(1));
            let mut target = prediction.clone();
            target[action] = Self::reward_target(exp.reward);

            let diff = &prediction - &target;
            let loss: f32 = diff.mapv(|v| if v.is_finite() { v * v } else { 0.0 }).sum();
            total_loss += if loss.is_finite() { loss } else { 0.0 };

            // Exact MSE-through-softmax gradient wrt logits:
            // dL/dz_i = sum_j 2(p_j - t_j)/n * p_j * (d_ij - p_i)
            let d = diff.mapv(|v| {
                if v.is_finite() {
                    2.0 * v / out_dim as f32
                } else {
                    0.0
                }
            });
            let mut delta = Array1::zeros(prediction.len());
            for i in 0..prediction.len() {
                let mut s = 0.0;
                for j in 0..prediction.len() {
                    let jac = prediction[j] * (if i == j { 1.0 } else { 0.0 } - prediction[i]);
                    s += d[j] * jac;
                }
                delta[i] = if s.is_finite() { s.clamp(-1.0, 1.0) } else { 0.0 };
            }

            // Backpropagate through every layer (ReLU mask on hidden).
            for l in (0..weights.len()).rev() {
                let input = posts[l].clone();
                let (rows, cols) = weights[l].dim();
                let rows = rows.min(delta.len());
                let cols = cols.min(input.len());
                // Delta for the layer below, from the CURRENT weights.
                let mut prev = Array1::zeros(input.len());
                if l > 0 {
                    for j in 0..cols {
                        let mut s = 0.0;
                        for i in 0..rows {
                            let w = weights[l][[i, j]];
                            s += if w.is_finite() { w * delta[i] } else { 0.0 };
                        }
                        let relu = if pres[l - 1].get(j).copied().unwrap_or(0.0) > 0.0 {
                            1.0
                        } else {
                            0.0
                        };
                        prev[j] = if s.is_finite() { (s * relu).clamp(-1.0, 1.0) } else { 0.0 };
                    }
                }
                // Gradient step on this layer's weights/biases.
                for i in 0..rows {
                    for j in 0..cols {
                        let h = if input[j].is_finite() {
                            input[j].clamp(-1.0, 1.0)
                        } else {
                            0.0
                        };
                        let step = lr * delta[i] * h;
                        if step.is_finite() {
                            weights[l][[i, j]] -= step;
                        }
                    }
                    if i < biases[l].len() && delta[i].is_finite() {
                        biases[l][i] -= lr * delta[i];
                    }
                }
                if l > 0 {
                    delta = prev;
                }
            }
        }

        self.update_weights(weights, biases);

        Ok(total_loss / batch.len() as f32)
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

    pub fn save(&self, path: &str) -> Result<()> {
        let data = bincode::serialize(self)?;
        std::fs::write(path, data)?;
        Ok(())
    }

    pub fn load(path: &str) -> Result<Self> {
        let data = std::fs::read(path)?;
        let mut network: Self = bincode::deserialize(&data).map_err(|e| anyhow::anyhow!("corrupt network db: {e}"))?;
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
        if !self.epsilon.is_finite() {
            self.epsilon = 0.1;
        } else {
            self.epsilon = self.epsilon.clamp(0.0, 0.5);
        }
    }
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
    fn backprop_reduces_loss_on_fixed_mapping() {
        // Deterministic: every experience is identical, so every sampled
        // batch is identical and the loss trajectory has no RNG dependence.
        let mut net = ModelProfileNetwork::new(4, vec![8, 8], 3);
        net.epsilon = 0.0;
        assert_eq!(net.train_step().unwrap(), 0.0);
        for _ in 0..8 {
            net.add_experience(Experience {
                state: Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]).into(),
                action: 0,
                reward: 1.5,
                next_state: Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]).into(),
                done: true,
            });
        }
        let first = net.train_step().unwrap();
        assert!(first.is_finite() && first > 0.0);
        let mut last = first;
        for _ in 0..30 {
            last = net.train_step().unwrap();
            assert!(last.is_finite() && last >= 0.0);
        }
        assert!(last < first, "loss did not decrease: {first} -> {last}");
        // Greedy pick for the trained state is the rewarded action.
        let state = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
        assert_eq!(net.select_model_profile(&state), 0);
    }

    #[test]
    fn reward_target_lives_on_softmax_scale() {
        assert_eq!(ModelProfileNetwork::reward_target(1.5), 1.0);
        assert_eq!(ModelProfileNetwork::reward_target(-0.5), 0.0);
        let mid = ModelProfileNetwork::reward_target(0.5);
        assert!((mid - 0.5).abs() < 1e-6);
    }
}
