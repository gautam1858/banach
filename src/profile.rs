//! Model profile from GGUF headers: what is always read per token, what is routed, how big
//! an expert is, and how much KV a token costs.

use crate::gguf::{self, Gguf, TensorInfo};
use serde::Serialize;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize)]
pub struct LayerExperts {
    pub layer: u32,
    /// Bytes of one routed expert in this layer (gate + up + down).
    pub expert_bytes: u64,
    pub n_experts: u64,
    /// Dominant ggml type of the routed tensors.
    pub type_name: String,
    pub bpw: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelProfile {
    pub name: String,
    pub arch: String,
    pub n_layers: u64,
    pub n_moe_layers: u64,
    pub n_experts: u64,
    pub top_k: u64,
    pub n_shared: u64,
    pub total_bytes: u64,
    /// Read for every token: everything except routed experts and the embedding table.
    pub dense_read_bytes: u64,
    /// Must be resident or cheaply addressable but is not streamed per token.
    pub embedding_bytes: u64,
    pub routed_bytes: u64,
    pub layers: Vec<LayerExperts>,
    /// Mean routed expert size across MoE layers.
    pub expert_bytes: u64,
    /// Routed bytes touched per token at native precision.
    pub routed_read_bytes: u64,
    pub kv_bytes_per_token: u64,
    pub context_length: u64,
    pub has_mtp: bool,
    pub params: u64,
    pub active_params: u64,
}

fn arch_u64(g: &Gguf, arch: &str, key: &str) -> Option<u64> {
    g.kv.get(&format!("{arch}.{key}")).and_then(|v| v.as_u64())
}

fn is_routed(name: &str) -> bool {
    name.contains("_exps.") && !name.contains("_shexp")
}

fn layer_of(name: &str) -> Option<u32> {
    name.strip_prefix("blk.")?.split('.').next()?.parse().ok()
}

/// Build a profile from one or more GGUF headers (split files are merged).
pub fn build(parts: &[Gguf]) -> Result<ModelProfile, String> {
    let g0 = parts.first().ok_or("no GGUF parts")?;
    let arch = g0
        .kv
        .get("general.architecture")
        .and_then(|v| v.as_str())
        .ok_or("missing general.architecture")?
        .to_string();
    let name = g0
        .kv
        .get("general.name")
        .and_then(|v| v.as_str())
        .unwrap_or(&arch)
        .to_string();
    let n_layers = arch_u64(g0, &arch, "block_count").ok_or("missing block_count")?;
    let n_experts = arch_u64(g0, &arch, "expert_count").unwrap_or(0);
    let top_k = arch_u64(g0, &arch, "expert_used_count").unwrap_or(0);
    let n_shared = arch_u64(g0, &arch, "expert_shared_count").unwrap_or(0);
    let context_length = arch_u64(g0, &arch, "context_length").unwrap_or(0);

    let tensors: Vec<&TensorInfo> = parts.iter().flat_map(|g| g.tensors.iter()).collect();
    let mut total = 0u64;
    let mut dense = 0u64;
    let mut embed = 0u64;
    let mut routed = 0u64;
    let mut params = 0u64;
    let mut routed_params = 0u64;
    let mut per_layer: BTreeMap<u32, (u64, BTreeMap<u32, u64>, u64)> = BTreeMap::new();
    let mut has_mtp = false;
    for t in &tensors {
        total = total.checked_add(t.nbytes).ok_or("size overflow")?;
        params = params.saturating_add(t.n_elements());
        if t.name.contains("nextn") || t.name.contains("mtp") {
            has_mtp = true;
        }
        if is_routed(&t.name) {
            routed += t.nbytes;
            routed_params = routed_params.saturating_add(t.n_elements());
            let l = layer_of(&t.name).ok_or_else(|| format!("routed tensor {} has no layer", t.name))?;
            let e = per_layer.entry(l).or_insert((0, BTreeMap::new(), 0));
            e.0 += t.nbytes;
            *e.1.entry(t.ggml_type).or_insert(0) += t.nbytes;
            // experts are the slowest-varying dimension
            e.2 = *t.dims.last().unwrap_or(&0);
        } else if t.name.starts_with("token_embd") {
            embed += t.nbytes;
        } else {
            dense += t.nbytes;
        }
    }
    // A tied output head reuses the embedding: it is read in full every token, so it moves
    // into the dense path (and out of the embedding-only bucket).
    if !tensors.iter().any(|t| t.name.starts_with("output.weight")) {
        dense += embed;
        embed = 0;
    }

    let mut layers = Vec::new();
    for (l, (bytes, types, ne)) in &per_layer {
        let n_e = if *ne > 0 { *ne } else { n_experts.max(1) };
        let (t, _) = types.iter().max_by_key(|(_, b)| **b).map(|(t, b)| (*t, *b)).unwrap_or((0, 0));
        layers.push(LayerExperts {
            layer: *l,
            expert_bytes: bytes / n_e,
            n_experts: n_e,
            type_name: gguf::type_name(t).to_string(),
            bpw: gguf::type_bpw(t).unwrap_or(0.0),
        });
    }
    let n_moe = layers.len() as u64;
    let expert_bytes = layers.iter().map(|l| l.expert_bytes).sum::<u64>().checked_div(n_moe).unwrap_or(0);
    let routed_read = layers.iter().map(|l| l.expert_bytes * top_k).sum();
    let n_experts = if n_experts == 0 { layers.first().map(|l| l.n_experts).unwrap_or(0) } else { n_experts };

    // KV: f16 K and V per layer per KV head. An estimate: MLA, sliding windows and compressed
    // attention all cost less; the planner reports it as such.
    let n_head_kv = arch_u64(g0, &arch, "attention.head_count_kv")
        .or_else(|| arch_u64(g0, &arch, "attention.head_count"))
        .unwrap_or(0);
    let emb = arch_u64(g0, &arch, "embedding_length").unwrap_or(0);
    let n_head = arch_u64(g0, &arch, "attention.head_count").unwrap_or(1).max(1);
    let k_len = arch_u64(g0, &arch, "attention.key_length").unwrap_or(emb / n_head);
    let v_len = arch_u64(g0, &arch, "attention.value_length").unwrap_or(emb / n_head);
    let kv_bytes_per_token = n_layers * n_head_kv * (k_len + v_len) * 2;

    let active_params = if routed_params > 0 && n_experts > 0 {
        params - routed_params + routed_params / n_experts * top_k
    } else {
        params
    };
    Ok(ModelProfile {
        name,
        arch,
        n_layers,
        n_moe_layers: n_moe,
        n_experts,
        top_k,
        n_shared,
        total_bytes: total,
        dense_read_bytes: dense,
        embedding_bytes: embed,
        routed_bytes: routed,
        layers,
        expert_bytes,
        routed_read_bytes: routed_read,
        kv_bytes_per_token,
        context_length,
        has_mtp,
        params,
        active_params,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::tests::{Builder, Mem};

    #[test]
    fn toy_moe_profile() {
        let bytes = Builder::new()
            .kv_str("general.architecture", "toy")
            .kv_u32("toy.block_count", 2)
            .kv_u32("toy.expert_count", 8)
            .kv_u32("toy.expert_used_count", 2)
            .kv_u32("toy.embedding_length", 64)
            .kv_u32("toy.attention.head_count", 4)
            .kv_u32("toy.attention.head_count_kv", 2)
            .tensor("token_embd.weight", &[64, 100], 1)
            .tensor("output.weight", &[64, 100], 1)
            .tensor("blk.0.attn_q.weight", &[64, 64], 1)
            .tensor("blk.0.ffn_gate_exps.weight", &[64, 32, 8], 8)
            .tensor("blk.0.ffn_up_exps.weight", &[64, 32, 8], 8)
            .tensor("blk.0.ffn_down_exps.weight", &[32, 64, 8], 8)
            .tensor("blk.1.ffn_up_exps.weight", &[64, 32, 8], 8)
            .build();
        let g = gguf::parse(&mut Mem(bytes), 0).unwrap();
        let p = build(&[g]).unwrap();
        assert_eq!(p.n_moe_layers, 2);
        assert_eq!(p.n_experts, 8);
        assert_eq!(p.layers[0].expert_bytes, 3 * (64 * 32 / 32 * 34));
        assert_eq!(p.layers[1].expert_bytes, 64 * 32 / 32 * 34);
        assert_eq!(p.embedding_bytes, 64 * 100 * 2); // untied: output.weight is separate
        assert_eq!(p.dense_read_bytes, 64 * 100 * 2 + 64 * 64 * 2);
        assert_eq!(p.routed_read_bytes, 2 * (p.layers[0].expert_bytes + p.layers[1].expert_bytes));
        assert_eq!(p.kv_bytes_per_token, 2 * 2 * (16 + 16) * 2);
        assert!((p.layers[0].bpw - 8.5).abs() < 1e-9);
    }
}
