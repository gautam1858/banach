//! Models `banach pick` considers when you do not name any: large open Mixture-of-Experts
//! GGUF files on Hugging Face. Only their headers are read, so listing them costs a few MB.

/// (label, spec) pairs; the spec is what `banach plan` accepts.
pub const MODELS: &[(&str, &str)] = &[
    ("Kimi K2 (UD-Q2_K_XL)",
     "hf:unsloth/Kimi-K2-Instruct-GGUF/UD-Q2_K_XL/Kimi-K2-Instruct-UD-Q2_K_XL-00001-of-00008.gguf"),
    ("DeepSeek V4 Flash (UD-Q4_K_XL)",
     "hf:unsloth/DeepSeek-V4-Flash-GGUF/UD-Q4_K_XL/DeepSeek-V4-Flash-UD-Q4_K_XL-00001-of-00005.gguf"),
    ("Qwen3 235B-A22B (UD-Q2_K_XL)",
     "hf:unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF/UD-Q2_K_XL/Qwen3-235B-A22B-Instruct-2507-UD-Q2_K_XL-00001-of-00002.gguf"),
    ("gpt-oss-120b (F16)", "hf:unsloth/gpt-oss-120b-GGUF/gpt-oss-120b-F16.gguf"),
    ("GLM-4.5-Air (UD-Q2_K_XL)", "hf:unsloth/GLM-4.5-Air-GGUF/GLM-4.5-Air-UD-Q2_K_XL.gguf"),
    ("Qwen3.6-35B-A3B (UD-Q4_K_M)", "hf:unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    ("Granite 3.1 1B-A400M (Q8_0)",
     "hf:QuantFactory/granite-3.1-1b-a400m-instruct-GGUF/granite-3.1-1b-a400m-instruct.Q8_0.gguf"),
];
