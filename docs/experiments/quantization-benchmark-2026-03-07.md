# Experiment: PyTorch Dynamic Quantization on Gemma-3-1B

**Date:** 2026-03-07
**Issue:** #24 (Explore smaller models and quantization for energy-efficient inference)
**Status:** Complete — naive quantization rejected, next steps identified

## Goal

Quick feasibility check: can PyTorch's built-in dynamic quantization make Gemma-3-1B viable for CPU inference without retraining?

## Setup

- **Model:** Uplifting v6 (Gemma-3-1B + LoRA, merged)
- **Hardware:** gpu-server CPU (AMD EPYC, used as CPU-only for this test)
- **Target hardware:** sadalsuud Ryzen 3 5300U (8-core, 30GB RAM)
- **Validation set:** 50 articles from uplifting v6 val split
- **Variants tested:** FP32 (baseline), FP16, INT8 (torch.quantization.quantize_dynamic)

## Results

| Metric | FP32 | FP16 | INT8 |
|--------|------|------|------|
| Model size | 4,000 MB | 2,000 MB | 1,208 MB |
| MAE | 0.749 | NaN (broken) | 1.378 (unusable) |
| Latency (ms/article) | 894 | 3,487 | 338 |
| Throughput (articles/sec) | 1.12 | 0.29 | 2.96 |

## Analysis

### FP16: broken on CPU
- Produces NaN predictions (numeric overflow — FP16 range is limited)
- 4x slower than FP32 because x86 CPUs lack native FP16 compute units; they emulate via conversion
- **Verdict:** Not viable for CPU inference

### INT8: fast but inaccurate
- 2.6x faster and 3.3x smaller — the speed/size wins are real
- MAE nearly doubles: 0.749 → 1.378 (+0.63)
- `torch.quantization.quantize_dynamic` quantizes ALL Linear layers uniformly (attention + MLP + score head)
- No calibration data is used — weights are naively per-tensor quantized
- **Verdict:** Too much accuracy loss. The speed gain doesn't matter if scores are wrong.

### FP32 baseline: too slow for production CPU
- 894ms/article on AMD EPYC (server-grade CPU)
- On sadalsuud's Ryzen 3 5300U, expect ~2-3 sec/article
- Daily scoring: 5,000 articles × 6 filters ≈ 25 hours — not practical

## Why naive quantization fails here

PyTorch's `quantize_dynamic` does per-tensor INT8 quantization on all `nn.Linear` layers. This is particularly harmful for:

1. **Attention layers** — Q/K/V projections are sensitive to precision; small quantization errors compound through softmax
2. **Score head** — the final regression layer maps to 0-10 scores; INT8 quantization of this layer directly distorts output range
3. **No calibration** — unlike ONNX Runtime or GPTQ, no representative data is used to determine optimal quantization parameters per layer

## Next Steps (from most to least promising)

### 1. ONNX Runtime INT8 (medium effort)
- Export merged model to ONNX format
- Use ONNX Runtime's calibrated INT8 quantization (feeds representative data through model to find optimal scale/zero-point per tensor)
- Expected: much better accuracy than naive PyTorch INT8
- Requires: `pip install optimum onnxruntime`

### 2. Smaller base model retraining (high effort, highest payoff)
- Train on 300-500M parameter models (SmolLM-360M, Qwen2-0.5B)
- Qwen2-0.5B was tested for uplifting v5 (MAE 0.760) — might be acceptable for some filters
- SmolLM-360M untested — could be 5-10x faster on CPU
- Requires full retraining per filter (~1 hour GPU per filter)

### 3. llama.cpp / GGUF (medium effort)
- Purpose-built C++ inference engine for transformer models on CPU
- Q4_K_M quantization is well-tested and preserves accuracy better
- Challenge: need custom integration (not a PyTorch model anymore)

### 4. torchao quantization (low effort, uncertain)
- PyTorch's successor to deprecated `torch.quantization`
- Supports INT4 weight-only quantization with groupwise scaling
- May preserve accuracy better than naive dynamic quantization
- Requires: `pip install torchao`

## Script

`scripts/experiments/quantization_benchmark.py` — reusable for future experiments. Loads any filter, merges LoRA, runs FP32/FP16/INT8 comparison on val set.

## Conclusion

Naive PyTorch quantization is not viable for our use case. The speed gain (2.6x) is real but the accuracy loss (+0.63 MAE) makes it unusable. CPU inference of the full 1B model at ~1 article/sec is also too slow for daily batch scoring.

**For CPU-viable inference, we need either calibrated quantization (ONNX Runtime) or a smaller base model.** The smaller model path is likely the bigger win — a 360M model at FP32 would already be ~3x faster than 1B at FP32, and quantizing a smaller model is also easier.
