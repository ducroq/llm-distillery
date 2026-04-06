"""
Verify foresight v1 model can load: check file paths, configs, and consistency.
Does NOT actually load the model (requires GPU).
"""

import json
import os
from pathlib import Path

FILTER_DIR = Path(__file__).resolve().parent.parent / "filters" / "foresight" / "v1"
MODEL_DIR = FILTER_DIR / "model"
PROBE_DIR = FILTER_DIR / "probe"

results = []


def check(name: str, passed: bool, detail: str = ""):
    status = "PASS" if passed else "FAIL"
    msg = f"[{status}] {name}"
    if detail:
        msg += f" -- {detail}"
    results.append((passed, msg))
    print(msg)


# --- Check 1: adapter_model.safetensors exists ---
safetensors_path = MODEL_DIR / "adapter_model.safetensors"
check(
    "adapter_model.safetensors exists",
    safetensors_path.is_file(),
    f"path={safetensors_path}, size={safetensors_path.stat().st_size if safetensors_path.is_file() else 'N/A'} bytes",
)

# --- Check 2: adapter_config.json fields ---
adapter_config_path = MODEL_DIR / "adapter_config.json"
try:
    with open(adapter_config_path, "r", encoding="utf-8") as f:
        ac = json.load(f)
    fields = {
        "base_model_name_or_path": ac.get("base_model_name_or_path"),
        "peft_type": ac.get("peft_type"),
        "r": ac.get("r"),
        "lora_alpha": ac.get("lora_alpha"),
        "task_type": ac.get("task_type"),
        "modules_to_save": ac.get("modules_to_save"),
    }
    print(f"\n  adapter_config.json fields:")
    for k, v in fields.items():
        print(f"    {k}: {v}")

    ok = (
        fields["base_model_name_or_path"] == "google/gemma-3-1b-pt"
        and fields["peft_type"] == "LORA"
        and fields["task_type"] == "SEQ_CLS"
        and isinstance(fields["r"], int)
        and isinstance(fields["lora_alpha"], int)
        and "score" in (fields["modules_to_save"] or [])
    )
    check("adapter_config.json valid", ok)
except Exception as e:
    check("adapter_config.json valid", False, str(e))

# --- Check 3: training_metadata.json fields ---
metadata_path = MODEL_DIR / "training_metadata.json"
try:
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    fields_meta = {
        "filter_name": meta.get("filter_name"),
        "num_dimensions": meta.get("num_dimensions"),
        "best_val_mae": meta.get("best_val_mae"),
        "train_examples": meta.get("train_examples"),
    }
    print(f"\n  training_metadata.json fields:")
    for k, v in fields_meta.items():
        print(f"    {k}: {v}")

    ok = (
        fields_meta["filter_name"] == "foresight"
        and fields_meta["num_dimensions"] == 6
        and isinstance(fields_meta["best_val_mae"], float)
        and isinstance(fields_meta["train_examples"], int)
    )
    check("training_metadata.json valid", ok)
except Exception as e:
    check("training_metadata.json valid", False, str(e))

# --- Check 4: calibration.json has entries for all 6 dimensions from base_scorer ---
EXPECTED_DIMS = [
    "time_horizon",
    "systems_awareness",
    "course_correction",
    "intergenerational_investment",
    "institutional_durability",
    "evidence_foundation",
]
cal_path = FILTER_DIR / "calibration.json"
try:
    with open(cal_path, "r", encoding="utf-8") as f:
        cal = json.load(f)
    cal_dims = set(cal.get("dimensions", {}).keys())
    missing = set(EXPECTED_DIMS) - cal_dims
    extra = cal_dims - set(EXPECTED_DIMS)
    detail = ""
    if missing:
        detail += f"missing={missing} "
    if extra:
        detail += f"extra={extra} "
    if not detail:
        detail = f"all 6 dimensions present: {sorted(cal_dims)}"
    check("calibration.json covers all 6 dimensions", len(missing) == 0, detail)
except Exception as e:
    check("calibration.json covers all 6 dimensions", False, str(e))

# --- Check 5: inference.py default model_path resolves correctly ---
# Simulate: Path(__file__).parent / "model" where __file__ = inference.py
inference_py = FILTER_DIR / "inference.py"
inferred_model_dir = inference_py.parent / "model"
has_safetensors = (inferred_model_dir / "adapter_model.safetensors").is_file()
check(
    "inference.py default model_path resolves to dir with adapter_model.safetensors",
    inferred_model_dir.is_dir() and has_safetensors,
    f"resolved={inferred_model_dir}, is_dir={inferred_model_dir.is_dir()}, has_safetensors={has_safetensors}",
)

# --- Check 6: probe files exist ---
probe_pkl = PROBE_DIR / "embedding_probe_e5small.pkl"
probe_sha = PROBE_DIR / "embedding_probe_e5small.pkl.sha256"
check(
    "embedding_probe_e5small.pkl exists",
    probe_pkl.is_file(),
    f"size={probe_pkl.stat().st_size if probe_pkl.is_file() else 'N/A'} bytes",
)
check(
    "embedding_probe_e5small.pkl.sha256 exists",
    probe_sha.is_file(),
    f"size={probe_sha.stat().st_size if probe_sha.is_file() else 'N/A'} bytes",
)

# --- Summary ---
total = len(results)
passed = sum(1 for p, _ in results if p)
failed = total - passed
print(f"\n{'='*60}")
print(f"SUMMARY: {passed}/{total} checks passed, {failed} failed")
if failed == 0:
    print("All checks PASSED - foresight v1 model is ready to load.")
else:
    print("Some checks FAILED - review above for details.")
