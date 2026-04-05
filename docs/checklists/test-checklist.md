# Test Checklist

Definition-of-done for evaluation and validation. Use before claiming a filter or model change is ready.

## Model Quality

- [ ] MAE evaluated on held-out test set (not val set used during training)
- [ ] Per-dimension MAE reported — no single dimension wildly off (cf. cultural-discovery #23)
- [ ] Score distribution inspected — no bimodal collapse (cf. thriving v1)
- [ ] Calibration fitted on val set and committed (`calibration.json`)

## Data Quality

- [ ] Training splits are 80/10/10 with no article overlap
- [ ] Oracle scores spot-checked — sample 10-20 articles across tiers, verify scores make sense
- [ ] Commerce prefilter applied — no commercial articles leaked into training data

## Integration

- [ ] `inference.py` loads and scores without errors
- [ ] `inference_hub.py` loads from HF Hub and produces same results as local
- [ ] Hybrid inference (if applicable) — probe threshold produces reasonable stage1/stage2 split
- [ ] Prefilter works — commercial articles blocked, non-commercial pass through

## Regression

- [ ] Existing filters unaffected — no shared code changes that break other filters
- [ ] `load_base_model_for_seq_cls()` used (not `AutoModelForSequenceClassification` directly)
- [ ] PEFT adapter in OLD key format (`.lora_A.weight`, not `.lora_A.default.weight`)
