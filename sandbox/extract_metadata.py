import json

with open(r'C:\local_dev\llm-distillery\datasets\working\uplifting_calibration_labeled_v2.jsonl', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        metadata = {k: v for k, v in data.items() if k != 'content'}
        print(json.dumps(metadata, indent=2))
