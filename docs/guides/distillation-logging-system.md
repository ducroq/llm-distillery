# Distillation Process Logging and Tracking System

This guide explains the comprehensive logging and tracking system for the distillation process, which allows you to monitor, analyze, and optimize your LLM distillation pipeline.

## Overview

The logging system tracks every article processed during distillation, recording:
- Success/failure status
- Error types and messages
- Retry attempts needed
- Processing time per article
- Whether JSON repair was used
- Session-level statistics

## Log Files Generated

When you run the batch labeler, it creates several log files in `datasets/{filter_name}/`:

### 1. `distillation.log` (Human-Readable)

**Purpose:** Detailed human-readable log of all processing activity

**Format:**
```
2025-10-27 10:20:03 | INFO     | SUCCESS | article_123 | attempts=1 | time=11.46s | repaired=False
2025-10-27 10:20:18 | INFO     | SUCCESS | article_456 | attempts=1 | time=15.52s | repaired=False
2025-10-27 10:20:31 | WARNING  | FAILED  | article_789 | error=json_parse_error | attempts=3 | time=42.15s
```

**What it contains:**
- Timestamp of each operation
- Log level (INFO, WARNING, ERROR)
- Article ID
- Success/failure status
- Number of retry attempts
- Processing time
- Whether JSON repair was used
- Error type (for failures)

**Use cases:**
- Quick review of processing status
- Debugging specific article failures
- Monitoring progress in real-time (tail -f)

### 2. `metrics.jsonl` (Structured Metrics)

**Purpose:** Machine-readable metrics for analysis and optimization

**Format:** JSON Lines (one JSON object per line)
```json
{"timestamp": "2025-10-27T09:20:03.336184Z", "article_id": "github_1175a04f0e39", "filter_name": "uplifting", "llm_provider": "gemini", "success": true, "attempts_made": 1, "time_taken_seconds": 11.46, "json_repaired": false}
{"timestamp": "2025-10-27T09:20:18.961907Z", "article_id": "github_8f3c05a0cedf", "filter_name": "uplifting", "llm_provider": "gemini", "success": true, "attempts_made": 1, "time_taken_seconds": 15.52, "json_repaired": false}
{"timestamp": "2025-10-27T09:22:45.123456Z", "article_id": "github_abc123", "filter_name": "uplifting", "llm_provider": "gemini", "success": false, "attempts_made": 3, "time_taken_seconds": 42.15, "json_repaired": false, "error_type": "json_parse_error", "error_message": "Expecting ',' delimiter: line 1 column 45"}
```

**Fields:**
- `timestamp`: ISO 8601 timestamp
- `article_id`: Article identifier
- `filter_name`: Which filter was applied (uplifting, sustainability, etc.)
- `llm_provider`: LLM used (claude, gemini, gpt4)
- `success`: Boolean - whether processing succeeded
- `attempts_made`: Number of retry attempts (1-3)
- `time_taken_seconds`: Processing time in seconds
- `json_repaired`: Boolean - whether JSON repair was needed
- `error_type`: (failures only) Type of error (see Error Types below)
- `error_message`: (failures only) Detailed error message

**Use cases:**
- Statistical analysis of success rates
- Identifying patterns in failures
- Performance optimization
- A/B testing different LLM providers
- Cost analysis (time × provider rate)

### 3. `session_summary.json` (Session Statistics)

**Purpose:** Aggregate statistics for the entire distillation session

**Format:**
```json
{
  "started_at": "2025-10-27T09:19:51.123456",
  "ended_at": "2025-10-27T11:45:32.654321",
  "duration_seconds": 9341.53,
  "articles_attempted": 150,
  "articles_succeeded": 142,
  "articles_failed": 8,
  "total_retries": 15,
  "errors_by_type": {
    "json_parse_error": 5,
    "timeout": 2,
    "json_extraction_failed": 1
  },
  "total_processing_time": 2847.32
}
```

**Fields:**
- `started_at` / `ended_at`: Session start/end timestamps
- `duration_seconds`: Total session duration
- `articles_attempted`: Total articles processed
- `articles_succeeded`: Successful completions
- `articles_failed`: Failed articles
- `total_retries`: Sum of all retry attempts
- `errors_by_type`: Breakdown of error types
- `total_processing_time`: Sum of all article processing times

**Use cases:**
- Session performance overview
- Cost estimation
- Identifying systemic issues
- Comparing runs with different settings

### 4. `error_logs/` (Failed Response Dumps)

**Purpose:** Full LLM responses for failed articles (for debugging)

**Location:** `datasets/{filter_name}/error_logs/`

**Format:** Plain text files named `{article_id}_attempt{N}_{timestamp}.txt`

**Example:**
```
article_789_attempt3_20250127_143022.txt
```

**Contents:**
```
Article ID: article_789
Attempt: 3
Error: Expecting ',' delimiter: line 1 column 45
Timestamp: 2025-10-27T14:30:22.123456

============================================================
Full Response:
============================================================
{
  "content_type": "solutions_story",
  "agency": 8
  "progress": 9,
  ...
```

**Use cases:**
- Debugging malformed JSON
- Identifying prompt issues
- Manual recovery of failed articles
- Improving JSON repair logic

## Error Types

The system classifies errors into the following categories:

| Error Type | Description | Common Causes |
|------------|-------------|---------------|
| `timeout` | LLM call exceeded timeout (60s) | Slow network, overloaded API, large article |
| `json_parse_error` | JSON parsing failed after repair | Malformed JSON, incomplete response, syntax errors |
| `json_extraction_failed` | Could not extract JSON from response | LLM returned only text, no JSON object found |
| `llm_api_error` | API call failed | Network error, authentication, rate limiting |
| `empty_response` | LLM returned empty response | API issue, content filtering, edge case |
| `unknown` | Unclassified error | Unexpected exception |

## Analysis Examples

### Example 1: Calculate Success Rate

```python
import json

success = 0
total = 0

with open('datasets/uplifting/metrics.jsonl', 'r') as f:
    for line in f:
        metric = json.loads(line)
        total += 1
        if metric['success']:
            success += 1

print(f"Success rate: {success/total*100:.1f}%")
```

### Example 2: Find Articles Requiring Retries

```python
import json

retried_articles = []

with open('datasets/uplifting/metrics.jsonl', 'r') as f:
    for line in f:
        metric = json.loads(line)
        if metric['attempts_made'] > 1:
            retried_articles.append({
                'id': metric['article_id'],
                'attempts': metric['attempts_made'],
                'success': metric['success']
            })

print(f"Articles requiring retries: {len(retried_articles)}")
for article in retried_articles:
    print(f"  {article['id']}: {article['attempts']} attempts - {'SUCCESS' if article['success'] else 'FAILED'}")
```

### Example 3: Analyze Error Distribution

```python
import json
from collections import Counter

errors = Counter()

with open('datasets/uplifting/metrics.jsonl', 'r') as f:
    for line in f:
        metric = json.loads(line)
        if not metric['success']:
            error_type = metric.get('error_type', 'unknown')
            errors[error_type] += 1

print("Error distribution:")
for error_type, count in errors.most_common():
    print(f"  {error_type}: {count}")
```

### Example 4: Calculate Average Processing Time

```python
import json

times = []

with open('datasets/uplifting/metrics.jsonl', 'r') as f:
    for line in f:
        metric = json.loads(line)
        times.append(metric['time_taken_seconds'])

avg_time = sum(times) / len(times)
print(f"Average processing time: {avg_time:.2f}s per article")
print(f"Estimated time for 1000 articles: {avg_time * 1000 / 3600:.1f} hours")
```

### Example 5: Find Slowest Articles

```python
import json

articles = []

with open('datasets/uplifting/metrics.jsonl', 'r') as f:
    for line in f:
        metric = json.loads(line)
        articles.append((metric['article_id'], metric['time_taken_seconds']))

# Sort by processing time
articles.sort(key=lambda x: x[1], reverse=True)

print("Top 10 slowest articles:")
for i, (article_id, time_taken) in enumerate(articles[:10], 1):
    print(f"  {i}. {article_id}: {time_taken:.2f}s")
```

### Example 6: Compare LLM Providers

```python
import json
from collections import defaultdict

providers = defaultdict(lambda: {'success': 0, 'failed': 0, 'total_time': 0})

with open('datasets/uplifting/metrics.jsonl', 'r') as f:
    for line in f:
        metric = json.loads(line)
        provider = metric['llm_provider']

        if metric['success']:
            providers[provider]['success'] += 1
        else:
            providers[provider]['failed'] += 1

        providers[provider]['total_time'] += metric['time_taken_seconds']

print("Provider comparison:")
for provider, stats in providers.items():
    total = stats['success'] + stats['failed']
    success_rate = stats['success'] / total * 100
    avg_time = stats['total_time'] / total
    print(f"\n{provider}:")
    print(f"  Success rate: {success_rate:.1f}%")
    print(f"  Avg time: {avg_time:.2f}s")
```

## Optimization Workflow

Use the logs to optimize your distillation process:

### 1. Identify Problems

```bash
# Check session summary
cat datasets/uplifting/session_summary.json

# Review recent failures
tail -50 datasets/uplifting/distillation.log | grep FAILED
```

### 2. Analyze Patterns

```python
# Run analysis scripts (examples above)
# Look for:
# - High failure rate on specific error types
# - Articles consistently timing out
# - Patterns in JSON repair usage
```

### 3. Iterate on Solutions

Based on patterns found:

**High `json_parse_error` rate:**
- Review prompt for clarity
- Check if response format is too complex
- Increase `max_tokens` further
- Improve `repair_json()` function

**Many `timeout` errors:**
- Increase timeout from 60s to 90s or 120s
- Reduce article content length in `build_prompt()`
- Switch to faster LLM provider

**High `json_extraction_failed` rate:**
- LLM not following instructions
- Improve system prompt
- Add few-shot examples to prompt
- Try different LLM provider

**Slow processing:**
- Switch to faster provider (Gemini is 5x faster than Claude)
- Reduce `max_tokens` if responses are verbose
- Pre-filter articles more aggressively

### 4. A/B Test Changes

Run small batches with different settings and compare metrics:

```bash
# Test 1: Current settings
python -m ground_truth.batch_scorer ... --batch-size 50 --max-batches 2

# Test 2: Increased timeout
# (edit code to change timeout_seconds=90)
python -m ground_truth.batch_scorer ... --batch-size 50 --max-batches 2

# Compare metrics.jsonl from both runs
```

## Integration with Analysis Tools

### Jupyter Notebooks

```python
import pandas as pd
import json

# Load metrics into DataFrame
metrics = []
with open('datasets/uplifting/metrics.jsonl', 'r') as f:
    for line in f:
        metrics.append(json.loads(line))

df = pd.DataFrame(metrics)

# Analyze with pandas
print(df.describe())
print(df.groupby('error_type').size())
df['success'].value_counts().plot(kind='bar')
```

### Database Storage

```python
import sqlite3
import json

conn = sqlite3.connect('distillation_metrics.db')
cursor = conn.cursor()

# Create table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS metrics (
        timestamp TEXT,
        article_id TEXT,
        filter_name TEXT,
        llm_provider TEXT,
        success INTEGER,
        attempts_made INTEGER,
        time_taken_seconds REAL,
        json_repaired INTEGER,
        error_type TEXT,
        error_message TEXT
    )
''')

# Load metrics
with open('datasets/uplifting/metrics.jsonl', 'r') as f:
    for line in f:
        metric = json.loads(line)
        cursor.execute('''
            INSERT INTO metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metric['timestamp'],
            metric['article_id'],
            metric['filter_name'],
            metric['llm_provider'],
            1 if metric['success'] else 0,
            metric['attempts_made'],
            metric['time_taken_seconds'],
            1 if metric['json_repaired'] else 0,
            metric.get('error_type'),
            metric.get('error_message')
        ))

conn.commit()

# Query with SQL
cursor.execute('''
    SELECT llm_provider,
           AVG(success) as success_rate,
           AVG(time_taken_seconds) as avg_time
    FROM metrics
    GROUP BY llm_provider
''')
print(cursor.fetchall())
```

## Best Practices

1. **Review logs regularly** - Don't wait until the end of a long run
2. **Set up monitoring** - Use `tail -f distillation.log` during runs
3. **Archive metrics** - Keep historical metrics for trend analysis
4. **Analyze failures first** - Focus on high-impact error types
5. **Test changes incrementally** - Small batches with different settings
6. **Document findings** - Keep notes on what works and what doesn't

## Troubleshooting

### Logs not appearing

**Problem:** No log files created

**Solution:** Check that output directory is writable and not read-only

### Metrics file corrupted

**Problem:** JSONL file has invalid JSON

**Solution:** Each line must be valid JSON. Use this to find bad lines:
```python
import json

with open('metrics.jsonl', 'r') as f:
    for i, line in enumerate(f, 1):
        try:
            json.loads(line)
        except json.JSONDecodeError as e:
            print(f"Line {i} is invalid: {e}")
```

### Session summary not created

**Problem:** session_summary.json missing

**Solution:** Summary is only written at the END of the run. If process was killed early, summary won't exist.

## Related Files

- Implementation: `ground_truth/batch_scorer.py:112-688`
- Error types: `ground_truth/batch_scorer.py:34-41`
- Metrics logging: `ground_truth/batch_scorer.py:544-649`
- Summary generation: `ground_truth/batch_scorer.py:651-688`

## See Also

- [JSON Error Handling Improvements](json-error-handling-improvements.md)
- [Tmux Usage Guide](tmux-usage.md)
