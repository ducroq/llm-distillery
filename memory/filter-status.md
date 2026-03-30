# Filter Status

## Production Filters

| Filter | Ver | MAE | Cal. MAE | Data | Hub Repo | Deployed |
|--------|-----|-----|----------|------|----------|----------|
| uplifting | v6 | 0.673 | — | 10.5K | `jeergrvgreg/uplifting-filter-v6` | 2026-02-19 |
| sustainability_technology | v3 | 0.72 | — | 10.6K | `jeergrvgreg/sustainability-technology-filter-v3` | 2026-02 |
| investment-risk | v6 | 0.497 | 0.465 | 10.4K | `jeergrvgreg/investment-risk-filter-v6` | 2026-02-21 |
| cultural-discovery | v4 | 0.74 | — | 8K | `jeergrvgreg/cultural-discovery-filter-v4` | 2026-02 |

| belonging | v1 | 0.534 | 0.489 | 7.4K | `jeergrvgreg/belonging-filter-v1` | 2026-03-04 |
| nature_recovery | v1 | 0.540 | 0.507 | 3.3K | `jeergrvgreg/nature-recovery-filter-v1` | 2026-03-06 |

All use Gemma-3-1B base + LoRA. All have local, Hub, and hybrid inference paths verified.

## ovr.news Lenses (English names)

| Lens | Filter | Status |
|------|--------|--------|
| **Thriving** | uplifting v6 (→ thriving v1 in progress) | Live (v6), thriving v1 next |
| **Belonging** | belonging v1 | Deployed, frontend pending |
| **Recovery** | nature_recovery v1 | Deployed, frontend pending |
| **Solutions** | sustainability_technology v3 | Live |
| **Discovery** | cultural-discovery v4 | Live |
| *(planned)* Wisdom | signs_of_wisdom v1 | Needs prompt |
| *(planned)* Education | future-of-education v1 | Concept only |

## In Development (priority: ovr.news)

| Filter | Ver | Status | ovr.news target |
|--------|-----|--------|-----------------|
| thriving | v1 | Filter created (ADR-012 rename). Next: 3-run oracle scoring → average → train → deploy | Replaces uplifting v6 on Thriving tab |
| signs_of_wisdom | v1 | Need harmonized prompt | Enrich Discovery or standalone |
| future-of-education | v1 | Concept only | Education tab |

## Other Filters (not ovr.news)

| Filter | Ver | Status |
|--------|-----|--------|
| ai-engineering-practice | v2 | Ready for oracle scoring |
| seece | v1 | Concept only |

## Backlog

- Commerce prefilter v2 — v1 needs rework for multilingual embeddings and context size
