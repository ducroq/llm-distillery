# AI Engineering Practice Filter v1

**Status:** Phase 4 Complete (Prefilter v1.1) - Ready for Training Data Generation

## Purpose

Surface articles providing evidence/insights on AI-augmented engineering practice for research on digital engineer competencies.

**Philosophy:** "Document the undocumented" - prioritize authentic practice evidence over hype

**Research Context:** Supports HAN University of Applied Sciences research on engineering expertise transition, investigating how professional engineers integrate AI tools into workflows.

## Scope

| Aspect | Coverage |
|--------|----------|
| Engineering domains | ALL (software, mechanical, electrical, embedded, etc.) |
| Geography | GLOBAL |
| Temporal | Fresh data (no historical filtering) |
| Education | ALL (formal education + corporate training) |

## Dimensions (Orthogonal Design)

This filter uses **orthogonal dimensions** - each measures something DIFFERENT and INDEPENDENT:

### Content Dimensions (WHAT is described)
| Dimension | Weight | Description |
|-----------|--------|-------------|
| workflow_detail | 25% | Specific AI tool usage patterns and processes described |
| validation_coverage | 20% | Methods for verifying/validating AI outputs |

### Quality Dimensions (HOW rigorous, WHO says it)
| Dimension | Weight | Description |
|-----------|--------|-------------|
| methodological_rigor | 20% | Systematic data collection vs. opinion (GATEKEEPER) |
| practitioner_voice | 20% | From actual engineers vs. vendors/analysts |
| educational_applicability | 15% | Can inform curriculum, training, pedagogy |

**Key Principle:** An article can score HIGH on one dimension and LOW on another:
- Rigorous survey (HIGH rigor) with no workflow details (LOW workflow)
- Practitioner blog (HIGH voice) with anecdotal evidence (LOW rigor)
- Academic paper (LOW voice) with extensive methodology (HIGH rigor)

## Tiers

| Tier | Threshold | Description |
|------|-----------|-------------|
| high | >= 7.0 | High-value research evidence |
| medium | >= 4.0 | Useful supporting evidence |
| low | >= 0.0 | Background or not relevant |

## Gatekeeper

**Evidence Gate:** If `methodological_rigor < 3`, overall score capped at 3.0

*Rationale: Pure opinion without data cannot provide research evidence*

---

## Prefilter v1.1

### Architecture: ALLOWLIST Approach

Unlike blocklist-based filters, this prefilter **requires positive signals**:

```
Article must have BOTH:
1. AI tool mention (Copilot, LLM, ChatGPT, agentic workflow, etc.)
2. Engineering context (developer, CAD, PCB, embedded, etc.)
```

Most random articles are NOT about AI in engineering, so we require explicit signals.

### v1.1 Changes (2025-12-19)

Based on Gemini feedback, major improvements to capture ALL engineering domains:

| Change | Rationale |
|--------|-----------|
| Expanded AI keywords (68→121) | Added 2025 tools: Cursor, Windsurf, Claude Code, agentic AI |
| Expanded engineering keywords (55→126) | Added ME/EE/Embedded: CFD, PCB, FPGA, MBSE, etc. |
| Removed fluid dynamics/thermodynamics from blocklist | These ARE engineering contexts |
| Added Physics-AI patterns | PINN, surrogate modeling, neural PDE |
| Added EDA/chip design AI | DSO.ai, Cadence Cerebrus, auto-routing |
| Strengthened ML research detection | Academic citation patterns (et al., Figure N:) |
| Fixed exception patterns | Require AI context to prevent false positives |

### Keyword Coverage

**AI Tools (121 keywords):**
- Software: Copilot, Cursor, Windsurf, Claude Code, Cline, Qodo, Bolt.new...
- Agentic: agentic ai, autonomous agent, multi-agent system...
- Physical: Ansys SimAI, Neural Concept, Bananaz AI, Monolith AI...
- Physics-ML: physics-informed, PINN, surrogate model, neural PDE...
- EDA: DSO.ai, Cadence Cerebrus, AI routing...

**Engineering Context (126 keywords):**
- Software: developer, coding, git, agile, IDE, refactoring...
- Mechanical: CAD, CAM, FEA, CFD, topology optimization, SolidWorks...
- Electrical: PCB, VLSI, EDA, FPGA, ASIC, Verilog, Altium, Synopsys...
- Embedded: firmware, microcontroller, RTOS, MBSE, V-model...
- Professional: ISO 26262, DO-178, safety-critical, compliance...

### Test Results

| Test | v1.0 | v1.1 | Status |
|------|------|------|--------|
| Random noise rejection | 0% | 0% | ✓ Blocks irrelevant |
| Copilot-relevant capture | 24% | 36% | ✓ +12% improvement |
| CFD/Fluid articles | 0 | 10 | ✓ Now captured |
| PCB/Circuit articles | ~0 | 8 | ✓ Now captured |
| FPGA/ASIC articles | ~0 | 429 | ✓ Now captured |
| Embedded/Firmware | ~0 | 94 | ✓ Now captured |
| Synthetic domain tests | N/A | 8/8 | ✓ All pass |

### Prefilter Flow

```
1. Validate structure and length (>300 chars)
2. Check EXCEPTIONS → Always allow if matched (practice studies)
3. Check blocked domains → Reject biorxiv, bloomberg, etc.
4. Check irrelevant topics → Reject finance, sports, warfare
5. REQUIRE AI mention → Reject if missing
6. REQUIRE engineering context → Reject if missing
7. Check ML research patterns → Reject pure academic papers
8. PASS → Send to oracle
```

---

## Calibration Results

From 100 prefiltered articles scored by Gemini Flash oracle:

**Content Type Distribution:**
| Type | Count |
|------|-------|
| research_study | 62% |
| practitioner_account | 36% |
| thought_piece | 1% |
| not_relevant | 1% |

**Dimension Statistics:**
| Dimension | Mean | Std |
|-----------|------|-----|
| workflow_detail | 5.99 | 1.04 |
| validation_coverage | 6.11 | 1.10 |
| methodological_rigor | 5.63 | 1.94 |
| practitioner_voice | 5.49 | 2.66 |
| educational_applicability | 6.72 | 0.83 |

**Key Correlations (Semantically Meaningful):**
- methodological_rigor ↔ practitioner_voice: **-0.83** (Academic papers lack practitioner voice)
- workflow_detail ↔ practitioner_voice: **+0.75** (Practitioners describe workflows)

---

## Development Progress

- [x] Phase 1: Planning
- [x] Phase 2: Architecture (config.yaml, prompt)
- [x] Phase 3: Oracle Calibration (100 articles)
- [x] Phase 4: Prefilter Development (v1.1)
- [ ] Phase 5: Training Data Generation (5K+ articles)
- [ ] Phase 6: Model Training (Qwen2.5-1.5B)
- [ ] Phase 7-9: Testing, Documentation, Deployment

## Files

| File | Description |
|------|-------------|
| config.yaml | Filter configuration and dimension definitions |
| prompt-compressed.md | Oracle prompt for Gemini/Claude scoring |
| prefilter.py | Rule-based prefiltering (v1.1) |
| calibration_report.md | Initial calibration results |
| inference.py | (TODO) Local model inference |

## Data Source

**Server:** `llm-distiller` (via Tailscale)

```bash
# Connect to server
ssh jeroen@llm-distiller

# Available datasets
~/llm-distillery/datasets/raw/master_dataset_20251009_20251124.jsonl  # 178,462 articles
~/llm-distillery/datasets/raw/master_dataset_20250929_20251008.jsonl  # 37,137 articles
# Total: 215,599 articles
```

## Related Research

- [Engineering Expertise in Transition](https://han.nl) - HAN University research project
- Key statistics: 90% AI adoption, 0 ethnographic studies, validation gap (+55% vs -19%)
