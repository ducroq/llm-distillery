const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
        Header, Footer, AlignmentType, PageOrientation, LevelFormat, HeadingLevel,
        BorderStyle, WidthType, ShadingType, VerticalAlign, PageNumber, PageBreak } = require('docx');
const fs = require('fs');

// Table styling
const tableBorder = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const cellBorders = { top: tableBorder, bottom: tableBorder, left: tableBorder, right: tableBorder };
const headerShading = { fill: "2E5A87", type: ShadingType.CLEAR };
const altRowShading = { fill: "F5F5F5", type: ShadingType.CLEAR };

// Helper: create table cell
function cell(content, opts = {}) {
  const { bold = false, width = 1560, header = false, align = AlignmentType.LEFT, shading = null } = opts;
  return new TableCell({
    borders: cellBorders,
    width: { size: width, type: WidthType.DXA },
    shading: shading,
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({
      alignment: align,
      children: [new TextRun({ text: String(content), bold: bold, color: header ? "FFFFFF" : "000000", size: 20 })]
    })]
  });
}

// Load chart images
const chartPath = 'C:/local_dev/llm-distillery/research/embedding_vs_finetuning/results/uplifting_v5_comparison_chart.png';
const chartBuffer = fs.readFileSync(chartPath);

const errorDistPath = 'C:/local_dev/llm-distillery/research/embedding_vs_finetuning/results/error_distribution.png';
const errorDistBuffer = fs.readFileSync(errorDistPath);

const errorVsScorePath = 'C:/local_dev/llm-distillery/research/embedding_vs_finetuning/results/error_vs_score.png';
const errorVsScoreBuffer = fs.readFileSync(errorVsScorePath);

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      { id: "Title", name: "Title", basedOn: "Normal",
        run: { size: 48, bold: true, color: "2E5A87", font: "Arial" },
        paragraph: { spacing: { before: 0, after: 200 }, alignment: AlignmentType.CENTER } },
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, color: "2E5A87", font: "Arial" },
        paragraph: { spacing: { before: 360, after: 120 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, color: "444444", font: "Arial" },
        paragraph: { spacing: { before: 240, after: 100 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 22, bold: true, color: "666666", font: "Arial" },
        paragraph: { spacing: { before: 200, after: 80 }, outlineLevel: 2 } }
    ]
  },
  numbering: {
    config: [
      { reference: "bullet-list",
        levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "numbered-list",
        levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "recommendations",
        levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] }
    ]
  },
  sections: [{
    properties: {
      page: { margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } }
    },
    headers: {
      default: new Header({ children: [new Paragraph({
        alignment: AlignmentType.RIGHT,
        children: [new TextRun({ text: "LLM Distillery Research Report", italics: true, size: 18, color: "888888" })]
      })] })
    },
    footers: {
      default: new Footer({ children: [new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "Page ", size: 18 }), new TextRun({ children: [PageNumber.CURRENT], size: 18 }),
                   new TextRun({ text: " of ", size: 18 }), new TextRun({ children: [PageNumber.TOTAL_PAGES], size: 18 })]
      })] })
    },
    children: [
      // Title
      new Paragraph({ heading: HeadingLevel.TITLE, children: [new TextRun("Embedding-Based Scoring vs Fine-Tuning")] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 400 },
        children: [new TextRun({ text: "Research Report - LLM Distillery Project", size: 24, color: "666666" })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 400 },
        children: [new TextRun({ text: "January 2026", size: 20, color: "888888" })] }),

      // Executive Summary
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Executive Summary")] }),
      new Paragraph({ spacing: { after: 200 }, children: [new TextRun(
        "This research investigates whether frozen embeddings combined with linear or neural probes can match the performance of fine-tuned Qwen2.5-1.5B models for semantic dimension scoring. The study evaluated 4 embedding models with 3 probe architectures on the uplifting_v5 dataset (10,000 articles, 6 dimensions)."
      )] }),
      new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun("Key Findings")] }),
      new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [new TextRun({ text: "Best embedding approach: ", bold: true }), new TextRun("E5-large-v2 + MLP probe achieved MAE of 0.86")] }),
      new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [new TextRun({ text: "Fine-tuned baseline: ", bold: true }), new TextRun("Qwen2.5-1.5B + LoRA achieved MAE of 0.68")] }),
      new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [new TextRun({ text: "Performance gap: ", bold: true }), new TextRun("26.4% higher error for embedding approach")] }),
      new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [new TextRun({ text: "Critical issue: ", bold: true }), new TextRun("Severe regression to the mean - only 3.6% of top-tier articles classified correctly")] }),
      new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [new TextRun({ text: "Recommendation: ", bold: true }), new TextRun("Fine-tuning is essential for finding high-quality content")] }),

      // Research Question
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Research Question")] }),
      new Paragraph({ spacing: { after: 200 }, children: [new TextRun(
        "Can frozen embeddings + linear/MLP probe match fine-tuned Qwen2.5-1.5B on semantic dimension scoring? This question is motivated by the potential benefits of embedding-based approaches: faster inference (<1ms vs 20-50ms), simpler training pipeline, and lower computational requirements."
      )] }),

      // Methodology
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Methodology")] }),
      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Baseline System")] }),
      new Table({
        columnWidths: [3120, 6240],
        rows: [
          new TableRow({ tableHeader: true, children: [
            cell("Component", { bold: true, width: 3120, header: true, shading: headerShading }),
            cell("Specification", { bold: true, width: 6240, header: true, shading: headerShading })
          ]}),
          new TableRow({ children: [
            cell("Model", { width: 3120 }), cell("Qwen2.5-1.5B + LoRA", { width: 6240 })
          ]}),
          new TableRow({ children: [
            cell("Trainable Parameters", { width: 3120, shading: altRowShading }), cell("18.5M (via LoRA)", { width: 6240, shading: altRowShading })
          ]}),
          new TableRow({ children: [
            cell("Training Time", { width: 3120 }), cell("2-3 hours on GPU", { width: 6240 })
          ]}),
          new TableRow({ children: [
            cell("Inference Time", { width: 3120, shading: altRowShading }), cell("20-50ms per article", { width: 6240, shading: altRowShading })
          ]}),
          new TableRow({ children: [
            cell("Baseline MAE", { width: 3120 }), cell("0.68", { width: 6240 })
          ]})
        ]
      }),

      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Embedding Models Tested")] }),
      new Table({
        columnWidths: [2800, 1400, 5160],
        rows: [
          new TableRow({ tableHeader: true, children: [
            cell("Model", { bold: true, width: 2800, header: true, shading: headerShading }),
            cell("Dimensions", { bold: true, width: 1400, header: true, shading: headerShading, align: AlignmentType.CENTER }),
            cell("Notes", { bold: true, width: 5160, header: true, shading: headerShading })
          ]}),
          new TableRow({ children: [
            cell("all-MiniLM-L6-v2", { width: 2800 }), cell("384", { width: 1400, align: AlignmentType.CENTER }), cell("Fast baseline, 22M params", { width: 5160 })
          ]}),
          new TableRow({ children: [
            cell("all-mpnet-base-v2", { width: 2800, shading: altRowShading }), cell("768", { width: 1400, align: AlignmentType.CENTER, shading: altRowShading }), cell("Quality baseline, 110M params", { width: 5160, shading: altRowShading })
          ]}),
          new TableRow({ children: [
            cell("bge-large-en-v1.5", { width: 2800 }), cell("1024", { width: 1400, align: AlignmentType.CENTER }), cell("MTEB top performer (BAAI)", { width: 5160 })
          ]}),
          new TableRow({ children: [
            cell("e5-large-v2", { width: 2800, shading: altRowShading }), cell("1024", { width: 1400, align: AlignmentType.CENTER, shading: altRowShading }), cell("Strong general embeddings (Microsoft)", { width: 5160, shading: altRowShading })
          ]})
        ]
      }),

      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Probe Architectures")] }),
      new Paragraph({ numbering: { reference: "numbered-list", level: 0 }, children: [new TextRun({ text: "Ridge Regression: ", bold: true }), new TextRun("Linear probe with L2 regularization (alpha cross-validated)")] }),
      new Paragraph({ numbering: { reference: "numbered-list", level: 0 }, children: [new TextRun({ text: "MLP (2-layer): ", bold: true }), new TextRun("256 → 128 → output, with dropout=0.2 and early stopping")] }),
      new Paragraph({ numbering: { reference: "numbered-list", level: 0 }, children: [new TextRun({ text: "LightGBM: ", bold: true }), new TextRun("Gradient boosted trees, one model per dimension")] }),

      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Dataset")] }),
      new Paragraph({ spacing: { after: 200 }, children: [new TextRun(
        "The uplifting_v5 dataset contains 10,000 articles scored on 6 semantic dimensions (human wellbeing impact, social cohesion impact, justice/rights impact, evidence level, benefit distribution, change durability). Data was split 80/10/10 for train/validation/test."
      )] }),

      // Page break before Results
      new Paragraph({ children: [new PageBreak()] }),

      // Results
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Results")] }),
      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Overall Comparison")] }),

      // Insert chart
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 200, after: 200 }, children: [
        new ImageRun({ type: "png", data: chartBuffer, transformation: { width: 550, height: 275 },
          altText: { title: "MAE Comparison Chart", description: "Bar chart comparing MAE across embedding models and probes", name: "comparison_chart" }
        })
      ]}),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 300 }, children: [
        new TextRun({ text: "Figure 1: MAE comparison across embedding models and probe types. The red dashed line indicates the fine-tuned baseline (0.68).", italics: true, size: 18 })
      ]}),

      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Detailed Results Table")] }),
      new Table({
        columnWidths: [2600, 1400, 1200, 1200, 1400, 1560],
        rows: [
          new TableRow({ tableHeader: true, children: [
            cell("Embedding Model", { bold: true, width: 2600, header: true, shading: headerShading }),
            cell("Probe", { bold: true, width: 1400, header: true, shading: headerShading, align: AlignmentType.CENTER }),
            cell("MAE", { bold: true, width: 1200, header: true, shading: headerShading, align: AlignmentType.CENTER }),
            cell("RMSE", { bold: true, width: 1200, header: true, shading: headerShading, align: AlignmentType.CENTER }),
            cell("Spearman", { bold: true, width: 1400, header: true, shading: headerShading, align: AlignmentType.CENTER }),
            cell("vs Baseline", { bold: true, width: 1560, header: true, shading: headerShading, align: AlignmentType.CENTER })
          ]}),
          new TableRow({ children: [
            cell("e5-large-v2", { width: 2600, bold: true }), cell("MLP", { width: 1400, align: AlignmentType.CENTER }),
            cell("0.860", { width: 1200, align: AlignmentType.CENTER, bold: true }), cell("1.108", { width: 1200, align: AlignmentType.CENTER }),
            cell("0.699", { width: 1400, align: AlignmentType.CENTER }), cell("+26.4%", { width: 1560, align: AlignmentType.CENTER })
          ]}),
          new TableRow({ children: [
            cell("e5-large-v2", { width: 2600, shading: altRowShading }), cell("Ridge", { width: 1400, align: AlignmentType.CENTER, shading: altRowShading }),
            cell("0.893", { width: 1200, align: AlignmentType.CENTER, shading: altRowShading }), cell("1.130", { width: 1200, align: AlignmentType.CENTER, shading: altRowShading }),
            cell("0.674", { width: 1400, align: AlignmentType.CENTER, shading: altRowShading }), cell("+31.4%", { width: 1560, align: AlignmentType.CENTER, shading: altRowShading })
          ]}),
          new TableRow({ children: [
            cell("bge-large-en-v1.5", { width: 2600 }), cell("MLP", { width: 1400, align: AlignmentType.CENTER }),
            cell("0.901", { width: 1200, align: AlignmentType.CENTER }), cell("1.169", { width: 1200, align: AlignmentType.CENTER }),
            cell("0.666", { width: 1400, align: AlignmentType.CENTER }), cell("+32.6%", { width: 1560, align: AlignmentType.CENTER })
          ]}),
          new TableRow({ children: [
            cell("e5-large-v2", { width: 2600, shading: altRowShading }), cell("LightGBM", { width: 1400, align: AlignmentType.CENTER, shading: altRowShading }),
            cell("0.905", { width: 1200, align: AlignmentType.CENTER, shading: altRowShading }), cell("1.145", { width: 1200, align: AlignmentType.CENTER, shading: altRowShading }),
            cell("0.670", { width: 1400, align: AlignmentType.CENTER, shading: altRowShading }), cell("+33.1%", { width: 1560, align: AlignmentType.CENTER, shading: altRowShading })
          ]}),
          new TableRow({ children: [
            cell("bge-large-en-v1.5", { width: 2600 }), cell("Ridge", { width: 1400, align: AlignmentType.CENTER }),
            cell("0.927", { width: 1200, align: AlignmentType.CENTER }), cell("1.175", { width: 1200, align: AlignmentType.CENTER }),
            cell("0.642", { width: 1400, align: AlignmentType.CENTER }), cell("+36.4%", { width: 1560, align: AlignmentType.CENTER })
          ]}),
          new TableRow({ children: [
            cell("all-mpnet-base-v2", { width: 2600, shading: altRowShading }), cell("MLP", { width: 1400, align: AlignmentType.CENTER, shading: altRowShading }),
            cell("0.939", { width: 1200, align: AlignmentType.CENTER, shading: altRowShading }), cell("1.218", { width: 1200, align: AlignmentType.CENTER, shading: altRowShading }),
            cell("0.623", { width: 1400, align: AlignmentType.CENTER, shading: altRowShading }), cell("+38.1%", { width: 1560, align: AlignmentType.CENTER, shading: altRowShading })
          ]}),
          new TableRow({ children: [
            cell("all-MiniLM-L6-v2", { width: 2600 }), cell("MLP", { width: 1400, align: AlignmentType.CENTER }),
            cell("0.975", { width: 1200, align: AlignmentType.CENTER }), cell("1.258", { width: 1200, align: AlignmentType.CENTER }),
            cell("0.589", { width: 1400, align: AlignmentType.CENTER }), cell("+43.4%", { width: 1560, align: AlignmentType.CENTER })
          ]}),
          new TableRow({ children: [
            cell("all-MiniLM-L6-v2", { width: 2600, shading: altRowShading }), cell("LightGBM", { width: 1400, align: AlignmentType.CENTER, shading: altRowShading }),
            cell("1.011", { width: 1200, align: AlignmentType.CENTER, shading: altRowShading }), cell("1.269", { width: 1200, align: AlignmentType.CENTER, shading: altRowShading }),
            cell("0.562", { width: 1400, align: AlignmentType.CENTER, shading: altRowShading }), cell("+48.6%", { width: 1560, align: AlignmentType.CENTER, shading: altRowShading })
          ]})
        ]
      }),

      // Per-dimension analysis
      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Per-Dimension Analysis (Best Model: E5-large-v2 + MLP)")] }),
      new Table({
        columnWidths: [3500, 1560, 1560, 1560, 1180],
        rows: [
          new TableRow({ tableHeader: true, children: [
            cell("Dimension", { bold: true, width: 3500, header: true, shading: headerShading }),
            cell("MAE", { bold: true, width: 1560, header: true, shading: headerShading, align: AlignmentType.CENTER }),
            cell("RMSE", { bold: true, width: 1560, header: true, shading: headerShading, align: AlignmentType.CENTER }),
            cell("Spearman", { bold: true, width: 1560, header: true, shading: headerShading, align: AlignmentType.CENTER }),
            cell("Rank", { bold: true, width: 1180, header: true, shading: headerShading, align: AlignmentType.CENTER })
          ]}),
          new TableRow({ children: [
            cell("evidence_level", { width: 3500 }), cell("0.738", { width: 1560, align: AlignmentType.CENTER }),
            cell("0.973", { width: 1560, align: AlignmentType.CENTER }), cell("0.731", { width: 1560, align: AlignmentType.CENTER }), cell("1 (best)", { width: 1180, align: AlignmentType.CENTER })
          ]}),
          new TableRow({ children: [
            cell("justice_rights_impact", { width: 3500, shading: altRowShading }), cell("0.803", { width: 1560, align: AlignmentType.CENTER, shading: altRowShading }),
            cell("1.053", { width: 1560, align: AlignmentType.CENTER, shading: altRowShading }), cell("0.745", { width: 1560, align: AlignmentType.CENTER, shading: altRowShading }), cell("2", { width: 1180, align: AlignmentType.CENTER, shading: altRowShading })
          ]}),
          new TableRow({ children: [
            cell("change_durability", { width: 3500 }), cell("0.833", { width: 1560, align: AlignmentType.CENTER }),
            cell("1.049", { width: 1560, align: AlignmentType.CENTER }), cell("0.668", { width: 1560, align: AlignmentType.CENTER }), cell("3", { width: 1180, align: AlignmentType.CENTER })
          ]}),
          new TableRow({ children: [
            cell("social_cohesion_impact", { width: 3500, shading: altRowShading }), cell("0.874", { width: 1560, align: AlignmentType.CENTER, shading: altRowShading }),
            cell("1.130", { width: 1560, align: AlignmentType.CENTER, shading: altRowShading }), cell("0.719", { width: 1560, align: AlignmentType.CENTER, shading: altRowShading }), cell("4", { width: 1180, align: AlignmentType.CENTER, shading: altRowShading })
          ]}),
          new TableRow({ children: [
            cell("human_wellbeing_impact", { width: 3500 }), cell("0.906", { width: 1560, align: AlignmentType.CENTER }),
            cell("1.149", { width: 1560, align: AlignmentType.CENTER }), cell("0.691", { width: 1560, align: AlignmentType.CENTER }), cell("5", { width: 1180, align: AlignmentType.CENTER })
          ]}),
          new TableRow({ children: [
            cell("benefit_distribution", { width: 3500, shading: altRowShading }), cell("1.005", { width: 1560, align: AlignmentType.CENTER, shading: altRowShading }),
            cell("1.269", { width: 1560, align: AlignmentType.CENTER, shading: altRowShading }), cell("0.640", { width: 1560, align: AlignmentType.CENTER, shading: altRowShading }), cell("6 (worst)", { width: 1180, align: AlignmentType.CENTER, shading: altRowShading })
          ]})
        ]
      }),

      // Page break before Error Distribution Analysis
      new Paragraph({ children: [new PageBreak()] }),

      // Error Distribution Analysis - NEW SECTION
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Error Distribution Analysis")] }),
      new Paragraph({ spacing: { after: 200 }, children: [new TextRun(
        "Beyond aggregate MAE metrics, understanding how errors are distributed reveals critical insights about the embedding approach's limitations."
      )] }),

      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Overall Error Statistics")] }),
      new Table({
        columnWidths: [4680, 4680],
        rows: [
          new TableRow({ tableHeader: true, children: [
            cell("Metric", { bold: true, width: 4680, header: true, shading: headerShading }),
            cell("Value", { bold: true, width: 4680, header: true, shading: headerShading, align: AlignmentType.CENTER })
          ]}),
          new TableRow({ children: [
            cell("Mean Error (bias)", { width: 4680 }), cell("-0.14", { width: 4680, align: AlignmentType.CENTER })
          ]}),
          new TableRow({ children: [
            cell("Median Error", { width: 4680, shading: altRowShading }), cell("-0.10", { width: 4680, align: AlignmentType.CENTER, shading: altRowShading })
          ]}),
          new TableRow({ children: [
            cell("Standard Deviation", { width: 4680 }), cell("1.22", { width: 4680, align: AlignmentType.CENTER })
          ]}),
          new TableRow({ children: [
            cell("Skewness", { width: 4680, shading: altRowShading }), cell("-0.22 (approximately symmetric)", { width: 4680, align: AlignmentType.CENTER, shading: altRowShading })
          ]}),
          new TableRow({ children: [
            cell("5th - 95th Percentile", { width: 4680 }), cell("-2.24 to +1.76", { width: 4680, align: AlignmentType.CENTER })
          ]})
        ]
      }),

      // Error distribution figure
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 300, after: 200 }, children: [
        new ImageRun({ type: "png", data: errorDistBuffer, transformation: { width: 550, height: 400 },
          altText: { title: "Error Distribution", description: "Histogram and Q-Q plot of prediction errors", name: "error_distribution" }
        })
      ]}),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 300 }, children: [
        new TextRun({ text: "Figure 2: Error distribution histogram (top left), Q-Q plot vs normal distribution (top right), and MAE by dimension (bottom right).", italics: true, size: 18 })
      ]}),

      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Critical Finding: Regression to the Mean")] }),
      new Paragraph({ spacing: { after: 200 }, children: [new TextRun(
        "The embedding approach exhibits severe regression to the mean. Errors vary systematically by true score, not randomly:"
      )] }),

      new Table({
        columnWidths: [2000, 1400, 1800, 1400, 2760],
        rows: [
          new TableRow({ tableHeader: true, children: [
            cell("True Score", { bold: true, width: 2000, header: true, shading: headerShading }),
            cell("Count", { bold: true, width: 1400, header: true, shading: headerShading, align: AlignmentType.CENTER }),
            cell("Mean Error", { bold: true, width: 1800, header: true, shading: headerShading, align: AlignmentType.CENTER }),
            cell("MAE", { bold: true, width: 1400, header: true, shading: headerShading, align: AlignmentType.CENTER }),
            cell("Interpretation", { bold: true, width: 2760, header: true, shading: headerShading })
          ]}),
          new TableRow({ children: [
            cell("0-3 (low)", { width: 2000 }), cell("1,984", { width: 1400, align: AlignmentType.CENTER }),
            cell("+0.66", { width: 1800, align: AlignmentType.CENTER, bold: true }), cell("0.86", { width: 1400, align: AlignmentType.CENTER }),
            cell("Overestimates low scores", { width: 2760 })
          ]}),
          new TableRow({ children: [
            cell("3-5 (medium)", { width: 2000, shading: altRowShading }), cell("2,374", { width: 1400, align: AlignmentType.CENTER, shading: altRowShading }),
            cell("-0.05", { width: 1800, align: AlignmentType.CENTER, shading: altRowShading }), cell("0.73", { width: 1400, align: AlignmentType.CENTER, shading: altRowShading }),
            cell("Nearly unbiased", { width: 2760, shading: altRowShading })
          ]}),
          new TableRow({ children: [
            cell("5-7 (good)", { width: 2000 }), cell("1,395", { width: 1400, align: AlignmentType.CENTER }),
            cell("-1.12", { width: 1800, align: AlignmentType.CENTER, bold: true }), cell("1.27", { width: 1400, align: AlignmentType.CENTER }),
            cell("Underestimates good scores", { width: 2760 })
          ]}),
          new TableRow({ children: [
            cell("7-10 (high)", { width: 2000, shading: altRowShading }), cell("253", { width: 1400, align: AlignmentType.CENTER, shading: altRowShading }),
            cell("-1.99", { width: 1800, align: AlignmentType.CENTER, bold: true, shading: altRowShading }), cell("2.01", { width: 1400, align: AlignmentType.CENTER, shading: altRowShading }),
            cell("Severely underestimates best", { width: 2760, shading: altRowShading })
          ]})
        ]
      }),

      // Error vs Score figure
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 300, after: 200 }, children: [
        new ImageRun({ type: "png", data: errorVsScoreBuffer, transformation: { width: 550, height: 330 },
          altText: { title: "Error vs True Score", description: "Scatter plot showing systematic bias by score range", name: "error_vs_score" }
        })
      ]}),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 300 }, children: [
        new TextRun({ text: "Figure 3: Error vs true score. The red line shows mean error by score range. Above zero = overestimate, below = underestimate. The embedding approach systematically pulls all predictions toward the middle.", italics: true, size: 18 })
      ]}),

      new Paragraph({ spacing: { after: 200 }, children: [
        new TextRun({ text: "This is the most important finding. ", bold: true }),
        new TextRun("The embedding approach systematically inflates low scores (making junk look better) and deflates high scores (burying the best content). This behavior directly undermines the use case of finding high-quality constructive news articles.")
      ]}),

      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Tier Classification Accuracy")] }),
      new Paragraph({ spacing: { after: 200 }, children: [new TextRun(
        "For practical filtering, what matters is tier assignment. Using thresholds at 3, 5, and 7:"
      )] }),

      new Table({
        columnWidths: [2600, 1800, 2100, 2860],
        rows: [
          new TableRow({ tableHeader: true, children: [
            cell("Tier", { bold: true, width: 2600, header: true, shading: headerShading }),
            cell("True Count", { bold: true, width: 1800, header: true, shading: headerShading, align: AlignmentType.CENTER }),
            cell("Exact Match", { bold: true, width: 2100, header: true, shading: headerShading, align: AlignmentType.CENTER }),
            cell("Within 1 Tier", { bold: true, width: 2860, header: true, shading: headerShading, align: AlignmentType.CENTER })
          ]}),
          new TableRow({ children: [
            cell("Tier 0 (<3, junk)", { width: 2600 }), cell("1,984", { width: 1800, align: AlignmentType.CENTER }),
            cell("83.0%", { width: 2100, align: AlignmentType.CENTER, bold: true }), cell("99.6%", { width: 2860, align: AlignmentType.CENTER })
          ]}),
          new TableRow({ children: [
            cell("Tier 1 (3-5, meh)", { width: 2600, shading: altRowShading }), cell("2,374", { width: 1800, align: AlignmentType.CENTER, shading: altRowShading }),
            cell("55.5%", { width: 2100, align: AlignmentType.CENTER, shading: altRowShading }), cell("100.0%", { width: 2860, align: AlignmentType.CENTER, shading: altRowShading })
          ]}),
          new TableRow({ children: [
            cell("Tier 2 (5-7, good)", { width: 2600 }), cell("1,395", { width: 1800, align: AlignmentType.CENTER }),
            cell("22.7%", { width: 2100, align: AlignmentType.CENTER, bold: true }), cell("87.0%", { width: 2860, align: AlignmentType.CENTER })
          ]}),
          new TableRow({ children: [
            cell("Tier 3 (>=7, great)", { width: 2600, shading: altRowShading }), cell("253", { width: 1800, align: AlignmentType.CENTER, shading: altRowShading }),
            cell("3.6%", { width: 2100, align: AlignmentType.CENTER, bold: true, shading: altRowShading }), cell("53.0%", { width: 2860, align: AlignmentType.CENTER, shading: altRowShading })
          ]})
        ]
      }),

      new Paragraph({ spacing: { before: 200, after: 200 }, children: [
        new TextRun({ text: "Only 3.6% of truly great articles (tier 3) are classified correctly. ", bold: true }),
        new TextRun("Half of them get pushed down 2+ tiers. For NexusMind/ovr.news, where the goal is to surface the best constructive news, the embedding approach would systematically bury exactly the content users want to find.")
      ]}),

      // Page break before Discussion
      new Paragraph({ children: [new PageBreak()] }),

      // Discussion
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Discussion")] }),

      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Key Observations")] }),
      new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [new TextRun({ text: "Embedding quality matters: ", bold: true }), new TextRun("Larger models (E5, BGE at 1024 dims) consistently outperformed smaller ones (MiniLM at 384 dims)")] }),
      new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [new TextRun({ text: "MLP beats linear: ", bold: true }), new TextRun("Non-linear probes (MLP) outperformed linear probes (Ridge) by 3-5%, suggesting dimensions require non-linear combinations of embedding features")] }),
      new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [new TextRun({ text: "LightGBM underperformed: ", bold: true }), new TextRun("Tree-based methods did not match neural approaches, possibly due to the continuous nature of embedding spaces")] }),
      new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [new TextRun({ text: "Dimension difficulty varies: ", bold: true }), new TextRun("'evidence_level' was easiest to predict (MAE 0.74), while 'benefit_distribution' was hardest (MAE 1.01)")] }),

      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Why Fine-Tuning Wins")] }),
      new Paragraph({ spacing: { after: 200 }, children: [new TextRun(
        "The 26% performance gap suggests that fine-tuning enables the model to learn task-specific representations that frozen embeddings cannot capture. General-purpose embeddings optimize for semantic similarity, not for predicting nuanced editorial dimensions like 'benefit distribution' or 'change durability'."
      )] }),

      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Efficiency Trade-offs")] }),
      new Table({
        columnWidths: [2500, 2600, 2200, 2060],
        rows: [
          new TableRow({ tableHeader: true, children: [
            cell("Metric", { bold: true, width: 2500, header: true, shading: headerShading }),
            cell("Fine-tuned Qwen", { bold: true, width: 2600, header: true, shading: headerShading, align: AlignmentType.CENTER }),
            cell("Embedding + Probe", { bold: true, width: 2200, header: true, shading: headerShading, align: AlignmentType.CENTER }),
            cell("Advantage", { bold: true, width: 2060, header: true, shading: headerShading, align: AlignmentType.CENTER })
          ]}),
          new TableRow({ children: [
            cell("Training Time", { width: 2500 }), cell("2-3 hours", { width: 2600, align: AlignmentType.CENTER }),
            cell("~5 minutes", { width: 2200, align: AlignmentType.CENTER }), cell("Embedding 20-30x", { width: 2060, align: AlignmentType.CENTER })
          ]}),
          new TableRow({ children: [
            cell("Inference Time", { width: 2500, shading: altRowShading }), cell("20-50ms", { width: 2600, align: AlignmentType.CENTER, shading: altRowShading }),
            cell("<1ms", { width: 2200, align: AlignmentType.CENTER, shading: altRowShading }), cell("Embedding 20-50x", { width: 2060, align: AlignmentType.CENTER, shading: altRowShading })
          ]}),
          new TableRow({ children: [
            cell("Model Size", { width: 2500 }), cell("1.5B + 18M LoRA", { width: 2600, align: AlignmentType.CENTER }),
            cell("~500MB + <1MB", { width: 2200, align: AlignmentType.CENTER }), cell("Embedding 3x", { width: 2060, align: AlignmentType.CENTER })
          ]}),
          new TableRow({ children: [
            cell("Accuracy (MAE)", { width: 2500, shading: altRowShading }), cell("0.68", { width: 2600, align: AlignmentType.CENTER, shading: altRowShading }),
            cell("0.86", { width: 2200, align: AlignmentType.CENTER, shading: altRowShading }), cell("Fine-tuned +26%", { width: 2060, align: AlignmentType.CENTER, shading: altRowShading })
          ]})
        ]
      }),

      // Recommendations
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Recommendations")] }),
      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Primary Recommendation: Continue with Fine-Tuning")] }),
      new Paragraph({ spacing: { after: 200 }, children: [new TextRun(
        "For production use where accuracy is the primary concern, fine-tuning provides a significant 26% advantage that justifies the additional training cost. The semantic dimensions used in LLM Distillery filters require nuanced understanding that general embeddings cannot fully capture."
      )] }),

      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("When to Consider Embeddings")] }),
      new Paragraph({ numbering: { reference: "recommendations", level: 0 }, children: [new TextRun({ text: "Rapid prototyping: ", bold: true }), new TextRun("Use embeddings for quick experiments before committing to fine-tuning")] }),
      new Paragraph({ numbering: { reference: "recommendations", level: 0 }, children: [new TextRun({ text: "Inference-critical applications: ", bold: true }), new TextRun("If sub-millisecond inference is required and 26% accuracy loss is acceptable")] }),
      new Paragraph({ numbering: { reference: "recommendations", level: 0 }, children: [new TextRun({ text: "Resource-constrained environments: ", bold: true }), new TextRun("When GPU training is unavailable but pre-computed embeddings can be generated")] }),
      new Paragraph({ numbering: { reference: "recommendations", level: 0 }, children: [new TextRun({ text: "Multi-filter systems: ", bold: true }), new TextRun("Embeddings can be shared across filters, amortizing computation cost")] }),

      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Future Research Directions")] }),
      new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [new TextRun("Test larger embedding models (e5-mistral-7b-instruct) to close the accuracy gap")] }),
      new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [new TextRun("Explore task-specific embedding fine-tuning (contrastive learning on scored articles)")] }),
      new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [new TextRun("Investigate ensemble approaches combining multiple embedding models")] }),
      new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [new TextRun("Test on additional filters (sustainability_technology, investment_risk) to assess generalization")] }),

      // Conclusion
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Conclusion")] }),
      new Paragraph({ spacing: { after: 200 }, children: [new TextRun(
        "This research demonstrates that embedding-based approaches cannot replace fine-tuned models for semantic dimension scoring—and the reasons go beyond the 26% MAE gap."
      )] }),
      new Paragraph({ spacing: { after: 200 }, children: [
        new TextRun({ text: "The critical finding is regression to the mean: ", bold: true }),
        new TextRun("the embedding approach systematically underestimates high scores and overestimates low scores. For tier 3 articles (score >=7), only 3.6% are classified correctly, and half are pushed down 2+ tiers. This makes embedding approaches fundamentally unsuitable for surfacing high-quality content.")
      ] }),
      new Paragraph({ spacing: { after: 400 }, children: [new TextRun(
        "For NexusMind and similar applications where the goal is finding the best constructive news, fine-tuning is not optional—it's essential. The embedding approach could serve as a fast pre-filter for rejecting obvious non-matches (83% accuracy on tier 0), but production scoring requires fine-tuned models."
      )] }),

      // Appendix
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Appendix: Experimental Details")] }),
      new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun("Environment")] }),
      new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [new TextRun("GPU: NVIDIA GeForce RTX 4080 (16GB VRAM)")] }),
      new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [new TextRun("Python 3.13, PyTorch 2.9.0+cu128")] }),
      new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [new TextRun("sentence-transformers, scikit-learn, LightGBM")] }),

      new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun("Hyperparameters")] }),
      new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [new TextRun("Ridge: alpha cross-validated from [0.01, 0.1, 1.0, 10.0, 100.0] (best: 100.0)")] }),
      new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [new TextRun("MLP: 256→128→output, dropout=0.2, lr=0.001, patience=10")] }),
      new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [new TextRun("LightGBM: 500 estimators, lr=0.05, max_depth=6, early_stopping=50")] }),

      new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun("Reproducibility")] }),
      new Paragraph({ spacing: { after: 200 }, children: [new TextRun("All experiments used random seed 42. Code available at: research/embedding_vs_finetuning/")] })
    ]
  }]
});

// Generate document
const outputPath = 'C:/local_dev/llm-distillery/research/embedding_vs_finetuning/results/Embedding_vs_Finetuning_Research_Report.docx';
Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync(outputPath, buffer);
  console.log('Report generated: ' + outputPath);
}).catch(err => console.error('Error:', err));
