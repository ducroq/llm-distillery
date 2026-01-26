const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
        AlignmentType, HeadingLevel, BorderStyle, WidthType, ShadingType,
        LevelFormat, PageBreak } = require('docx');
const fs = require('fs');

// Table styling helpers
const tableBorder = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const cellBorders = { top: tableBorder, bottom: tableBorder, left: tableBorder, right: tableBorder };

function headerCell(text, width) {
  return new TableCell({
    borders: cellBorders,
    width: { size: width, type: WidthType.DXA },
    shading: { fill: "D5E8F0", type: ShadingType.CLEAR },
    children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ text, bold: true, size: 22 })]
    })]
  });
}

function dataCell(text, width, align = AlignmentType.LEFT) {
  return new TableCell({
    borders: cellBorders,
    width: { size: width, type: WidthType.DXA },
    children: [new Paragraph({
      alignment: align,
      children: [new TextRun({ text, size: 22 })]
    })]
  });
}

function boldDataCell(text, width, align = AlignmentType.LEFT) {
  return new TableCell({
    borders: cellBorders,
    width: { size: width, type: WidthType.DXA },
    children: [new Paragraph({
      alignment: align,
      children: [new TextRun({ text, bold: true, size: 22 })]
    })]
  });
}

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 24 } } },
    paragraphStyles: [
      { id: "Title", name: "Title", basedOn: "Normal",
        run: { size: 56, bold: true, color: "000000", font: "Arial" },
        paragraph: { spacing: { before: 240, after: 120 }, alignment: AlignmentType.CENTER } },
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, color: "000000", font: "Arial" },
        paragraph: { spacing: { before: 360, after: 240 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, color: "000000", font: "Arial" },
        paragraph: { spacing: { before: 280, after: 180 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, color: "000000", font: "Arial" },
        paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 2 } }
    ]
  },
  numbering: {
    config: [
      { reference: "bullet-list",
        levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "numbered-list-1",
        levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "numbered-list-2",
        levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] }
    ]
  },
  sections: [{
    properties: {
      page: { margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } }
    },
    children: [
      // Title
      new Paragraph({
        heading: HeadingLevel.TITLE,
        children: [new TextRun("Context Length Research Report")]
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 400 },
        children: [new TextRun({ text: "Dataset: uplifting_v5", italics: true, size: 24 })]
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 400 },
        children: [new TextRun({ text: "Generated: 2025-01-26", italics: true, size: 22 })]
      }),

      // Executive Summary
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Executive Summary")] }),
      new Paragraph({
        spacing: { after: 200 },
        children: [new TextRun("This report documents experiments with different context lengths for fine-tuning Qwen2.5-1.5B on the uplifting_v5 dataset. The research question: Can we improve model quality by using longer context windows, and what is the optimal trade-off between quality and inference cost?")]
      }),
      new Paragraph({
        spacing: { after: 200 },
        children: [new TextRun({ text: "Key Findings:", bold: true })]
      }),
      new Paragraph({ numbering: { reference: "bullet-list", level: 0 },
        children: [new TextRun("Baseline (512 tokens): MAE 0.680")] }),
      new Paragraph({ numbering: { reference: "bullet-list", level: 0 },
        children: [new TextRun("1024 tokens: MAE 0.652 (-4.1%)")] }),
      new Paragraph({ numbering: { reference: "bullet-list", level: 0 },
        children: [new TextRun("2048 tokens: MAE 0.627 (-7.8%)")] }),
      new Paragraph({ numbering: { reference: "bullet-list", level: 0 },
        children: [new TextRun({ text: "Head+Tail (256+256): MAE 0.655 (-3.7%) with baseline inference speed", bold: true })] }),
      new Paragraph({
        spacing: { before: 200, after: 200 },
        children: [new TextRun({ text: "Recommendation: ", bold: true }), new TextRun("Use head+tail extraction for production. It achieves nearly the same quality as 1024-token training while maintaining baseline inference speed.")]
      }),

      // Motivation
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Motivation")] }),
      new Paragraph({
        spacing: { after: 200 },
        children: [new TextRun("Qwen2.5-1.5B supports up to 128K tokens, but our baseline training used only 512 tokens. This was discovered to cause significant truncation of longer articles, potentially losing important information about outcomes and conclusions that typically appear at the end of news articles.")]
      }),

      // Article Length Analysis
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Article Length Analysis")] }),
      new Paragraph({
        spacing: { after: 200 },
        children: [new TextRun("Analysis of the uplifting_v5 training dataset reveals that high-scoring articles tend to be longer, making truncation particularly problematic for quality content.")]
      }),
      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Length Distribution")] }),
      new Table({
        columnWidths: [2340, 2340, 2340, 2340],
        rows: [
          new TableRow({ children: [
            headerCell("Length Category", 2340),
            headerCell("% of Articles", 2340),
            headerCell("% of High-Scorers (≥6)", 2340),
            headerCell("Truncated at 512", 2340)
          ]}),
          new TableRow({ children: [
            dataCell("≤ 512 tokens", 2340),
            dataCell("77.7%", 2340, AlignmentType.CENTER),
            dataCell("43%", 2340, AlignmentType.CENTER),
            dataCell("No", 2340, AlignmentType.CENTER)
          ]}),
          new TableRow({ children: [
            dataCell("512-1024 tokens", 2340),
            dataCell("12.0%", 2340, AlignmentType.CENTER),
            dataCell("49%", 2340, AlignmentType.CENTER),
            dataCell("Yes", 2340, AlignmentType.CENTER)
          ]}),
          new TableRow({ children: [
            dataCell("1024-2048 tokens", 2340),
            dataCell("8.6%", 2340, AlignmentType.CENTER),
            dataCell("7.5%", 2340, AlignmentType.CENTER),
            dataCell("Yes", 2340, AlignmentType.CENTER)
          ]}),
          new TableRow({ children: [
            dataCell("> 2048 tokens", 2340),
            dataCell("1.7%", 2340, AlignmentType.CENTER),
            dataCell("0%", 2340, AlignmentType.CENTER),
            dataCell("Yes", 2340, AlignmentType.CENTER)
          ]})
        ]
      }),
      new Paragraph({
        spacing: { before: 200, after: 200 },
        children: [new TextRun({ text: "Critical insight: ", bold: true }), new TextRun("High-scoring articles tend to be longer. At 512 tokens, 56.9% of high-scorers are truncated, potentially losing crucial information about documented outcomes.")]
      }),

      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Truncation Impact on High-Scorers")] }),
      new Table({
        columnWidths: [4680, 4680],
        rows: [
          new TableRow({ children: [
            headerCell("Context Limit", 4680),
            headerCell("High-Scorers Truncated", 4680)
          ]}),
          new TableRow({ children: [
            dataCell("512 tokens", 4680),
            dataCell("56.9%", 4680, AlignmentType.CENTER)
          ]}),
          new TableRow({ children: [
            dataCell("1024 tokens", 4680),
            dataCell("31.6%", 4680, AlignmentType.CENTER)
          ]}),
          new TableRow({ children: [
            dataCell("2048 tokens", 4680),
            dataCell("7.5%", 4680, AlignmentType.CENTER)
          ]}),
          new TableRow({ children: [
            dataCell("4096 tokens", 4680),
            dataCell("0.0%", 4680, AlignmentType.CENTER)
          ]})
        ]
      }),

      // Page break before methodology
      new Paragraph({ children: [new PageBreak()] }),

      // Methodology
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Methodology")] }),
      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Training Configuration")] }),
      new Paragraph({
        spacing: { after: 200 },
        children: [new TextRun("All experiments used identical training configuration except for context length and batch size (adjusted for GPU memory):")]
      }),
      new Paragraph({ numbering: { reference: "bullet-list", level: 0 },
        children: [new TextRun("Model: Qwen/Qwen2.5-1.5B with LoRA (1.18% trainable parameters)")] }),
      new Paragraph({ numbering: { reference: "bullet-list", level: 0 },
        children: [new TextRun("Training data: 7,999 articles, Validation: 1,000 articles")] }),
      new Paragraph({ numbering: { reference: "bullet-list", level: 0 },
        children: [new TextRun("Epochs: 3")] }),
      new Paragraph({ numbering: { reference: "bullet-list", level: 0 },
        children: [new TextRun("Learning rate: 2e-05")] }),
      new Paragraph({ numbering: { reference: "bullet-list", level: 0 },
        children: [new TextRun("Warmup steps: 500")] }),
      new Paragraph({ numbering: { reference: "bullet-list", level: 0 },
        children: [new TextRun("GPU: 16GB VRAM")] }),

      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Experiments")] }),
      new Table({
        columnWidths: [2340, 2340, 2340, 2340],
        rows: [
          new TableRow({ children: [
            headerCell("Experiment", 2340),
            headerCell("Max Length", 2340),
            headerCell("Batch Size", 2340),
            headerCell("Description", 2340)
          ]}),
          new TableRow({ children: [
            dataCell("Baseline", 2340),
            dataCell("512", 2340, AlignmentType.CENTER),
            dataCell("8", 2340, AlignmentType.CENTER),
            dataCell("Standard truncation", 2340)
          ]}),
          new TableRow({ children: [
            dataCell("1024tok", 2340),
            dataCell("1024", 2340, AlignmentType.CENTER),
            dataCell("4", 2340, AlignmentType.CENTER),
            dataCell("2x context", 2340)
          ]}),
          new TableRow({ children: [
            dataCell("2048tok", 2340),
            dataCell("2048", 2340, AlignmentType.CENTER),
            dataCell("1", 2340, AlignmentType.CENTER),
            dataCell("4x context", 2340)
          ]}),
          new TableRow({ children: [
            boldDataCell("head_tail", 2340),
            dataCell("512 (256+256)", 2340, AlignmentType.CENTER),
            dataCell("4", 2340, AlignmentType.CENTER),
            dataCell("First + last tokens", 2340)
          ]})
        ]
      }),

      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Head+Tail Extraction Method")] }),
      new Paragraph({
        spacing: { after: 200 },
        children: [new TextRun("The head+tail approach extracts the first 256 tokens (introduction/context) and last 256 tokens (conclusion/outcomes) from each article, joining them with a separator:")]
      }),
      new Paragraph({
        spacing: { after: 100 },
        shading: { fill: "F0F0F0", type: ShadingType.CLEAR },
        children: [new TextRun({ text: "head_text = tokenizer.decode(tokens[:256])", font: "Courier New", size: 20 })]
      }),
      new Paragraph({
        spacing: { after: 100 },
        shading: { fill: "F0F0F0", type: ShadingType.CLEAR },
        children: [new TextRun({ text: "tail_text = tokenizer.decode(tokens[-256:])", font: "Courier New", size: 20 })]
      }),
      new Paragraph({
        spacing: { after: 200 },
        shading: { fill: "F0F0F0", type: ShadingType.CLEAR },
        children: [new TextRun({ text: 'return head_text + " [...] " + tail_text', font: "Courier New", size: 20 })]
      }),
      new Paragraph({
        spacing: { after: 200 },
        children: [new TextRun({ text: "Rationale: ", bold: true }), new TextRun("News articles typically present key context at the beginning and documented outcomes/conclusions at the end. The middle often contains detailed explanation that is less critical for scoring.")]
      }),

      // Page break before results
      new Paragraph({ children: [new PageBreak()] }),

      // Results
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Results")] }),
      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Model Comparison")] }),
      new Table({
        columnWidths: [1872, 1872, 1872, 1872, 1872],
        rows: [
          new TableRow({ children: [
            headerCell("Model", 1872),
            headerCell("Max Length", 1872),
            headerCell("Best Val MAE", 1872),
            headerCell("vs Baseline", 1872),
            headerCell("Inference Speed", 1872)
          ]}),
          new TableRow({ children: [
            dataCell("Baseline", 1872),
            dataCell("512", 1872, AlignmentType.CENTER),
            dataCell("0.680", 1872, AlignmentType.CENTER),
            dataCell("-", 1872, AlignmentType.CENTER),
            dataCell("1x", 1872, AlignmentType.CENTER)
          ]}),
          new TableRow({ children: [
            dataCell("1024tok", 1872),
            dataCell("1024", 1872, AlignmentType.CENTER),
            dataCell("0.652", 1872, AlignmentType.CENTER),
            dataCell("-4.1%", 1872, AlignmentType.CENTER),
            dataCell("~2x slower", 1872, AlignmentType.CENTER)
          ]}),
          new TableRow({ children: [
            boldDataCell("head_tail", 1872),
            boldDataCell("512 (256+256)", 1872, AlignmentType.CENTER),
            boldDataCell("0.655", 1872, AlignmentType.CENTER),
            boldDataCell("-3.7%", 1872, AlignmentType.CENTER),
            boldDataCell("1x", 1872, AlignmentType.CENTER)
          ]}),
          new TableRow({ children: [
            dataCell("2048tok", 1872),
            dataCell("2048", 1872, AlignmentType.CENTER),
            dataCell("0.627", 1872, AlignmentType.CENTER),
            dataCell("-7.8%", 1872, AlignmentType.CENTER),
            dataCell("~4x slower", 1872, AlignmentType.CENTER)
          ]})
        ]
      }),

      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Key Observations")] }),
      new Paragraph({ numbering: { reference: "numbered-list-1", level: 0 },
        children: [new TextRun({ text: "Longer context consistently improves MAE ", bold: true }), new TextRun("- each doubling of context reduces error, but with diminishing returns.")] }),
      new Paragraph({ numbering: { reference: "numbered-list-1", level: 0 },
        children: [new TextRun({ text: "Head+tail nearly matches 1024tok ", bold: true }), new TextRun("- only 0.003 MAE difference (0.655 vs 0.652) while using same 512 token length.")] }),
      new Paragraph({ numbering: { reference: "numbered-list-1", level: 0 },
        children: [new TextRun({ text: "Training cost scales with context ", bold: true }), new TextRun("- 2048 tokens requires batch_size=1 on 16GB GPU.")] }),
      new Paragraph({ numbering: { reference: "numbered-list-1", level: 0 },
        children: [new TextRun({ text: "Inference cost scales quadratically ", bold: true }), new TextRun("- attention is O(n²) in sequence length.")] }),

      // Inference Cost Analysis
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Inference Cost Analysis")] }),
      new Paragraph({
        spacing: { after: 200 },
        children: [new TextRun("CPU inference benchmarks with INT8 quantization:")]
      }),
      new Table({
        columnWidths: [3120, 3120, 3120],
        rows: [
          new TableRow({ children: [
            headerCell("Context Length", 3120),
            headerCell("Time per Article", 3120),
            headerCell("10K Articles", 3120)
          ]}),
          new TableRow({ children: [
            dataCell("512 tokens", 3120),
            dataCell("1.6s", 3120, AlignmentType.CENTER),
            dataCell("4.4 hours", 3120, AlignmentType.CENTER)
          ]}),
          new TableRow({ children: [
            dataCell("1024 tokens", 3120),
            dataCell("3.5s", 3120, AlignmentType.CENTER),
            dataCell("9.7 hours", 3120, AlignmentType.CENTER)
          ]}),
          new TableRow({ children: [
            dataCell("2048 tokens", 3120),
            dataCell("~7s", 3120, AlignmentType.CENTER),
            dataCell("~19 hours", 3120, AlignmentType.CENTER)
          ]}),
          new TableRow({ children: [
            dataCell("4096 tokens", 3120),
            dataCell("~15s", 3120, AlignmentType.CENTER),
            dataCell("~42 hours", 3120, AlignmentType.CENTER)
          ]})
        ]
      }),
      new Paragraph({
        spacing: { before: 200, after: 200 },
        children: [new TextRun({ text: "Critical insight: ", bold: true }), new TextRun("Head+tail extraction maintains the 512-token inference speed (1.6s/article) while capturing information from longer articles that would otherwise be truncated.")]
      }),

      // Page break before conclusions
      new Paragraph({ children: [new PageBreak()] }),

      // Conclusions
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Conclusions")] }),
      new Paragraph({ numbering: { reference: "numbered-list-2", level: 0 },
        children: [new TextRun({ text: "Longer context helps quality but has diminishing returns. ", bold: true }), new TextRun("512→1024 gave 4.1% improvement, 1024→2048 gave only 3.7% more.")] }),
      new Paragraph({ numbering: { reference: "numbered-list-2", level: 0 },
        children: [new TextRun({ text: "Head+tail is the optimal strategy. ", bold: true }), new TextRun("Achieves 3.7% improvement over baseline with no inference cost increase.")] }),
      new Paragraph({ numbering: { reference: "numbered-list-2", level: 0 },
        children: [new TextRun({ text: "Article structure matters. ", bold: true }), new TextRun("Intro (context) and conclusion (outcomes) contain most of the signal for uplifting content scoring.")] }),
      new Paragraph({ numbering: { reference: "numbered-list-2", level: 0 },
        children: [new TextRun({ text: "2048tok is impractical for production. ", bold: true }), new TextRun("4x slower inference for only 4.3% additional improvement over head+tail.")] }),

      // Recommendations
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Recommendations")] }),
      new Paragraph({
        spacing: { after: 200 },
        children: [new TextRun({ text: "Primary recommendation: ", bold: true }), new TextRun("Use head+tail (256+256) extraction for production deployment.")]
      }),
      new Paragraph({ numbering: { reference: "bullet-list", level: 0 },
        children: [new TextRun("Best quality-speed tradeoff (MAE 0.655 at baseline speed)")] }),
      new Paragraph({ numbering: { reference: "bullet-list", level: 0 },
        children: [new TextRun("3.7% better than baseline with no inference cost increase")] }),
      new Paragraph({ numbering: { reference: "bullet-list", level: 0 },
        children: [new TextRun("Only 0.003 MAE worse than 1024tok but 2x faster")] }),
      new Paragraph({ numbering: { reference: "bullet-list", level: 0 },
        children: [new TextRun("Captures intro (context) + conclusion (outcomes) which are most relevant for scoring")] }),
      new Paragraph({
        spacing: { before: 200, after: 200 },
        children: [new TextRun({ text: "Alternative: ", bold: true }), new TextRun("If maximum quality is required and inference cost is not a concern, use 2048tok (MAE 0.627).")]
      }),

      // Models Produced
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Models Produced")] }),
      new Paragraph({ numbering: { reference: "bullet-list", level: 0 },
        children: [new TextRun({ text: "uplifting_v5_1024tok ", font: "Courier New", size: 20 }), new TextRun("- MAE 0.652")] }),
      new Paragraph({ numbering: { reference: "bullet-list", level: 0 },
        children: [new TextRun({ text: "uplifting_v5_2048tok ", font: "Courier New", size: 20 }), new TextRun("- MAE 0.627")] }),
      new Paragraph({ numbering: { reference: "bullet-list", level: 0 },
        children: [new TextRun({ text: "uplifting_v5_head_tail ", font: "Courier New", size: 20 }), new TextRun("- MAE 0.655 (recommended)")] }),
      new Paragraph({
        spacing: { before: 200 },
        children: [new TextRun({ text: "Location: ", bold: true }), new TextRun({ text: "research/embedding_vs_finetuning/models/", font: "Courier New", size: 20 })]
      })
    ]
  }]
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync("Context_Length_Research_Report.docx", buffer);
  console.log("Report generated: Context_Length_Research_Report.docx");
});
