#!/usr/bin/env python3
"""Generate a PDF report of top articles by weighted impact score using post-classifier."""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib import colors

# Import post-classifier
sys.path.insert(0, str(Path(__file__).parent.parent / 'filters' / 'uplifting' / 'v1'))
from post_classifier import UpliftingPostClassifierV1

def load_labeled_data(file_path: str) -> List[Dict]:
    """Load labeled articles from JSONL file."""
    articles = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                articles.append(json.loads(line))
    return articles

def calculate_impact_score(dimensions: Dict[str, int]) -> float:
    """Calculate weighted average impact score."""
    weights = {
        'agency': 1.0,
        'progress': 1.0,
        'collective_benefit': 1.5,
        'connection': 0.8,
        'innovation': 1.2,
        'justice': 1.3,
        'resilience': 1.0,
        'wonder': 0.9
    }

    total_weighted = 0
    total_weight = 0

    for dim, score in dimensions.items():
        if dim in weights:
            total_weighted += score * weights[dim]
            total_weight += weights[dim]

    return total_weighted / total_weight if total_weight > 0 else 0

def get_top_articles_by_weighted_impact(articles: List[Dict], n: int = 10) -> List[Tuple[Dict, float, float, str]]:
    """Get top N articles by weighted impact score using post-classifier."""
    classifier = UpliftingPostClassifierV1()
    scored_articles = []

    for article in articles:
        analysis = None
        if 'uplifting_analysis' in article and 'dimensions' in article['uplifting_analysis']:
            analysis = article['uplifting_analysis']['dimensions']
        elif 'analysis' in article:
            analysis = article['analysis']

        if not analysis:
            continue

        # Extract dimension scores
        dimensions = {}
        for dim, data in analysis.items():
            if isinstance(data, dict) and 'score' in data:
                dimensions[dim] = data['score']
            elif isinstance(data, int):
                dimensions[dim] = data

        if dimensions:
            # Calculate base impact score
            base_impact = calculate_impact_score(dimensions)

            # Apply post-classifier for weighted score and category
            weighted_score, weight_reason = classifier.calculate_weighted_score(article, base_impact)
            category, category_details = classifier.classify_emotional_tone(article)

            scored_articles.append((article, base_impact, weighted_score, category))

    # Sort by weighted score descending
    scored_articles.sort(key=lambda x: x[2], reverse=True)
    return scored_articles[:n]

def get_analysis_reasoning(article: Dict) -> str:
    """Extract analysis reasoning from article."""
    if 'uplifting_analysis' in article and 'reasoning' in article['uplifting_analysis']:
        return article['uplifting_analysis']['reasoning']
    return "No reasoning provided."

def get_category_color(category: str) -> str:
    """Get color hex code for category badge."""
    colors = {
        'celebrating_progress': '#27AE60',  # Green
        'inspiring_through_adversity': '#E67E22',  # Orange
        'heavy_but_inspiring': '#D35400',  # Dark orange
        'neutral_technical': '#95A5A6',  # Gray
        'unknown': '#BDC3C7'  # Light gray
    }
    return colors.get(category, '#95A5A6')

def get_category_label(category: str) -> str:
    """Get human-readable label for category."""
    labels = {
        'celebrating_progress': 'Celebrating Progress',
        'inspiring_through_adversity': 'Inspiring Through Adversity',
        'heavy_but_inspiring': 'Heavy But Inspiring',
        'neutral_technical': 'Neutral/Technical',
        'unknown': 'Mixed Emotions'
    }
    return labels.get(category, category.replace('_', ' ').title())

def generate_pdf_report(articles: List[Dict], output_file: str):
    """Generate a professional PDF report of top articles with post-classifier categorization."""

    # Get top 10 articles by weighted impact
    top_articles = get_top_articles_by_weighted_impact(articles, n=10)

    # Calculate category distribution
    category_counts = {}
    for _, _, _, category in top_articles:
        category_counts[category] = category_counts.get(category, 0) + 1

    # Create PDF
    doc = SimpleDocTemplate(
        output_file,
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=1*inch,
        bottomMargin=0.75*inch
    )

    # Container for the 'Flowable' objects
    story = []

    # Define styles
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#2C3E50'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )

    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.HexColor('#7F8C8D'),
        spaceAfter=20,
        alignment=TA_CENTER,
        fontName='Helvetica'
    )

    article_title_style = ParagraphStyle(
        'ArticleTitle',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#2980B9'),
        spaceAfter=8,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )

    impact_score_style = ParagraphStyle(
        'ImpactScore',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#27AE60'),
        spaceAfter=8,
        fontName='Helvetica-Bold'
    )

    url_style = ParagraphStyle(
        'URL',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#3498DB'),
        spaceAfter=8,
        fontName='Helvetica',
        wordWrap='LTR'
    )

    reasoning_style = ParagraphStyle(
        'Reasoning',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#34495E'),
        spaceAfter=20,
        alignment=TA_JUSTIFY,
        fontName='Helvetica',
        leading=14
    )

    category_style = ParagraphStyle(
        'Category',
        parent=styles['Normal'],
        fontSize=9,
        spaceAfter=6,
        fontName='Helvetica-Bold'
    )

    # Add title
    story.append(Paragraph("Top 10 Uplifting News Articles", title_style))
    story.append(Paragraph(
        "Selected by Weighted Impact Score with Emotional Tone Classification",
        subtitle_style
    ))
    story.append(Spacer(1, 0.2*inch))

    # Add description
    description = """
    These articles were selected from a curated dataset of 10,138 high-quality news articles,
    filtered for content quality and analyzed using AI across 8 dimensions. Each article's
    base impact score is weighted by emotional tone category to prioritize emotionally uplifting
    and actionable content. Technical/academic articles receive 0.5x weight multiplier.
    """
    story.append(Paragraph(description, reasoning_style))

    # Add category distribution
    cat_dist = "Category distribution: " + ", ".join([
        f"{get_category_label(cat)} ({count})"
        for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
    ])
    story.append(Paragraph(cat_dist, subtitle_style))
    story.append(Spacer(1, 0.3*inch))

    # Add each article
    for idx, (article, base_score, weighted_score, category) in enumerate(top_articles, 1):
        # Article number and scores
        score_text = f"<b>Article #{idx}</b> — Weighted Score: {weighted_score:.2f}/10"
        if abs(base_score - weighted_score) > 0.01:
            score_text += f" (base: {base_score:.2f})"
        story.append(Paragraph(score_text, impact_score_style))

        # Category badge
        category_label = get_category_label(category)
        category_color = get_category_color(category)
        story.append(Paragraph(
            f"<font color='{category_color}'>● {category_label}</font>",
            category_style
        ))

        # Article title
        title = article.get('title', 'Untitled')
        story.append(Paragraph(f"<b>{title}</b>", article_title_style))

        # URL (make it clickable)
        url = article.get('url', 'No URL provided')
        story.append(Paragraph(f"<link href='{url}' color='#3498DB'>{url}</link>", url_style))

        # Reasoning
        reasoning = get_analysis_reasoning(article)
        story.append(Paragraph(f"<i>Why this article matters:</i> {reasoning}", reasoning_style))

        # Add separator line
        if idx < len(top_articles):
            story.append(Spacer(1, 0.15*inch))
            story.append(HRFlowable(
                width="100%",
                thickness=0.5,
                color=colors.HexColor('#BDC3C7'),
                spaceAfter=0.15*inch
            ))

    # Add footer
    story.append(Spacer(1, 0.3*inch))
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.HexColor('#95A5A6'),
        alignment=TA_CENTER,
        fontName='Helvetica-Oblique'
    )
    story.append(Paragraph(
        "Generated with Post-Classifier v1.0 • Filtered 10k dataset • Gemini Flash Analysis",
        footer_style
    ))

    # Build PDF
    doc.build(story)
    print(f"PDF report generated: {output_file}")

if __name__ == '__main__':
    input_file = 'datasets/ground_truth_filtered_10k/labeled_articles.jsonl'
    output_file = 'reports/top_10_uplifting_articles_v2.pdf'

    print(f"Loading data from {input_file}...")
    articles = load_labeled_data(input_file)
    print(f"Loaded {len(articles):,} articles")

    print(f"Applying post-classifier and generating PDF report...")
    generate_pdf_report(articles, output_file)
    print("Done!")
