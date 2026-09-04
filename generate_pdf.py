import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

PDF_PATH = r"c:\Users\dell\Downloads\archive (1)\IEEE_CIS_Fraud_Detection_Interview_Master_Guide.pdf"

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#718096"))
        
        # Header (pages 2+)
        if self._pageNumber > 1:
            self.drawString(54, 750, "IEEE-CIS Fraud Detection — AI Project Master Guide & Technical Interview Prep")
            self.setStrokeColor(colors.HexColor("#CBD5E0"))
            self.setLineWidth(0.5)
            self.line(54, 742, 612 - 54, 742)
            
        # Footer
        footer_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(612 - 54, 36, footer_text)
        self.drawString(54, 36, "Confidential — Prepared for Resume, Portfolio & Technical Interview Mastery")
        self.setStrokeColor(colors.HexColor("#CBD5E0"))
        self.setLineWidth(0.5)
        self.line(54, 48, 612 - 54, 48)
        self.restoreState()

def build_pdf():
    doc = SimpleDocTemplate(
        PDF_PATH,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom Color Palette
    PRIMARY = colors.HexColor("#1A365D")    # Dark Navy
    SECONDARY = colors.HexColor("#2B6CB0")  # Slate Blue
    ACCENT = colors.HexColor("#D69E2E")     # Warm Gold
    TEXT_COLOR = colors.HexColor("#2D3748") # Dark Gray
    BG_LIGHT = colors.HexColor("#F7FAFC")   # Light Gray/Off-White
    BORDER_COLOR = colors.HexColor("#E2E8F0")

    # Typography Styles
    style_title = ParagraphStyle(
        'DocTitle',
        parent=styles['Title'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=PRIMARY,
        alignment=0,
        spaceAfter=6
    )

    style_subtitle = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=SECONDARY,
        spaceAfter=15
    )

    style_h1 = ParagraphStyle(
        'H1',
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        textColor=PRIMARY,
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )

    style_h2 = ParagraphStyle(
        'H2',
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=SECONDARY,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    style_body = ParagraphStyle(
        'BodyText',
        fontName='Helvetica',
        fontSize=9.5,
        leading=14,
        textColor=TEXT_COLOR,
        spaceAfter=6
    )

    style_pitch = ParagraphStyle(
        'PitchText',
        fontName='Helvetica-Oblique',
        fontSize=10,
        leading=15,
        textColor=PRIMARY,
        spaceAfter=6
    )

    style_qa_q = ParagraphStyle(
        'QA_Q',
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=14,
        textColor=PRIMARY,
        spaceBefore=8,
        spaceAfter=3,
        keepWithNext=True
    )

    style_qa_a = ParagraphStyle(
        'QA_A',
        fontName='Helvetica',
        fontSize=9,
        leading=13.5,
        textColor=TEXT_COLOR,
        spaceAfter=8
    )

    story = []

    # Title Banner
    story.append(Paragraph("IEEE-CIS Financial Fraud Detection Engine", style_title))
    story.append(Paragraph("A-to-Z Technical Master Breakdown & Technical Interview Handbook", style_subtitle))
    story.append(HRFlowable(width="100%", thickness=2, color=SECONDARY, spaceAfter=15))

    # SECTION 1: Executive Project Summary
    story.append(Paragraph("1. Executive Project Summary", style_h1))
    
    # Pitch Box
    pitch_html = (
        "<b>Interview Pitch (\"Tell me about your project\" — 1-Minute Answer):</b><br/>"
        "<i>\"In my project, I engineered an end-to-end Financial Fraud Detection Engine using XGBoost on the "
        "IEEE-CIS benchmark dataset (1.1 Million rows, 430+ features). The main challenge was extreme class imbalance "
        "(3.5% fraud) and system memory constraints. I built a disk-cached ETL pipeline that downcasted data types, "
        "extracted time-delta features, and pruned 159 redundant sparse columns. Using 3-Fold Stratified K-Fold CV "
        "with XGBoost's histogram-based tree algorithm (tree_method='hist'), the model achieved an Out-Of-Fold (OOF) "
        "ROC-AUC of 0.9421 and a Fraud Precision of 0.92 in under 4 minutes while keeping RAM below 1.5 GB.\"</i>"
    )
    
    pitch_table = Table(
        [[Paragraph(pitch_html, style_pitch)]],
        colWidths=[504]
    )
    pitch_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#EBF8FF")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#3182CE")),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(pitch_table)
    story.append(Spacer(1, 10))

    # Workflow Breakdown Table
    story.append(Paragraph("End-to-End Pipeline Architecture", style_h2))
    pipeline_data = [
        [Paragraph("<b>Stage</b>", style_body), Paragraph("<b>Implementation Details</b>", style_body), Paragraph("<b>Key Technical Highlight</b>", style_body)],
        [Paragraph("1. Data Ingestion", style_body), Paragraph("Chunked CSV reading (50k rows/batch), float64 to float32, int64 to int8/int16 downcasting.", style_body), Paragraph("50% RAM reduction during load", style_body)],
        [Paragraph("2. Disk Caching", style_body), Paragraph("Sequential train/test processing, cached to disk via joblib.", style_body), Paragraph("Train & test never sit in RAM together", style_body)],
        [Paragraph("3. Feature Engineering", style_body), Paragraph("Extracted DT_hour, DT_day_of_week, DT_W, DT_M; Log(Amt+1) scaling; 43 label encodings.", style_body), Paragraph("Cyclic time-delta capture", style_body)],
        [Paragraph("4. Feature Selection", style_body), Paragraph("Pruned 159 Vesta columns with >= 50% missing values (437 -> 278 features).", style_body), Paragraph("Matrix memory dropped to 626 MB", style_body)],
        [Paragraph("5. Model Training", style_body), Paragraph("3-Fold Stratified K-Fold CV using low-level xgb.DMatrix and tree_method='hist'.", style_body), Paragraph("Eliminated C++ memory allocation errors", style_body)],
        [Paragraph("6. Test Inference", style_body), Paragraph("Chunked Booster predictions in 100k batches across 3 fold models.", style_body), Paragraph("506,691 test rows processed in seconds", style_body)],
    ]
    t_pipe = Table(pipeline_data, colWidths=[90, 260, 154])
    t_pipe.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), SECONDARY),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, BG_LIGHT]),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_pipe)
    story.append(Spacer(1, 15))

    # SECTION 2: Deep-Dive & Results Explanation
    story.append(Paragraph("2. Technical Deep-Dive & Performance Results", style_h1))
    
    metrics_data = [
        [Paragraph("<b>Metric</b>", style_body), Paragraph("<b>Achieved Value</b>", style_body), Paragraph("<b>Technical & Business Significance</b>", style_body)],
        [Paragraph("<b>OOF ROC-AUC</b>", style_body), Paragraph("<b>0.94208</b>", style_body), Paragraph("Exceptional separation between Fraud and Legitimate distributions (Fold 1: 0.944, Fold 2: 0.939, Fold 3: 0.942).", style_body)],
        [Paragraph("<b>Precision (Fraud)</b>", style_body), Paragraph("<b>0.92</b>", style_body), Paragraph("Out of 100 transactions flagged as fraud, 92 are actual fraud. Minimizes false declines for customers.", style_body)],
        [Paragraph("<b>Recall (Fraud)</b>", style_body), Paragraph("<b>0.48</b> (at 0.5 thresh)", style_body), Paragraph("Catches 48% of total fraud at 0.5 threshold. Lowering threshold (e.g. 0.15) increases recall to 80%+.", style_body)],
        [Paragraph("<b>Overall Accuracy</b>", style_body), Paragraph("<b>98.0%</b>", style_body), Paragraph("High overall accuracy; dataset contains 96.5% legitimate transactions.", style_body)],
        [Paragraph("<b>Training Latency</b>", style_body), Paragraph("<b>3.84 mins</b>", style_body), Paragraph("Fast training on standard CPU using histogram binning (tree_method='hist').", style_body)],
    ]
    t_met = Table(metrics_data, colWidths=[110, 94, 300])
    t_met.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, BG_LIGHT]),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_met)
    story.append(Spacer(1, 15))

    # SECTION 3: Technical Interview Preparation (Q&A)
    story.append(Paragraph("3. Technical Interview Question & Answer Bank", style_h1))
    story.append(HRFlowable(width="100%", thickness=1, color=PRIMARY, spaceAfter=10))

    # Group A
    story.append(Paragraph("Category A: Fundamental & Architectural Questions", style_h2))

    q1_a = (
        "<b>Q1: Why choose XGBoost over Deep Learning (Neural Networks) or Random Forest?</b><br/>"
        "<b>Answer:</b> On tabular data with high missingness and mixed types (categorical + continuous), "
        "Gradient Boosted Decision Trees (GBDTs) consistently outperform Deep Learning models. XGBoost handles missing values "
        "natively without imputation artifacts, and tree_method='hist' enables fast execution on CPU with minimal RAM footprint."
    )
    story.append(Paragraph(q1_a, style_qa_a))

    q2_a = (
        "<b>Q2: What is the difference between tree_method='hist' and tree_method='exact'?</b><br/>"
        "<b>Answer:</b> 'exact' evaluates every unique value as a split point, costing O(N * D) per node with high memory overhead. "
        "'hist' discretizes continuous features into fixed bins (max_bin=128), reducing split finding to O(bin_count * D). "
        "This cuts memory usage by >70% and speeds up training by 5x with zero loss in ROC-AUC."
    )
    story.append(Paragraph(q2_a, style_qa_a))

    q3_a = (
        "<b>Q3: What loss function is used, and how does XGBoost optimize it?</b><br/>"
        "<b>Answer:</b> Binary Cross-Entropy (Log-Loss): L(y, p) = -[y log(p) + (1-y) log(1-p)]. XGBoost computes a 2nd-order "
        "Taylor expansion of the loss, using 1st order gradients (g_i) and 2nd order hessians (h_i) to determine optimal split gains "
        "and leaf node weights."
    )
    story.append(Paragraph(q3_a, style_qa_a))

    # Group B
    story.append(Paragraph("Category B: Data & Preprocessing Questions", style_h2))

    q4_a = (
        "<b>Q4: How did you handle the extreme Class Imbalance (3.5% Fraud)?</b><br/>"
        "<b>Answer:</b> 1) Used 3-Fold Stratified K-Fold CV to preserve the 3.5% fraud ratio in every fold. "
        "2) Selected ROC-AUC and Precision-Recall as core evaluation metrics instead of Accuracy. "
        "3) Post-training probability threshold tuning to adjust Precision vs. Recall according to business cost functions."
    )
    story.append(Paragraph(q4_a, style_qa_a))

    q5_a = (
        "<b>Q5: How did you resolve Out-Of-Memory (OOM) errors during local execution?</b><br/>"
        "<b>Answer:</b> 1) Disk-cached train and test DataFrames separately using joblib so they were never in RAM simultaneously. "
        "2) Downcasted types (float64 -> float32, int64 -> int8/int16). "
        "3) Pruned 159 sparse Vesta columns with >= 50% missing values (dropping matrix size to 626 MB). "
        "4) Passed C-contiguous float32 NumPy matrices to low-level xgb.DMatrix to eliminate intermediate memory copies."
    )
    story.append(Paragraph(q5_a, style_qa_a))

    # Group C
    story.append(Paragraph("Category C: Deployment & System Design Questions", style_h2))

    q6_a = (
        "<b>Q6: How would you deploy this model for low-latency real-time inference?</b><br/>"
        "<b>Answer:</b> Package the feature transformations and exported XGBoost Booster JSON model into a FastAPI microservice "
        "containerized with Docker. Single transaction JSON payloads are transformed into a 1D NumPy float32 array, passed to "
        "booster.predict(), returning a fraud probability in under 5 milliseconds."
    )
    story.append(Paragraph(q6_a, style_qa_a))

    q7_a = (
        "<b>Q7: How would you scale inference for 10,000 requests/sec?</b><br/>"
        "<b>Answer:</b> 1) Convert model to Treelite or ONNX Runtime to compile decision trees into native C instructions (3-5x faster). "
        "2) Use Redis in-memory cache for O(1) user profile and categorical lookups. "
        "3) Implement dynamic micro-batching (50 requests/batch) for parallel execution."
    )
    story.append(Paragraph(q7_a, style_qa_a))

    # Group D
    story.append(Paragraph("Category D: Trick & Edge-Case Questions", style_h2))

    q8_a = (
        "<b>Q8: What if the dataset grows by 10x (e.g. 50 Million rows)?</b><br/>"
        "<b>Answer:</b> 1) Migrate pipeline to PySpark or Dask-XGBoost distributed cluster. "
        "2) Use XGBoost's external memory QuantileDMatrix iterator to stream Parquet chunks from S3/disk. "
        "3) Utilize Feast or Hopsworks Feature Store to decouple feature computation from model training."
    )
    story.append(Paragraph(q8_a, style_qa_a))

    q9_a = (
        "<b>Q9: How will you detect and handle Concept Drift in production?</b><br/>"
        "<b>Answer:</b> 1) Data Drift: Track Population Stability Index (PSI) and Wasserstein distance on key features (e.g. TransactionAmt). "
        "2) Performance Drift: Monitor 7-day rolling Precision/Recall against chargeback logs. "
        "3) Retraining: Trigger automated retraining (Airflow / MLflow) weekly or when PSI > 0.25."
    )
    story.append(Paragraph(q9_a, style_qa_a))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"SUCCESS: PDF generated at: {PDF_PATH}")

if __name__ == '__main__':
    build_pdf()
