import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

PDF_PATH = r"c:\Users\dell\Downloads\archive (1)\IEEE_CIS_Fraud_Detection_System_Architecture.pdf"

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
            self.drawString(54, 750, "IEEE-CIS Fraud Detection Engine — End-to-End System Architecture Document")
            self.setStrokeColor(colors.HexColor("#CBD5E0"))
            self.setLineWidth(0.5)
            self.line(54, 742, 612 - 54, 742)
            
        # Footer
        footer_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(612 - 54, 36, footer_text)
        self.drawString(54, 36, "System Architecture Specification — IEEE-CIS Fraud Detection Project")
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

    # Color Palette
    PRIMARY = colors.HexColor("#0F2C59")    # Deep Navy
    SECONDARY = colors.HexColor("#1F6E8C")  # Teal Accent
    DARK_TEXT = colors.HexColor("#1A1A1A")
    LIGHT_BG = colors.HexColor("#F8F9FA")
    BORDER_COLOR = colors.HexColor("#DEE2E6")
    HIGHLIGHT_BG = colors.HexColor("#E3F2FD")

    # Typography Styles
    style_title = ParagraphStyle(
        'DocTitle',
        parent=styles['Title'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=PRIMARY,
        alignment=0,
        spaceAfter=4
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
        fontSize=15,
        leading=19,
        textColor=PRIMARY,
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )

    style_h2 = ParagraphStyle(
        'H2',
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=SECONDARY,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    style_body = ParagraphStyle(
        'BodyText',
        fontName='Helvetica',
        fontSize=9,
        leading=13.5,
        textColor=DARK_TEXT,
        spaceAfter=5
    )

    style_code = ParagraphStyle(
        'CodeBlock',
        fontName='Courier',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#2B6CB0"),
        spaceAfter=4
    )

    story = []

    # Title Banner
    story.append(Paragraph("IEEE-CIS Financial Fraud Detection Engine", style_title))
    story.append(Paragraph("End-to-End System, Pipeline & Production Deployment Architecture Specification", style_subtitle))
    story.append(HRFlowable(width="100%", thickness=2, color=PRIMARY, spaceAfter=12))

    # SECTION 1: Architecture Blueprint
    story.append(Paragraph("1. High-Level Architecture Overview", style_h1))
    
    overview_text = (
        "The IEEE-CIS Fraud Detection Engine is built on a <b>Disk-Cached, Memory-Optimized Streaming ETL & Machine Learning Architecture</b>. "
        "It processes over 1.1 million financial transactions (1.3 GB raw CSVs) under strict physical RAM constraints (< 1.5 GB RAM cap) "
        "and delivers real-time fraud probability scores with an Out-of-Fold (OOF) ROC-AUC of <b>0.9421</b>."
    )
    story.append(Paragraph(overview_text, style_body))

    # Diagram Table
    arch_diagram_html = (
        "<b>SYSTEM COMPONENT FLOW:</b><br/><br/>"
        "<b>[ 1. Data Layer ]</b> Raw CSV Files (train/test transaction & identity tables)<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;│<br/>"
        "<b>[ 2. Streaming Ingestion ]</b> Chunked Reader (50k rows/batch) + Type Downcaster (float32/int8)<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;│<br/>"
        "<b>[ 3. ETL & Disk Cache ]</b> Isolated Train/Test Phase Pipeline ──> Joblib Binary Disk Cache<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;│<br/>"
        "<b>[ 4. Feature Pipeline ]</b> Time Extractor + Log Scaler + Categorical Encoder + Sparse Pruning (437 -> 278 cols)<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;│<br/>"
        "<b>[ 5. Model Engine ]</b> 3-Fold Stratified K-Fold CV + Low-Level xgb.DMatrix + tree_method='hist'<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;│<br/>"
        "<b>[ 6. Inference & Serve ]</b> Chunked Batch Prediction Engine (100k rows/batch) ──> submission.csv / REST API"
    )
    
    diagram_box = Table([[Paragraph(arch_diagram_html, style_code)]], colWidths=[504])
    diagram_box.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), HIGHLIGHT_BG),
        ('BOX', (0,0), (-1,-1), 1, SECONDARY),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
    ]))
    story.append(diagram_box)
    story.append(Spacer(1, 10))

    # SECTION 2: Component Architecture Breakdown
    story.append(Paragraph("2. Detailed Component Specifications", style_h1))

    comp_table_data = [
        [Paragraph("<b>Component</b>", style_body), Paragraph("<b>Architectural Responsibility</b>", style_body), Paragraph("<b>Tech Stack & Specification</b>", style_body)],
        [
            Paragraph("<b>Ingestion & Type Downcasting</b>", style_body),
            Paragraph("Reads raw CSV chunks, downcasts float64 -> float32 and int64 -> int8/int16 based on min/max bounds.", style_body),
            Paragraph("Pandas `read_csv(chunksize=50000)`, `np.iinfo`, `np.finfo` downcasting", style_body)
        ],
        [
            Paragraph("<b>Disk-Cache Storage</b>", style_body),
            Paragraph("Decouples train and test ingestion so they never occupy RAM simultaneously. Flushes memory via `gc.collect()`.", style_body),
            Paragraph("`joblib.dump(compress=1)` storing binary `.pkl` files in `_temp_cache/`", style_body)
        ],
        [
            Paragraph("<b>Feature Engineering</b>", style_body),
            Paragraph("Extracts DT_M, DT_W, DT_D, DT_hour, DT_day_of_week from TransactionDT; applies log1p scaling to transaction amounts.", style_body),
            Paragraph("NumPy vectorized math, `LabelEncoder` fitted over combined vocabularies", style_body)
        ],
        [
            Paragraph("<b>Feature Pruner</b>", style_body),
            Paragraph("Filters out 159 sparse Vesta (V) features with missing rate >= 50%, reducing feature dimension from 437 to 278.", style_body),
            Paragraph("Pandas `.isnull().mean()` missingness analysis (Matrix: 626 MB)", style_body)
        ],
        [
            Paragraph("<b>Model Training Engine</b>", style_body),
            Paragraph("Trains 3-Fold Stratified K-Fold CV using low-level `xgb.DMatrix` and histogram-based binning (`max_bin=128`).", style_body),
            Paragraph("XGBoost 3.2.0 C++ API (`tree_method='hist'`, `max_depth=6`, `n_jobs=2`)", style_body)
        ],
        [
            Paragraph("<b>Inference Pipeline</b>", style_body),
            Paragraph("Loads serialized JSON booster models and computes ensemble predictions in 100k row batches.", style_body),
            Paragraph("`xgb.Booster.load_model()`, vectorised chunk prediction", style_body)
        ],
    ]

    t_comp = Table(comp_table_data, colWidths=[110, 244, 150])
    t_comp.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT_BG]),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_comp)
    story.append(Spacer(1, 10))

    # SECTION 3: Memory Optimization Architecture
    story.append(Paragraph("3. Memory Optimization & Zero-OOM Strategy", style_h1))
    
    mem_text = (
        "Standard machine learning pipelines fail on 8 GB RAM machines due to simultaneous memory residency of "
        "train DataFrames (1.0 GB), test DataFrames (0.9 GB), and intermediate pandas/C++ consolidation buffers (1.2 GB). "
        "Our architecture solves this through four strict execution layers:"
    )
    story.append(Paragraph(mem_text, style_body))

    mem_table_data = [
        [Paragraph("<b>Optimization Layer</b>", style_body), Paragraph("<b>Standard Approach</b>", style_body), Paragraph("<b>Our Architecture</b>", style_body), Paragraph("<b>RAM Saved</b>", style_body)],
        [Paragraph("<b>Pipeline Coupling</b>", style_body), Paragraph("Train & Test in RAM together", style_body), Paragraph("Phase 1 Train -> Disk Cache -> Free RAM -> Phase 2 Test", style_body), Paragraph("<b>1.0 GB</b>", style_body)],
        [Paragraph("<b>Data Downcasting</b>", style_body), Paragraph("Default float64 / int64", style_body), Paragraph("Aggressive downcasting to float32 / int16 / int8", style_body), Paragraph("<b>450 MB</b>", style_body)],
        [Paragraph("<b>Feature Pruning</b>", style_body), Paragraph("All 437 features in RAM", style_body), Paragraph("Pruned 159 sparse V columns (>= 50% null)", style_body), Paragraph("<b>400 MB</b>", style_body)],
        [Paragraph("<b>Matrix & DMatrix API</b>", style_body), Paragraph("Pandas DataFrame -> High-level fit()", style_body), Paragraph("C-contiguous NumPy float32 -> low-level `xgb.DMatrix`", style_body), Paragraph("<b>1.2 GB (C++ buffer)</b>", style_body)],
    ]
    t_mem = Table(mem_table_data, colWidths=[100, 130, 184, 90])
    t_mem.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), SECONDARY),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT_BG]),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_mem)
    story.append(Spacer(1, 10))

    # SECTION 4: Model & CV Architecture
    story.append(Paragraph("4. Model & Cross-Validation Architecture", style_h1))

    model_spec_text = (
        "<b>Cross-Validation Scheme:</b> 3-Fold Stratified K-Fold. Preserves 3.50% fraud target ratio in every fold.<br/>"
        "<b>XGBoost Hyperparameters:</b> objective='binary:logistic', eval_metric='auc', learning_rate=0.05, max_depth=6, "
        "max_bin=128, subsample=0.8, colsample_bytree=0.8, tree_method='hist', n_jobs=2.<br/>"
        "<b>Model Serialization:</b> Fold models stored as `xgb_model_fold_0.json`, `xgb_model_fold_1.json`, `xgb_model_fold_2.json`."
    )
    story.append(Paragraph(model_spec_text, style_body))
    story.append(Spacer(1, 6))

    cv_results_data = [
        [Paragraph("<b>Fold ID</b>", style_body), Paragraph("<b>Validation Set Size</b>", style_body), Paragraph("<b>Validation ROC-AUC</b>", style_body), Paragraph("<b>Early Stop Iteration</b>", style_body)],
        [Paragraph("Fold 1", style_body), Paragraph("196,847 rows", style_body), Paragraph("<b>0.94442</b>", style_body), Paragraph("Iteration 499", style_body)],
        [Paragraph("Fold 2", style_body), Paragraph("196,847 rows", style_body), Paragraph("<b>0.93988</b>", style_body), Paragraph("Iteration 499", style_body)],
        [Paragraph("Fold 3", style_body), Paragraph("196,846 rows", style_body), Paragraph("<b>0.94208</b>", style_body), Paragraph("Iteration 499", style_body)],
        [Paragraph("<b>Overall OOF</b>", style_body), Paragraph("<b>590,540 rows</b>", style_body), Paragraph("<b>0.94208 ± 0.00185</b>", style_body), Paragraph("<b>3.84 Mins Total</b>", style_body)],
    ]
    t_cv = Table(cv_results_data, colWidths=[90, 130, 144, 140])
    t_cv.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT_BG]),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_cv)
    story.append(Spacer(1, 10))

    # SECTION 5: Production & Deployment Architecture
    story.append(Paragraph("5. Production & Cloud Deployment Architecture", style_h1))

    prod_diagram_html = (
        "<b>PRODUCTION MICROSERVICE DESIGN (Low Latency < 5ms):</b><br/><br/>"
        "<b>[ Client / Gateway ]</b> HTTP POST Request (Transaction Payload)<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;│<br/>"
        "<b>[ FastAPI Container ]</b> Dockerized microservice endpoint (`/predict`)<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;├──> <b>[ Redis Cache ]</b> O(1) fetch for pre-computed user profile & aggregate features<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;├──> <b>[ Treelite / ONNX Engine ]</b> Compiled native C decision tree model execution<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;└──> <b>[ Risk Decision Engine ]</b> Score Thresholding:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;• Score < 0.15 ──> <b>APPROVE</b> (Green)<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;• 0.15 <= Score < 0.60 ──> <b>STEP-UP 2FA / MANUAL REVIEW</b> (Yellow)<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;• Score >= 0.60 ──> <b>DECLINE / BLOCK</b> (Red)"
    )

    prod_box = Table([[Paragraph(prod_diagram_html, style_code)]], colWidths=[504])
    prod_box.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F0FDF4")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#16A34A")),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
    ]))
    story.append(prod_box)

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"SUCCESS: Architecture PDF generated at: {PDF_PATH}")

if __name__ == '__main__':
    build_pdf()
