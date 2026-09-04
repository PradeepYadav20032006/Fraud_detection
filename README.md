# 🛡️ IEEE-CIS Financial Fraud Detection Engine

[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost 3.2](https://img.shields.io/badge/XGBoost-3.2.0-FF6F00?style=flat&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![ROC-AUC 0.9421](https://img.shields.io/badge/ROC--AUC-0.9421-22c55e?style=flat&logo=scikitlearn&logoColor=white)](#-model-performance)
[![Precision 0.92](https://img.shields.io/badge/Precision-0.92-blue?style=flat)](#-model-performance)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end, high-performance **Machine Learning System** built to detect online e-commerce transaction fraud on the benchmark [IEEE-CIS Fraud Detection Dataset](https://www.kaggle.com/c/ieee-fraud-detection) (1.1 Million transactions, 430+ features).

Engineered with a **Disk-Cached, Low-Memory ETL Streaming Pipeline** that operates under strict RAM constraints (< 1.5 GB peak memory) while achieving an **Out-Of-Fold (OOF) ROC-AUC of 0.9421**.

---

## 📌 Executive Summary

- **Domain**: Financial Technology / E-Commerce Payment Security
- **Goal**: Predict binary fraud target (`isFraud`: 0 vs 1) on online card transactions.
- **Dataset Size**: 1,097,231 total rows (Train: 590,540 rows, Test: 506,691 rows, 434 features).
- **Class Imbalance**: **3.50% Fraud** (20,663 fraud vs 569,877 legitimate transactions).
- **Primary Model**: 3-Fold Stratified K-Fold XGBoost with histogram binning (`tree_method='hist'`).
- **Memory Footprint**: Capped below **1.5 GB RAM** (zero OOM risk on 8 GB RAM machines).
- **Training Speed**: **3.84 minutes** end-to-end on standard CPU.

---

## 📊 Model Performance & Results

| Metric | Score | Significance |
|---|---|---|
| **Overall OOF ROC-AUC** | **0.94208** | Exceptional separation of Fraud vs. Legitimate probability distributions. |
| **Fold 1 ROC-AUC** | **0.94442** | Validation Set (196,847 rows) |
| **Fold 2 ROC-AUC** | **0.93988** | Validation Set (196,847 rows) |
| **Fold 3 ROC-AUC** | **0.94208** | Validation Set (196,846 rows) |
| **Fraud Precision** | **0.92** | 92% of transactions flagged as fraud are actual fraud (minimal customer false alarms). |
| **Overall Accuracy** | **98.0%** | Overall correct predictions |

---

## 🏗️ System Architecture & Workflow

```
[ Raw CSV Files ] ──► [ Chunked Ingestion & Downcasting ] ──► [ Disk Cache (_temp_cache/*.pkl) ]
                                                                       │
[ submission.csv ] ◄── [ Batch Inference Engine ] ◄── [ 3-Fold XGBoost ] ◄───┴──► [ Sparse Feature Pruner (437 -> 278) ]
```

### Key Architectural Layers:
1. **Sequential Processing & Disk Caching**: Process train and test datasets independently and cache intermediate DataFrames to disk (`joblib`), ensuring `train` and `test` never sit in RAM simultaneously (**Saves ~1.0 GB RAM**).
2. **Data Type Downcasting**: Automatically downcast `float64` $\to$ `float32` and `int64` $\to$ `int8`/`int16` based on value bounds (**Saves ~450 MB RAM**).
3. **Sparse Feature Pruning**: Filter out 159 sparse `V` (Vesta) features with $\ge 50\%$ missing rates, reducing matrix dimensions from 437 to 278 high-signal features (**Saves ~400 MB RAM**).
4. **Low-Level `xgb.DMatrix` Integration**: Direct conversion to C-contiguous 32-bit NumPy matrices with low-level `xgb.DMatrix` and histogram binning (`max_bin=128`), eliminating C++ Quantile buffer allocation spikes (**Saves ~1.2 GB RAM**).

---

## 📁 Repository Structure

```
├── run_fraud_detection.py                  # Core production training & inference script
├── generate_pdf.py                         # PDF generator for Interview Master Guide
├── generate_architecture_pdf.py            # PDF generator for System Architecture Specs
├── xgboost_fraud_detection_colab.ipynb     # Original exploration notebook
├── IEEE_CIS_Fraud_Detection_Interview_Master_Guide.pdf  # PDF Handbook for Interviews
├── IEEE_CIS_Fraud_Detection_System_Architecture.pdf      # PDF Architecture Specifications
├── plots/                                  # Generated diagnostic & EDA plots
│   ├── 01_target_distribution.png
│   ├── 02_transaction_amount_distribution.png
│   ├── 03_roc_curve.png
│   ├── 04_confusion_matrix.png
│   └── 05_feature_importance.png
├── submission.csv                          # Kaggle submission predictions
├── .gitignore                              # Git ignore rules for datasets & venvs
└── README.md                               # Project documentation
```

---

## 🚀 Quickstart & Execution Guide

### 1. Prerequisites
- Python 3.10+
- Kaggle IEEE-CIS Fraud Detection dataset files in the root folder (`train_transaction.csv`, `train_identity.csv`, `test_transaction.csv`, `test_identity.csv`).

### 2. Setup Virtual Environment
```bash
python -m venv fraud_venv
# On Windows:
.\fraud_venv\Scripts\activate
# On Linux/macOS:
source fraud_venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install xgboost scikit-learn pandas numpy matplotlib seaborn joblib reportlab
```

### 4. Run the Pipeline
```bash
python run_fraud_detection.py
```

### 5. Generate Documentation PDFs
```bash
python generate_pdf.py
python generate_architecture_pdf.py
```

---

## 📄 Documentation PDFs Included

- 📄 [`IEEE_CIS_Fraud_Detection_Interview_Master_Guide.pdf`](IEEE_CIS_Fraud_Detection_Interview_Master_Guide.pdf) — Comprehensive 1-minute pitch, metric deep-dives, and 9 technical interview Q&As across 4 categories.
- 📄 [`IEEE_CIS_Fraud_Detection_System_Architecture.pdf`](IEEE_CIS_Fraud_Detection_System_Architecture.pdf) — End-to-end component flow diagrams, memory layer specifications, and FastAPI/Docker production deployment designs.

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
