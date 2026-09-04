"""
IEEE-CIS Fraud Detection - XGBoost (Local Version - Ultra Memory Disk-Cached)
Converted from xgboost_fraud_detection_colab.ipynb for local execution.
Optimized for systems with limited RAM (~8GB) and heavy background processes.

Key Strategy:
1. Disk-caching (joblib) for intermediate train/test DataFrames.
2. Sequential loading & merging so train & test are NEVER in RAM at the same time.
3. Disk-based model saving and chunked test inference.
"""

import sys
import os
import gc
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

print(f"Python Version:  {sys.version}")
print(f"Pandas Version:  {pd.__version__}")
print(f"XGBoost Version: {xgb.__version__}")

try:
    import psutil
    mem = psutil.virtual_memory()
    print(f"Total RAM: {mem.total / 1024**3:.1f} GB, Available: {mem.available / 1024**3:.1f} GB")
except ImportError:
    pass

# Paths
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(DATA_DIR, '_temp_cache')
PLOTS_DIR = os.path.join(DATA_DIR, 'plots')
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

train_trans_path = os.path.join(DATA_DIR, 'train_transaction.csv')
train_id_path = os.path.join(DATA_DIR, 'train_identity.csv')
test_trans_path = os.path.join(DATA_DIR, 'test_transaction.csv')
test_id_path = os.path.join(DATA_DIR, 'test_identity.csv')
sub_path = os.path.join(DATA_DIR, 'sample_submission.csv')

for fpath in [train_trans_path, train_id_path, test_trans_path, test_id_path, sub_path]:
    if not os.path.exists(fpath):
        print(f"ERROR: File not found: {fpath}")
        sys.exit(1)
print("All data files verified!")

# =============================================================================
# Helper Functions
# =============================================================================
def reduce_mem_usage(df, verbose=False):
    """Downcast numeric columns to reduce memory usage."""
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f'  Memory downcasted: {start_mem:.0f} MB -> {end_mem:.0f} MB ({100 * (start_mem - end_mem) / start_mem:.0f}% reduction)')
    return df


def load_csv_chunked(filepath, rename_cols=False, chunksize=50000):
    """Load CSV in chunks to minimize peak memory."""
    print(f"  Reading {os.path.basename(filepath)}...")
    chunks = []
    for i, chunk in enumerate(pd.read_csv(filepath, chunksize=chunksize, low_memory=False)):
        if rename_cols:
            chunk.columns = [col.replace('-', '_') for col in chunk.columns]
        chunk = reduce_mem_usage(chunk, verbose=False)
        chunks.append(chunk)
        rows = (i + 1) * chunksize
        if rows % 100000 == 0:
            print(f"    {rows:,} rows read...")
    df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()
    print(f"  -> Shape: {df.shape}, Memory: {df.memory_usage().sum() / 1024**2:.0f} MB")
    return df


def create_time_features(df):
    """Extract time-based features from TransactionDT."""
    df['DT_M'] = (df['TransactionDT'] / (3600 * 24 * 30)).astype(np.int8)
    df['DT_W'] = (df['TransactionDT'] / (3600 * 24 * 7)).astype(np.int8)
    df['DT_D'] = (df['TransactionDT'] / (3600 * 24)).astype(np.int16)
    df['DT_hour'] = ((df['TransactionDT'] / 3600) % 24).astype(np.int8)
    df['DT_day_of_week'] = ((df['TransactionDT'] / (3600 * 24)) % 7).astype(np.int8)
    return df


CAT_COLS = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain',
            'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
            'DeviceType', 'DeviceInfo'] + [f'id_{i:02d}' for i in range(12, 39)]

TRAIN_CACHE = os.path.join(TEMP_DIR, 'train_df.pkl')
TEST_CACHE = os.path.join(TEMP_DIR, 'test_df.pkl')
CAT_MAP_CACHE = os.path.join(TEMP_DIR, 'cat_map.pkl')

# =============================================================================
# PHASE 1: TRAIN Data Prep & Save to Disk
# =============================================================================
if not os.path.exists(TRAIN_CACHE):
    print("\n" + "="*60)
    print("PHASE 1: Preparing TRAIN data")
    print("="*60)
    train_trans = load_csv_chunked(train_trans_path)
    train_id = load_csv_chunked(train_id_path)
    
    print("\n  Merging train_transaction & train_identity...")
    train_df = train_trans.merge(train_id, on='TransactionID', how='left')
    del train_trans, train_id
    gc.collect()
    
    train_df = create_time_features(train_df)
    train_df['TransactionAmt_log'] = np.log1p(train_df['TransactionAmt']).astype(np.float32)
    train_df = reduce_mem_usage(train_df, verbose=True)
    
    print(f"  Caching train_df to disk ({train_df.shape})...")
    joblib.dump(train_df, TRAIN_CACHE, compress=1)
    del train_df
    gc.collect()
else:
    print(f"\nPHASE 1: TRAIN cache already exists at {TRAIN_CACHE}")

# =============================================================================
# PHASE 2: TEST Data Prep & Save to Disk
# =============================================================================
if not os.path.exists(TEST_CACHE):
    print("\n" + "="*60)
    print("PHASE 2: Preparing TEST data (TRAIN is cleared from RAM)")
    print("="*60)
    test_trans = load_csv_chunked(test_trans_path)
    test_id = load_csv_chunked(test_id_path, rename_cols=True)
    
    print("\n  Merging test_transaction & test_identity...")
    test_df = test_trans.merge(test_id, on='TransactionID', how='left')
    del test_trans, test_id
    gc.collect()
    
    test_df = create_time_features(test_df)
    test_df['TransactionAmt_log'] = np.log1p(test_df['TransactionAmt']).astype(np.float32)
    test_df = reduce_mem_usage(test_df, verbose=True)
    
    print(f"  Caching test_df to disk ({test_df.shape})...")
    joblib.dump(test_df, TEST_CACHE, compress=1)
    del test_df
    gc.collect()
else:
    print(f"\nPHASE 2: TEST cache already exists at {TEST_CACHE}")

# =============================================================================
# PHASE 3: Label Encoding Across Train & Test
# =============================================================================
if not os.path.exists(CAT_MAP_CACHE):
    print("\n" + "="*60)
    print("PHASE 3: Building Categorical Mappings")
    print("="*60)
    cat_values = {}
    
    print("  Collecting train categorical values...")
    train_df = joblib.load(TRAIN_CACHE)
    cat_cols_present = [col for col in CAT_COLS if col in train_df.columns]
    for col in cat_cols_present:
        cat_values[col] = set(train_df[col].astype(str).unique())
    del train_df
    gc.collect()
    
    print("  Collecting test categorical values...")
    test_df = joblib.load(TEST_CACHE)
    for col in cat_cols_present:
        if col in test_df.columns:
            cat_values[col].update(test_df[col].astype(str).unique())
    del test_df
    gc.collect()
    
    print("  Fitting LabelEncoders...")
    encoders = {}
    for col, vals in cat_values.items():
        le = LabelEncoder()
        le.fit(list(vals))
        encoders[col] = le
        
    joblib.dump(encoders, CAT_MAP_CACHE)
    print(f"  Encoded {len(encoders)} categorical columns")
    
    print("  Applying encodings to TRAIN...")
    train_df = joblib.load(TRAIN_CACHE)
    for col, le in encoders.items():
        if col in train_df.columns:
            train_df[col] = le.transform(train_df[col].astype(str)).astype(np.int16)
    train_df = reduce_mem_usage(train_df)
    joblib.dump(train_df, TRAIN_CACHE, compress=1)
    del train_df
    gc.collect()
    
    print("  Applying encodings to TEST...")
    test_df = joblib.load(TEST_CACHE)
    for col, le in encoders.items():
        if col in test_df.columns:
            test_df[col] = le.transform(test_df[col].astype(str)).astype(np.int16)
    test_df = reduce_mem_usage(test_df)
    joblib.dump(test_df, TEST_CACHE, compress=1)
    del test_df
    gc.collect()
else:
    print(f"\nPHASE 3: Encodings already completed ({CAT_MAP_CACHE})")

# =============================================================================
# PHASE 4: EDA Plots
# =============================================================================
print("\n" + "="*60)
print("PHASE 4: Generating EDA Plots")
print("="*60)

train_df = joblib.load(TRAIN_CACHE)

# Target Distribution Plot
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
sns.countplot(x='isFraud', data=train_df, ax=ax[0], palette=['#2ecc71', '#e74c3c'])
ax[0].set_title('Target Class Counts (0: Legitimate, 1: Fraud)', fontsize=14)
ax[0].set_ylabel('Count')

fraud_counts = train_df['isFraud'].value_counts(normalize=True) * 100
ax[1].pie(fraud_counts, labels=['Legitimate (0)', 'Fraud (1)'], autopct='%1.2f%%',
          colors=['#2ecc71', '#e74c3c'], startangle=90, explode=(0, 0.1))
ax[1].set_title('Target Class Percentage', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '01_target_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: plots/01_target_distribution.png")

print(f"  Total Transactions: {len(train_df):,}")
print(f"  Fraud Transactions: {train_df['isFraud'].sum():,} ({fraud_counts.iloc[1]:.2f}%)")

# Transaction Amount Distribution Plot
plt.figure(figsize=(12, 5))
sns.kdeplot(train_df[train_df['isFraud'] == 0]['TransactionAmt'], label='Legitimate', color='green', fill=True, clip=(0, 1000))
sns.kdeplot(train_df[train_df['isFraud'] == 1]['TransactionAmt'], label='Fraud', color='red', fill=True, clip=(0, 1000))
plt.title('Transaction Amount Distribution ($0 - $1000)', fontsize=14)
plt.xlabel('Transaction Amount ($)')
plt.ylabel('Density')
plt.legend()
plt.savefig(os.path.join(PLOTS_DIR, '02_transaction_amount_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: plots/02_transaction_amount_distribution.png")

# Prepare features & target (drop sparse V columns with >50% missing to optimize memory)
features_to_drop = ['TransactionID', 'isFraud', 'TransactionDT']
all_cols = [col for col in train_df.columns if col not in features_to_drop]

v_cols = [c for c in all_cols if c.startswith('V')]
non_v_cols = [c for c in all_cols if not c.startswith('V')]

# Keep V columns with < 50% missing values
v_nulls = train_df[v_cols].isnull().mean()
v_cols_keep = v_nulls[v_nulls < 0.50].index.tolist()

features = non_v_cols + v_cols_keep
print(f"  Feature Selection: {len(all_cols)} -> {len(features)} high-signal features (dropped {len(v_cols) - len(v_cols_keep)} sparse V columns)")

print("  Converting features to float32 NumPy matrix...")
X_mat = train_df[features].to_numpy(dtype=np.float32)
y = train_df['isFraud'].to_numpy(dtype=np.int8)
del train_df
gc.collect()

# =============================================================================
# PHASE 5: XGBoost Model Training (3-Fold Stratified K-Fold)
# =============================================================================
print("\n" + "="*60)
print("PHASE 5: XGBoost Training (3-Fold CV - Memory Optimized)")
print("="*60)
print(f"Feature matrix shape: {X_mat.shape} (Memory: {X_mat.nbytes / 1024**2:.0f} MB)")
print(f"Target vector shape:  {y.shape}")

N_SPLITS = 3
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_preds = np.zeros(len(y), dtype=np.float32)
feature_importance_df = pd.DataFrame()

xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'learning_rate': 0.05,
    'max_depth': 6,
    'max_bin': 128,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'missing': -999,
    'tree_method': 'hist',
    'device': 'cpu',
    'n_jobs': 2,
    'random_state': 42
}

print(f"XGBoost Parameters: {xgb_params}")
fold_auc_scores = []
models = []
start_time = time.time()

for fold, (train_idx, val_idx) in enumerate(skf.split(X_mat, y)):
    print(f"\n--- Fold {fold + 1} / {N_SPLITS} ---")
    X_train, y_train = X_mat[train_idx], y[train_idx]
    X_val, y_val = X_mat[val_idx], y[val_idx]

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features, missing=-999)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=features, missing=-999)
    del X_train, y_train, X_val, y_val
    gc.collect()

    bst = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=500,
        evals=[(dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=100
    )

    val_pred = bst.predict(dval)
    oof_preds[val_idx] = val_pred

    fold_auc = roc_auc_score(y[val_idx], val_pred)
    fold_auc_scores.append(fold_auc)
    print(f"Fold {fold + 1} ROC-AUC: {fold_auc:.5f}")

    # Save fold model to disk
    model_path = os.path.join(TEMP_DIR, f'xgb_model_fold_{fold}.json')
    bst.save_model(model_path)
    models.append(model_path)

    scores = bst.get_score(importance_type='gain')
    fold_importance = pd.DataFrame({
        'feature': list(scores.keys()),
        'importance': list(scores.values())
    })
    feature_importance_df = pd.concat([feature_importance_df, fold_importance], axis=0)

    del dtrain, dval, bst
    gc.collect()

overall_auc = roc_auc_score(y, oof_preds)
elapsed = time.time() - start_time
print(f"\n" + "="*60)
print(f"Training Finished in {elapsed/60:.2f} minutes")
print(f"Mean Fold ROC-AUC: {np.mean(fold_auc_scores):.5f} +/- {np.std(fold_auc_scores):.5f}")
print(f"Overall OOF ROC-AUC: {overall_auc:.5f}")
print("="*60)

# Clean train matrix from RAM
del X_mat, y
gc.collect()

# =============================================================================
# PHASE 6: Model Evaluation & Diagnostic Plots
# =============================================================================
print("\n" + "="*60)
print("PHASE 6: Generating Diagnostic Plots")
print("="*60)

# Reload target y for evaluation metrics
train_df_y = joblib.load(TRAIN_CACHE)['isFraud'].values

# ROC Curve
fpr, tpr, _ = roc_curve(train_df_y, oof_preds)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'XGBoost OOF ROC-AUC = {overall_auc:.5f}', color='darkorange', lw=2)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)', fontsize=12)
plt.ylabel('True Positive Rate (TPR)', fontsize=12)
plt.title('Out-Of-Fold ROC Curve', fontsize=14)
plt.legend(loc="lower right", fontsize=12)
plt.grid(True)
plt.savefig(os.path.join(PLOTS_DIR, '03_roc_curve.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: plots/03_roc_curve.png")

# Confusion Matrix
optimal_threshold = 0.5
oof_binary = (oof_preds >= optimal_threshold).astype(int)
cm = confusion_matrix(train_df_y, oof_binary)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Legitimate', 'Fraud'],
            yticklabels=['Legitimate', 'Fraud'])
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.title(f'Confusion Matrix (Threshold = {optimal_threshold})', fontsize=14)
plt.savefig(os.path.join(PLOTS_DIR, '04_confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: plots/04_confusion_matrix.png")

print("\n--- Classification Report ---")
print(classification_report(train_df_y, oof_binary, target_names=['Legitimate (0)', 'Fraud (1)']))

# Feature Importance
mean_importance = feature_importance_df.groupby('feature')['importance'].mean().reset_index()
top_features = mean_importance.sort_values(by='importance', ascending=False).head(25)
plt.figure(figsize=(10, 8))
sns.barplot(x='importance', y='feature', data=top_features, palette='viridis')
plt.title('Top 25 Most Important Features in XGBoost Model', fontsize=14)
plt.xlabel('Mean Feature Importance (Gain)', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '05_feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: plots/05_feature_importance.png")

del train_df_y, oof_preds, feature_importance_df
gc.collect()

# =============================================================================
# PHASE 7: Test Inference & Submission Generation
# =============================================================================
print("\n" + "="*60)
print("PHASE 7: Test Inference & Submission Generation")
print("="*60)

test_df = joblib.load(TEST_CACHE)
test_ids = test_df['TransactionID'].values
X_test_mat = test_df[features].to_numpy(dtype=np.float32)
del test_df
gc.collect()

print(f"Test feature matrix shape: {X_test_mat.shape} (Memory: {X_test_mat.nbytes / 1024**2:.0f} MB)")
test_preds = np.zeros(len(X_test_mat), dtype=np.float32)

for fold, m_path in enumerate(models):
    print(f"  Predicting with Fold {fold + 1} model...")
    bst = xgb.Booster()
    bst.load_model(m_path)
    
    # Predict in chunks of 100k rows to keep RAM super low
    chunk_size = 100000
    preds_fold = np.zeros(len(X_test_mat), dtype=np.float32)
    for start in range(0, len(X_test_mat), chunk_size):
        end = min(start + chunk_size, len(X_test_mat))
        dsub = xgb.DMatrix(X_test_mat[start:end], feature_names=features, missing=-999)
        preds_fold[start:end] = bst.predict(dsub)
        del dsub
        
    test_preds += preds_fold / N_SPLITS
    del bst
    gc.collect()

sub = pd.DataFrame({
    'TransactionID': test_ids,
    'isFraud': test_preds
})

sub_filename = os.path.join(DATA_DIR, 'submission.csv')
sub.to_csv(sub_filename, index=False)
print(f"\n  Saved: {sub_filename}")

print("\n--- Submission Sanity Check ---")
print(f"  Total Rows: {len(sub):,}")
print(f"  Null Count: {sub['isFraud'].isnull().sum()}")
print(f"  Min Pred:   {sub['isFraud'].min():.4f}")
print(f"  Max Pred:   {sub['isFraud'].max():.4f}")
print(f"  Mean Pred:  {sub['isFraud'].mean():.4f}")
print("\nFirst 10 rows:")
print(sub.head(10))

# Clean up temp cache
import shutil
try:
    shutil.rmtree(TEMP_DIR)
    print("\n  Cleaned temporary cache folder.")
except Exception as e:
    print(f"  Note: Could not remove temp folder: {e}")

print("\n" + "="*60)
print("SUCCESSFULLY COMPLETED!")
print("All diagnostic plots saved to: plots/")
print("Kaggle submission file saved to: submission.csv")
print("="*60)
