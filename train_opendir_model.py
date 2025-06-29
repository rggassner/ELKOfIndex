#!venv/bin/python3
import pandas as pd
from pathlib import Path
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
import joblib
import numpy as np

# --- Parameters ---
DATASET_CSV = "data/training.csv"  # CSV file with columns 'id' and 'isopendir'
WEB_CONTENT_TYPES = ["min", "raw"]  # Two types of content: minified and raw HTML
NGRAM_OPTIONS = [(1, 1), (1, 2)]  # Try both unigrams and bigrams
N_FEATURES_POWERS = range(16, 21)  # We'll try 2**16 to 2**20 features
MODEL_DIR = Path("model")  # Folder to save models
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MAX_CHARS = 30000  # Max characters from HTML to consider
BATCH_SIZE = 10000  # Number of samples per batch
RANDOM_STATE = 42  # For reproducibility

# --- Load dataset ---
df_all = pd.read_csv(DATASET_CSV)  # Contains 'id' and 'isopendir' columns

# --- Read HTML content safely ---
def safe_read_text(path):
    try:
        return path.read_text(encoding='utf-8', errors='ignore')[:MAX_CHARS]
    except Exception:
        return ""

# --- Collect performance results ---
results = []

# --- Main loop: Try each content type (min/raw) ---
for content_type in WEB_CONTENT_TYPES:
    print(f"\n\U0001F310 Content type: {content_type}")
    html_dir = Path(f"data/{content_type}_webcontent")  # Folder with HTML files

    # Prepare dataframe with file paths and labels
    df = df_all.copy()
    df['html_path'] = df['id'].apply(lambda id: html_dir / id)
    df = df[df['html_path'].apply(lambda p: p.exists())].copy()
    df = df.reset_index(drop=True)

    y_all = df['isopendir'].astype(int).values  # Binary labels
    paths = df['html_path'].values  # Corresponding file paths
    classes = np.array([0, 1])  # Needed for partial_fit

    # Compute balanced class weights (very important if dataset is imbalanced)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_all)
    class_weight_dict = dict(zip(classes, class_weights))
    print("Class weights:", class_weight_dict)

    # Try each n_features and ngram_range combination
    for n_power in N_FEATURES_POWERS:
        n_features = 2**n_power

        for ngram_range in NGRAM_OPTIONS:
            print(f"\nTraining: content_type={content_type}, ngram={ngram_range}, n_features={n_features}")
            model_name = f"open_dir_classifier_{content_type}_ngram_{ngram_range[0]}_{ngram_range[1]}_nfeat_{n_power}.pkl"
            model_path = MODEL_DIR / model_name

            # HashingVectorizer: maps text to a fixed-size sparse vector without storing a vocabulary
            vectorizer = HashingVectorizer(
                n_features=n_features,
                alternate_sign=False,
                ngram_range=ngram_range,
                norm='l2',
                lowercase=True,
                encoding='utf-8',
                decode_error='ignore'
            )

            # SGDClassifier: very efficient for large, sparse datasets with partial_fit
            model = SGDClassifier(
                loss='log_loss',  # Logistic regression
                max_iter=5,       # Only 5 epochs per batch
                class_weight=class_weight_dict,
                random_state=RANDOM_STATE
            )

            # Shuffle data indices
            rng = np.random.default_rng(RANDOM_STATE)
            indices = np.arange(len(paths))
            rng.shuffle(indices)

            # --- Train in batches ---
            for i in range(0, len(indices), BATCH_SIZE):
                batch_idx = indices[i:i + BATCH_SIZE]
                batch_paths = paths[batch_idx]
                batch_y = y_all[batch_idx]

                # Read and filter non-empty documents
                batch_texts = [safe_read_text(Path(p)) for p in batch_paths]
                batch_texts, batch_y = zip(*[(txt, y) for txt, y in zip(batch_texts, batch_y) if txt.strip()])
                if not batch_texts:
                    continue

                # Vectorize and train
                X = vectorizer.transform(batch_texts)
                model.partial_fit(X, batch_y, classes=classes)
                print(f"Trained on batch {i // BATCH_SIZE + 1} ({len(batch_texts)} docs)")

            # --- Evaluate on 10% of data ---
            eval_size = int(0.1 * len(paths))
            eval_paths = paths[:eval_size]
            eval_y = y_all[:eval_size]
            eval_texts = [safe_read_text(Path(p)) for p in eval_paths]
            eval_texts, eval_y = zip(*[(txt, y) for txt, y in zip(eval_texts, eval_y) if txt.strip()])

            X_eval = vectorizer.transform(eval_texts)
            y_pred = model.predict(X_eval)

            # Gather performance metrics
            report = classification_report(eval_y, y_pred, output_dict=True)
            f1 = f1_score(eval_y, y_pred)

            print(f"F1 Score: {f1:.4f}")
            joblib.dump((vectorizer, model), model_path, compress=('xz', 3))
            print(f"Model saved to {model_path}")

            # Record result for later comparison
            results.append({
                "content_type": content_type,
                "ngram_range": str(ngram_range),
                "n_features": n_features,
                "f1_score": f1,
                "precision_1": report['1']['precision'],
                "recall_1": report['1']['recall'],
                "support_1": report['1']['support'],
                "model_path": str(model_path)
            })

# --- Build summary table ---
df_results = pd.DataFrame(results)
df_results = df_results.sort_values(by='f1_score', ascending=False)

print("\nSummary of Results (Top Models):")
print(df_results[[
    'content_type', 'ngram_range', 'n_features', 'f1_score', 'precision_1', 'recall_1', 'model_path'
]].to_string(index=False))

# Best model per n_features
print("\nBest model for each n_features value:")
best_per_nfeat = df_results.loc[df_results.groupby("n_features")['f1_score'].idxmax()]
print(best_per_nfeat[['n_features', 'f1_score', 'content_type', 'ngram_range', 'model_path']].to_string(index=False))

# Best overall model
best_overall = df_results.iloc[0]
print("\nOverall best model:")
print(best_overall.to_string())
