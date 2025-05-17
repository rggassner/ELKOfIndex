#! venv/bin/python3
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# --- Parameters ---
DATASET_CSV = "data/training.csv"
WEB_CONTENT_TYPES = ["min", "raw"]
MODEL_DIR = Path("model")
MODEL_DIR.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists

# --- Load full dataset once ---
df_all = pd.read_csv(DATASET_CSV)

# --- Function to safely read HTML ---
def safe_read_text(path):
    try:
        return path.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return ""

# --- Loop over content types ---
for content_type in WEB_CONTENT_TYPES:
    print(f"\n🚀 Training for content type: {content_type}")

    html_dir = Path(f"data/{content_type}_webcontent")
    model_path = MODEL_DIR / f"open_dir_classifier_{content_type}.pkl"

    df = df_all.copy()
    df['html_path'] = df['id'].apply(lambda id: html_dir / id)

    # Filter out missing files
    df = df[df['html_path'].apply(lambda p: p.exists())].copy()

    # Load and extract text
    df['text'] = df['html_path'].apply(safe_read_text)

    # Drop empty documents
    df = df[df['text'].str.strip().astype(bool)]

    # --- Train/Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['isopendir'], test_size=0.2, random_state=42)

    # --- Classifier Pipeline ---
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=20000)),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # --- Evaluation ---
    print(f"📊 Binary Classifier Report ({content_type}):")
    print(classification_report(y_test, y_pred))

    # --- Save model ---
    joblib.dump(pipeline, model_path)
    print(f"✅ Model saved to {model_path}")

