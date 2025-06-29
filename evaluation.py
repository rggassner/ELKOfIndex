#!venv/bin/python3
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report
import joblib
import html
import re
import gc

# --- Config ---
TEST_CSV = "data/evaluation.csv"
HTML_REPORT_PATH = "misclassified.html"
MODEL_DIR = Path("model")
BATCH_SIZE = 1000

# --- Load and Prepare Base Data ---
df_base = pd.read_csv(TEST_CSV)

# --- HTML Report Content Accumulator ---
report_sections = ["<h1>Misclassified URLs Report</h1>"]

# --- Reusable: safe text read ---
def safe_read_text(file_path):
    try:
        return file_path.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return ""

# --- Reusable: HTML table generator ---
def generate_html_table(df, title):
    rows = "\n".join(
        f'<tr><td><a href="{html.escape(url)}" target="_blank" rel="noopener noreferrer">{html.escape(url)}</a></td></tr>'
        for url in df['url']
    )
    return f"""
    <h2>{html.escape(title)}</h2>
    <table border="1" cellpadding="5" cellspacing="0">
        <thead><tr><th>URL</th></tr></thead>
        <tbody>
            {rows}
        </tbody>
    </table>
    """

# --- Loop through all model files ---
for model_file in sorted(MODEL_DIR.glob("open_dir_classifier_*.pkl")):
    match = re.match(r"open_dir_classifier_(\w+)_ngram_\d_\d_nfeat_\d+.pkl", model_file.name)
    if not match:
        continue

    content_type = match.group(1)
    print(f"\n Evaluating model: {model_file.name} for content type: {content_type}")

    html_dir = Path(f"data/{content_type}_webcontent")

    # Load model
    vectorizer, model = joblib.load(model_file)

    # Prepare evaluation set
    df = df_base.copy()
    df['html_path'] = df['id'].apply(lambda x: html_dir / x)
    df = df[df['html_path'].apply(lambda p: p.exists())].copy()
    df = df.reset_index(drop=True)

    y_true_all = []
    y_pred_all = []
    misclassified_fp = []
    misclassified_fn = []

    # Evaluate in batches
    for i in range(0, len(df), BATCH_SIZE):
        batch = df.iloc[i:i + BATCH_SIZE].copy()
        batch['text'] = batch['html_path'].apply(safe_read_text)
        batch = batch[batch['text'].str.strip().astype(bool)]
        if batch.empty:
            continue

        X = vectorizer.transform(batch['text'])
        y_true = batch['isopendir'].astype(int).values
        y_pred = model.predict(X)

        y_true_all.extend(y_true)
        y_pred_all.extend(y_pred)

        batch['predicted'] = y_pred
        misclassified_fp.append(batch[(batch['isopendir'] == False) & (batch['predicted'] == True)])
        misclassified_fn.append(batch[(batch['isopendir'] == True) & (batch['predicted'] == False)])

        del batch, X, y_true, y_pred
        gc.collect()

    # Full evaluation metrics
    print(classification_report(y_true_all, y_pred_all))

    # Combine misclassified DataFrames
    false_positives = pd.concat(misclassified_fp, ignore_index=True)
    false_negatives = pd.concat(misclassified_fn, ignore_index=True)

    print(f" False Positives: {len(false_positives)}")
    print(f" False Negatives: {len(false_negatives)}")

    # Append to HTML report
    report_sections.append(generate_html_table(false_positives, f"{model_file.name} - False Positives (Predicted Open Dir, but it's NOT)"))
    report_sections.append(generate_html_table(false_negatives, f"{model_file.name} - False Negatives (Missed Real Open Dir)"))

    del false_positives, false_negatives
    gc.collect()

# --- Final HTML ---
html_content = f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\">
    <title>Misclassified URLs</title>
</head>
<body>
    {'\n'.join(report_sections)}
</body>
</html>
"""

with open(HTML_REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"\n HTML report saved to {HTML_REPORT_PATH}")
