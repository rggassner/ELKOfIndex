#!venv/bin/python3
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report
import joblib
import html
import re

# --- Config ---
TEST_CSV = "data/evaluation.csv"
HTML_REPORT_PATH = "misclassified.html"
MODEL_DIR = Path("model")

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
    print(f"\n🚀 Evaluating model: {model_file.name} for content type: {content_type}")

    html_dir = Path(f"data/{content_type}_webcontent")

    # Load model
    vectorizer, model = joblib.load(model_file)

    # Prepare dataframe
    df = df_base.copy()
    df['html_path'] = df['id'].apply(lambda x: html_dir / x)
    df = df[df['html_path'].apply(lambda p: p.exists())].copy()
    df['text'] = df['html_path'].apply(safe_read_text)
    df = df[df['text'].str.strip().astype(bool)]

    # Predict
    X = vectorizer.transform(df['text'])
    y_true = df['isopendir']
    y_pred = model.predict(X)
    df['predicted'] = y_pred

    print(classification_report(y_true, y_pred))

    # Misclassifications
    false_positives = df[(df['isopendir'] == False) & (df['predicted'] == True)]
    false_negatives = df[(df['isopendir'] == True) & (df['predicted'] == False)]

    print(f"❌ False Positives: {len(false_positives)}")
    print(f"❌ False Negatives: {len(false_negatives)}")

    # Append to HTML report
    report_sections.append(generate_html_table(false_positives, f"{model_file.name} - False Positives (Predicted Open Dir, but it's NOT)"))
    report_sections.append(generate_html_table(false_negatives, f"{model_file.name} - False Negatives (Missed Real Open Dir)"))

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

print(f"\n💾 HTML report saved to {HTML_REPORT_PATH}")
