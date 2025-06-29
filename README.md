# ELKOfIndex

Because `Index of /` was never meant to be private.


**ELKOfIndex** is a machine learning-powered tool for identifying open directories on the web using data stored in Elasticsearch. It combines the capabilities of the ELK stack with supervised learning to train, evaluate, and classify web content as open directories or not. It includes support for minimal and raw web content, HTML export of misclassifications, and modular design for extensibility.

The solution uses a combination of `HashingVectorizer` and `SGDClassifier` from scikit-learn, optimized for performance across various text representations and parameter configurations. It supports batch training to handle large datasets with minimal memory usage and is designed for reproducibility and portability.


##  Features


-  Collects and stores web content from Elasticsearch.
  
-  Trains classifiers (Logistic Regression) using `scikit-learn`.
  
-  Evaluates model performance with rich classification reports.
  
-  Detects and reports false positives and false negatives.
  
-  Supports both minimal (`min`) and raw (`raw`) web content versions.
  
-  Outputs HTML reports for easy review of misclassified URLs.
  
-  Built with Python 3, Pandas, Scikit-learn, and Elasticsearch Python client.
  

---

##  Project Structure


```
ELKOfIndex/
│
├── data/
│ ├── evaluation.csv # Evaluation dataset with ids, labels, and URLs
│ ├── training.csv # Training dataset
│ ├── min_webcontent/ # Text content extracted from minimal HTML
│ └── raw_webcontent/ # Text content extracted from full/raw HTML
│
├── model/
│ ├── open_dir_classifier_min.pkl # Trained classifier on minimal content
│ └── open_dir_classifier_raw.pkl # Trained classifier on raw content
│
├── get_documents.py # Downloads and prepares data from Elasticsearch
├── train_opendir_model.py # Trains models for both 'min' and 'raw' content
├── evaluation.py # Runs evaluation and generates HTML report
├── config.py # Configuration for Elasticsearch
├── requirements.txt # Dependencies
└── README.md # You're here!
```

---

##  Usage

### 1. Install Dependencies

```
apt install python3-venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Elasticsearch

Set your Elasticsearch parameters in config.py:

```
ELASTICSEARCH_HOST = "your_host"
ELASTICSEARCH_PORT = 9200
ELASTICSEARCH_USER = "user"
ELASTICSEARCH_PASSWORD = "password"
URLS_INDEX = "your_index"
```

### 3. Get Documents

```
./get_documents.py
```

This script:

Extracts records from Elasticsearch that contain the words field.

Saves minimal and raw HTML content to files.

Creates a training/evaluation split in CSV format.


### 4. Train Models

```
./train_opendir_model.py
```

This will:

Load the training data.

Train one model for each web content type (min and raw).

Save them in the model/ directory.

### 5. Evaluate

```
./evaluation.py
```

This will:

Load the evaluation dataset.

Predict and compare results against true labels.

Print classification metrics.

Generate misclassified.html with lists of false positives and negatives.



HTML Report


The generated misclassified.html includes clickable links to:

False Positives (predicted open dir but it's not)

False Negatives (missed actual open dirs)

It distinguishes between models trained on min and raw content.

If you want to manually update isopendir variable you could use the following script in kibana.

```
POST crawler/_update_by_query
{
  "script": {
    "source": "ctx._source.isopendir = true",
    "lang": "painless"
  },
  "query": {
    "term": {
      "url": "https://youropendir.com"
    }
  }
}
```


---

## Classification Task

The model predicts whether an HTML page is an "open directory" based on textual features.

- **Target:** Binary classification
  - `0` = Not an open directory
  - `1` = Open directory
- **Input:** HTML content as plain text

---

## Training Pipeline

- Vectorization: `HashingVectorizer`
  - Avoids vocabulary storage (privacy-safe)
  - Fixed memory usage regardless of dataset size
  - Configurable `ngram_range` and `n_features`

- Classifier: `SGDClassifier` with `log_loss`
  - Supports `partial_fit` for batch training
  - Balanced class weights computed manually

- Training:
  - Batches of 10,000 HTML documents
  - Early filtering of empty or non-existent files
  - Evaluation on 10% validation sample per model

---

## Hyperparameter Tuning

We explored combinations of:

| Parameter     | Values Tested                   |
|---------------|---------------------------------|
| `ngram_range` | `(1, 1)`, `(1, 2)`               |
| `n_features`  | `2^16`, `2^17`, `2^18`, `2^19`, `2^20` |
| `content_type`| `min` (clean HTML), `raw` (original HTML) |

**Total trained models:** 20

---

## Summary of Results (Top Models)

| Content Type | N-gram Range | Features | F1 Score | Precision (class=1) | Recall (class=1) | Model Path |
|--------------|--------------|----------|----------|----------------------|------------------|------------|
| `min`        | (1, 2)       | 65,536   | **0.9801** | 0.9667               | 0.9940           | `model/open_dir_classifier_min_ngram_1_2_nfeat_16.pkl` |
| `min`        | (1, 2)       | 131,072  | 0.9799   | 0.9663               | 0.9940           | ...        |
| `raw`        | (1, 2)       | 262,144  | 0.9736   | 0.9540               | 0.9940           | ...        |
| `min`        | (1, 1)       | 262,144  | 0.9774   | 0.9614               | 0.9940           | ...        |

### Best Model (Overall)

- **Content Type**: `min`
- **n-gram range**: `(1, 2)`
- **Hashing dimensions**: `2^16 = 65,536`
- **F1 Score**: `0.9801`
- **Precision (class=1)**: `0.9667`
- **Recall (class=1)**: `0.9940`

---

## Privacy Considerations

This project uses `HashingVectorizer` instead of `TfidfVectorizer` to:
- Prevent the model from storing or exposing original terms from the training data
- Make the saved models **safe for public sharing or deployment**

---


Notes


The script handles URLs with commas and other special characters safely via CSV quoting.

Easily extensible to support more classifiers or web content preprocessing strategies.

Designed for batch processing large-scale crawled data via Elasticsearch [crawling2elk](https://github.com/rggassner/crawling2elk) project.



Author


Made with ❤️ by Rafael — Cybersecurity & ML Enthusiast

