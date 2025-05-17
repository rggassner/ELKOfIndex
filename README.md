# ELKOfIndex

Because `Index of /` was never meant to be private.

**ELKOfIndex** is a machine learning-powered tool for identifying open directories on the web using data stored in Elasticsearch. It combines the capabilities of the ELK stack with supervised learning to train, evaluate, and classify web content as open directories or not. It includes support for minimal and raw web content, HTML export of misclassifications, and modular design for extensibility.

## 🔧 Features

- ✅ Collects and stores web content from Elasticsearch.
- 🧠 Trains classifiers (Logistic Regression) using `scikit-learn`.
- 📊 Evaluates model performance with rich classification reports.
- ❌ Detects and reports false positives and false negatives.
- 📁 Supports both minimal (`min`) and raw (`raw`) web content versions.
- 🌐 Outputs HTML reports for easy review of misclassified URLs.
- 🐍 Built with Python 3, Pandas, Scikit-learn, and Elasticsearch Python client.

---

## 📁 Project Structure

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

## 🚀 Usage

### 1. Install Dependencies

```
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


📊 HTML Report

The generated misclassified.html includes clickable links to:

False Positives (predicted open dir but it's not)

False Negatives (missed actual open dirs)

It distinguishes between models trained on min and raw content.


📌 Notes
The script handles URLs with commas and other special characters safely via CSV quoting.

Easily extensible to support more classifiers or web content preprocessing strategies.

Designed for batch processing large-scale crawled data via Elasticsearch crawling2elk project.


👨‍💻 Author

Made with ❤️ by Rafael — Cybersecurity & ML Enthusiast

