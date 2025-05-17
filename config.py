import os

# Elasticsearch connection settings
ELASTICSEARCH_HOST = "127.0.0.1"
ELASTICSEARCH_PORT = 9200
ELASTICSEARCH_USER = os.getenv("ELASTICSEARCH_USER", "elastic")
ELASTICSEARCH_PASSWORD = os.getenv("ELASTICSEARCH_PASSWORD", "yourpasswordhere")
ELASTICSEARCH_CA_CERT_PATH = None
VERIFY_CERTS = False

# Elasticsearch behavior
URLS_INDEX = 'crawler'
MAX_ES_RETRIES = 10
ES_RETRY_DELAY = 1
ES_BUCKET_SIZE = 1000

# Data output paths
DATA_DIR = "data"
MIN_CONTENT_DIR = os.path.join(DATA_DIR, "min_webcontent")
RAW_CONTENT_DIR = os.path.join(DATA_DIR, "raw_webcontent")
DATASET_PATH = os.path.join(DATA_DIR, "dataset.csv")
TRAIN_PATH = os.path.join(DATA_DIR, "training.csv")
EVAL_PATH = os.path.join(DATA_DIR, "evaluation.csv")
