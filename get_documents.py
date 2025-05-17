#!venv/bin/python3
import os
import csv
import warnings
import urllib3
import time
import math
import random
import traceback

from elasticsearch import Elasticsearch
from config import *

# Disable warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore", category=Warning, message="Connecting to .* using TLS with verify_certs=False is insecure")

class DatabaseConnection:
    def __init__(self):
        es_config = {
            "hosts": [f"https://{ELASTICSEARCH_HOST}:{ELASTICSEARCH_PORT}"],
            "basic_auth": (ELASTICSEARCH_USER, ELASTICSEARCH_PASSWORD),
            "verify_certs": VERIFY_CERTS,
            "request_timeout": 30,
            "headers": {"Content-Type": "application/json"},
        }

        if ELASTICSEARCH_CA_CERT_PATH:
            es_config["ca_certs"] = ELASTICSEARCH_CA_CERT_PATH

        self.es = Elasticsearch(**es_config)
        self.con = self.es  # Compatibility alias

    def close(self):
        self.es.close()

    def search(self, *args, **kwargs):
        headers = kwargs.pop('headers', None)
        return self.es.options(headers=headers).search(*args, **kwargs) if headers else self.es.search(*args, **kwargs)

    def scroll(self, *args, **kwargs):
        headers = kwargs.pop('headers', None)
        return self.es.options(headers=headers).scroll(*args, **kwargs) if headers else self.es.scroll(*args, **kwargs)

def get_and_save_documents_with_non_empty_words():
    db = DatabaseConnection()

    # Create required directories
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MIN_CONTENT_DIR, exist_ok=True)
    os.makedirs(RAW_CONTENT_DIR, exist_ok=True)

    all_records = []

    try:
        query_body = {
            "query": {
                "exists": {
                    "field": "words"
                }
            },
            "size": ES_BUCKET_SIZE,
            "_source": ["_id", "min_webcontent", "raw_webcontent",
                        "isopendir", "url"]
        }

        response = db.search(
            index=URLS_INDEX,
            body=query_body,
            scroll="2m"
        )

        scroll_id = response.get('_scroll_id')
        results = response['hits']['hits']
        total_saved = 0

        while results:
            for doc in results:
                doc_id = doc["_id"]
                source = doc["_source"]
                min_web = source.get("min_webcontent", "")
                raw_web = source.get("raw_webcontent", "")
                url = source.get("url", "")
                isopendir = source.get("isopendir", False)
                if min_web:
                    all_records.append([doc_id, isopendir, url])

                    with open(os.path.join(MIN_CONTENT_DIR, doc_id), "w", encoding="utf-8") as f:
                        f.write(min_web)

                    if raw_web:
                        with open(os.path.join(RAW_CONTENT_DIR, doc_id), "w", encoding="utf-8") as f:
                            f.write(raw_web)

                    total_saved += 1

            print(f"Saved {len(results)} documents. Total so far: {total_saved}")

            scroll_response = db.scroll(scroll_id=scroll_id, scroll="2m")
            scroll_id = scroll_response.get('_scroll_id')
            results = scroll_response['hits']['hits']

        # Shuffle for fair training/eval split
        random.shuffle(all_records)

        split_index = math.floor(len(all_records) * 0.7)
        training_data = all_records[:split_index]
        evaluation_data = all_records[split_index:]

        with open(DATASET_PATH, 'w', newline='', encoding='utf-8') as dataset_file, \
             open(TRAIN_PATH, 'w', newline='', encoding='utf-8') as training_file, \
             open(EVAL_PATH, 'w', newline='', encoding='utf-8') as evaluation_file:

            headers = ['id', 'isopendir', 'url']
            csv.writer(dataset_file, quoting=csv.QUOTE_ALL).writerow(headers)
            csv.writer(training_file, quoting=csv.QUOTE_ALL).writerow(headers)
            csv.writer(evaluation_file, quoting=csv.QUOTE_ALL).writerow(headers)

            for row in all_records:
                csv.writer(dataset_file, quoting=csv.QUOTE_ALL).writerow(row)
            for row in training_data:
                csv.writer(training_file, quoting=csv.QUOTE_ALL).writerow(row)
            for row in evaluation_data:
                csv.writer(evaluation_file, quoting=csv.QUOTE_ALL).writerow(row)

        return total_saved, len(training_data), len(evaluation_data)

    except Exception as e:
        print(f"[ERROR] {e}")
        traceback.print_exc()
        return 0, 0, 0
    finally:
        db.close()

if __name__ == "__main__":
    start_time = time.time()
    saved, train_count, eval_count = get_and_save_documents_with_non_empty_words()
    elapsed = time.time() - start_time

    print(f"\n✅ Process completed in {elapsed:.2f} seconds")
    print(f"📦 Total documents saved: {saved}")
    print(f"🎓 Training set: {train_count} records (70%)")
    print(f"🧪 Evaluation set: {eval_count} records (30%)")
    print(f"\n📂 Files saved in:")
    print(f"  - {MIN_CONTENT_DIR}")
    print(f"  - {RAW_CONTENT_DIR}")
    print(f"  - {DATASET_PATH}")
    print(f"  - {TRAIN_PATH}")
    print(f"  - {EVAL_PATH}")

