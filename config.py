import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
    FAISS_INDEX_PATH = 'mental_health_cases.index'
    FAISS_METADATA_PATH = 'mental_health_cases_metadata.pkl'
    JSON_DATA_PATH = 'cases_new.json'
    MAX_RESULTS = 10
    DEFAULT_SIMILARITY_THRESHOLD = 0.19
    MAX_QUESTIONS = 8