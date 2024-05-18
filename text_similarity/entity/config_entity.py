from dataclasses import dataclass
from text_similarity.constants import *
import os

@dataclass
class DataIngestionConfig:
    def __init__(self):
        self.BUCKET_NAME = BUCKET_NAME
        self.ARCHIEVE_NAME = DATASET_ARCHIEVE_NAME
        self.INGESTION_ARTIFACTS_DIRECTORY : str = os.path.join(os.getcwd(), ARTIFACTS_DIRECTORY, INGESTION_ARTIFACTS_DIRECTORY)
        self.ARTIFACTS_DIRECTORY: str = os.path.join(self.INGESTION_ARTIFACTS_DIRECTORY, INGESTION_DATASET_1)
        self.NEW_ARTIFACTS_DIRECTORY: str = os.path.join(self.INGESTION_ARTIFACTS_DIRECTORY, INGESTION_DATASET_2)
        self.ARCHIVE_DIRECTORY = os.path.join(self.INGESTION_ARTIFACTS_DIRECTORY)
        self.ARCHIEVE_FILE_PATH = os.path.join(self.INGESTION_ARTIFACTS_DIRECTORY, self.ARCHIEVE_NAME)